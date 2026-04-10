---
title: "Branch Lifecycle"
description: "Create, fork, prefill, produce/commit, prune — the full lifecycle of an inference branch."
---

`Branch` is the fundamental inference handle. It owns everything needed for independent generation: a KV cache sequence, sampler chain, logits snapshot, and perplexity tracker. Every agent, every scratchpad extraction, every diverge attempt -- all are branches. Understanding the branch lifecycle is prerequisite to understanding prefix sharing, multi-agent pools, and KV pressure.

## Branch as a KV sequence

A branch maps to a `seq_id` in the unified KV cache. The KV cache is a shared memory pool holding key/value vectors at indexed positions. A `seq_id` is a tag on those cells -- it says "this cell belongs to sequence N." Multiple seq_ids can tag the same cell (this is how fork works), and each seq_id can have cells that no other sequence shares (this is where branches diverge).

```
KV cache (simplified):

Position:  0    1    2    3    4    5    6    7    8
           ┌────┬────┬────┬────┬────┬────┬────┬────┬────┐
Cells:     │ k0 │ k1 │ k2 │ k3 │ k4 │ k5 │ k6 │ k7 │ k8 │
           │ v0 │ v1 │ v2 │ v3 │ v4 │ v5 │ v6 │ v7 │ v8 │
           └────┴────┴────┴────┴────┴────┴────┴────┴────┘
seq_ids:    0,1   0,1  0,1  0,1  0,1   1         0    0
                                       ↑         ↑    ↑
                                    branch 1   branch 0
                                    diverges   diverges
```

Positions 0-4 are shared between seq 0 and seq 1 (tagged with both). Position 5 belongs only to seq 1. Positions 7-8 belong only to seq 0. Each branch reads from the shared prefix and writes to its own divergent cells.

## `Branch.create` -- cold start

`Branch.create` allocates a new branch at a given position with its own sampler chain:

```typescript
const branch = Branch.create(ctx, position, params, nBatch, grammar);
```

- **`ctx`** -- `SessionContext` providing access to the KV cache and model.
- **`position`** -- Starting position in the KV cache. For a fresh conversation, this is 0 (cold start). For continuing a conversation, this is the current token count.
- **`params`** -- Sampling parameters (temperature, topP, topK, minP). Controls the sampler chain created for this branch.
- **`nBatch`** -- Per-branch batch size for `prefill()`. Defaults to the context's nBatch. Controls how many tokens are sent per `llama_decode` call during prefill.
- **`grammar`** -- Optional GBNF grammar string. When provided, `sample()` returns only grammar-valid tokens.

A newly created branch has no tokens in its KV cache. Call `prefill()` to decode prompt tokens before generating.

```typescript
// Cold start: create at position 0, prefill system prompt
const root = Branch.create(ctx, 0, { temperature: 0.5 });
const tokens = ctx.tokenizeSync(formattedPrompt);
await root.prefill(tokens);
// root.position is now tokens.length
```

This is the pattern `withSharedRoot` uses internally: create a root at position 0, prefill the shared system prompt, then pass the root to the body for forking.

## `fork` / `forkSync` -- O(1) branch creation

Forking creates a new branch that shares the parent's KV prefix:

```typescript
const child = parent.forkSync();
// child.position === parent.position
// child shares parent's KV cells at positions 0..parent.position-1
```

Under the hood, `forkSync` calls `llama_kv_self_seq_cp` which tags existing KV cells with a new `seq_id`. No KV tensor buffers are copied. The child starts at the parent's position and shares all cells up to that point. Only tokens decoded after the fork point allocate new cells exclusive to the child.

Fork clones the parent's:
- **Logits snapshot** -- The child can immediately call `produce()` or `sample()` and get the same distribution the parent would.
- **Sampler chain** -- Temperature, penalties, PRNG state are all cloned. Call `reseedSampler()` on children for stochastic diversity.
- **Grammar state** -- If the parent has an active grammar, the child inherits it and can diverge independently.
- **Perplexity tracker** -- The child starts with the parent's accumulated perplexity.

Fork does NOT clone:
- **Steer biases** -- Dynamic logit adjustments from `steer()` are per-branch and not inherited. Each branch manages its own steer state.

### Reseeding after fork

Without reseeding, all children forked from the same parent produce identical outputs -- they share the same PRNG state:

```typescript
const children = [];
for (let i = 0; i < 3; i++) {
  const child = parent.forkSync();
  child.reseedSampler(1000 + i);  // different seed per child
  children.push(child);
}
```

Only affects stochastic samplers (temperature > 0). Greedy sampling (temperature = 0) produces identical outputs regardless of seed.

### Fork trees

Branches form trees, not just flat lists. Fork from root for best-of-N. Fork from children for tree search. Fork from a draft for speculative decoding:

```
root (pos 0..100)
 ├── child A (pos 100..150)
 │    ├── grandchild A0 (pos 150..180)
 │    └── grandchild A1 (pos 150..175)
 └── child B (pos 100..160)
```

Each level shares the prefix from its parent. Grandchild A0 shares positions 0-99 with every branch, positions 100-149 with child A and grandchild A1, and owns positions 150-179 exclusively.

## `prefill` -- decode tokens into KV

`prefill` bulk-decodes tokens into the branch's KV cache and captures the final logits:

```typescript
await branch.prefill(tokens);
// branch.position += tokens.length
// branch.logits = distribution from last token
```

Tokens are chunked to `nBatch` (set at `Branch.create`). For 500 tokens with `nBatch=64`, this is 8 `llama_decode` calls (7x64 + 1x52).

`prefill` advances `position` by `tokens.length`. After prefill, the branch's logits snapshot holds the output distribution from the final decoded token -- ready for the next `produce()` or `sample()` call.

`prefill` does NOT accept tokens into the repeat-penalty window. It is for external tokens (user input, tool results) where repeat-penalty tracking is unwanted. For model-generated tokens, use the `produce()` / `commit()` protocol which does accept into the penalty window.

### Batched prefill via `BranchStore`

When multiple branches need prefill (e.g., agent setup or tool result injection), `BranchStore.prefill` packs all branches into a single `llama_decode` call:

```typescript
const store = new BranchStore(ctx);
await store.prefill([
  [branchA, systemPromptTokens],  // 200 tokens
  [branchB, shortQueryTokens],    //  12 tokens
  [branchC, longDocumentTokens],  // 800 tokens
]);
```

The store uses a two-pass bin-packing algorithm: first fit items into chunks of size `nBatch`, then dispatch each chunk via `decode_scatter`. Variable-length token arrays are handled automatically -- no padding, no wasted batch slots.

## `produce` / `commit` -- the generation protocol

Generation separates sampling from state advancement:

**`produce()` / `produceSync()`** -- Sample the next token from the branch's logits snapshot. Returns `{ token, text, isStop }`. Does NOT write to KV. Does NOT advance position. The branch is unchanged after produce -- you can inspect the result before deciding to commit.

**`commit(token)`** -- Accept the token into the sampler's repeat-penalty window, decode it into KV (writing one cell), and capture new logits. Advances position by 1. After commit, the branch is ready for the next `produce()`.

```typescript
// Manual produce/commit loop
while (true) {
  const { token, text, isStop } = branch.produceSync();
  if (isStop) break;
  await branch.commit(token);
  process.stdout.write(text);
}
```

The agent pool uses this protocol in its tick loop: PRODUCE calls `produceSync()` on all active agents, collecting tokens. COMMIT batch-decodes all produced tokens in a single `store.commit(entries)` call.

### Async iterator

For simpler cases, Branch provides an async iterator with commit-before-yield semantics:

```typescript
for await (const { token, text } of branch) {
  process.stdout.write(text);
}
```

Every yielded token is already committed -- written to KV and accepted into the sampler. Breaking out of the loop is clean. `useAgent()` uses this internally.

### Batched commit via `BranchStore`

When multiple branches produce tokens simultaneously (the agent pool's COMMIT phase), `BranchStore.commit` packs all tokens into a single `llama_decode` call:

```typescript
await store.commit([
  [branchA, tokenA],
  [branchB, tokenB],
  [branchC, tokenC],
]);
// One llama_decode() for all 3 tokens
```

This is the core of the agent pool's throughput: N agents, 1 GPU dispatch per tick. The GPU parallelizes across sequences within the batch.

## `prune` / `pruneSync` -- free KV cells

Pruning discards a branch's divergent KV entries and frees its handle:

```typescript
branch.pruneSync();
// branch.disposed === true
// Any subsequent call throws "Branch has been disposed"
```

Only cells divergent from the shared prefix are freed. Sibling branches are unaffected. If positions 0-99 are shared between branches A, B, and C, pruning branch A frees only A's cells at positions 100+. Cells at 0-99 remain tagged with B and C's seq_ids.

### RESTRICT mode

`pruneSync()` throws if the branch has children:

```typescript
// This throws:
parent.pruneSync();
// Error: Branch.prune(): branch 1 has 2 active child(ren) [2, 3].
// Prune children first or use pruneSubtree().
```

This prevents accidentally orphaning child branches. Prune children first, then the parent. Or use `pruneSubtreeSync()` for cascade delete.

### `pruneSubtreeSync` -- cascade delete

Prunes the branch and all its descendants in post-order (children first, then parent):

```typescript
root.pruneSubtreeSync();
// All descendants pruned, then root pruned
// root.disposed === true
```

`withSharedRoot` uses this in its finally block to clean up the entire shared prefix tree when the scope exits.

## The scope tree pattern

Branch lifecycle is managed through Effection's structured concurrency. The `ensure` + `pruneSync` pattern guarantees cleanup:

```typescript
function* setupAgent(parent: Branch, task: AgentTaskSpec, ctx: SessionContext) {
  const branch = parent.forkSync();
  yield* ensure(() => { if (!branch.disposed) branch.pruneSync(); });

  // ... setup agent state ...
  return { agent, suffixTokens };
}
```

`ensure` registers a cleanup callback that fires when the enclosing scope exits -- whether the scope completes normally, throws an error, or is cancelled by a parent. Every branch created by `setupAgent` has its own `ensure`. When the agent pool's resource scope exits, all ensure callbacks fire and all branches are pruned.

This makes orphaned-branch leaks structurally impossible. You cannot exit a scope without running its ensure callbacks. Effection enforces this at the runtime level: no operation may outlive its scope.

### `withSharedRoot` cleanup

```typescript
function* withSharedRoot(opts, body) {
  const root = Branch.create(ctx, 0, opts.params);
  await root.prefill(sharedTokens);

  try {
    yield* ScratchpadParent.set(root);
    return yield* body(root, sharedTokens.length);
  } finally {
    if (!root.disposed) root.pruneSubtreeSync();
  }
}
```

`pruneSubtreeSync` in the finally block handles the case where agent branches (children of root) still exist when the scope exits. Rather than requiring every consumer to prune children before root, the subtree prune cascades through the entire tree.

### `createAgent` cleanup

`createAgent` wraps `useAgent` in `scoped()`. When the scope exits, the `ensure()` callback prunes the root and all agent branches:

```typescript
function* createAgent(opts) {
  return yield* scoped(function*() {
    return yield* useAgent(opts);
    // On scope exit: ensure() prunes root subtree
  });
}
```

Whether generation succeeds, fails, or is cancelled, branches are pruned. For scratchpad extraction (`parent` provided), this means the temporary fork is always cleaned up and the parent's KV is untouched.

## Branch as the foundation

Everything in the agent framework builds on Branch:

- **Prefix sharing**: `withSharedRoot` creates a root branch, prefills the shared prompt, agents fork from it. See [Prefix Sharing](/reference/prefix-sharing).
- **Agent pools**: Each agent is a forked branch. The tick loop calls `produceSync()` and `store.commit()` on branch arrays. See [Concurrency Model](/reference/concurrency).
- **KV pressure**: `ContextPressure` reads `cellsUsed` which is incremented by branch decode operations. See [KV Pressure](/reference/kv-pressure).
- **Scratchpad extraction**: `createAgent({ parent })` forks a temporary branch for grammar-constrained extraction. See [Scratchpad Extraction](/reference/scratchpad-extraction).
- **Grammar constraining**: `setGrammar()` and `setGrammarLazy()` are branch methods. Grammar state is cloned on fork. See [Grammar & Tool Ordering](/reference/grammar-and-ordering).

The produce/commit separation, O(1) fork, and scope-tree cleanup are the primitives that make multi-agent generation on shared GPU compute possible.
