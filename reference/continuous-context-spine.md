---
title: "Continuous Context Spine"
description: "How KV state accumulates and propagates through recursive delegation — the physical mechanics that make cascaded research work."
---

## The Spine

When an agent researches a topic, calls tools, reads pages, and then delegates sub-questions via `web_research`, the sub-agents don't start cold. They fork from the parent agent's branch. Every token the parent decoded — system prompt, tool calls, tool results, generated reasoning — is physically present in the sub-agents' KV cache at their original positions. The sub-agents attend over the parent's full state. No re-encoding. No lossy summarization. The parent's attention state IS the sub-agents' starting context.

The **spine** is the chain of KV state that persists through a recursive delegation sequence. Each delegation extends the spine. Each level's findings become part of the context for the next level.

```
Session trunk (prior conversation, if warm)
  ↓ fork
Shared root (system prompt + tool schemas — decoded once)
  ↓ fork
Agent A (task: "survey voice agent architectures")
  position 0-100: shared root prefix
  position 100-300: Agent A's suffix (system+user prompt)
  position 300-500: web_search result (prefilled via SETTLE)
  position 500-700: fetch_page result (survey article)
  position 700-750: Agent A's reasoning about what it read
  
  Agent A calls web_research(["Moshi architecture", "Sesame prosody"])
  ↓ fork (warm path — only turn separator prefilled)
  
  Inner shared root (forkHead=750, position=755)
    ↓ fork × 2
    
    Sub-agent B (task: "Moshi architecture")
      Attends over positions 0-755 (entire spine)
      Sees: system prompt + survey article + Agent A's reasoning
      Searches "Moshi" with vocabulary from the survey already in attention
      
    Sub-agent C (task: "Sesame prosody")  
      Attends over positions 0-755 (same spine)
      Searches "Sesame" with the same accumulated context
```

Sub-agents B and C don't re-discover the survey. They don't re-search for "voice agent architectures." The survey is at positions 500-700 in their KV — their `produceSync()` attends over it on every token. When Sub-agent B searches for "Moshi," the model's query formulation is informed by the survey content because those key-value pairs are in the attention window.

## Physical Mechanics

### Fork is metadata-only

`Branch.forkSync()` creates a new sequence ID that shares all KV cells up to the fork point. No tensor copy. The child reads the parent's K and V vectors at shared positions. Only tokens decoded after the fork point write new cells exclusive to the child.

Cost of forking: ~0. The child's unique overhead is its suffix tokens (system prompt reformatting + user message), typically 150-300 tokens.

### Prune frees only unique cells

When a sub-agent prunes, it releases cells from `forkHead` to `position` — its unique tokens above the fork point. The parent's cells (positions 0 to forkHead) are unaffected. Siblings forked from the same parent are unaffected.

```
Before prune:
  Agent A: 400 unique cells (positions 100-500)
  Sub-agent B: 300 unique cells (positions 755-1055)
  Sub-agent C: 300 unique cells (positions 755-1055)
  Total unique: 1000 cells

After pruning Sub B and Sub C:
  Agent A: 400 unique cells (unchanged)
  Total unique: 400 cells
  Freed: 600 cells for Agent A to continue
```

### The warm path saves the prefix

When `agentPool` is called with `parent: context.branch` (the DelegateTool's warm path), the inner pool's shared root forks from the calling agent's branch. It only prefills the turn separator (~5 tokens). The system prompt, tool schemas, and all prior tool results are already in the parent's KV — inherited for free.

Cold path (no parent): shared root at position 0, prefills full system prompt + tool schemas (~300-500 tokens).

Warm path (parent provided): shared root forks from parent, prefills separator only (~5 tokens). Saves ~300-500 tokens per recursive level.

### pruneOnReport and mid-pool KV recovery

When `pruneOnReport: true`, an agent's branch is pruned immediately after it calls `report()`. The agent's unique cells are freed while the pool is still running. Other agents gain headroom.

In recursive delegation: the inner pool's sub-agents report and prune individually. As each sub-agent finishes, its cells are freed for the remaining sub-agents. The inner pool shrinks as it progresses.

After the inner pool completes, `withSharedRoot`'s finally block prunes the inner root and all remaining descendants. The calling agent's branch is fully restored — its cells were never touched.

### SETTLE extends the branch

Tool results are prefilled into the calling agent's branch via `store.prefill([[branch, tokens]])`. The branch's `position` advances. The next `produceSync()` attends over the newly prefilled tokens. This is how tool results become part of the attention state — they're decoded into KV at the branch's current position, not re-serialized as text.

When `web_research` returns, DelegateTool's result (JSON-serialized sub-agent findings) is prefilled into the calling agent's branch during SETTLE. The calling agent resumes with sub-agent findings in its KV, at the position where the delegation result was prefilled.

## Two Spine Types

The system uses two distinct spine mechanisms that compose at different levels.

### KV spine — within a task's delegation

When an agent calls `web_research`/`research`, DelegateTool creates an inner pool with `parent: context.branch`. Sub-agents fork from the calling agent's branch and inherit its full KV state.

```
Position 0-100:    Shared root (system prompt + tool schemas)
Position 100-300:  Agent suffix (system+user prompt with task)
Position 300-500:  web_search result (prefilled via SETTLE)
Position 500-900:  fetch_page result (survey article content)
Position 900-950:  Agent reasoning + web_research call

  Agent calls web_research(["Moshi architecture", "Sesame prosody"])
  ↓ fork (warm path — only turn separator prefilled)

  Inner shared root (forkHead=950, position=955)
    ↓ fork × 2

    Sub-agent B (task: "Moshi architecture")
      Attends over positions 0-955 (entire KV spine)
      Searches with vocabulary from the survey already in attention

    Sub-agent C (task: "Sesame prosody")
      Attends over positions 0-955 (same spine)
```

The KV spine is the chain of decoded state within a single agent's branch. Sub-agents inherit it at zero cost. Delegation results are JSON-serialized and prefilled back into the calling agent's branch during SETTLE, extending the spine for subsequent tool calls.

### KV spine — between task stages (`extendRoot`)

The harness sequences tasks via a `reduce` combinator inside a `withSharedRoot` scope. A query-scoped `queryRoot` branch is created once; all tasks fork from it, and between tasks, findings are prefilled directly into `queryRoot` as user+assistant turns via the `extendRoot` helper.

```
queryRoot (position 0)
  └─ Task 1 pool forks from queryRoot (position 0)
       → Agent searches, fetches, reports → findings A
       → extendRoot: prefill [user: "Task 1", assistant: findings A] into queryRoot
         queryRoot advances to position 500

  └─ Task 2 pool forks from queryRoot (position 500)
       → Agent inherits Task 1's findings via KV share (zero re-encoding)
       → Searches for entities from Task 1 → findings B
       → extendRoot: prefill [user: "Task 2", assistant: findings B] into queryRoot
         queryRoot advances to position 1100

  └─ Task 3 pool forks from queryRoot (position 1100)
       → Agent inherits Tasks 1+2 findings via KV share
       → Deepens specific entities → findings C
```

Each task's pool forks from the EXTENDED `queryRoot`. Findings are decoded once into `queryRoot`; every subsequent fork shares them at zero marginal cost. No text re-encoding per agent.

### `extendRoot` is queue-and-drain serialized

`PoolContext.extendRoot` does NOT issue a native `store.prefill` from the orchestrator's fiber. It queues a request onto `pendingExtends` and suspends on an Effection [`action()`][action] until the tick loop's Phase 0 (SPAWN+EXTEND) drains it. The drain batches all pending extends with all pending fork suffixes into a single `store.prefill(prefillPairs)` call, then resolves each suspended action with its delta token count.

This matters in flat-mode DAGs where multiple sibling tasks complete near-simultaneously. Without queue-and-drain, two `extendRoot` calls firing from separate fibers would race into `store.prefill` and violate the single-fiber discipline (only the tick loop's fiber issues native model calls). The action-based rendezvous serializes them through the next Phase 0 without blocking the orchestrator's other work.

[action]: https://frontside.com/effection/api/v4/action

### Where to put the spine: harness `withSharedRoot` vs `agentPool({ systemPrompt })`

Spine extensions write to the pool's `spineRoot`. Two configurations decide which branch that is:

- **`agentPool` invoked WITHOUT `systemPrompt`** (the common case for chains that need cross-pool spine persistence). `spineRoot = warmParent` — the root the harness passed in via `parent:`. The harness's outer `withSharedRoot` owns this root for its full scope, so extensions persist past pool exit and a post-pool `useAgent({ parent: queryRoot })` forks the spine and attends to all chain extensions.
- **`agentPool` invoked WITH `systemPrompt`** (catalog mode). `agentPool` internally creates a *nested* `withSharedRoot` whose root carries the catalog. `spineRoot = inner root`. Extensions write to that inner root, which gets pruned at `agentPool` exit. A post-pool `useAgent({ parent: queryRoot })` would fork the OUTER queryRoot and find an empty branch — chain extensions disappeared with the inner root.

Two patterns both work; pick by where the synth runs:

```typescript
// Pattern A: synth as a DAG node INSIDE the pool (compare's pattern)
yield* withSharedRoot({ parent: session.trunk }, function*(queryRoot) {
  return yield* agentPool({
    orchestrate: dag(nodes),    // research → compare → synth as nodes
    systemPrompt: skillCatalog, // catalog mode is fine: synth's branch
                                // forks the inner spineRoot
    parent: queryRoot,
    tools, terminalTool: 'report',
  });
});

// Pattern B: synth as a separate useAgent AFTER the pool
yield* withSharedRoot(
  {
    parent: session.trunk,
    systemPrompt: skillCatalog,    // ← catalog on harness root
    toolsJson: toolkit.toolsJson,  // ← so spine extensions land here
  },
  function*(queryRoot) {
    yield* agentPool({
      orchestrate: chain(tasks),
      // NO systemPrompt — non-shared mode, spineRoot = queryRoot
      parent: queryRoot,
      tools, terminalTool: 'report',
    });
    // synth forks queryRoot and sees all chain extensions via attention
    return yield* useAgent({ parent: queryRoot, /* ... */ });
  },
);
```

See [Skill Catalog](/reference/skill-catalog) for the catalog convention and [Concurrency](/reference/concurrency#rootfmt--catalog-mode) for the `RootFmt` context that drives shared-mode behavior.

### How they compose

Within a task, the KV spine operates through delegation — sub-agents inherit the calling agent's KV state. Between tasks, `extendRoot` extends `queryRoot` with tokenized user+assistant turns. The synthesis pool forks from the fully-extended `queryRoot` and attends over the complete research chain. After synthesis, `queryRoot` is pruned (scope exit frees all intermediate KV). Only the final Q&A persists on the session trunk via `session.commitTurn`.

Within-query spine compounds evidence task-by-task via KV share. Cross-query spine stays minimal (one Q&A turn per query on the trunk).

## Session Trunk as Long-Term Spine

The Session trunk persists across queries in a multi-turn conversation:

1. `handleQuery` runs — agent pools fork from trunk, investigate, prune
2. All pools complete — branches pruned, trunk survives
3. `session.commitTurn(query, answer)` appends to trunk
4. Next query — pools fork from updated trunk, inherit prior conversation

The trunk is the long-term spine across queries. The textual spine is the medium-term spine across task stages within a single query. The KV spine is the short-term spine within a single task's delegation chain.

## Anti-patterns

### Wasteful prefill

If the inner pool's shared root re-prefills the system prompt when the warm path could fork from the parent, ~300-500 tokens are wasted per recursive level. Always pass `parent: context.branch` in the DelegateTool.

### Spine bloat from large tool results

Each tool result prefilled into an agent's branch permanently extends its KV spine. A fetch_page returning 2000 tokens consumes 2000 cells. Sub-agents at deeper delegation levels inherit the cost. Use scratchpad extraction (fork, attend, extract, prune) for large results that should be compressed before entering the spine.
