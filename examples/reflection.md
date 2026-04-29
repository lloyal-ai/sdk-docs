---
title: "Reflection"
description: "Research → draft → critique → revise — manual branch lifecycle with diverge-based verification."
---

Research, then draft a response, then critique the draft, then revise. The critic forks from the draft's live branch. The reviser forks from the critic's branch. No re-prompting — each phase continues from the physical KV state of the previous phase.

This example demonstrates **manual branch lifecycle**: `Branch.create`, `forkSync`, `buildUserDelta`, and `diverge` with a parent branch. It contrasts with [react-agent](/examples/react-agent), which delegates all branch management to `useAgent`.

**Source**: `examples/reflection/`

## Prerequisites

- A GGUF instruction-tuned model with native tool calling
- A GGUF reranker model
- A text corpus (directory or single file)

## Run it

```bash
npx tsx examples/reflection/main.ts ./models/Qwen3-4B-Q4_K_M.gguf \
  --corpus ./my-docs/ \
  --query "Explain the tradeoffs in this design"
```

Same flags as react-agent: `--reranker`, `--verbose`, `--trace`, `--jsonl`. Without `--query`, REPL.

## Code walkthrough

### `harness.ts` — four sequential phases

Each phase is a generator function. The next phase receives the previous phase's branch and continues from its live KV state:

```typescript
export function* handleQuery(query: string, opts: HarnessOpts): Operation<void> {
  const r  = yield* research(query, opts);
  const d  = yield* draft(r.findings, query, opts);
  const cr = yield* critique(d.branch, opts);   // fork from draft
  const v  = yield* revise(cr.branch, opts);    // fork from critique
}
```

Branches are physical KV state. Passing a branch forward means the next phase sees everything the previous phase generated — at zero re-encoding cost.

### Phase 1: Research

Same shape as the react-agent example — a single `useAgent` over the corpus tools:

```typescript
const agent = yield* useAgent({
  systemPrompt: RESEARCH.system,
  task: query,
  tools: [...opts.tools, reportTool],
  terminalTool: 'report',
  maxTurns: opts.maxTurns,
  policy: new DefaultAgentPolicy({ budget: { context: { softLimit: 2048 } } }),
});
const findings = agent.result ?? '(no findings)';
```

This phase produces findings text that feeds into the draft.

### Phase 2: Draft — manual branch, no agent

The draft does NOT use `useAgent`. It manages a branch directly:

```typescript
const ctx: SessionContext = yield* Ctx.expect();

const branch = Branch.create(ctx, 0, { temperature: 0.6 });
yield* ensure(() => { if (!branch.disposed) branch.pruneSync(); });

const messages = [
  { role: 'system', content: DRAFT.system },
  { role: 'user',   content: DRAFT.user.replace('{{findings}}', findings).replace('{{query}}', query) },
];
const { prompt } = ctx.formatChatSync(JSON.stringify(messages));
const tokens = ctx.tokenizeSync(prompt, true);
yield* call(() => branch.prefill(tokens));

let output = '';
for (;;) {
  const { token, text, isStop } = branch.produceSync();
  if (isStop) break;
  yield* call(() => branch.commit(token));
  output += text;
}
```

What's happening:

1. `Branch.create(ctx, 0, ...)` — fresh cold branch at position 0 with custom sampling.
2. `formatChatSync` + `tokenizeSync` — compile the prompt through the model's chat template.
3. `branch.prefill(tokens)` — fill the KV cache with the prompt.
4. Manual decode loop: `produceSync()` samples a token, `commit()` writes it to KV.
5. The branch is **returned**, not pruned — the critique phase needs the live KV state. The `ensure` cleanup fires only when the outer scope completes.

### Phase 3: Critique — fork + buildUserDelta + diverge

```typescript
const critiqueRoot = draftBranch.forkSync();
yield* ensure(() => { if (!critiqueRoot.disposed) critiqueRoot.pruneSync(); });

const delta = buildUserDelta(ctx, CRITIQUE.user);
yield* call(() => critiqueRoot.prefill(delta));

const result: DivergeResult = yield* diverge({
  parent: critiqueRoot,
  attempts: opts.critiqueAttempts,
  params: { temperature: 0.7 },
});

return { branch: result.best, output: result.bestOutput, tokenCount: result.totalTokens };
```

Three operations:

1. **`draftBranch.forkSync()`** — O(1) metadata copy. The fork shares all KV cells with the draft (system prompt, user message, full draft output already cached). Only tokens generated after the fork point allocate new cells.
2. **`buildUserDelta(ctx, CRITIQUE.user)`** — builds a token array for a warm user-turn injection (`getTurnSeparator() + formatChatSync() + tokenizeSync()`). Prefilled into the fork, the model sees the draft output followed by critique instructions as a natural conversation turn.
3. **`diverge({ parent, attempts: 3 })`** — generates 3 independent critique samples from the same starting point. Each attempt forks from `critiqueRoot`, generates at temperature 0.7, scores by perplexity. `diverge` returns the best sample (lowest perplexity) and its branch; non-winners are pruned.

The branch chain: `draft output → [fork] → critique instructions → [fork × 3] → best critique`. All three phases share the original KV prefix; only divergent tokens consume new cells.

### Phase 4: Revise — fork + manual decode

```typescript
const reviseBranch = critiqueBranch.forkSync();
yield* ensure(() => { if (!reviseBranch.disposed) reviseBranch.pruneSync(); });
const delta = buildUserDelta(ctx, REVISE.user);
yield* call(() => reviseBranch.prefill(delta));

let output = '';
for (;;) {
  const { token, text, isStop } = reviseBranch.produceSync();
  if (isStop) break;
  yield* call(() => reviseBranch.commit(token));
  output += text;
}
```

Same pattern as critique but a sequential decode (no divergence). The model sees the full chain — draft, critique, revision instructions — and generates a revised response that addresses the critique's specific feedback.

### Task prompts

Four markdown files in `tasks/`:

- **`research.md`** — same prompt as the react-agent example
- **`draft.md`** — synthesize research findings into a response. Templates: `{{findings}}`, `{{query}}`
- **`critique.md`** — evaluate accuracy, completeness, coherence, unsupported claims; quote and suggest improvements
- **`revise.md`** — incorporate valid criticism while keeping what worked

## How it differs from react-agent

| Aspect | react-agent | reflection |
|---|---|---|
| Branch management | Delegated to `useAgent` | Manual `Branch.create`, `forkSync`, `buildUserDelta` |
| Agent count | 1 | 1 research agent + 3 manual phases |
| Verification | None | `diverge` produces 3 critique samples, picks by perplexity |
| KV continuity | Branch pruned on scope exit | Branch chain persists across phases (draft → critique → revise) |
| Tool use | All phases | Only research; draft/critique/revise are text-only |

The architectural difference: reflection treats branches as first-class values. The draft branch is returned and passed to critique. The critique branch is returned and passed to revise. This is only possible because branches are physical KV state — forking is O(1) and the model sees everything that came before without re-processing.

## Customization

- **Critique diversity** — `CRITIQUE_ATTEMPTS` in `main.ts` (default: 3). More attempts = more KV during `diverge`, higher chance of catching weaknesses.
- **Sampling** — draft uses `temperature: 0.6` (conservative), critique `0.7` (exploratory). Adjust per phase.
- **Skip phases** — drop `critique` and `revise` from `handleQuery` for research-then-draft only.
- **Domain-specific critique** — edit `tasks/critique.md`'s evaluation criteria.

## Related pages

- [React Agent](/examples/react-agent) — single-agent baseline this example builds on
- [Branch Lifecycle](/reference/branch-lifecycle) — `Branch.create`, `forkSync`, KV cell sharing
- [Sessions](/learn/sessions) — `buildUserDelta` and warm continuation
- [Grammar & Tool Ordering](/reference/grammar-and-ordering) — how `diverge` samples are scored
