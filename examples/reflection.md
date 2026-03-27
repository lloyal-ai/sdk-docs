---
title: "Reflection"
description: "Research → draft → critique → revise — manual branch lifecycle with diverge-based verification."
---

Research, then draft a response, then critique the draft, then revise. The critic forks from the draft's live branch. The reviser forks from the critic's branch. No re-prompting -- each phase continues from the physical KV state of the previous phase.

This example demonstrates manual branch lifecycle management: `Branch.create`, `forkSync`, `buildUserDelta`, and `diverge` with a parent branch. It contrasts with the react-agent example, which delegates all branch management to `useAgentPool`.

**Source**: `examples/reflection/`

## Prerequisites

- A GGUF instruction-tuned model with tool calling support
- A GGUF reranker model for semantic search
- A text corpus (directory or single file)

## Run it

```bash
npx tsx examples/reflection/main.ts ./models/Qwen3-4B-Q4_K_M.gguf \
  --corpus ./my-docs/ \
  --query "Explain the tradeoffs in this design"
```

Options are the same as the react-agent example: `--reranker`, `--verbose`, `--trace`, `--jsonl`. Without `--query`, drops into a REPL.

## Code walkthrough

### `main.ts` -- identical structure to react-agent

The CLI entry point is structurally identical to the react-agent example: parse args, load models, load corpus, build tools, initialize agents, run REPL. The only difference is the `critiqueAttempts` parameter passed to the harness:

```typescript
const harnessOpts: HarnessOpts = {
  session,
  toolMap,
  toolsJson,
  events,
  maxTurns: MAX_TOOL_TURNS,
  critiqueAttempts: CRITIQUE_ATTEMPTS,  // 3
  trace,
};
```

This controls how many independent critique samples `diverge` produces for the verification step.

### `harness.ts` -- four-phase workflow

The harness implements four sequential phases. Each is a standalone generator function.

**`handleQuery` -- composition:**

```typescript
export function* handleQuery(query: string, opts: HarnessOpts): Operation<void> {
  const r = yield* research(query, opts);
  const d = yield* draft(r.findings, query, opts);
  const cr = yield* critique(d.branch, opts);
  const v = yield* revise(cr.branch, opts);
}
```

Each phase returns its branch and output. The next phase receives the previous phase's branch and builds on its KV state. This is the key insight: branches are physical KV state, and passing a branch forward means the next phase sees everything the previous phase generated.

**Phase 1: Research (agent pool, same as react-agent):**

```typescript
function* research(query: string, opts: HarnessOpts): Operation<{
  findings: string; pool: AgentPoolResult; timeMs: number
}> {
  const { result: pool } = yield* withSharedRoot(
    { systemPrompt: RESEARCH.system, tools: opts.toolsJson },
    function*(root) {
      const pool = yield* useAgentPool({
        tasks: [{ systemPrompt: RESEARCH.system, content: query, tools: opts.toolsJson, parent: root }],
        tools: opts.toolMap,
        maxTurns: opts.maxTurns,
        terminalTool: 'report',
        trace: opts.trace,
        pressure: { softLimit: 2048 },
        reportPrompt: REPORT,
        pruneOnReport: true,
      });
      return { result: pool };
    },
  );

  const findings = pool.agents[0]?.findings ?? '(no findings)';
  return { findings, pool, timeMs };
}
```

This is identical to the react-agent's `handleQuery`. One agent, corpus tools, `withSharedRoot` + `useAgentPool`. The research phase produces findings text that feeds into the next phase.

**Phase 2: Draft (manual branch, no agent pool):**

```typescript
function* draft(
  findings: string, query: string, opts: HarnessOpts,
): Operation<{ branch: Branch; output: string; tokenCount: number; timeMs: number }> {
  const ctx: SessionContext = yield* Ctx.expect();

  const branch = Branch.create(ctx, 0, { temperature: 0.6 });
  yield* ensure(() => { if (!branch.disposed) branch.pruneSync(); });

  const userContent = DRAFT.user
    .replace('{{findings}}', findings)
    .replace('{{query}}', query);

  const messages = [
    { role: 'system', content: DRAFT.system },
    { role: 'user', content: userContent },
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

  return { branch, output, tokenCount, timeMs };
}
```

This phase does NOT use `useAgentPool`. Instead it manages the branch directly:

1. `Branch.create(ctx, 0, { temperature: 0.6 })` -- creates a cold branch at position 0 with custom sampling parameters
2. `ctx.formatChatSync` + `ctx.tokenizeSync` -- formats the prompt through the model's chat template and tokenizes it
3. `branch.prefill(tokens)` -- fills the KV cache with the prompt tokens
4. Manual decode loop: `produceSync()` samples the next token, `commit()` writes it to KV
5. Returns the branch -- the caller needs the live KV state for the critique phase

The `ensure` block guarantees the branch is pruned if the scope exits early. But note that the branch is returned and used by the next phase -- `ensure` only fires when the outer scope completes.

**Phase 3: Critique (fork + buildUserDelta + diverge):**

```typescript
function* critique(
  draftBranch: Branch, opts: HarnessOpts,
): Operation<{ branch: Branch; output: string; tokenCount: number; timeMs: number }> {
  const ctx: SessionContext = yield* Ctx.expect();

  const critiqueRoot = draftBranch.forkSync();
  yield* ensure(() => { if (!critiqueRoot.disposed) critiqueRoot.pruneSync(); });
  const delta = buildUserDelta(ctx, CRITIQUE.user);
  yield* call(() => critiqueRoot.prefill(delta));

  const result: DivergeResult = yield* diverge({
    parent: critiqueRoot,
    attempts: opts.critiqueAttempts,
    params: { temperature: 0.7 },
  });

  return { branch: result.best, output: result.bestOutput, tokenCount: result.totalTokens, timeMs };
}
```

Three key operations happen here:

1. **`draftBranch.forkSync()`** -- O(1) metadata copy. The fork shares all KV cells with the draft branch (system prompt, user message, and the entire draft output are already in cache). Only tokens generated after the fork point allocate new cells.

2. **`buildUserDelta(ctx, CRITIQUE.user)`** -- builds a token array for a warm user turn injection. This is the canonical warm-continuation pattern: `getTurnSeparator() + formatChatSync() + tokenizeSync()`. The result is prefilled into the fork, so the model sees the draft output followed by the critique instructions as if it were a natural conversation turn.

3. **`diverge({ parent, attempts: 3 })`** -- generates 3 independent critique samples from the same starting point. Each attempt forks from `critiqueRoot`, generates at temperature 0.7, and scores the output by perplexity. `diverge` returns the best sample (lowest perplexity) and its branch. The non-winning branches are pruned automatically.

The branch chain at this point: `draft output -> [fork] -> critique instructions -> [fork x3] -> best critique`. All three phases share the original KV prefix. Only the divergent tokens after each fork point consume new cells.

**Phase 4: Revise (fork + buildUserDelta + manual decode):**

```typescript
function* revise(
  critiqueBranch: Branch, opts: HarnessOpts,
): Operation<{ output: string; tokenCount: number; timeMs: number }> {
  const ctx: SessionContext = yield* Ctx.expect();

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

  return { output, tokenCount, timeMs };
}
```

Same pattern as critique: fork from the critique branch, inject revision instructions via `buildUserDelta`, manual decode loop. The model sees the full chain -- draft, critique, revision instructions -- and generates a revised response that addresses the critique's specific feedback.

### Task prompts

Four markdown files in `tasks/`:

- **`research.md`** -- same research prompt as the react-agent example
- **`draft.md`** -- instructs the model to synthesize research findings into a comprehensive response. User content includes `{{findings}}` and `{{query}}` placeholders.
- **`critique.md`** -- evaluates accuracy, completeness, logical coherence, and unsupported claims. Instructs the critic to quote the parts it is critiquing and suggest concrete improvements.
- **`revise.md`** -- instructs the model to incorporate valid criticism while keeping what was already good.

## How it differs from react-agent

| Aspect | react-agent | reflection |
|--------|-------------|------------|
| Branch management | Delegated to `useAgentPool` | Manual `Branch.create`, `forkSync`, `buildUserDelta` |
| Agent count | 1 | 1 research agent + 3 manual phases |
| Verification | None | `diverge` produces 3 critique samples, selects by perplexity |
| KV continuity | Agent branch is pruned after `withSharedRoot` exits | Branch chain persists across phases (draft -> critique -> revise) |
| Tool use | All 4 phases | Only research phase uses tools; draft/critique/revise are text-only |

The key architectural difference is that reflection manages branches as first-class values. The draft branch is returned and passed to critique. The critique branch is returned and passed to revise. This is only possible because branches are physical KV state -- forking is O(1), and the model sees everything that came before without re-processing.

## Customization

**Adjust critique diversity**: Change `CRITIQUE_ATTEMPTS` in `main.ts` (default: 3). More attempts means more KV cells used during `diverge`, but a higher chance of catching weaknesses.

**Change sampling parameters**: The draft uses `temperature: 0.6` (conservative), critique uses `temperature: 0.7` (more exploratory). Adjust these in the respective phase functions.

**Skip phases**: Remove the `critique` and `revise` calls from `handleQuery` to get a research-then-draft pipeline. Or remove just `revise` for research-critique without revision.

**Change prompts**: Edit the markdown files in `tasks/`. The critique prompt's evaluation criteria (accuracy, completeness, coherence, unsupported claims) can be customized for your domain.

## Related pages

- [Branch Lifecycle](/reference/branch-lifecycle) -- `Branch.create`, `forkSync`, and how KV cells are shared
- [Thinking in lloyal](/learn/thinking-in-lloyal) -- the generator model and structured concurrency
- [Sessions](/learn/sessions) -- `buildUserDelta` and warm continuation
- [Grammar & Tool Ordering](/reference/grammar-and-ordering) -- how `diverge` samples are scored and selected
