---
title: "Supervisor"
description: "Grammar-constrained routing, heterogeneous agent pools, and warm trunk synthesis."
---

Classify a query, route it to specialist agents, run them in parallel, and synthesize their findings on the warm session trunk. This example demonstrates grammar-constrained routing decisions and heterogeneous agent pools where different agents receive different system prompts within the same `withSharedRoot`.

**Source**: `examples/supervisor/`

## Prerequisites

- A GGUF instruction-tuned model with tool calling support
- A GGUF reranker model for semantic search
- A text corpus (directory or single file)

## Run it

```bash
npx tsx examples/supervisor/main.ts ./models/Qwen3-4B-Q4_K_M.gguf \
  --corpus ./my-docs/ \
  --query "Compare the performance characteristics of these two approaches"
```

Options: `--reranker`, `--verbose`, `--trace`, `--jsonl`. Without `--query`, drops into a REPL.

## Code walkthrough

### `main.ts` -- CLI entry point

Structurally identical to the react-agent and reflection entry points: argument parsing, model loading, corpus ingestion, tool creation, REPL loop. The harness opts are simpler than reflection (no `critiqueAttempts`):

```typescript
const harnessOpts: HarnessOpts = {
  session,
  toolMap,
  toolsJson,
  events,
  maxTurns: MAX_TOOL_TURNS,
  trace,
};
```

### `harness.ts` -- three-phase workflow

The harness implements three phases: **Classify**, **Dispatch**, **Synthesize**.

**Specialist definitions:**

```typescript
const SPECIALISTS: Record<string, string> = {
  factual: 'Find specific facts, definitions, data points. Quote exact passages. Do not infer.',
  analytical: 'Trace reasoning chains. Identify causes and effects. Connect evidence to conclusions.',
  comparative: 'Identify entities being compared. List dimensions. Note similarities and differences.',
};
```

Each specialist type maps to a behavioral directive that is prepended to the user query. The model chooses which specialists to activate based on the query's needs.

**`handleQuery` -- composition:**

```typescript
export function* handleQuery(query: string, opts: HarnessOpts): Operation<void> {
  const cl = yield* classify(query, opts);
  const d = yield* dispatch(query, cl.routes, opts);
  const s = yield* synthesize(d.pool, cl.routes, query, opts);
}
```

Three sequential phases. The classify step produces routes (specialist types). Dispatch spawns specialist agents. Synthesize generates the final answer on the session trunk.

### Phase 1: Classify (grammar-constrained routing)

```typescript
function* classify(
  query: string, opts: HarnessOpts,
): Operation<{ routes: string[]; rationale: string; tokenCount: number; timeMs: number }> {
  const ctx: SessionContext = yield* Ctx.expect();

  const schema = {
    type: 'object',
    properties: {
      specialists: {
        type: 'array',
        items: { type: 'string', enum: ['factual', 'analytical', 'comparative'] },
        minItems: 1,
        maxItems: 3,
      },
      rationale: { type: 'string' },
    },
    required: ['specialists', 'rationale'],
  };
  const result = yield* agent({
    systemPrompt: CLASSIFY_PROMPT,
    task: query,
    schema,
  });
}
```

This is the central pattern: a JSON schema is passed to `agent`, which compiles it to a GBNF grammar and constrains every token. The model can only select from `['factual', 'analytical', 'comparative']` -- invalid specialist names are structurally impossible.

`agent()` is the single-agent primitive. It creates a branch, generates until stop, and returns an Agent with the raw output. Parse the JSON from `result.rawOutput`. If parsing fails, the code falls back to a default route:

```typescript
try {
  const parsed = result.parsed as { specialists: string[]; rationale: string };
  routes = parsed.specialists.filter(s => s in SPECIALISTS);
  if (!routes.length) routes = ['factual'];
} catch {
  routes = ['factual'];
  rationale = 'Defaulting to factual specialist';
}
```

Low temperature (0.3) keeps routing decisions consistent. The `rationale` field gives the model space to reason about its choice, improving classification accuracy even though the rationale is only used for debugging/display.

### Phase 2: Dispatch (heterogeneous agent pool)

```typescript
function* dispatch(
  query: string, routes: string[], opts: HarnessOpts,
): Operation<{ pool: AgentPoolResult; timeMs: number }> {
  const { result: pool } = yield* withSharedRoot(
    { systemPrompt: SPECIALIST.system, tools: opts.toolsJson },
    function*(root) {
      const tasks = routes.map((route, i) => ({
        systemPrompt: SPECIALIST.system,
        content: `${SPECIALISTS[route]}\n\nQuestion: ${query}`,
        tools: opts.toolsJson,
        parent: root,
        seed: Date.now() + i,
      }));

      const pool = yield* useAgentPool({
        tasks,
        tools: opts.toolMap,
        maxTurns: opts.maxTurns,
        terminalTool: 'report',
        trace: opts.trace,
        policy: new DefaultAgentPolicy({
          budget: { context: { softLimit: 2048 } },
          recovery: { prompt: REPORT },
        }),
        pruneOnReport: true,
      });

      return { result: pool };
    },
  );
}
```

The key insight is how heterogeneity works within a shared root:

1. **`withSharedRoot`** creates a KV prefix with the specialist system prompt and tool schemas. This prefix is identical for all agents.

2. **Each task gets a different `content`** -- the specialist behavioral directive is prepended to the query. A factual specialist sees `"Find specific facts, definitions, data points. Quote exact passages. Do not infer.\n\nQuestion: Compare the performance..."`, while a comparative specialist sees `"Identify entities being compared. List dimensions. Note similarities and differences.\n\nQuestion: Compare the performance..."`.

3. **`seed: Date.now() + i`** ensures each agent samples differently even with the same temperature, avoiding redundant investigation paths.

All agents share the same system prompt, the same tool schemas, and the same tool implementations. They differ only in their user message content and random seed. This is why `setupAgent` formats the full chat independently for each task -- agents need different user messages, but the same KV prefix is physically shared underneath.

If classify selects `['factual', 'comparative']`, dispatch spawns two agents. If it selects all three, three agents run in parallel within the same pool. The pool's tick loop batch-decodes all agents simultaneously.

### Phase 3: Synthesize (warm trunk generation)

```typescript
function* synthesize(
  pool: AgentPoolResult, routes: string[], query: string, opts: HarnessOpts,
): Operation<{ tokenCount: number; timeMs: number }> {
  const findings = pool.agents
    .map((a, i) => `[${routes[i]}] ${(a.result || '').trim()}`)
    .join('\n\n');

  yield* call(() => opts.session.prefillUser(
    `Specialist findings:\n${findings}\n\nSynthesize answering: ${query}`
  ));

  const trunk = opts.session.trunk!;
  for (;;) {
    const { token, text, isStop } = trunk.produceSync();
    if (isStop) break;
    yield* call(() => trunk.commit(token));
    tokenCount++;
  }
}
```

This phase differs from research and dispatch -- it generates directly on the session trunk rather than creating new branches:

1. **`session.prefillUser(...)`** appends a user turn to the existing session trunk. This is the warm continuation pattern: `buildUserDelta` under the hood constructs `[turn_separator, formatted_message, tokenized]` and prefills it into the trunk's KV cache. The model sees the entire prior conversation (if any) plus the new specialist findings.

2. **Manual decode loop on `trunk`** generates the synthesis directly into the persistent conversation state. When this completes, the trunk contains the full Q&A pair, ready for the next query in the REPL.

The specialist findings are labeled with their route type (`[factual]`, `[analytical]`, `[comparative]`) so the synthesis model knows which perspective each finding represents.

### Task prompts

Three markdown files in `tasks/`:

- **`classify.md`** -- describes the three specialist types and instructs the model to select 1-3 based on the question's needs. The `---` separator introduces the user prompt with a `{{query}}` placeholder.
- **`specialist.md`** -- generic research specialist prompt (same as react-agent's research.md). The behavioral differentiation comes from the content field, not the system prompt.
- **`synthesize.md`** -- instructs the model to combine specialist findings, integrating complementary information and preferring the most specific evidence when specialists overlap.

## How it differs from react-agent and reflection

| Aspect | react-agent | reflection | supervisor |
|--------|-------------|------------|------------|
| Routing | None | None | Grammar-constrained `agent()` |
| Agent count | 1 | 1 + manual phases | 1-3 (dynamic, based on classification) |
| Agent differentiation | N/A | N/A | Different `content` per specialist |
| Synthesis | Agent's own report | Draft -> critique -> revise chain | Warm trunk generation via `session.prefillUser` |
| Multi-turn | No | No | Yes (trunk persists between queries) |
| Branch management | `useAgentPool` | Manual (`Branch.create`, `forkSync`) | `agentPool` + `agent()` + `commitTurn` |

The supervisor example is the first to combine multiple framework primitives in a single pipeline: `agent()` for classification, `agentPool` for parallel specialist research, and `commitTurn` for trunk advancement. It also demonstrates warm trunk continuation -- the session trunk persists across REPL queries, giving follow-up questions full context.

## Customization

**Add specialist types**: Add entries to the `SPECIALISTS` record and update the `enum` in the classify schema. The grammar ensures the model can only select from the defined types.

**Change routing logic**: Modify the `classify` function's schema. You could add priority weights, required specialists, or minimum/maximum counts.

**Replace warm synthesis with an agent**: Swap the manual decode loop in `synthesize` for a `withSharedRoot` + `useAgentPool` call if you want the synthesizer to use tools (e.g., for fact-checking).

**Chain with verification**: Add a `diverge`-based verification step after synthesis, similar to the deep-research example's eval phase.

**Adjust temperature by specialist**: Currently all specialists share the same sampling parameters from the pool. Fork the tasks with explicit `params` overrides for per-specialist temperature control.

## Related pages

- [Grammar & Tool Ordering](/reference/grammar-and-ordering) -- how JSON schemas become GBNF grammars
- [Concurrency Model](/reference/concurrency) -- how the agent pool batch-decodes multiple specialists
- [Prefix Sharing](/reference/prefix-sharing) -- why the shared root matters for multi-agent efficiency
- [Sessions](/learn/sessions) -- `session.prefillUser` and warm trunk continuation
- [KV Pressure](/reference/kv-pressure) -- how pressure settings govern specialist lifecycle
