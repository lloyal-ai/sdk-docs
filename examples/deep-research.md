---
title: "Deep Research"
description: "The reference pipeline — planning, sequential KV-chained research, narrative synthesis, and convergence evaluation."
---

The reference implementation. Source-agnostic deep research across web, local corpus, or both -- with chain-shaped planning, sequential KV-chained research via the spine, narrative-arc synthesis, and multi-sample convergence evaluation.

This is the most complex example. It composes every major framework primitive: `agent`, `agentPool`, `reduce`, `Source`, and `Session.commitTurn`. If you want to understand how a production research pipeline works end to end, this is the code to study.

**Source**: `examples/deep-research-web/`

## Prerequisites

- A GGUF instruction-tuned model with tool calling support
- A GGUF reranker model for semantic search
- At least one source: a `TAVILY_API_KEY` environment variable for web research, a `--corpus` path for local files, or both

## Run it

Three source configurations:

```bash
# Web only (requires Tavily API key)
TAVILY_API_KEY=tvly-... npx tsx examples/deep-research-web/main.ts \
  ./models/Qwen3-4B-Q4_K_M.gguf \
  --query "What are the health effects of intermittent fasting?"

# Corpus only (local files)
npx tsx examples/deep-research-web/main.ts \
  ./models/Qwen3-4B-Q4_K_M.gguf \
  --corpus ./my-docs/ \
  --query "What does this codebase do?"

# Both (web + corpus)
TAVILY_API_KEY=tvly-... npx tsx examples/deep-research-web/main.ts \
  ./models/Qwen3-4B-Q4_K_M.gguf \
  --corpus ./my-docs/ \
  --query "How does this project compare to the state of the art?"
```

Without `--query`, drops into an interactive REPL with multi-turn support. Follow-up questions continue from the prior conversation's KV state.

Options:

- `--reranker <path>` -- custom reranker model path
- `--trace` -- emit a JSONL trace file (`trace-<timestamp>.jsonl`) with every prompt, tool call, and branch event
- `--verbose` -- show stderr from the inference backend
- `--jsonl` -- machine-readable output (run once, no REPL)
- `--findings-budget <chars>` -- maximum character budget for reranked passages passed to synthesis (default: 4000)

## Code walkthrough

### `main.ts` -- source assembly

The entry point's distinguishing feature is source assembly. Instead of building tools directly, it constructs `Source` instances and passes them to the harness.

**Source configuration:**

```typescript
const sources: Source<SourceContext, Chunk>[] = [];

if (corpusDir) {
  const resources = loadResources(corpusDir);
  const chunks = chunkResources(resources);
  sources.push(new CorpusSource(resources, chunks));
}

if (hasTavily) {
  sources.push(new WebSource(new TavilyProvider()));
}
```

`CorpusSource` provides local file tools (`search`, `grep`, `read_file`). `WebSource` provides web tools (`web_search`, `fetch_page`). Both implement the same `Source` interface — the harness calls `agentPool()` with `source.tools` for each, using source-specific prompts and recursion config. Orchestration lives in the harness, not the source.

Source order matters: each source's tools are available to all research tasks. The convention is corpus first (fast, local) then web (slower, broader).

**Concurrency parameters:**

```typescript
const VERIFY_COUNT = 3;    // independent samples for convergence eval
const MAX_TOOL_TURNS = 10; // tool calls before forced report

const ctx: SessionContext = yield* call(() =>
  createContext({
    modelPath,
    nCtx,
    nSeqMax: Math.max(AGENT_COUNT, VERIFY_COUNT) * 4 + 3,
    typeK: "q4_0",
    typeV: "q4_0",
  }),
);
```

`nSeqMax` is sized to accommodate the maximum number of concurrent branches: research agent per task, synthesis, and verify samples.

**Trace support:**

```typescript
const traceWriter = trace
  ? new JsonlTraceWriter(fs.openSync(`trace-${Date.now()}.jsonl`, 'w'))
  : undefined;
const { session, events } = yield* initAgents<WorkflowEvent>(ctx, { traceWriter });
```

When `--trace` is passed, a `JsonlTraceWriter` captures every prompt, completion, tool call, branch create/prune, and source bind event to a JSONL file. This is passed to `initAgents` and propagated through the `Trace` context to all framework primitives.

### `harness.ts` -- the full pipeline

The harness is structured as a set of generator functions composed by `handleQuery`. Task prompts and Eta templates are loaded at module level:

```typescript
const PLAN = loadPrompt('plan');         // --- separator → { system, user }
const FALLBACK = loadPrompt('fallback'); // cold-start root system prompt
const VERIFY = loadPrompt('verify');
const EVAL = loadPrompt('eval');
const RECOVERY = loadPrompt('recovery'); // grammar extraction on agent drop
const SYNTHESIZE = loadPrompt('synthesize');
const CORPUS_WORKER_TEMPLATE = loadTemplate('corpus-worker');  // raw Eta string
const WEB_WORKER_TEMPLATE = loadTemplate('web-worker');        // raw Eta string
```

Two loader types: `loadPrompt` splits on `---` to produce `{ system, user }` for task prompts. `loadTemplate` returns raw Eta template strings that are rendered at call time via `renderTemplate` with runtime context (tools, maxTurns, date, taskIndex).

### Stage 1: Plan (PlanTool + intent routing)

```typescript
const planTool = new PlanTool({ prompt: PLAN, session, maxQuestions: 10 });
const planContext = context
  ? `Today's date: ${today}\n\n${context}`
  : `Today's date: ${today}`;
const plan = (yield* planTool.execute({ query, context: planContext })) as PlanResult;
const r = route(plan, query, opts.maxTurns);
```

`PlanTool` is a grammar-constrained generator that decomposes the query into a SEQUENCE of research tasks with intent classification. The grammar schema constrains output to:

```json
{
  "tasks": [
    { "description": "...", "intent": "research" | "clarify" }
  ]
}
```

Task 1 is always landscape discovery — it surveys the broad category and establishes vocabulary. Tasks 2+ each build on what prior tasks will have surfaced, using pronoun-like references ("for the entities surfaced by task 1", "deepen the benchmarks from task 2"). This forces a real dependency chain rather than independent slices.

Today's date is threaded into the plan context so recency-sensitive searches anchor on the current year.

**Routing logic:**

```typescript
function route(plan: PlanResult, query: string, maxTurns: number): Route {
  const research = plan.tasks.filter(t => t.intent === 'research');
  const clarify = plan.tasks.filter(t => t.intent === 'clarify');

  if (research.length === 0 && clarify.length > 0)
    return { type: 'clarify', questions: clarify.map(t => t.description) };

  const tasks = research.length > 0
    ? research
    : [{ description: query, intent: 'research' as const }];
  return { type: 'research', tasks, maxTurns };
}
```

Three outcomes:

- **All clarify** -- the query is ambiguous. Returns `{ type: 'clarify' }` to the REPL, which prompts the user for more information and re-runs `handleQuery` with the clarification as `context`.
- **Decompose** -- research tasks execute sequentially via the spine. Each gets `maxTurns` tool calls.
- **Passthrough** -- empty tasks array means the query is focused enough to investigate directly. The original query becomes a single task.

### Stage 2: Research (sequential KV-chained spine)

The research phase is the heart of the pipeline. It runs tasks SEQUENTIALLY via the `reduce` combinator, with each task's findings prefilled into a shared `queryRoot` branch so later tasks inherit prior research via KV attention.

```typescript
const { answer, totalTokens, totalToolCalls } =
  yield* withSharedRoot(
    { systemPrompt: baseWorkerPrompt, tools: queryToolkit.toolsJson, parent: warmParent },
    function* (queryRoot) {
      const stats = yield* reduce(
        r.tasks,
        { totalTokens: 0, totalToolCalls: 0 },
        function* (acc, task, i) {
          const result = yield* runResearchTask({
            task, taskIndex: i, taskCount: r.tasks.length,
            queryRoot, primarySource, primaryScorer,
            allDataTools, workerToolCtx, baseWorkerPrompt,
            maxTurns, trace, ctx, store, send,
          });
          return {
            totalTokens: acc.totalTokens + result.totalTokens,
            totalToolCalls: acc.totalToolCalls + result.totalToolCalls,
          };
        },
      );
      // ... synthesis follows inside the same withSharedRoot scope
    },
  );
```

Key details:

1. **`withSharedRoot` creates a query-scoped temp root** (`queryRoot`) that lives only for this `handleQuery` call. If `session.trunk` exists, `queryRoot` forks from it (warm continuation). All research tasks and synthesis fork from `queryRoot`.

2. **`reduce(r.tasks, acc, fn)`** sequences tasks. Each iteration receives the accumulated stats and a `ResearchTask` from the plan. The reduce body runs one `agentPool` per task with one agent.

3. **Per-task worker prompts** are rendered via `renderTemplate(WEB_WORKER_TEMPLATE, ctx)` with runtime context including `taskIndex` (for spine-awareness on tasks 1+) and `date` (current date for recency-anchored search queries).

4. **`extendSpine`** prefills task findings into `queryRoot` between tasks:

```typescript
function* extendSpine(ctx, store, queryRoot, userContent, assistantContent) {
  const messages = [
    { role: 'user', content: userContent },
    { role: 'assistant', content: assistantContent },
  ];
  const sep = ctx.getTurnSeparator();
  const { prompt } = ctx.formatChatSync(JSON.stringify(messages), {});
  const turnTokens = [...sep, ...ctx.tokenizeSync(prompt, false)];
  yield* call(() => store.prefill([[queryRoot, turnTokens]]));
  return turnTokens.length;
}
```

After each task, its findings are tokenized as a user+assistant turn and prefilled into `queryRoot`. The next task's pool forks from the EXTENDED `queryRoot` and sees all prior findings via KV attention — no text re-encoding, no per-agent prefill cost.

5. **Terminal tool protection**: the research policy is configured with `terminalTool: 'report'`, which protects agents mid-generation of the report tool call from `shouldExit` time-budget kills. Agents that start writing their report after a nudge are allowed to complete.

**Recovery — scratchpad extraction for hard-cut agents:**

Hard-cut recovery is controlled by the policy's `onRecovery()` method. Configure via `DefaultAgentPolicyOpts.recovery`:

```typescript
function createResearchPolicy() {
  return new DefaultAgentPolicy({
    budget: {
      context: { softLimit: 2048, hardLimit: 1024 },
      time: { softLimit: 120_000, hardLimit: 180_000 },
    },
    recovery: { prompt: RECOVERY },
    terminalTool: 'report',
  });
}
```

After an agent finishes or is killed, the policy decides per-agent whether to extract. The pool prefills the recovery prompt into the agent's branch, runs a grammar-constrained `{ result }` extraction, and records the result with `scratchpad` provenance. A confabulation guard (default: 100 tokens, 2 tool calls) skips agents with insufficient context.

### Stage 3: Synthesize

Synthesis runs inside the same `withSharedRoot` scope as research, so it forks from `queryRoot` with the full task-by-task research spine already in KV:

```typescript
const synthCtx = { query };
const synth = yield* agentPool({
  tasks: [{ content: renderTemplate(SYNTHESIZE.user, synthCtx) }],
  tools: [reportTool],
  systemPrompt: renderTemplate(SYNTHESIZE.system, synthCtx),
  parent: queryRoot,
  terminalTool: 'report',
  maxTurns: opts.maxTurns,
  trace: opts.trace,
});
const synthAnswer = synth.agents[0]?.result || '';
```

The synth agent reads prior findings by attending over the user/assistant turns in its inherited KV context — no text re-encoding, no findings blob in the prompt. The `synthesize.eta` prompt instructs the model to:

1. Read every research turn above
2. Decide what they holistically mean for the user's question
3. Write a research report with narrative arc: the first `##` section leads with the direct answer, subsequent sections advance, qualify, or challenge it
4. Use flexible body form: prose, `###` subsections, markdown tables, counterpoints
5. End with `## Conclusion` (including `### Limitations`) and `## Sources`

### Stage 4: Verify + Eval

After synthesis, the pipeline checks whether the answer is consistent by generating multiple independent verify samples and comparing them.

**Verify via seeded pool:**

```typescript
const verifyContent = renderTemplate(VERIFY.user, {
  agentFindings: answer || '(none)',
  sourcePassages: '(spine)',
  query,
});
const verifyPool = yield* agentPool({
  tasks: Array.from({ length: opts.verifyCount }, (_, i) => ({
    content: verifyContent,
    seed: 2000 + i,
  })),
  systemPrompt: VERIFY.system,
});
```

Each verify task gets a different seed, producing N independent answers to the same question given the same evidence. The verify prompt asks for a direct answer rather than a cited report.

**Grammar-constrained convergence check:**

```typescript
const evalAgent = yield* agent({
  systemPrompt: EVAL.system,
  task: renderTemplate(EVAL.user, { responses: responsesText }),
  schema: { type: 'object', properties: { converged: { type: 'boolean' } }, required: ['converged'] },
});
```

The eval agent reads all verify samples and produces a single boolean: `{ "converged": true }` or `{ "converged": false }`. The grammar ensures the output is valid JSON with exactly one boolean field.

If samples converge, the synthesis is likely grounded in evidence. If they diverge, the research may have been insufficient or contradictory. The convergence result is reported in the stats output but does not currently trigger re-research -- this is a hook for pipeline extensions.

### Finalize (trunk commit for multi-turn)

```typescript
if (answer) yield* call(() => session.commitTurn(query, answer));
```

`Session.commitTurn(query, answer)` handles both cold and warm paths internally. On cold start (no trunk), it creates a new branch, prefills the Q&A pair, and promotes as trunk. On warm continuation, it appends the turn to the existing trunk with proper turn separators. The intermediate research spine (in `queryRoot`) is pruned when `withSharedRoot` exits — only the clean Q&A survives on the long-term trunk.

### Prompts

Eta templates and task prompts in `prompts/`:

| File | Type | Purpose |
|------|------|---------|
| `plan.eta` | Task prompt | Decompose query into a chain of research tasks. Task 1 = landscape discovery. Tasks 2+ reference prior task findings via pronoun phrases. |
| `web-worker.eta` | Eta template | Worker system prompt for web source agents. Rendered with `tools`, `maxTurns`, `date`, `taskIndex`, `siblingTasks`. Tasks 1+ get spine-awareness guidance. |
| `corpus-worker.eta` | Eta template | Worker system prompt for corpus source agents. Similar structure, corpus-specific tool rules. |
| `synthesize.eta` | Task prompt | Narrative-arc synthesis: holistic analysis → direct answer → advancing sections → conclusion + limitations + sources. |
| `recovery.eta` | Task prompt | Grammar extraction prompt for agents dropped before reporting. Forces `{ result }` JSON output. |
| `verify.eta` | Task prompt | Simplified answer prompt for independent verify samples. |
| `eval.eta` | Task prompt | Consistency checker: compare verify samples and determine convergence. |
| `fallback.eta` | Task prompt | Minimal cold-start root prompt for unknown sources. |

## The full data flow

```
User query
    |
    v
 [PlanTool]  grammar-constrained chain decomposition
    |         -> { tasks: [{ description, intent }] }
    |
    v
 [route()]   intent routing
    |         -> clarify? return to user
    |         -> research tasks (or passthrough)
    |
    v
 [withSharedRoot]  create queryRoot (warm from trunk or cold)
    |
    v
 [reduce]    sequential task spine:
    |           for each task:
    |             renderWorkerPrompt (taskIndex, date, tools)
    |             agentPool (1 agent, forks from queryRoot)
    |               -> policy with terminalTool protection
    |               -> agent.observe(ctx) detects tool selection mid-turn
    |               -> recovery extraction on agent drop
    |             extendSpine: prefill findings into queryRoot
    |             next task inherits extended queryRoot via KV share
    |
    v
 [synthesize]  forks from spine-extended queryRoot
    |            narrative-arc report: direct answer → advancing sections
    |            -> cited markdown report
    |
    v
 [verify]    N seeded verify samples via agentPool
    |
    v
 [eval]      grammar-constrained convergence check
    |          -> { converged: boolean }
    |
    v
 [commitTurn]  persist Q&A to session trunk
```

## Source implementation details

### `CorpusSource`

- **Tools**: `search` (semantic via reranker), `read_file`, `grep`
- **Chunks**: Pre-split paragraph-level chunks from loaded resources. Available immediately after construction.
- **Bind**: Tokenizes chunks through the reranker and prepends a `SearchTool` to the tool list.

### `WebSource`

- **Tools**: `web_search` (via Tavily or custom provider), `fetch_page` (with optional reranker-based chunk scoring)
- **Chunks**: Populated as agents fetch pages. `getChunks()` converts the `FetchedPage` buffer into paragraph-level chunks via `chunkFetchedPages` for post-use reranking.
- **Bind**: Clears the page buffer and wires the reranker to `FetchPageTool`.

Both sources are platform-agnostic — no `node:fs` dependency. They work identically on Node.js and React Native.

### Reranker-based chunk scoring in FetchPageTool

When `FetchPageTool` has a reranker and the agent provides a `query` argument, fetched pages are structurally chunked on heading boundaries and scored against the query. Only the top-K most relevant chunks within a 2048-token budget are returned — reducing KV pressure without lossy summarization.

## Customization

**Change sources**: Remove or add sources in `main.ts`. The harness code is source-agnostic — all source tools are unioned and available to every research agent.

**Create a custom source**: Implement `Source` with `name`, `tools`, `bind()`, and `getChunks()`. See [Custom Source](../guides/custom-source.md).

**Adjust task count**: `PlanTool.maxItems` (default 6) bounds how many research tasks the planner can generate. More tasks = deeper spine chain but longer wall time.

**Add pipeline stages**: Insert new generator functions inside the `withSharedRoot` callback in `handleQuery`, between `reduce` and synthesis. Each stage can `extendSpine` to add findings to `queryRoot`.

**Trigger re-research on divergence**: The eval result (`converged: boolean`) is currently informational. Add a loop in `handleQuery` that re-runs research with refined tasks when `converged` is false.

**Customize prompts**: Edit the Eta templates and task prompts in `prompts/`. Template variables are rendered via `renderTemplate()` at call time. Task prompts split on `---` into system and user sections.

## Related pages

- [Pipelines](/learn/pipelines) -- gentler walkthrough of pipeline concepts
- [Sources](/learn/sources) -- the `Source` abstraction
- [Continuous Context Spine](/reference/continuous-context-spine) -- KV spine mechanics and extendSpine pattern
- [Scratchpad Extraction](/reference/scratchpad-extraction) -- recovery extraction for dropped agents
- [Agent Policy](/reference/agent-policy) -- shouldExit, terminalTool, shouldExplore, recovery
- [Grammar & Tool Ordering](/reference/grammar-and-ordering) -- how PlanTool and eval grammars work
- [Concurrency Model](/reference/concurrency) -- agent pool tick loop, Agent.observe/finalize
- [KV Pressure](/reference/kv-pressure) -- pressure settings and agent drops
- [Branch Lifecycle](/reference/branch-lifecycle) -- fork, prune, setLogits/mergeLogits, commitTurn
- [Tracing](/learn/tracing) -- JSONL trace output and `JsonlTraceWriter`
