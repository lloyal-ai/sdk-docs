---
title: "Deep Research"
description: "The reference pipeline — planning, parallel research, source bridging, synthesis, and convergence evaluation."
---

The reference implementation. Source-agnostic deep research across web, local corpus, or both -- with planning, parallel research, source bridging, synthesis with grounding tools, and multi-sample convergence evaluation.

This is the most complex example. It composes every major framework primitive: `generate`, `withSharedRoot`, `useAgentPool`, `diverge`, `createToolkit`, `Source`, and `Session.promote`/`appendTurn`. If you want to understand how a production research pipeline works end to end, this is the code to study.

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

`CorpusSource` provides local file tools (`search`, `grep`, `read_file`). `WebSource` provides web tools (`web_search`, `fetch_page`). Both implement the same `Source` interface — the harness calls `spawnAgents()` with `source.tools` for each, using source-specific prompts and recursion config. Orchestration lives in the harness, not the source.

Source order matters: sources are researched sequentially, and bridge passes carry discoveries from earlier sources into later ones. The convention is corpus first (fast, local) then web (slower, broader).

**Concurrency parameters:**

```typescript
const AGENT_COUNT = 3;     // max sub-questions from planning
const VERIFY_COUNT = 3;    // independent samples for convergence eval
const MAX_TOOL_TURNS = 20; // tool calls before forced report

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

`nSeqMax` is sized to accommodate the maximum number of concurrent branches: research agents, plus their sub-agents (from `web_research` or recursive `research` tools), plus synthesis, plus `diverge` samples.

**Trace support:**

```typescript
const traceWriter = trace
  ? new JsonlTraceWriter(fs.openSync(`trace-${Date.now()}.jsonl`, 'w'))
  : undefined;
const { session, events } = yield* initAgents<WorkflowEvent>(ctx, { traceWriter });
```

When `--trace` is passed, a `JsonlTraceWriter` captures every prompt, completion, tool call, branch create/prune, and source bind event to a JSONL file. This is passed to `initAgents` and propagated through the `Trace` context to all framework primitives.

### `harness.ts` -- the full pipeline

The harness is structured as a set of generator functions composed by `handleQuery`. Seven task prompts are loaded at module level:

```typescript
const PLAN = loadTask('plan');
const ROOT = loadTask('root');
const BRIDGE = loadTask('bridge');
const SYNTHESIZE = loadTask('synthesize');
const VERIFY = loadTask('verify');
const EVAL = loadTask('eval');
const REPORT = loadTask('report');
```

### Stage 1: Plan (PlanTool + intent routing)

```typescript
const planTool = new PlanTool({
  prompt: PLAN,
  session: opts.session,
  maxQuestions: opts.agentCount,
});
const plan = (yield* planTool.execute({ query, context })) as PlanResult;
const r = route(plan, query, opts.maxTurns);
```

`PlanTool` is a grammar-constrained generator that decomposes the query into sub-questions with intent classification. The grammar schema constrains output to:

```json
{
  "questions": [
    { "text": "...", "intent": "research" | "clarify" }
  ]
}
```

The `maxQuestions` parameter bounds the array length in the grammar, preventing the model from generating more sub-questions than the pipeline can handle concurrently.

If the session has an existing trunk (warm continuation), `PlanTool` formats the plan as a warm turn on the trunk rather than a cold start -- the model sees the prior conversation and can produce follow-up-aware decompositions.

**Routing logic:**

```typescript
function route(plan: PlanResult, query: string, maxTurns: number): Route {
  const research = plan.questions.filter(q => q.intent === 'research');
  const clarify = plan.questions.filter(q => q.intent === 'clarify');

  if (research.length === 0 && clarify.length > 0)
    return { type: 'clarify', questions: clarify.map(q => q.text) };

  const questions = research.length > 0 ? research.map(q => q.text) : [query];
  const effectiveMaxTurns = questions.length === 1 ? maxTurns * 2 : maxTurns;
  return { type: 'research', questions, maxTurns: effectiveMaxTurns };
}
```

Three outcomes:

- **All clarify** -- the query is ambiguous. Returns `{ type: 'clarify' }` to the REPL, which prompts the user for more information and re-runs `handleQuery` with the clarification as `context`.
- **Decompose** -- research questions are dispatched to parallel agents. Each gets `maxTurns` tool calls.
- **Passthrough** -- empty questions array means the query is focused enough to investigate directly. The original query becomes a single question with doubled turn budget (`maxTurns * 2`).

### Stage 2: Research (source iteration + bridge)

The research function is the heart of the pipeline. It iterates over configured sources, runs parallel agents for each, and structures discoveries between sources via bridge passes.

**Cold research (first query):**

```typescript
function* research(
  questions: string[], query: string, opts: WorkflowOpts, maxTurns?: number,
): Operation<{ agentFindings: string; sourcePassages: string; totalTokens: number; totalToolCalls: number; timeMs: number }> {
  const chunks = yield* withSharedRoot(
    { systemPrompt: ROOT.system },
    function*(root) {
      // Bind each source with runtime context
      for (const source of opts.sources)
        yield* source.bind({
          reranker: opts.reranker,
          reporterPrompt: REPORT, reportTool,
          maxTurns: effectiveMaxTurns, trace: opts.trace,
        });

      let activeQuestions = questions;

      // Research each source sequentially
      for (let i = 0; i < opts.sources.length; i++) {
        const source = opts.sources[i];
        const result = (yield* spawnAgents({
          questions: activeQuestions,
        })) as SourceResearchResult;

        // Collect findings...
        // Bridge between sources (if not the last source)...
      }

      return opts.sources.flatMap(s => s.getChunks());
    },
  );
}
```

Key details:

1. **`withSharedRoot({ systemPrompt: ROOT.system })`** creates a minimal shared prefix. Note: no tools in the root prompt. Tools are provided per-source by the research tool implementations.

2. **`source.bind({ reranker })`** late-binds runtime dependencies. For `CorpusSource`, this tokenizes chunks through the reranker and builds a `SearchTool`. For `WebSource`, this wires the reranker to `FetchPageTool` for chunk scoring.

3. **`spawnAgents({ tools: source.tools, ... })`** orchestrates parallel agents with the source's tools. The harness provides the system prompt, recursion shape, and entailment scorer:
   - **Corpus agents** get: `search` (semantic via reranker), `read_file`, `grep`, `report`, and an optional recursive delegate tool
   - **Web agents** get: `web_search` (Tavily), `fetch_page` (with reranker chunk scoring + `alsoOnPage` discovery headings), `report`, and an optional recursive delegate tool

4. The `result` contains findings from each agent plus token/tool-call counts.

**Bridge passes between sources:**

```typescript
if (i < opts.sources.length - 1 && sectionFindings) {
  const sourceChunks = source.getChunks();
  const passages = yield* rerankChunks(sourceChunks, query, opts.reranker, 10, opts.findingsMaxChars);

  const bridgeContent = BRIDGE.user
    .replace('{{agentFindings}}', sectionFindings)
    .replace('{{sourcePassages}}', passages)
    .replace('{{query}}', query);

  const discoveries = yield* withSharedRoot(
    { systemPrompt: BRIDGE.system, tools: reportOnlyToolkit.toolsJson },
    function*(bridgeRoot) {
      const pool = yield* useAgentPool({
        tasks: [{ systemPrompt: BRIDGE.system, content: bridgeContent, tools: reportOnlyToolkit.toolsJson, parent: bridgeRoot }],
        tools: reportOnlyToolkit.toolMap,
        terminalTool: 'report',
        maxTurns: effectiveMaxTurns,
        trace: opts.trace,
        pressure: { softLimit: 1024 },
        reportPrompt: REPORT,
      });
      return pool.agents[0]?.findings || '';
    },
  );

  if (discoveries) {
    activeQuestions = questions.map(q =>
      `${q}\n\nPrior research discoveries:\n${discoveries}`
    );
  }
}
```

The bridge is a distinct agent that runs between sources. It receives the previous source's findings and reranked passages, and structures them into three categories:

1. What was established (preserve evidence verbatim)
2. Where the evidence is incomplete (identified limitations)
3. What was not covered (gaps for the next source)

The structured discoveries are injected into the questions for the next source. This prevents the second source from re-investigating established ground and focuses it on evidence gaps.

**Warm research (follow-up queries):**

When the session already has a trunk (warm continuation), `warmResearch` is used instead. It skips `withSharedRoot` for the outer scope (the trunk already provides context) but still creates `withSharedRoot` scopes inside each source's research tool and bridge pass. The logic is otherwise identical.

**`reportPrompt` -- scratchpad extraction for hard-cut agents:**

Hard-cut recovery is now built into the pool. Pass `reportPrompt` with system and user strings, and the pool handles extraction automatically:

```typescript
const pool = yield* useAgentPool({
  tasks, tools: toolMap,
  terminalTool: 'report',
  reportPrompt: {
    system: 'You are a research reporter. Call the report tool with all findings.',
    user: 'Report your findings.',
  },
  pruneOnReport: true,   // free KV mid-pool
});
```

After all agents finish or are killed, the pool forks from each unreported agent's branch, runs a grammar-constrained `{ findings }` extraction, and records the result with `scratchpad` provenance. A confabulation guard skips agents with fewer than 100 tokens or 2 tool calls — insufficient context would produce hallucinated findings. Extraction runs before the `pool:close` trace event, so findings are populated in trace output.

### Stage 3: Findings Eval + Synthesize

Before synthesis, the pipeline evaluates whether research agents agree on key claims:

```typescript
const findingsEval = yield* evaluateFindings(res.agentFindings, opts);
const conflicts = !findingsEval.converged ? findingsEval.conflicts : undefined;

const s = yield* synthesize(res.agentFindings, res.sourcePassages, query, opts, conflicts);
```

The `evaluateFindings` function uses a grammar-constrained generation that returns `{converged: boolean, conflicts: string[]}`. A contradiction exists when one agent claims something IS the case with evidence while another claims it is NOT the case. Different coverage angles are complementary, not contradictory.

**Synthesis adapts to inference state** via an [Eta template](synthesize.eta):

```typescript
const rendered = renderTemplate(SYNTHESIZE_TEMPLATE, {
  agentFindings,
  sourcePassages: sourcePassages || null,  // omitted when corpus-only
  conflicts: conflicts || null,             // omitted when converged
  query,
});

const sourceTools = conflicts
  ? opts.sources.flatMap(s => s.tools)
  : [];
const synthToolkit = createToolkit([...sourceTools, reportTool]);
```

- **Converged** -- report-only toolkit, standard prompt. No grounding needed.
- **Diverged** -- grounding tools added (search, read_file, grep for corpus; web_search, fetch_page for web). The synthesizer reads the source directly to resolve each conflict.
- **Multi-source** -- source passages section included with citation instructions.
- **Single-source** -- source passages section omitted entirely, not filled with "(none)".

The Eta template controls which system prompt clauses, user content sections, and numbered instructions appear based on which variables are truthy. No phantom sections, no wasted attention on absent capabilities.

### Stage 4: Eval (diverge for convergence)

After synthesis, the pipeline checks whether the answer is consistent by generating multiple independent samples and comparing them.

**Verify samples via `diverge`:**

```typescript
const verifyContent = VERIFY.user
  .replace('{{agentFindings}}', agentFindings || '(none)')
  .replace('{{sourcePassages}}', sourcePassages || '(none)')
  .replace('{{query}}', query);

const samples = yield* diverge({
  prompt,
  attempts: opts.verifyCount,
  params: { temperature: 0.7 },
});
```

`diverge` creates N independent branches from the same prompt and generates at temperature 0.7. Each sample is an independent answer to the same question given the same evidence. The verify prompt is simpler than the synthesize prompt -- it asks for a direct answer rather than a cited report.

**Grammar-constrained convergence check:**

```typescript
const evalSchema = {
  type: 'object',
  properties: { converged: { type: 'boolean' } },
  required: ['converged'],
};
const grammar: string = yield* call(() => ctx.jsonSchemaToGrammar(JSON.stringify(evalSchema)));

const result = yield* generate({
  prompt,
  grammar,
  params: { temperature: 0 },
  parse: (output: string) => {
    try { return JSON.parse(output).converged as boolean; }
    catch { return null; }
  },
});
```

The eval agent reads all verify samples and produces a single boolean: `{ "converged": true }` or `{ "converged": false }`. Temperature 0 makes the evaluation deterministic. The grammar ensures the output is valid JSON with exactly one boolean field.

If samples converge, the synthesis is likely grounded in evidence. If they diverge, the research may have been insufficient or contradictory. The convergence result is reported in the stats output but does not currently trigger re-research -- this is a hook for pipeline extensions.

### Finalize (trunk promotion for multi-turn)

```typescript
function* promoteTrunk(
  query: string, response: string, opts: WorkflowOpts,
): Operation<void> {
  const ctx: SessionContext = yield* Ctx.expect();
  const messages = [
    { role: 'user', content: query },
    { role: 'assistant', content: response },
  ];
  const { prompt } = yield* call(() => ctx.formatChat(JSON.stringify(messages), { enableThinking: false }));
  const tokens = yield* call(() => ctx.tokenize(prompt, false));
  const trunk = Branch.create(ctx, 0, {});
  yield* call(() => trunk.prefill(tokens));
  yield* call(() => opts.session.promote(trunk));
}

function* appendTurn(
  query: string, response: string, opts: WorkflowOpts,
): Operation<void> {
  const ctx: SessionContext = yield* Ctx.expect();
  const sep = ctx.getTurnSeparator();
  const messages = [
    { role: 'user', content: query },
    { role: 'assistant', content: response },
  ];
  const { prompt } = ctx.formatChatSync(JSON.stringify(messages), { enableThinking: false });
  const tokens = ctx.tokenizeSync(prompt, false);
  yield* call(() => opts.session.trunk!.prefill([...sep, ...tokens]));
}
```

On the first query (cold), `promoteTrunk` creates a new branch at position 0, prefills the query/response pair, and promotes it as the session trunk via `session.promote`. This sets up the persistent conversation state.

On follow-up queries (warm), `appendTurn` appends the new turn to the existing trunk using `getTurnSeparator()` + formatted tokens. The separator ensures proper turn boundaries in the model's chat format.

The routing in `handleQuery`:

```typescript
if (warm) {
  yield* appendTurn(query, findings, opts);
} else if (findings) {
  yield* promoteTrunk(query, findings, opts);
}
```

### Task prompts

Seven markdown files in `tasks/`:

| File | Purpose |
|------|---------|
| `plan.md` | Decompose query into sub-questions with `research`/`clarify` intent. Uses `{{count}}` for max questions, `{{query}}` for the user query. |
| `root.md` | Minimal shared root system prompt: `"You are a research assistant."` |
| `bridge.md` | Structure discoveries between sources. Distinguishes established evidence, incomplete evidence, and uncovered topics. |
| `synthesize.md` | Cross-reference research notes against source passages. Produce cited markdown report. |
| `verify.md` | Simplified answer prompt for generating independent samples. |
| `eval.md` | Consistency checker. Compare verify samples and determine if they convey the same core meaning. |
| `report.md` | Reporter extraction prompt for hard-cut agents. Preserves detail -- explicitly says "do not compress or summarize." |

## The full data flow

```
User query
    |
    v
 [PlanTool]  grammar-constrained decomposition
    |         -> { questions: [{ text, intent }] }
    |
    v
 [route()]   intent routing
    |         -> clarify? return to user
    |         -> research questions (or passthrough)
    |
    v
 [research]  for each source:
    |           source.bind({ reranker })
    |           scorer = source.createScorer(query)
    |           spawnAgents({ tools: source.tools, scorer, ... })
    |             -> withSharedRoot + useAgentPool (parallel agents)
    |             -> entailment scoring at steering boundaries
    |             -> reportPrompt (scratchpad extraction for hard-cut agents)
    |           if not last source:
    |             rerankChunks -> bridge agent -> inject discoveries
    |
    v
 [rerankChunks]  score all buffered chunks against original query
    |
    v
 [findingsEval]  grammar-constrained convergence check on agent findings
    |              -> { converged: boolean, conflicts: string[] }
    |
    v
 [synthesize]  report-only (converged) or grounding tools (diverged)
    |            Eta template controls prompt/tools based on inference state
    |            -> cited markdown report
    |
    v
 [diverge]  N independent verify samples at temp 0.7
    |
    v
 [evaluate]  grammar-constrained convergence check
    |          -> { converged: boolean }
    |
    v
 [promote/append]  persist Q&A to session trunk
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

**Change sources**: Remove or add sources in `main.ts`. The harness code is source-agnostic — it iterates `opts.sources` and calls `spawnAgents()` for each.

**Create a custom source**: Implement `Source` with `name`, `tools`, `bind()`, and `getChunks()`. See [Custom Source](../guides/custom-source.md).

**Adjust agent count**: Change `AGENT_COUNT` in `main.ts`. This bounds `PlanTool`'s `maxQuestions` and affects how many parallel research agents run per source.

**Adjust findings budget**: Use `--findings-budget <chars>` to control how much reranked text the synthesizer sees. Smaller budgets reduce KV pressure but may miss relevant context.

**Add pipeline stages**: Insert new generator functions between `research` and `synthesize` in `handleQuery`. The pattern is consistent: receive findings, produce structured output, pass forward.

**Trigger re-research on divergence**: The eval result (`converged: boolean`) is currently informational. Add a loop in `handleQuery` that re-runs research with refined questions when `converged` is false.

**Customize prompts**: Edit the seven markdown files in `tasks/`. The `{{placeholder}}` values are replaced at runtime. The bridge prompt's three-category structure (established, incomplete, uncovered) is particularly important for multi-source quality.

## Related pages

- [Pipelines](/learn/pipelines) -- gentler walkthrough of pipeline concepts
- [Sources](/learn/sources) -- the `Source` abstraction
- [RIG Bridges](/reference/rig/bridges) -- bridge pass pattern between sources
- [Scratchpad Extraction](/reference/scratchpad-extraction) -- `generate({ parent })` and `ScratchpadParent`
- [Grammar & Tool Ordering](/reference/grammar-and-ordering) -- how PlanTool and eval grammars work
- [Concurrency Model](/reference/concurrency) -- agent pool tick loop and batch decoding
- [KV Pressure](/reference/kv-pressure) -- pressure settings and agent drops
- [Branch Lifecycle](/reference/branch-lifecycle) -- fork, prune, and the promote/append pattern
- [Tracing](/learn/tracing) -- JSONL trace output and `JsonlTraceWriter`
