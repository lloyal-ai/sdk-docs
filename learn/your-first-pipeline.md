# Your First Pipeline

The quick start ran a single agent against a local corpus. That works for simple questions, but real research benefits from structure: decomposing a question into sub-questions, investigating them in parallel, synthesizing findings into a coherent report, and verifying that the answer is consistent.

This guide walks through the reference implementation in `examples/deep-research-web/`, which implements a four-stage pipeline: **Plan**, **Research**, **Synthesize**, and **Eval**. By the end, you'll understand how each stage works and how to modify the pipeline for your own use cases.

## What a pipeline adds

A single agent produces decent results for focused questions. A pipeline adds:

- **Planning** -- grammar-constrained decomposition that breaks a broad question into parallel sub-questions, with intent classification (research vs. clarify)
- **Parallel research** -- multiple agents investigating different angles simultaneously, each with their own tool calls and findings
- **Source composition** -- research across multiple backends (web, local corpus, or both) with bridge passes that carry discoveries from one source into the next
- **Synthesis** -- a dedicated agent that cross-references research notes against source passages, producing a cited report
- **Verification** -- multiple independent answer samples checked for convergence via a grammar-constrained eval pass

## The four stages

```
User query
    |
    v
 [Plan]  -----> grammar-constrained decomposition
    |
    v
 [Research]  --> parallel agents per sub-question, per source
    |
    v
 [Synthesize] -> cited report from findings + reranked passages
    |
    v
 [Eval]  -----> diverge N samples, check convergence
    |
    v
 Final answer (promoted to session trunk for follow-ups)
```

Each stage is a generator function in `harness.ts` that composes framework primitives (`generate`, `withSharedRoot`, `useAgentPool`, `diverge`). The entry point is `handleQuery`, which orchestrates all four stages and emits workflow events for the TUI.

## Running it

The pipeline supports three source configurations: web-only, corpus-only, or both.

```bash
# Web research (requires Tavily API key)
TAVILY_API_KEY=tvly-... npx tsx examples/deep-research-web/main.ts ./models/Qwen3-4B-Q4_K_M.gguf \
  --query "What are the health effects of intermittent fasting?"

# Corpus research (local files)
npx tsx examples/deep-research-web/main.ts ./models/Qwen3-4B-Q4_K_M.gguf \
  --corpus ./my-docs/ \
  --query "What does this codebase do?"

# Both sources (web + corpus)
TAVILY_API_KEY=tvly-... npx tsx examples/deep-research-web/main.ts ./models/Qwen3-4B-Q4_K_M.gguf \
  --corpus ./my-docs/ \
  --query "How does this project compare to the state of the art?"
```

Without `--query`, the pipeline drops into an interactive REPL where you can ask follow-up questions. The session trunk preserves KV state between turns, so follow-ups continue from the prior conversation.

Options:
- `--reranker <path>` -- path to reranker model (default: `models/qwen3-reranker-0.6b-q4_k_m.gguf`)
- `--trace` -- emit a JSONL trace file with every prompt, tool call, and branch event
- `--verbose` -- show stderr output from the inference backend
- `--findings-budget <chars>` -- maximum character budget for reranked passages passed to synthesis

## Stage 1: Plan

The plan stage analyzes the user's query and decides what to do with it. It uses `PlanTool` -- a grammar-constrained generation that produces structured JSON.

```typescript
const planTool = new PlanTool({
  prompt: PLAN,
  session: opts.session,
  maxQuestions: opts.agentCount,
});
const plan = (yield* planTool.execute({ query, context })) as PlanResult;
```

`PlanTool` builds a JSON grammar from a schema that constrains the model to produce an array of sub-questions, each classified as `"research"` or `"clarify"`:

```typescript
// PlanTool's internal schema
const schema = {
  type: 'object',
  properties: {
    questions: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          text: { type: 'string' },
          intent: { type: 'string', enum: ['research', 'clarify'] },
        },
        required: ['text', 'intent'],
      },
      maxItems: this._maxQuestions,
    },
  },
  required: ['questions'],
};
```

The grammar guarantee means the model can only produce valid JSON matching this schema. No parsing failures, no retry loops.

### Routing

After planning, the harness routes based on intent:

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
- **All clarify** -- returns to the user for more information
- **Decompose** -- research questions are dispatched to parallel agents
- **Passthrough** -- empty array means the query is focused enough to research directly (gets the original query as a single question, with doubled turn budget)

## Stage 2: Research

The research stage runs parallel agents for each sub-question. The key insight is that research is **source-agnostic** -- the harness iterates over configured sources, and each source provides its own research tool that knows how to investigate questions in its domain.

### Source setup

Sources are configured in `main.ts` and passed to the harness:

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

Each `Source` exposes a `researchTool` (spawns parallel agents), `groundingTools` (independent verification tools for synthesis), and `getChunks()` (post-research content for reranking).

### Running research

Inside a `withSharedRoot`, the harness binds each source with runtime context, then executes research tools sequentially:

```typescript
const chunks = yield* withSharedRoot(
  { systemPrompt: ROOT.system },
  function*(root) {
    for (const source of opts.sources)
      yield* source.bind({
        reranker: opts.reranker,
        reporterPrompt: REPORT, reportTool,
        maxTurns: effectiveMaxTurns, trace: opts.trace,
      });

    let activeQuestions = questions;

    for (let i = 0; i < opts.sources.length; i++) {
      const source = opts.sources[i];
      const result = (yield* source.researchTool.execute({
        questions: activeQuestions,
      })) as SourceResearchResult;
      // ...collect findings...
    }

    return opts.sources.flatMap(s => s.getChunks());
  },
);
```

When you call `source.researchTool.execute()`, the source internally creates its own `withSharedRoot` and `useAgentPool`, spawning one agent per question. Each agent has access to source-specific tools:

- **Corpus**: `search` (semantic via reranker), `read_file`, `grep`, `report`
- **Web**: `web_search` (Tavily), `fetch_page`, `web_research` (recursive sub-agents), `report`

Agents run until they call `report` (the terminal tool) or hit the turn limit. Agents that hit the limit without reporting are forced through a reporter pass that extracts partial findings from their KV state.

### Bridge passes

When multiple sources are configured, discoveries from one source are structured into durable context for the next. Between sources, a bridge agent runs:

```typescript
if (i < opts.sources.length - 1 && sectionFindings) {
  // Rerank findings from this source
  const passages = yield* rerankChunks(sourceChunks, query, opts.reranker, 10, opts.findingsMaxChars);

  // Bridge agent structures discoveries for the next source
  const discoveries = yield* withSharedRoot(
    { systemPrompt: BRIDGE.system, tools: reportOnlyToolkit.toolsJson },
    function*(bridgeRoot) {
      const pool = yield* useAgentPool({
        tasks: [{ systemPrompt: BRIDGE.system, content: bridgeContent, tools: reportOnlyToolkit.toolsJson, parent: bridgeRoot }],
        tools: reportOnlyToolkit.toolMap,
        terminalTool: 'report',
        maxTurns: effectiveMaxTurns,
        trace: opts.trace,
      });
      yield* reportPass(pool, opts);
      return pool.agents[0]?.findings || '';
    },
  );

  // Inject discoveries into questions for the next source
  if (discoveries) {
    activeQuestions = questions.map(q =>
      `${q}\n\nPrior research discoveries:\n${discoveries}`
    );
  }
}
```

This prevents the second source from re-investigating what the first source already established, and highlights evidence gaps that need deeper investigation.

## Stage 3: Synthesize

After all research completes, the harness collects agent findings and reranks buffered content against the original query. The synthesis agent receives both:

```typescript
const content = SYNTHESIZE.user
  .replace('{{agentFindings}}', agentFindings || '(none)')
  .replace('{{sourcePassages}}', sourcePassages || '(none)')
  .replace('{{query}}', query);
```

The synthesis agent runs in its own `withSharedRoot` with access to grounding tools from all sources plus the report tool. This allows it to independently verify claims from the research notes:

```typescript
const groundingTools = opts.sources.flatMap(s => s.groundingTools);
const synthToolkit = createToolkit([...groundingTools, reportTool]);

const synthPool = yield* withSharedRoot(
  { systemPrompt: SYNTHESIZE.system, tools: synthToolkit.toolsJson },
  function*(root) {
    const pool = yield* useAgentPool({
      tasks: [{ systemPrompt: SYNTHESIZE.system, content, tools: synthToolkit.toolsJson, parent: root }],
      tools: synthToolkit.toolMap,
      terminalTool: 'report',
      maxTurns: opts.maxTurns,
      trace: opts.trace,
    });
    yield* reportPass(pool, opts);
    return pool;
  },
);
```

The synthesize prompt instructs the model to cross-reference research notes (for structure and analytical connections) against source passages (for ground truth and citations). The result is a markdown report with citations linked to source URLs.

## Stage 4: Eval

The eval stage checks whether the synthesis produced a consistent answer. It uses `diverge` to generate multiple independent answer samples from a verify prompt, then grammar-constrains an evaluation:

```typescript
// Generate N independent answer samples
const samples = yield* diverge({
  prompt,
  attempts: opts.verifyCount,
  params: { temperature: 0.7 },
});

// Grammar-constrained convergence check
const evalSchema = {
  type: 'object',
  properties: { converged: { type: 'boolean' } },
  required: ['converged'],
};
const grammar = yield* call(() => ctx.jsonSchemaToGrammar(JSON.stringify(evalSchema)));

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

`diverge` forks multiple branches from the same prompt and generates independently at temperature 0.7. The eval agent then reads all responses and produces a single boolean: did the samples converge on the same core answer?

This is a cheap entropy check -- if three independent samples agree, the synthesis is likely grounded. If they diverge, the research may have been insufficient or contradictory.

### Finalize

After eval, the harness promotes the findings to the session trunk for multi-turn support:

```typescript
if (warm) {
  yield* appendTurn(query, findings, opts);
} else if (findings) {
  yield* promoteTrunk(query, findings, opts);
}
```

On the first query (cold), `promoteTrunk` creates a new branch with the query/answer pair and promotes it as the session trunk. On follow-up queries (warm), `appendTurn` appends the new turn to the existing trunk. Either way, the next query starts with full KV context of the conversation history.

## The full flow

Putting it all together, `handleQuery` runs the complete pipeline:

```typescript
export function* handleQuery(
  query: string,
  opts: WorkflowOpts,
): Operation<QueryResult> {
  // Plan
  const plan = (yield* planTool.execute({ query, context })) as PlanResult;
  const r = route(plan, query, opts.maxTurns);

  if (r.type === 'clarify')
    return { type: 'clarify', questions: r.questions };

  // Research -> Synthesize -> Eval -> Finalize
  const res = yield* research(r.questions, query, opts, r.maxTurns);
  const s = yield* synthesize(res.agentFindings, res.sourcePassages, query, opts);

  // Promote to trunk for follow-ups
  const findings = s.pool.agents[0]?.findings || '';
  yield* promoteTrunk(query, findings, opts);

  return { type: 'done' };
}
```

Each stage is a generator function that composes framework primitives. The TUI subscribes to a `Channel<WorkflowEvent>` and renders progress -- all presentation is decoupled from pipeline logic.

## Customizing the pipeline

### Change the prompts

Each stage loads its prompt from a markdown file in `tasks/`. The convention is system prompt above `---`, user content below:

```markdown
You are a research synthesizer. You will receive two types of input:
...
---
Research notes:

{{agentFindings}}

Source passages:

{{sourcePassages}}

Write a detailed research report answering: "{{query}}"
```

Modify these files to change model behavior without touching pipeline code. The `{{placeholder}}` values are replaced at runtime.

### Add a source

Create a class that extends `Source<SourceContext, Chunk>`:

```typescript
class MySource extends Source<SourceContext, Chunk> {
  readonly name = "my-source";

  get researchTool(): Tool { /* return your research tool */ }
  get groundingTools(): Tool[] { /* return tools for synthesis verification */ }
  *bind(ctx: SourceContext): Operation<void> { /* late-bind runtime deps */ }
  getChunks(): Chunk[] { /* return chunks for post-research reranking */ }
}
```

Then add it to the sources array in `main.ts`. The harness iterates sources in order, running bridge passes between them automatically.

### Adjust concurrency

The constants in `main.ts` control parallelism:

```typescript
const AGENT_COUNT = 3;    // max sub-questions from planning
const VERIFY_COUNT = 3;   // independent samples for eval
const MAX_TOOL_TURNS = 20; // tool calls before forced report
```

More agents require more KV cache capacity. Adjust `nCtx` and `nSeqMax` in the `createContext` call to match.

## Next steps

> [Adding Tools](../guides/adding-tools.md) covers implementing custom tools with grammar-constrained output and scratchpad extraction.

> [Concurrency](../concepts/concurrency.md) explains the four-phase tick loop, context pressure, and how the agent pool manages shared GPU resources.

> [Prefix Sharing](../concepts/prefix-sharing.md) details how `withSharedRoot` shares KV prefix across agents and why it matters for context efficiency.
