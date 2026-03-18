# Multi-Source Research

A single source -- web search, or a local corpus -- gives you one perspective on a question. When you have both, the research gets richer: the corpus provides grounded evidence from known documents, and the web fills gaps with public knowledge. This page covers how to compose multiple sources in a pipeline, how discoveries flow from one source to the next, and why the execution order matters.

## Source composition

Sources are added to the pipeline as an ordered array. Each source is a self-contained research unit: it provides its own research tool (which spawns parallel agents), grounding tools (for synthesis verification), and a chunk buffer (for post-research reranking). The harness does not know or care what kind of source it is running -- it calls the same interface on each one.

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

The `Source` base class defines the contract:

```typescript
abstract class Source<TCtx, TChunk> {
  abstract readonly name: string;
  abstract get researchTool(): Tool;

  *bind(_ctx: TCtx): Operation<void> {}
  getChunks(): TChunk[] { return []; }
  get groundingTools(): Tool[] { return []; }
}
```

- **`researchTool`** -- the atomic research swarm. Calling `execute()` spawns parallel agents with source-specific prompts and tools.
- **`groundingTools`** -- independent verification tools exposed to the synthesis stage. Corpus provides `search`, `read_file`, `grep`. Web provides `web_search`, `fetch_page`.
- **`bind(ctx)`** -- late-binds runtime dependencies (reranker, reporter prompt, trace flag) that are not available at construction time. Called once before research begins.
- **`getChunks()`** -- returns buffered content after research completes, for reranker scoring against the original query.

Array order determines execution order. Put your most constrained, highest-precision source first.

## Sequential execution

Sources run one at a time, not in parallel. This is a KV budget decision.

Each source's research tool spawns its own `withSharedRoot` and `useAgentPool` inside the harness's outer `withSharedRoot`. That inner pool occupies KV cells for the duration of its research phase -- shared root prefix, agent branches, tool result prefills. When the pool completes and the inner `withSharedRoot` scope exits, those cells are freed via branch pruning.

The freed KV budget is then available for the next source's research pool. If both sources ran simultaneously, their combined KV footprint would either exceed `nCtx` or require halving the agent count per source.

```
Source 1 (corpus)     Source 2 (web)
┌─────────────────┐
│ research pool    │
│ [agents + tools] │
│ KV: ~4000 cells  │
└─────────────────┘
     prune ↓ free
     bridge ↓ structure discoveries
                    ┌─────────────────┐
                    │ research pool    │
                    │ [agents + tools] │
                    │ KV: ~4000 cells  │
                    └─────────────────┘
```

Sequential execution with budget recovery lets each source use the full KV capacity. The tradeoff is wall-clock time -- but since both sources share one GPU, true parallelism would not help throughput anyway.

## Bridge agents

Between sources, the harness runs a bridge agent. Its job is to structure the raw findings from the previous source into durable context that the next source can use. Raw findings are agent scratchpad output -- stream-of-consciousness notes optimized for the model that wrote them. A bridge agent reorganizes that material into a three-tier discovery structure that any research agent can build on.

The bridge runs in its own `withSharedRoot` with only the `report` tool:

```typescript
const discoveries = yield* withSharedRoot(
  { systemPrompt: BRIDGE.system, tools: reportOnlyToolkit.toolsJson },
  function*(bridgeRoot) {
    const pool = yield* useAgentPool({
      tasks: [{
        systemPrompt: BRIDGE.system,
        content: bridgeContent,
        tools: reportOnlyToolkit.toolsJson,
        parent: bridgeRoot,
      }],
      tools: reportOnlyToolkit.toolMap,
      terminalTool: 'report',
      maxTurns: effectiveMaxTurns,
      trace: opts.trace,
      pressure: { softLimit: 1024 },
    });
    yield* reportPass(pool, opts);
    return pool.agents[0]?.findings || '';
  },
);
```

The bridge receives two inputs: the raw agent findings from the previous source, and reranked source passages (top-10 chunks scored against the original query). It produces a single structured report via the `report` tool.

## The bridge prompt

The bridge prompt instructs the model to classify every major claim or topic into three tiers:

1. **What was established** -- specific data points, statistics, quotes, study details. Evidence that the first source delivered with full substantiation. Preserved verbatim.
2. **Where the evidence is incomplete** -- topics that were covered but have identified limitations. Six sections of evidence with no experimental validation is a well-researched claim with an evidence limitation, not a gap. The bridge flags the limitation without re-flagging the topic.
3. **What was not covered** -- topics mentioned but not substantiated, or entirely absent from the research.

The distinction between tier 2 and tier 3 is the critical design choice. A corpus source that covers a topic thoroughly but without a particular evidence type (e.g., no RCTs) should not cause the web source to re-research the whole topic. It should cause the web source to look specifically for that evidence type.

```markdown
For each major claim or topic, report:
1. **What was established** — specific data points, study details, statistics, quotes.
   Preserve evidence verbatim.
2. **Where the evidence is incomplete** — acknowledged limitations, absent study
   designs, uncertain mechanisms, missing comparisons.
3. **What was not covered** — topics mentioned but not substantiated, or entirely
   absent from the research.

The distinction between (2) and (3) is critical. A topic with six sections of
evidence but no experimental validation is NOT a gap — it is a well-researched
claim with an identified evidence limitation. Flag the limitation, not the topic.
```

This three-tier structure prevents the second source from doing redundant work on established claims while directing it toward genuine gaps.

## How discoveries condition the next source

After the bridge agent reports, its structured discoveries are injected into the questions for the next source. Each original sub-question gets the discoveries appended as prior context:

```typescript
if (discoveries) {
  activeQuestions = questions.map(q =>
    `${q}\n\nPrior research discoveries:\n${discoveries}`
  );
}
```

The next source's research agents see the original question plus a structured summary of what is already known. This changes their behavior in measurable ways:

- Agents skip searches for well-established claims (tier 1) and focus on gaps
- Agents that encounter a tier-2 limitation know what specific evidence to look for
- Agents investigating tier-3 topics can start from a fresh angle without duplicating the first source's approach

The `activeQuestions` array replaces the original `questions` array for the next source only. The original questions are preserved for subsequent bridge passes if more than two sources are configured.

## Example: corpus first, web second

The reference pipeline in `examples/deep-research-web/` supports a concrete pattern: researching a local document corpus first, then filling gaps from the web. Consider a question about a DOJ complaint:

```
Query: "How does the DOJ connect Apple's iPod-era success
        to its current monopoly practices?"
```

**Phase 1: Corpus research.** `CorpusSource` spawns agents with `search`, `read_file`, `grep`, and `report` tools. Agents search the complaint document semantically, read relevant sections, grep for specific patterns, and report findings. The corpus covers the complaint's arguments thoroughly but cannot provide external context -- historical analysis, legal commentary, market data.

**Bridge.** The bridge agent structures discoveries:
- *Established*: the complaint's specific claims about iPod lock-in, the digital rights management strategy, the transition to App Store
- *Incomplete*: no external validation of the complaint's historical claims, no market share data beyond what the complaint cites
- *Not covered*: academic analysis of the antitrust theory, precedent cases, Apple's public response

**Phase 2: Web research.** `WebSource` spawns agents with `web_search`, `fetch_page`, `web_research` (recursive sub-agents), and `report`. The prior discoveries steer agents: they do not re-read the complaint's iPod claims. Instead, they search for antitrust legal analysis, market data sources, and Apple's response -- the tier-2 and tier-3 gaps identified by the bridge.

**Synthesis.** The synthesis agent receives findings from both sources plus reranked passages from all buffered content. Grounding tools from both sources are available (`search`, `read_file`, `grep`, `web_search`, `fetch_page`), so the synthesizer can independently verify claims from either source:

```typescript
const groundingTools = opts.sources.flatMap(s => s.groundingTools);
const synthToolkit = createToolkit([...groundingTools, reportTool]);
```

The result is a report that cross-references the complaint's specific legal claims (from corpus) against external analysis and market data (from web), with citations to both source types.

## Adding your own source

To compose a new source into the pipeline, extend `Source<SourceContext, Chunk>`:

```typescript
class DatabaseSource extends Source<SourceContext, Chunk> {
  readonly name = "database";

  get researchTool(): Tool {
    if (!this._researchTool)
      throw new Error("DatabaseSource: bind() must be called first");
    return this._researchTool;
  }

  get groundingTools(): Tool[] {
    return [this._queryTool];
  }

  *bind(ctx: SourceContext): Operation<void> {
    // Build your research toolkit: query tools + report + recursive research
    const research = new ResearchTool({
      systemPrompt: DB_RESEARCH_PROMPT,
      reporterPrompt: ctx.reporterPrompt,
      maxTurns: ctx.maxTurns,
      trace: ctx.trace,
    });
    const toolkit = createToolkit([this._queryTool, ctx.reportTool, research]);
    research.setToolkit(toolkit);
    this._researchTool = research;
  }

  getChunks(): Chunk[] {
    // Return buffered query results as chunks for reranking
    return this._buffer.map(row => ({
      resource: row.table,
      heading: row.id,
      text: row.content,
      tokens: [],
      startLine: 0,
      endLine: 0,
    }));
  }
}
```

Then add it to the sources array. The harness handles the rest -- sequential execution, bridge passes, findings collection, and reranking all work through the `Source` interface without modification.

Array position matters. Put the source with the most specific, highest-confidence data first. Bridge passes carry established facts forward -- they cannot carry web discoveries backward into a corpus search.

## Next steps

> [Warm Conversations](./warm-conversations.md) covers how the session trunk preserves KV state between queries, enabling follow-up research that builds on prior answers without re-processing the full conversation.
