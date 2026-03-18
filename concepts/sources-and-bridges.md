# Sources and Bridges

Research pipelines need to investigate across different data backends -- local files, the web, databases, APIs. The `Source` abstraction standardizes how these backends plug into the agent pool, and bridges structure discoveries between sequential sources so the second source investigates gaps rather than re-covering established ground.

## The `Source` abstraction

`Source<TCtx, TChunk>` is the base class from `@lloyal-labs/lloyal-agents`:

```typescript
import { Source } from '@lloyal-labs/lloyal-agents';
import type { Tool } from '@lloyal-labs/lloyal-agents';

abstract class Source<TCtx = Record<string, unknown>, TChunk = unknown> {
  abstract readonly name: string;
  abstract get researchTool(): Tool;

  *bind(_ctx: TCtx): Operation<void> {}
  getChunks(): TChunk[] { return []; }
  get groundingTools(): Tool[] { return []; }
}
```

Five members define the contract:

**`name`** -- Human-readable label (e.g. `'web'`, `'corpus'`). Used for labeling findings in the synthesis phase so the report can attribute claims to specific sources.

**`researchTool`** -- The source's atomic research tool. This is a self-contained swarm: when called, it spawns a pool of agents with source-specific prompts, tools, and self-referential recursion. The orchestrating harness sees only this tool plus the terminal `report` tool. It does not need to know whether the source searches the web, greps through files, or queries a database.

**`bind(ctx)`** -- Late-bind runtime dependencies that are not available at construction time. Called before tools are used. The context `TCtx` carries whatever the source needs -- parent branch for forking, reranker for tokenization, reporter prompt, trace settings. `bind` is an Effection generator (`*bind`), so it can perform async work like tokenizing chunks or loading resources.

**`getChunks()`** -- Returns buffered content from the research phase for post-research reranking. Called after research completes. The chunk type `TChunk` is source-specific -- `Chunk` for corpus and web sources, or whatever your custom source buffers during research.

**`groundingTools`** -- Tools for independent verification. The synthesis and evaluation phases may need to fact-check claims against the source without running a full research pass. For a corpus source, these are `search`, `read_file`, and `grep`. For a web source, `web_search` and `fetch_page`. Returned as an array so the harness can merge grounding tools from multiple sources into a single verification toolkit.

## `CorpusSource`

Corpus research over local files using grep, semantic search, file reading, and recursive sub-agent spawning.

### Construction

```typescript
import { CorpusSource, loadResources, chunkResources } from '@lloyal-labs/rig';

const resources = loadResources('./docs');
const chunks = chunkResources(resources);
const corpus = new CorpusSource(resources, chunks);
```

`resources` are loaded file contents with metadata. `chunks` are paragraph-level segments split from those files, ready for reranker scoring.

### `bind`

```typescript
*bind(ctx: SourceContext): Operation<void> {
  yield* call(() => ctx.reranker.tokenizeChunks(this._chunks));
  this._tools.unshift(new SearchTool(this._chunks, ctx.reranker));

  const research = new ResearchTool({
    systemPrompt: researchPrompt.system,
    reporterPrompt: ctx.reporterPrompt,
    maxTurns: ctx.maxTurns,
    trace: ctx.trace,
  });
  const toolkit = createToolkit([...this._tools, ctx.reportTool, research]);
  research.setToolkit(toolkit);
  this._researchTool = research;
}
```

`bind` tokenizes all chunks through the reranker (necessary for scoring), prepends a `SearchTool` to the tool list (so semantic search appears first in the toolkit -- see [Grammar Constraining](grammar-constraining.md) for why tool ordering matters), and constructs the self-referential `ResearchTool`.

The `ResearchTool` is self-referential: it appears in its own toolkit. When a research agent decides it needs deeper investigation, it calls `research` which spawns sub-agents with the same tools. Recursion depth is bounded by KV pressure, not by an artificial limit.

### `groundingTools`

Returns `[SearchTool, ReadFileTool, GrepTool]` -- the same tools research agents use, available for verification passes without spawning a full research pool.

### `getChunks`

Returns the pre-split corpus chunks. Since corpus content is static (loaded at construction), chunks do not change during research.

## `WebSource`

Web research using search APIs, page fetching with scratchpad extraction, and recursive sub-agent spawning.

### Construction

```typescript
import { WebSource, TavilyProvider } from '@lloyal-labs/rig';

const web = new WebSource(new TavilyProvider());
```

The search provider is pluggable. `TavilyProvider` is included; implement the `SearchProvider` interface for other backends.

### Internal tools

`WebSource` wraps the base web tools with extraction wrappers:

- **`BufferingWebSearch`** -- Wraps `WebSearchTool`. When a `ScratchpadParent` is available, forks a scratchpad to extract `{ urls, summary }` from raw search results. The compact output reduces KV pressure on the calling agent.

- **`BufferingFetchPage`** -- Wraps `FetchPageTool`. Buffers the full page content in a `FetchedPage` array for post-research reranking, then uses a scratchpad to extract `{ summary, links }`. The agent receives the compact extraction; the reranker scores from the buffered full text.

Both wrappers fall back to raw results if no scratchpad parent is available or if extraction fails. See [Scratchpad Extraction](scratchpad-extraction.md) for details on the pattern.

### `bind`

```typescript
*bind(ctx: SourceContext): Operation<void> {
  this._buffer.length = 0;  // clear prior-run content

  if (!this._researchTool) {
    const webResearch = new WebResearchTool({
      name: 'web_research',
      description: 'Spawn parallel web research agents...',
      systemPrompt: this._researchPrompt.system,
      reporterPrompt: ctx.reporterPrompt,
      maxTurns: ctx.maxTurns,
      trace: ctx.trace,
    });
    const toolkit = createToolkit([
      this._webSearch,
      this._fetchPage,
      ctx.reportTool,
      webResearch,
    ]);
    webResearch.setToolkit(toolkit);
    this._researchTool = webResearch;
  }
}
```

The page buffer is cleared on every `bind` call so prior-run content does not leak into a new research pass. The `WebResearchTool` is constructed on first bind only -- the toolkit is stateless once built.

Note the tool ordering: `web_search`, `fetch_page`, `report`, then `web_research`. The terminal tool (`report`) is placed before its prefix-sharing sibling (`web_research`), following the ordering rule from [Grammar Constraining](grammar-constraining.md).

### `groundingTools`

Returns `[BufferingWebSearch, BufferingFetchPage]`. Verification agents can search the web and fetch pages without spawning full research sub-agents.

### `getChunks`

Converts the buffered `FetchedPage` array into paragraph-level `Chunk` instances via `chunkFetchedPages`. Unlike corpus chunks (static), web chunks are accumulated during research -- every page an agent fetches adds to the buffer. After research completes, the reranker scores these chunks to select the most relevant passages for synthesis.

## Source composition

A pipeline can use one source, multiple sources, or custom implementations:

```typescript
const sources: Source[] = [];

if (corpusDir) {
  const resources = loadResources(corpusDir);
  const chunks = chunkResources(resources);
  sources.push(new CorpusSource(resources, chunks));
}

if (process.env.TAVILY_API_KEY) {
  sources.push(new WebSource(new TavilyProvider()));
}
```

When multiple sources are configured, they run sequentially. Each source gets the full KV budget:

```
Source 0 (corpus):  bind → research → getChunks → prune all inner branches
Source 1 (web):     bind → research → getChunks → prune all inner branches
```

After source N completes and its inner branches are pruned, KV is freed for source N+1. Sequential execution means sources never compete for KV -- each gets the full `nCtx` minus whatever the outer harness consumes.

## Bridge agents

When multiple sources are used, a bridge exit gate runs between them. The bridge structures discoveries from the completed source as durable context for the next source's investigation.

### The three-tier discovery structure

The bridge extracts three tiers:

1. **What was established** -- Specific data points, statistics, quotes, study details. Evidence preserved verbatim. These are claims the source substantiated with concrete evidence.

2. **Where evidence is incomplete** -- Acknowledged limitations, absent study designs, uncertain mechanisms. These are well-researched claims with identified evidence gaps. The distinction from tier 3 is critical: a topic with six sections of evidence but no experimental validation is not a gap -- it is a well-researched claim with an identified limitation.

3. **What was not covered** -- Topics mentioned but not substantiated, or entirely absent from the source. These are genuine gaps that the next source should investigate.

The distinction between tiers 2 and 3 prevents the next source from re-investigating what the previous source already covered. A limitation within established evidence (tier 2) directs the next source to seek specific supplementary evidence. A genuine gap (tier 3) directs the next source to investigate the topic from scratch.

### Conditioning the next source

Bridge discoveries are appended to the next source's sub-questions:

```typescript
activeQuestions = questions.map(
  q => `${q}\n\nPrior research discoveries:\n${discoveries}`,
);
```

The next source's research agents see the original question plus a structured summary of what was already found and what remains unknown. This prevents redundant investigation: an agent researching "iPod-era market dominance" that sees the corpus already established the timeline and market share data will focus on mechanisms and counterfactuals instead of re-establishing basic facts.

## Custom sources

Extend `Source` to create sources for any data backend:

```typescript
import { Source } from '@lloyal-labs/lloyal-agents';
import type { Tool } from '@lloyal-labs/lloyal-agents';

interface Row { table: string; data: Record<string, unknown>; }

class DatabaseSource extends Source<SourceContext, Row> {
  readonly name = 'database';

  get researchTool(): Tool {
    if (!this._researchTool)
      throw new Error('DatabaseSource: bind() must be called first');
    return this._researchTool;
  }

  *bind(ctx: SourceContext) {
    // Build tools with ctx.reportTool, ctx.reporterPrompt, etc.
    const queryTool = new SqlQueryTool(this._connection);
    const schemaTool = new SchemaInspectTool(this._connection);

    const research = new DatabaseResearchTool({
      systemPrompt: 'You investigate databases...',
      reporterPrompt: ctx.reporterPrompt,
      maxTurns: ctx.maxTurns,
    });
    const toolkit = createToolkit([
      queryTool, schemaTool, ctx.reportTool, research,
    ]);
    research.setToolkit(toolkit);
    this._researchTool = research;
  }

  get groundingTools(): Tool[] {
    return [this._queryTool, this._schemaTool];
  }

  getChunks(): Row[] {
    return this._bufferedResults;
  }
}
```

The implementation pattern is the same as `CorpusSource` and `WebSource`:

1. Construct with data-backend-specific dependencies (connection, resources, API keys).
2. In `bind`, build the tool registry and the self-referential research tool using `createToolkit`.
3. In `getChunks`, return whatever the source buffered during research for post-research reranking.
4. In `groundingTools`, expose lightweight tools for verification without spawning full research pools.

The research tool internally calls `withSharedRoot` + `useAgentPool` -- the same primitives the built-in sources use. Your custom source participates in the same KV pressure system, event channel, and structured concurrency guarantees as `CorpusSource` and `WebSource`.
