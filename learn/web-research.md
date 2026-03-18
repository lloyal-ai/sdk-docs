# Web Research

This guide covers adding web search to a research pipeline. It walks through the `WebSource` implementation -- how agents search, fetch, and extract; how scratchpad compression saves KV budget; and how fetched content is buffered for post-research reranking.

## Adding web search to your pipeline

`WebSource` is a drop-in `Source` implementation that gives research agents access to the web. You need a search provider (Tavily is the built-in one) and three lines of setup:

```typescript
import { WebSource, TavilyProvider } from '@lloyal-labs/rig';

const web = new WebSource(new TavilyProvider());
sources.push(web);
```

`TavilyProvider` reads its API key from the constructor argument or the `TAVILY_API_KEY` environment variable. If neither is set, it throws at search time:

```typescript
// Explicit key
new TavilyProvider('tvly-...');

// Or via environment
// TAVILY_API_KEY=tvly-... npx tsx your-app.ts
new TavilyProvider();
```

The `SearchProvider` interface is simple enough to swap in any backend:

```typescript
interface SearchProvider {
  search(query: string, maxResults: number): Promise<SearchResult[]>;
}

interface SearchResult {
  title: string;
  url: string;
  snippet: string;
}
```

Implement `SearchProvider` to plug in Brave, SerpAPI, or any other search API.

Once you have sources, the harness passes them through the standard pipeline -- `source.bind()` wires runtime dependencies, `source.researchTool` dispatches agents, `source.getChunks()` returns content for reranking:

```typescript
const harnessOpts: WorkflowOpts = {
  session,
  reranker,
  events,
  agentCount: 3,
  verifyCount: 3,
  maxTurns: 20,
  trace: false,
  sources,  // [web] or [corpus, web] -- order matters for bridges
};

yield* handleQuery(query, harnessOpts);
```

## How web agents work

Each web research agent gets four tools and a five-step process. The prompt tells the agent exactly what to do:

1. **`web_search`** -- search the web with focused queries targeting specific aspects of the question.
2. **`fetch_page`** -- read the most promising results. Follow links within pages when they lead to more authoritative content.
3. **`web_search` again** -- search with refined queries based on what was learned. Target gaps in findings.
4. **`web_research`** -- spawn parallel sub-agents if areas need deeper investigation.
5. **`report`** -- submit findings with source URLs and direct quotes as evidence.

The `web_research` tool is recursive. When an agent calls it with `{"questions": ["q1", "q2"]}`, each question gets its own sub-agent in a `withSharedRoot` pool. Sub-agents have the same four tools, so they can search, fetch, and even spawn their own sub-agents (though context pressure typically prevents deep recursion).

The toolkit is wired with a circular reference. `WebSource.bind()` constructs the `WebResearchTool`, builds the toolkit, then closes the loop:

```typescript
// Inside WebSource.bind()
const webResearch = new WebResearchTool({
  name: "web_research",
  description: "Spawn parallel web research agents...",
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
```

The `setToolkit()` call gives the research tool access to the full toolkit, including itself. Sub-agents spawned by `web_research` can call `web_search`, `fetch_page`, `report`, and `web_research` -- the same tools as the parent agent.

## Scratchpad extraction

Raw web results are large. A single `web_search` call returns up to 8 results with titles, URLs, and snippets. A single `fetch_page` can return up to 6000 characters of article text. These tool results get prefilled into the agent's KV cache, consuming context budget that the agent needs for reasoning and subsequent tool calls.

`BufferingWebSearch` and `BufferingFetchPage` solve this with scratchpad extraction -- a pattern where a grammar-constrained generation on a forked branch compresses the tool result before the agent sees it.

### How BufferingWebSearch works

When the agent calls `web_search`, the wrapper:

1. Delegates to the inner `WebSearchTool` to get raw results from the search provider.
2. Reads `ScratchpadParent` from the Effection context -- this is the innermost active root branch, set by `withSharedRoot`.
3. Forks a scratchpad branch from that parent.
4. Grammar-constrains a generation that distills the raw results into a compact JSON object:

```json
{
  "urls": ["https://...", "https://..."],
  "summary": "Brief summary of what the search found"
}
```

5. Returns the compact object to the agent instead of the full result array.

The agent sees only the URLs worth fetching and a brief summary. The full search results never enter the agent's KV cache.

If no `ScratchpadParent` is available (e.g. running outside `withSharedRoot`) or extraction fails, the wrapper falls back to returning raw results. The scratchpad is an optimization, not a requirement.

### How BufferingFetchPage works

When the agent calls `fetch_page`, the wrapper:

1. Delegates to the inner `FetchPageTool` to fetch and extract the page via Readability.
2. Pushes the full page content into a shared `FetchedPage[]` buffer (for later reranking).
3. Forks a scratchpad branch from `ScratchpadParent`.
4. Grammar-constrains a generation that extracts a summary and links:

```json
{
  "summary": "Concise summary of the page content",
  "links": ["https://...", "https://..."]
}
```

5. Returns the compact result to the agent.

The agent gets a summary and links to follow, not the full 6000-character article. The full text is preserved in the buffer for the reranker to work with after research completes.

### The extraction prompts

Two separate prompts drive the scratchpad extractions. For `fetch_page`, the extraction prompt (`extract.md`) is minimal:

> You extract key information from web page content. Produce a concise summary and list any URLs/links found in the text that are worth following. Output JSON only.

The user section receives the page URL, title, and full content via template variables (`{{url}}`, `{{title}}`, `{{content}}`).

For `web_search`, the search extraction prompt (`search-extract.md`) focuses on URL selection:

> You select the most relevant search results for a research query. Pick URLs most likely to contain substantive information and summarize the key findings. Output JSON only.

The user section receives the query and formatted search results via `{{query}}` and `{{results}}`.

Both extractions run at temperature 0.3 for consistency. The grammar constraint ensures valid JSON output -- no parsing failures from free-form generation.

### Why scratchpads work

The key insight is that scratchpad branches are ephemeral. They fork from the shared root (which already contains the system prompt tokens), generate a compressed result, and are immediately pruned. The fork is O(1) metadata -- it tags existing KV cells with a new sequence ID, no tensor copy. The generation adds a few hundred tokens at most. Pruning releases those cells.

The calling agent never sees the scratchpad. It receives the compact JSON as a tool result, which is prefilled into its own branch. A 6000-character page becomes a 200-character summary. The KV savings compound across multiple search and fetch calls per agent, across multiple agents in the pool.

## The web research prompt

The full system prompt given to web research agents (from `web-research.md`):

> You are a research assistant investigating questions using the web. Your tools:
> - **web_search**: search the web -- returns results with titles, snippets, and URLs
> - **fetch_page**: fetch a URL and extract its article content -- use to read promising search results or follow links
> - **web_research**: spawn parallel sub-agents that each run their own web_search/fetch_page cycle -- call with `{"questions": ["q1", "q2", ...]}`
> - **report**: submit your final findings with evidence and source URLs
>
> Process -- follow every step in order:
> 1. Search the web with focused queries targeting specific aspects of the question.
> 2. Read the most promising results with fetch_page. Follow links within pages when they lead to more authoritative content.
> 3. Search again with refined queries based on what you learned. Target gaps in your findings.
> 4. Call web_research with sub-questions if you judge there are areas that need deeper investigation.
> 5. Report with source URLs and direct quotes as evidence. State what you found and what you checked.

The prompt enforces a search-fetch-refine-delegate-report loop. The agent must search before it fetches, refine before it delegates, and cite sources when it reports. The `web_research` step is optional -- not every question needs recursive sub-agents.

## Content buffering and reranking

`WebSource` maintains an internal `FetchedPage[]` buffer. Every successful `fetch_page` call pushes the full page content into this buffer, regardless of whether scratchpad extraction succeeds.

```typescript
interface FetchedPage {
  url: string;    // Resolved URL of the fetched page
  title: string;  // Page title (may be empty)
  text: string;   // Full extracted article text
}
```

After research completes, the harness calls `source.getChunks()`, which converts the buffer into `Chunk` instances via `chunkFetchedPages()`. The chunking splits each page's text on blank-line paragraph boundaries, filtering paragraphs shorter than 40 characters. If no paragraphs survive the filter, the full text is emitted as a single chunk (if long enough).

```typescript
// Inside WebSource
getChunks(): Chunk[] {
  return chunkFetchedPages(this._buffer);
}
```

These chunks are then scored by the reranker against the original query. The top-scoring passages are assembled into a context string for the synthesis and verification phases:

```typescript
// In the harness, after research completes
const chunks = opts.sources.flatMap(s => s.getChunks());
const sourcePassages = yield* rerankChunks(chunks, query, opts.reranker, 10, maxChars);
```

The reranker receives paragraph-level chunks from all sources (web pages, corpus files, or both). It scores them uniformly and returns the most relevant passages regardless of origin. The synthesis agent sees a single ranked evidence block, not separate "web findings" and "corpus findings".

The buffer is cleared on every `bind()` call, so prior-run content does not leak into a new research pass:

```typescript
// Inside WebSource.bind()
this._buffer.length = 0;
```

## Web-only vs corpus+web

`WebSource` and `CorpusSource` are both `Source` implementations. The harness accepts an array of sources and processes them in order.

**Web-only** -- set `TAVILY_API_KEY` and pass a single `WebSource`:

```typescript
const sources = [new WebSource(new TavilyProvider())];
```

Research agents use `web_search` and `fetch_page` to find information. The reranker scores fetched page chunks. Good for open-domain questions where the answer lives on the public web.

**Corpus-only** -- pass a single `CorpusSource`:

```typescript
const resources = loadResources(corpusDir);
const chunks = chunkResources(resources);
const sources = [new CorpusSource(resources, chunks)];
```

Research agents use `search` (semantic reranker), `read_file`, and `grep` to investigate local documents. Good for questions about a known body of documents.

**Corpus + web** -- pass both, corpus first:

```typescript
const sources = [
  new CorpusSource(resources, chunks),
  new WebSource(new TavilyProvider()),
];
```

The harness processes sources in array order. Corpus research runs first. Between sources, a bridge agent distills the corpus findings into durable context that is prepended to each web research question. Web agents start with knowledge of what the corpus already revealed, so they target gaps rather than re-discovering the same information.

Source ordering matters: put the source with the most constrained, authoritative content first. Corpus documents are known and bounded. Web content is unbounded and variable. Running corpus first gives the bridge agent a solid foundation to guide web research.

The synthesis phase gets grounding tools from all sources, so the synthesizer can independently verify claims against both the corpus and the web:

```typescript
// In the harness synthesis phase
const groundingTools = opts.sources.flatMap(s => s.groundingTools);
const synthToolkit = createToolkit([...groundingTools, reportTool]);
```

For `WebSource`, `groundingTools` returns `[web_search, fetch_page]`. For `CorpusSource`, it returns `[search, read_file, grep]`. The synthesizer sees all of them.

## Next steps

- [Multi-source research](multi-source.md) -- combining sources with bridges, exit gates, and cross-source reranking
