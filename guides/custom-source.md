---
title: "Build a Custom Source"
description: "Implement Source<TCtx, TChunk> to plug any data backend into an agent pipeline."
---

This guide walks through implementing a custom `Source`. A source is a data backend -- it provides tools for agents to interact with your data. The harness orchestrates agents with those tools via `spawnAgents()`.

Sources don't own prompts or orchestration. The harness decides the prompt, the recursion shape, and the agent policy. Your source just provides tools and chunks.

## The Source base class

From `packages/agents/src/source.ts`:

```typescript
export abstract class Source<TCtx, TChunk> {
  abstract readonly name: string;
  abstract get tools(): Tool[];

  *bind(_ctx: TCtx): Operation<void> {}
  getChunks(): TChunk[] { return []; }
}
```

| Member | Required | Purpose |
|--------|----------|---------|
| `name` | Yes | Labels output to attribute which source produced what (e.g. "database", "api") |
| `tools` | Yes | Data access tools the harness passes to `spawnAgents()` |
| `bind(ctx)` | Optional | Late-binds runtime deps (reranker, API clients) not available at construction |
| `getChunks()` | Optional | Returns buffered chunks for post-use reranking |

The type parameters: `TCtx` is the context type passed to `bind()`, `TChunk` is the chunk type returned by `getChunks()`.

## Step-by-step implementation

### 1. Define your data types

```typescript
import type { Tool } from '@lloyal-labs/lloyal-agents';
import { Source } from '@lloyal-labs/lloyal-agents';
import type { Chunk } from '@lloyal-labs/rig';

interface DatabaseRow {
  id: string;
  table: string;
  content: string;
}
```

### 2. Create data access tools

These are the tools agents use to interact with your data. See the [Custom Tool](./custom-tool.md) guide for full details. At minimum you need a retrieval tool:

```typescript
class QueryTool extends Tool<{ sql: string }> {
  readonly name = 'query';
  readonly description = 'Run a read-only SQL query against the database';
  readonly parameters = {
    type: 'object',
    properties: {
      sql: { type: 'string', description: 'SELECT query to execute' },
    },
    required: ['sql'],
  };

  private _db: Database;
  constructor(db: Database) {
    super();
    this._db = db;
  }

  *execute(args: { sql: string }): Operation<unknown> {
    const rows = yield* call(() => this._db.query(args.sql));
    return { rows: rows.slice(0, 50) };
  }
}

class DescribeTool extends Tool<{ table: string }> {
  readonly name = 'describe';
  readonly description = 'Show column names and types for a table';
  readonly parameters = {
    type: 'object',
    properties: {
      table: { type: 'string', description: 'Table name' },
    },
    required: ['table'],
  };

  private _db: Database;
  constructor(db: Database) {
    super();
    this._db = db;
  }

  *execute(args: { table: string }): Operation<unknown> {
    const columns = yield* call(() => this._db.describe(args.table));
    return { table: args.table, columns };
  }
}
```

### 3. Implement the Source class

```typescript
class DatabaseSource extends Source<{ reranker: Reranker }, Chunk> {
  readonly name = 'database';

  private _db: Database;
  private _queryTool: QueryTool;
  private _describeTool: DescribeTool;
  private _results: Chunk[] = [];

  constructor(db: Database) {
    super();
    this._db = db;
    this._queryTool = new QueryTool(db);
    this._describeTool = new DescribeTool(db);
  }

  get tools(): Tool[] {
    return [this._queryTool, this._describeTool];
  }

  *bind(ctx: { reranker: Reranker }): Operation<void> {
    // Wire reranker if you have chunks to tokenize
    // For a database source, this might be a no-op
  }

  getChunks(): Chunk[] {
    return this._results;
  }
}
```

### 4. Use in a pipeline with `spawnAgents()`

The harness orchestrates agents with your source's tools:

```typescript
import { spawnAgents } from '@lloyal-labs/lloyal-agents';

const db = await connectDatabase(connectionString);
const source = new DatabaseSource(db);
yield* source.bind({ reranker });

const pool = yield* spawnAgents({
  tools: source.tools,
  systemPrompt: `You are a database analyst. Use query() to run SQL
    and describe() to inspect table schemas. Report your findings.`,
  tasks: questions,
  terminalTool: { name: 'report', tool: reportTool },
  maxTurns: 10,
  recursive: {
    name: 'investigate',
    description: 'Delegate sub-questions to parallel agents.',
    extractTasks: (args) => args.questions as string[],
  },
  reportPrompt: REPORT,
  pruneOnReport: true,
});
```

The source provides tools. The harness provides the prompt, the recursion shape, and the orchestration config. The same source works in any pipeline -- research, code review, support triage -- with different prompts for each.

### 5. Combine with other sources

```typescript
const sources = [];

const db = await connectDatabase(connectionString);
sources.push(new DatabaseSource(db));

if (process.env.TAVILY_API_KEY) {
  sources.push(new WebSource(new TavilyProvider()));
}

// In the pipeline:
for (const source of sources) {
  yield* source.bind({ reranker });
  const pool = yield* spawnAgents({
    tools: source.tools,
    systemPrompt: PROMPTS[source.name] ?? DEFAULT_PROMPT,
    tasks: questions,
    ...opts,
  });
}
```

## The bind lifecycle

`bind()` is called by the harness before agents start. It receives whatever context the source needs -- typically just the reranker:

```typescript
*bind(ctx: { reranker: Reranker }): Operation<void> {
  yield* call(() => ctx.reranker.tokenizeChunks(this._chunks));
  this._searchTool = new SearchTool(this._chunks, ctx.reranker);
}
```

Use `bind()` to:
- Initialize the reranker with your chunks (if using semantic search)
- Set up any runtime state that depends on pipeline configuration

Make `bind()` idempotent -- the harness may call it multiple times across warm queries. Guard with a `_bound` flag.

## Buffering chunks for reranking

`getChunks()` is called after agents complete. The harness passes chunks through the reranker to select the most relevant passages for synthesis.

The `Chunk` type:

```typescript
interface Chunk {
  resource: string;   // Source identifier (file path, URL, table name)
  heading: string;    // Section heading for display
  text: string;       // Chunk text content
  tokens: number[];   // Tokenized form (populated by reranker.tokenizeChunks)
  startLine: number;  // Position within resource
  endLine: number;
}
```

Buffer data during agent execution (in tool execute methods) and return it from `getChunks()`. The WebSource pattern: `BufferingFetchPage` pushes fetched pages into a buffer during execution, `getChunks()` converts the buffer to chunks via `chunkFetchedPages()`.

For a database source, you might buffer query results:

```typescript
*execute(args: { sql: string }): Operation<unknown> {
  const rows = yield* call(() => this._db.query(args.sql));
  // Buffer for post-use reranking
  for (const row of rows) {
    this._resultBuffer.push({
      resource: `${row.table}/${row.id}`,
      heading: `${row.table} row ${row.id}`,
      text: JSON.stringify(row),
      tokens: [],
      startLine: 0,
      endLine: 0,
    });
  }
  return { rows: rows.slice(0, 50) };
}
```

## Grounding tools for synthesis

When agents produce conflicting findings, the synthesis agent may need to verify claims by reading the source directly. The harness passes `source.tools` to the synthesizer conditionally:

```typescript
const groundingTools = conflicts
  ? opts.sources.flatMap(s => s.tools)
  : [];
const synthToolkit = createToolkit([...groundingTools, reportTool]);
```

This gives the synthesis agent direct access to your data tools when needed -- without running a full agent swarm. When findings converge, grounding tools are not added.
