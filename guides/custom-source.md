# Custom Source

This guide walks through implementing a custom `Source` for the research pipeline. A source is a self-contained data backend -- it provides its own research tool (an agent swarm with source-specific prompts and toolkit), grounding tools for synthesis verification, and post-research chunks for reranking.

The harness sees sources as interchangeable. It calls `source.bind()`, executes `source.researchTool`, collects `source.getChunks()` for reranking, and uses `source.groundingTools` during synthesis. Your custom source plugs into this contract without changes to the harness.

## The Source base class

From `packages/agents/src/source.ts`:

```typescript
export abstract class Source<TCtx, TChunk> {
  abstract readonly name: string;
  abstract get researchTool(): Tool;

  *bind(_ctx: TCtx): Operation<void> {}
  getChunks(): TChunk[] { return []; }
  get groundingTools(): Tool[] { return []; }
}
```

| Member | Required | Purpose |
|--------|----------|---------|
| `name` | Yes | Labels findings in the research output (e.g. "database", "api") |
| `researchTool` | Yes | The tool the harness executes -- spawns an agent pool internally |
| `bind(ctx)` | Optional | Late-binds runtime deps (reranker, reporter prompt) not available at construction |
| `getChunks()` | Optional | Returns buffered chunks for post-research reranking |
| `groundingTools` | Optional | Tools available during synthesis for independent verification |

The type parameters: `TCtx` is the context type passed to `bind()`, `TChunk` is the chunk type returned by `getChunks()`. The built-in sources use `SourceContext` and `Chunk` from `@lloyal-labs/rig`.

## Step-by-step implementation

### 1. Define your data types

```typescript
import type { Tool } from '@lloyal-labs/lloyal-agents';
import { Source } from '@lloyal-labs/lloyal-agents';
import type { SourceContext, Chunk } from '@lloyal-labs/rig';

interface DatabaseRow {
  id: string;
  table: string;
  content: string;
}
```

### 2. Create grounding tools

These are the tools research agents use to interact with your data. See the [Custom Tool](./custom-tool.md) guide for full details. At minimum you need a retrieval tool:

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

### 3. Create the research tool

The research tool spawns an agent pool with `withSharedRoot` + `useAgentPool`. It is self-referential -- it includes itself in the toolkit so agents can spawn sub-agents for deeper investigation.

```typescript
import { createToolkit, withSharedRoot, useAgentPool } from '@lloyal-labs/lloyal-agents';
import type { Toolkit, PressureThresholds } from '@lloyal-labs/lloyal-agents';

class DatabaseResearchTool extends Tool<{ questions: string[] }> {
  readonly name = 'research';
  readonly description = 'Spawn parallel agents to investigate database questions';
  readonly parameters = {
    type: 'object',
    properties: {
      questions: {
        type: 'array',
        items: { type: 'string' },
        description: 'Questions to investigate in parallel',
      },
    },
    required: ['questions'],
  };

  private _systemPrompt: string;
  private _maxTurns: number;
  private _trace: boolean;
  private _toolkit: Toolkit | null = null;

  constructor(opts: { systemPrompt: string; maxTurns: number; trace: boolean }) {
    super();
    this._systemPrompt = opts.systemPrompt;
    this._maxTurns = opts.maxTurns;
    this._trace = opts.trace;
  }

  setToolkit(toolkit: Toolkit): void {
    this._toolkit = toolkit;
  }

  *execute(args: { questions: string[] }): Operation<unknown> {
    if (!this._toolkit)
      throw new Error('DatabaseResearchTool: call setToolkit() first');
    const questions = args.questions;
    if (!Array.isArray(questions) || questions.length === 0)
      return { error: 'questions must be a non-empty array' };

    const toolkit = this._toolkit;
    const systemPrompt = this._systemPrompt;

    return yield* withSharedRoot(
      { systemPrompt, tools: toolkit.toolsJson },
      function*(root) {
        const pool = yield* useAgentPool({
          tasks: questions.map(q => ({
            systemPrompt,
            content: q,
            tools: toolkit.toolsJson,
            parent: root,
          })),
          tools: toolkit.toolMap,
          terminalTool: 'report',
          maxTurns: this._maxTurns,
          trace: this._trace,
        });

        return {
          findings: pool.agents.map(a => a.findings).filter(Boolean),
          agentCount: pool.agents.length,
          totalTokens: pool.totalTokens,
          totalToolCalls: pool.totalToolCalls,
        };
      }.bind(this),
    );
  }
}
```

### 4. Wire the toolkit with correct ordering

Tool ordering in the toolkit array affects model behavior. Terminal tools (`report`) must not be last in the array -- place them before any self-referential tool that shares a name prefix. This prevents LLM recency bias from favoring early termination.

```typescript
// Correct: report before research
const toolkit = createToolkit([queryTool, describeTool, reportTool, researchTool]);

// Wrong: report last -- model favors "report" over "research" at decision boundaries
const toolkit = createToolkit([queryTool, describeTool, researchTool, reportTool]);
```

### 5. Implement the Source class

```typescript
class DatabaseSource extends Source<SourceContext, Chunk> {
  readonly name = 'database';

  private _db: Database;
  private _queryTool: QueryTool;
  private _describeTool: DescribeTool;
  private _researchTool: DatabaseResearchTool | null = null;
  private _results: Chunk[] = [];
  private _bound = false;

  constructor(db: Database) {
    super();
    this._db = db;
    this._queryTool = new QueryTool(db);
    this._describeTool = new DescribeTool(db);
  }

  get researchTool(): Tool {
    if (!this._researchTool)
      throw new Error('DatabaseSource: bind() must be called first');
    return this._researchTool;
  }

  get groundingTools(): Tool[] {
    return [this._queryTool, this._describeTool];
  }

  *bind(ctx: SourceContext): Operation<void> {
    if (this._bound) return;

    const systemPrompt = `You are a database research agent. Use query() to run SQL and describe() to inspect table schemas. Report your findings when you have enough information.`;

    const research = new DatabaseResearchTool({
      systemPrompt,
      maxTurns: ctx.maxTurns,
      trace: ctx.trace,
    });

    // Tool ordering: grounding tools, then report, then research
    const toolkit = createToolkit([
      this._queryTool,
      this._describeTool,
      ctx.reportTool,
      research,
    ]);
    research.setToolkit(toolkit);
    this._researchTool = research;
    this._bound = true;
  }

  getChunks(): Chunk[] {
    return this._results;
  }
}
```

### 6. Use in a pipeline

```typescript
const sources: Source<SourceContext, Chunk>[] = [];

const db = await connectDatabase(connectionString);
sources.push(new DatabaseSource(db));

// Optionally combine with other sources
if (process.env.TAVILY_API_KEY) {
  sources.push(new WebSource(new TavilyProvider()));
}

const result = yield* handleQuery(query, { ...opts, sources });
```

## The bind lifecycle

`bind()` is called by the harness before research starts. It receives a `SourceContext`:

```typescript
interface SourceContext {
  reranker: Reranker;
  reporterPrompt: { system: string; user: string };
  reportTool: Tool;
  maxTurns: number;
  trace: boolean;
}
```

Use `bind()` to:
- Initialize the reranker with your chunks (if using semantic search)
- Build the research toolkit with the shared `reportTool`
- Set up any runtime state that depends on pipeline configuration

Make `bind()` idempotent -- the harness may call it multiple times across warm queries. Guard with a `_bound` flag.

## Buffering chunks for reranking

`getChunks()` is called after research completes. The harness passes chunks through the reranker to select the most relevant passages for synthesis.

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

Buffer data during research (in tool execute methods) and return it from `getChunks()`. The WebSource pattern: `BufferingFetchPage` pushes fetched pages into a buffer during research, `getChunks()` converts the buffer to chunks via `chunkFetchedPages()`.

For a database source, you might buffer query results:

```typescript
*execute(args: { sql: string }): Operation<unknown> {
  const rows = yield* call(() => this._db.query(args.sql));
  // Buffer for post-research reranking
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

`groundingTools` returns tools the synthesis agent can use to independently verify research findings. These are typically the same retrieval tools your research agents use, minus the research tool itself (synthesis does not spawn sub-agents).

```typescript
get groundingTools(): Tool[] {
  return [this._queryTool, this._describeTool];
}
```

The harness aggregates grounding tools from all sources:

```typescript
const groundingTools = opts.sources.flatMap(s => s.groundingTools);
const synthToolkit = createToolkit([...groundingTools, reportTool]);
```

This gives the synthesis agent direct access to all data backends for cross-referencing claims against primary sources.
