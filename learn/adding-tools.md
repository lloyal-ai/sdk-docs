# Adding Tools

Tools are how agents interact with the world -- search a corpus, fetch a web page, spawn sub-agents, or signal that work is done. This guide walks through building tools from simple to complex, grounded in the actual `Tool` base class and the reference implementations in `@lloyal-labs/rig`.

## The Tool base class

Every tool extends the abstract `Tool<TArgs>` class from `@lloyal-labs/lloyal-agents`:

```typescript
import type { Operation } from 'effection';

export abstract class Tool<TArgs = Record<string, unknown>> {
  abstract readonly name: string;
  abstract readonly description: string;
  abstract readonly parameters: JsonSchema;

  abstract execute(args: TArgs, context?: ToolContext): Operation<unknown>;

  get schema(): ToolSchema { /* auto-generated */ }
}
```

Four things to implement:

| Member | Purpose |
| --- | --- |
| `name` | Function identifier the model sees in tool calls |
| `description` | Natural-language description shown to the model |
| `parameters` | JSON Schema describing the tool's arguments |
| `execute()` | Generator method that does the work and returns a result |

The `schema` getter is auto-generated from the first three -- it produces the OpenAI-compatible function schema that `formatChat()` expects. You never construct it manually.

`execute()` returns an Effection `Operation<unknown>`, which means you implement it as a generator method (`*execute()`). The return value is JSON-serialized and prefilled back into the agent's KV cache as a tool result. The agent sees the full return value and decides what to do next.

`ToolContext` provides the calling agent's identity and an optional progress callback:

```typescript
interface ToolContext {
  agentId: number;
  onProgress?: (p: { filled: number; total: number }) => void;
}
```

## A simple tool

`GrepTool` is the minimal example -- synchronous logic, no async calls, no sub-agents. It scans loaded resources for regex matches and returns results directly.

```typescript
import type { Operation } from 'effection';
import { Tool } from '@lloyal-labs/lloyal-agents';
import type { JsonSchema } from '@lloyal-labs/lloyal-agents';
import type { Resource } from '../resources/types';

export class GrepTool extends Tool<{ pattern: string; ignoreCase?: boolean }> {
  readonly name = 'grep';
  readonly description = 'Search the entire corpus for a regex pattern. Returns every matching line with line numbers and total match count.';
  readonly parameters: JsonSchema = {
    type: 'object',
    properties: {
      pattern: { type: 'string', description: 'Regex pattern' },
      ignoreCase: { type: 'boolean', description: 'Case-insensitive matching (default: true)' },
    },
    required: ['pattern'],
  };

  private _resources: Resource[];

  constructor(resources: Resource[]) {
    super();
    this._resources = resources;
  }

  *execute(args: { pattern: string; ignoreCase?: boolean }): Operation<unknown> {
    const pattern = args.pattern?.trim();
    if (!pattern) return { error: 'pattern must not be empty' };
    const flags = (args.ignoreCase === false) ? 'g' : 'gi';
    let re: RegExp;
    try { re = new RegExp(pattern, flags); }
    catch { return { error: `Invalid regex: ${pattern}` }; }

    const matches: { file: string; line: number; text: string }[] = [];
    for (const res of this._resources) {
      const lines = res.content.split('\n');
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].match(re)) {
          matches.push({ file: res.name, line: i + 1, text: lines[i].trim() });
        }
      }
    }

    return { totalMatches: matches.length, matches: matches.slice(0, 50) };
  }
}
```

@category Rig

Key points:

- **`TArgs` type parameter** matches the JSON Schema. `{ pattern: string; ignoreCase?: boolean }` maps to the `properties` and `required` in `parameters`. The agent pool parses the model's JSON tool call and passes typed args to `execute()`.
- **Generator body, synchronous return.** Even though `execute()` returns `Operation<unknown>`, the body is plain synchronous code. The `*` makes it a generator; `return` yields the result. No `yield*` needed when there is no async work.
- **Constructor injects dependencies.** The tool receives `resources` at construction time, not at call time. Tools are instantiated once and reused across the agent pool's lifetime.
- **Error results, not thrown errors.** Return `{ error: '...' }` for expected failures (bad input, no matches). The agent sees the error as a tool result and can adjust. Thrown errors are caught by the pool's `scoped()` boundary and terminate the agent.

## A tool with async work

`SearchTool` performs semantic search using a reranker -- an async operation that scores chunks against the query. The `call()` function from Effection bridges the async reranker into the generator.

```typescript
import { call } from 'effection';
import type { Operation } from 'effection';
import { Tool, Trace } from '@lloyal-labs/lloyal-agents';
import type { JsonSchema, ToolContext } from '@lloyal-labs/lloyal-agents';
import type { Chunk } from '../resources/types';
import type { Reranker, ScoredChunk } from './types';

export class SearchTool extends Tool<{ query: string }> {
  readonly name = 'search';
  readonly description = 'Search the knowledge base. Returns sections ranked by relevance.';
  readonly parameters: JsonSchema = {
    type: 'object',
    properties: { query: { type: 'string', description: 'Search query' } },
    required: ['query'],
  };

  private _chunks: Chunk[];
  private _reranker: Reranker;

  constructor(chunks: Chunk[], reranker: Reranker) {
    super();
    this._chunks = chunks;
    this._reranker = reranker;
  }

  *execute(args: { query: string }, context?: ToolContext): Operation<unknown> {
    const query = args.query?.trim();
    if (!query) return { error: 'query must not be empty' };

    const results: ScoredChunk[] = yield* call(async () => {
      let last: ScoredChunk[] = [];
      for await (const { results, filled, total } of this._reranker.score(query, this._chunks)) {
        if (context?.onProgress) context.onProgress({ filled, total });
        last = results;
      }
      return last;
    });

    return results;
  }
}
```

@category Rig

Key points:

- **`yield* call(async () => { ... })`** is the pattern for async work inside a tool. `call()` takes a Promise-returning function, wraps it as an Operation, and respects cancellation -- if the agent pool exits while the reranker is scoring, the operation is halted cleanly.
- **`context?.onProgress`** reports progress to the agent pool. The pool can emit progress events to the UI while a long-running tool is in flight. This is optional -- tools that complete quickly can ignore the context entirely.
- **`yield* Trace.expect()`** reads the trace writer from Effection context. Tools that want to emit trace events (timing, intermediate results) use this to write structured JSONL. This is optional instrumentation, not required for the tool to work.

When to use `call()`:

| Situation | Pattern |
| --- | --- |
| Pure synchronous logic | `return result` -- no `yield*` needed |
| Single async call | `yield* call(() => someAsyncFn())` |
| Async iteration | `yield* call(async () => { for await (...) { ... } return result })` |
| Nested agent spawning | `yield* withSharedRoot(...)` -- see recursive tools below |

## Terminal tools

A terminal tool signals that an agent is done. When the agent pool detects a call to the designated `terminalTool`, it intercepts the call, extracts the arguments as findings, and marks the agent as finished. The tool's `execute()` is never actually called.

```typescript
import type { Operation } from 'effection';
import { Tool } from '@lloyal-labs/lloyal-agents';
import type { JsonSchema } from '@lloyal-labs/lloyal-agents';

export class ReportTool extends Tool<{ findings: string }> {
  readonly name = 'report';
  readonly description = 'Submit your final research findings. Call this when you have gathered enough information.';
  readonly parameters: JsonSchema = {
    type: 'object',
    properties: { findings: { type: 'string', description: 'Your research findings and answer' } },
    required: ['findings'],
  };

  *execute(): Operation<unknown> { return {}; }
}
```

@category Rig

The `execute()` body is a no-op because the agent pool intercepts terminal tool calls before dispatch. The tool exists so its schema is included in the tool list the model sees.

Terminal tools work through the `terminalTool` option on `useAgentPool`:

```typescript
const pool = yield* useAgentPool({
  tasks: [...],
  tools: toolMap,
  terminalTool: 'report',  // <-- name of the terminal tool
  maxTurns: 20,
});
```

The pool enforces one rule: **the terminal tool cannot be the agent's first tool call** (unless the tool registry contains only the terminal tool, as in reporter sub-agents). This prevents agents from immediately reporting without doing any work. If an agent tries to call `report` before calling any other tool, the pool rejects the call and prefills an error message asking the agent to do research first.

After the pool finishes, each agent's findings are available on `agent.findings`:

```typescript
const findings = pool.agents.map(a => a.findings).filter(Boolean);
```

Agents that hit `maxTurns` or context pressure without calling the terminal tool end up with `findings === undefined`. The harness can then force-extract findings from these "hard-cut" agents using a reporter pass.

## Recursive tools

The most powerful pattern is a tool that spawns its own agent pool. `ResearchTool` takes an array of questions and runs a parallel pool of research agents, each with access to the full toolkit.

```typescript
import type { Operation } from 'effection';
import { Tool, useAgentPool, withSharedRoot } from '@lloyal-labs/lloyal-agents';
import type { JsonSchema, Toolkit, PressureThresholds } from '@lloyal-labs/lloyal-agents';

export class ResearchTool extends Tool<{ questions: string[] }> {
  readonly name = 'research';
  readonly description = 'Spawn parallel research agents to investigate sub-questions.';
  readonly parameters: JsonSchema = {
    type: 'object',
    properties: {
      questions: {
        type: 'array',
        items: { type: 'string' },
        description: 'Sub-questions to research in parallel',
      },
    },
    required: ['questions'],
  };

  private _systemPrompt: string;
  private _maxTurns: number;
  private _toolkit: Toolkit | null = null;

  constructor(opts: { systemPrompt: string; maxTurns?: number }) {
    super();
    this._systemPrompt = opts.systemPrompt;
    this._maxTurns = opts.maxTurns ?? 20;
  }

  setToolkit(toolkit: Toolkit): void {
    this._toolkit = toolkit;
  }

  *execute(args: { questions: string[] }): Operation<unknown> {
    const questions = args?.questions;
    if (!Array.isArray(questions) || questions.length === 0) {
      return { error: 'questions must be a non-empty array of strings' };
    }
    if (!this._toolkit) throw new Error('ResearchTool: setToolkit() must be called before execute');

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
          maxTurns: 20,
        });

        return {
          findings: pool.agents.map(a => a.findings).filter(Boolean),
          agentCount: pool.agents.length,
        };
      },
    );
  }
}
```

@category Rig

This works because `execute()` returns an `Operation`, and Operations compose. The inner `withSharedRoot` + `useAgentPool` runs inside the outer pool's DISPATCH phase. The scope tree guarantees cleanup:

```
outer pool (DISPATCH phase)
  └─ tool.execute()
      └─ withSharedRoot()        ← inner root branch
          └─ useAgentPool()      ← inner pool with N sub-agents
              ├─ sub-agent 0
              ├─ sub-agent 1
              └─ sub-agent 2
```

When the inner pool finishes (or the outer pool is cancelled), all inner branches are pruned automatically. The tool returns aggregated findings to the outer agent as a normal tool result.

Key points:

- **`setToolkit()` wires the toolkit after construction.** This is needed because the toolkit includes the research tool itself (circular reference). Construct all tools first, build the toolkit with `createToolkit()`, then call `setToolkit()` to close the loop.
- **`withSharedRoot` creates a new KV prefix.** The inner pool's agents share a root that contains their system prompt and tool schemas, forked from the outer pool's context. This is efficient -- the shared prefix is computed once and reused across all sub-agents.
- **`parent: root`** in each task spec forks the sub-agent from the inner root, not from the outer agent's branch. Sub-agents start with a clean context containing only the system prompt, not the outer agent's conversation history.
- **The outer agent blocks during inner pool execution.** While the inner pool runs, the outer agent that called the research tool is suspended. Other agents in the outer pool are also suspended (tools execute sequentially in DISPATCH). The inner pool has full access to the GPU for its tick loop.

## Building a toolkit

`createToolkit()` aggregates tool instances into the two structures the agent pool needs:

```typescript
import { createToolkit } from '@lloyal-labs/lloyal-agents';

const { toolMap, toolsJson } = createToolkit([
  new SearchTool(chunks, reranker),
  new ReadFileTool(resources),
  new GrepTool(resources),
  new ReportTool(),
]);
```

- **`toolMap`** is a `Map<string, Tool>` used by `useAgentPool` to dispatch tool calls at runtime. When the model emits `{"name": "grep", "arguments": {...}}`, the pool looks up `toolMap.get('grep')` and calls its `execute()`.
- **`toolsJson`** is a JSON string of the tool schemas, passed to `formatChat()` via the task spec's `tools` field. This is what the model sees in its prompt -- the available functions with their descriptions and parameter schemas.

Always use `createToolkit()`. Never manually construct `toolMap` or `toolsJson` -- the toolkit ensures the dispatch map and schema string are consistent.

### Tool ordering

The order of tools in the array matters for the model. Tools listed first tend to get called more often by smaller models. Place the tools the agent should reach for first at the top of the array, and put less common tools later.

One important rule: **do not place the terminal tool last in the array.** Some models interpret the last tool in the list as a "default" and call it prematurely. The reference implementations place `report` after `grep` (third position in a four-tool toolkit) rather than at the end:

```typescript
// Good -- report is not last
createToolkit([searchTool, readFileTool, grepTool, reportTool]);

// Risky -- model may default to report too early
createToolkit([searchTool, readFileTool, reportTool, grepTool]);
```

### Wiring recursive tools

When a toolkit contains a recursive tool (like `ResearchTool`), you need a two-step setup:

```typescript
const searchTool = new SearchTool(chunks, reranker);
const readFileTool = new ReadFileTool(resources);
const grepTool = new GrepTool(resources);
const reportTool = new ReportTool();
const researchTool = new ResearchTool({
  systemPrompt: researchSystemPrompt,
  maxTurns: 20,
});

// Build toolkit with all tools including research
const toolkit = createToolkit([searchTool, readFileTool, grepTool, reportTool, researchTool]);

// Close the loop -- research tool now knows about the full toolkit
researchTool.setToolkit(toolkit);
```

The `setToolkit()` call must happen before any agent executes the research tool. Since tools are constructed before the pool starts, this is always safe.

## Writing your own tool

Here is the recipe:

1. **Extend `Tool<TArgs>`** with a type parameter matching your JSON Schema.
2. **Set `name`, `description`, and `parameters`** as readonly properties. The description should tell the model when and why to call the tool. The parameters schema should use `description` fields on each property -- models use these to fill in arguments correctly.
3. **Implement `*execute()`** as a generator method. Return the result directly for sync work, or use `yield* call(...)` for async work. Return error objects for expected failures; throw only for programming errors.
4. **Inject dependencies via the constructor.** Resources, API clients, rerankers -- anything the tool needs at runtime. Keep `execute()` focused on the tool's logic.
5. **Add to `createToolkit()`** alongside your other tools.

A minimal custom tool:

```typescript
import type { Operation } from 'effection';
import { call } from 'effection';
import { Tool } from '@lloyal-labs/lloyal-agents';
import type { JsonSchema } from '@lloyal-labs/lloyal-agents';

class DatabaseLookup extends Tool<{ table: string; id: string }> {
  readonly name = 'db_lookup';
  readonly description = 'Look up a record by ID from the database.';
  readonly parameters: JsonSchema = {
    type: 'object',
    properties: {
      table: { type: 'string', description: 'Table name' },
      id: { type: 'string', description: 'Record ID' },
    },
    required: ['table', 'id'],
  };

  private _db: Database;

  constructor(db: Database) {
    super();
    this._db = db;
  }

  *execute(args: { table: string; id: string }): Operation<unknown> {
    const record = yield* call(() => this._db.get(args.table, args.id));
    if (!record) return { error: `No record found: ${args.table}/${args.id}` };
    return record;
  }
}
```

## Next steps

- [Thinking in lloyal](thinking-in-lloyal.md) -- structured concurrency, generators, and scope-based cleanup
- [Quick Start](quick-start.md) -- run a research agent with the built-in corpus tools
