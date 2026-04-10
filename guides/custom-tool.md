---
title: "Build a Custom Tool"
description: "Create tools for agents — extend Tool<TArgs>, handle async work, build recursive tools, and pass to createAgentPool()."
---

This guide walks through creating tools for agents to use during generation. Tools extend the `Tool<TArgs>` base class and are passed directly to `createAgent()` or `createAgentPool()`.

## Tool anatomy

From `packages/agents/src/Tool.ts`, the base class:

```typescript
abstract class Tool<TArgs = Record<string, unknown>> {
  abstract readonly name: string;
  abstract readonly description: string;
  abstract readonly parameters: JsonSchema;
  abstract execute(args: TArgs, context?: ToolContext): Operation<unknown>;
}
```

| Member | Purpose |
|--------|---------|
| `name` | Function identifier in tool calls. Must be unique within a toolkit. |
| `description` | Shown to the model in the system prompt. Drives tool selection. |
| `parameters` | JSON Schema describing expected arguments. |
| `execute()` | Generator function that performs the work and returns the result. |
| `probe` | Optional getter returning `string \| null`. When non-null, the pool prefills this text after the tool result settles — nudging the model to reason before the next tool call. Default: `null` (noop). |

The `schema` getter (inherited) auto-generates the OpenAI-compatible function schema from these fields. You never construct the schema manually.

## Step 1: Extend Tool

Define your argument type as the generic parameter:

```typescript
import type { Operation } from 'effection';
import { Tool } from '@lloyal-labs/lloyal-agents';
import type { JsonSchema } from '@lloyal-labs/lloyal-agents';

class TimestampTool extends Tool<{ format?: string }> {
  readonly name = 'timestamp';
  readonly description = 'Get the current timestamp in the specified format';
  readonly parameters: JsonSchema = {
    type: 'object',
    properties: {
      format: {
        type: 'string',
        description: 'Output format: "iso", "unix", or "human" (default: "iso")',
      },
    },
    required: [],
  };

  *execute(args: { format?: string }): Operation<unknown> {
    const now = new Date();
    switch (args.format) {
      case 'unix': return { timestamp: Math.floor(now.getTime() / 1000) };
      case 'human': return { timestamp: now.toLocaleString() };
      default: return { timestamp: now.toISOString() };
    }
  }
}
```

Key points:
- `execute()` is a generator function (note the `*`). It returns `Operation<unknown>`.
- The return value is JSON-serialized and prefilled into the agent's KV cache as a tool result.
- Keep return values compact. Large results consume KV tokens and can trigger pressure-based agent shutdowns.

## Step 2: Handle async work with call()

Tools run inside the agent pool's Effection scope. For async operations, wrap in `call()`:

```typescript
import { call } from 'effection';

class HttpTool extends Tool<{ url: string; method?: string }> {
  readonly name = 'http_request';
  readonly description = 'Make an HTTP request and return the response body';
  readonly parameters: JsonSchema = {
    type: 'object',
    properties: {
      url: { type: 'string', description: 'URL to request' },
      method: { type: 'string', description: 'HTTP method (default: GET)' },
    },
    required: ['url'],
  };

  *execute(args: { url: string; method?: string }): Operation<unknown> {
    const response = yield* call(async () => {
      const res = await fetch(args.url, { method: args.method || 'GET' });
      if (!res.ok) return { error: `HTTP ${res.status}`, status: res.status };
      const text = await res.text();
      // Truncate to keep KV cost bounded
      return { body: text.slice(0, 4000), status: res.status };
    });
    return response;
  }
}
```

`call()` from Effection handles Promises, Operations, and plain values uniformly. Always use it for async work -- do not `await` inside a generator.

## Step 3: Accept constructor dependencies

Tools are instantiated before the agent pool runs. Pass data dependencies through the constructor:

```typescript
class GrepTool extends Tool<{ pattern: string; ignoreCase?: boolean }> {
  readonly name = 'grep';
  readonly description = 'Search the corpus for a regex pattern';
  readonly parameters: JsonSchema = {
    type: 'object',
    properties: {
      pattern: { type: 'string', description: 'Regex pattern' },
      ignoreCase: { type: 'boolean', description: 'Case-insensitive (default: true)' },
    },
    required: ['pattern'],
  };

  private _resources: Resource[];

  constructor(resources: Resource[]) {
    super();
    this._resources = resources;
  }

  *execute(args: { pattern: string; ignoreCase?: boolean }): Operation<unknown> {
    const flags = (args.ignoreCase === false) ? 'g' : 'gi';
    let re: RegExp;
    try { re = new RegExp(args.pattern, flags); }
    catch { return { error: `Invalid regex: ${args.pattern}` }; }

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

## Step 4: Build a recursive tool

Tools can spawn sub-agents. Instead of writing a custom recursive tool, use `createAgentPool()` with `recursive`. It handles the circular toolkit wiring internally:

```typescript
import { createAgentPool } from '@lloyal-labs/lloyal-agents';

const pool = yield* createAgentPool({
  tasks: questions.map(q => ({ content: q })),
  tools: [searchTool, grepTool, readFileTool, reportTool],
  systemPrompt: RESEARCH_PROMPT,
  terminalTool: 'report',
  maxTurns: 20,
  recursive: {
    name: 'research',
    description: 'Spawn parallel sub-agents to investigate sub-questions.',
    argsSchema: {
      type: 'object',
      properties: { questions: { type: 'array', items: { type: 'string' } } },
      required: ['questions'],
    },
    extractTasks: (args) => args.questions as string[],
  },
});
```

`createAgentPool()` creates the recursive delegate tool, adds it to the toolkit, and wires the circular reference. Agents see `research` as a callable tool. When they call it, another `createAgentPool()` runs internally with the same config. Recursion to arbitrary depth, bounded by KV.

## Step 5: Pass tools to the pool

Pass Tool instances directly to `createAgent` or `createAgentPool`. The SDK builds the toolkit internally — serializes schemas for the prompt and builds the dispatch map:

```typescript
const pool = yield* createAgentPool({
  tasks: questions.map(q => ({ content: q })),
  tools: [
    new SearchTool(chunks, reranker),
    new ReadFileTool(resources),
    new GrepTool(resources),
    reportTool,
  ],
  systemPrompt: RESEARCH_PROMPT,
  terminalTool: 'report',
  maxTurns: 20,
});
```

When `recursive` is set, the recursive delegate tool is added to the toolkit automatically — you don't include it in the tools array.

## Tool ordering rules

The order of tools in the `tools` array matters. Tool schemas are serialized into the system prompt, and LLM recency bias affects tool selection at generation decision boundaries.

**Rule: terminal tools must not be last in the array.** Place them before any tool that shares a name prefix.

```typescript
// Correct: report before research (both start with "r")
tools: [search, readFile, grep, report, research]

// Wrong: report last -- model favors early termination
tools: [search, readFile, grep, research, report]
```

This was the root cause of a real regression where agents lost depth-first investigation behavior. The grammar is order-agnostic (it allows all valid tokens), but the system prompt ordering shifts the model's logit distribution over tool names.

## Terminal tool pattern

A terminal tool signals agent completion. The agent pool intercepts the call, extracts findings, and marks the agent done. The tool's `execute()` is never called.

```typescript
class ReportTool extends Tool<{ result: string }> {
  readonly name = 'report';
  readonly description = 'Submit your final research findings';
  readonly parameters: JsonSchema = {
    type: 'object',
    properties: {
      result: { type: 'string', description: 'Your findings and answer' },
    },
    required: ['result'],
  };

  *execute(): Operation<unknown> { return {}; }
}
```

Register it as `terminalTool` in the pool options:

```typescript
const pool = yield* createAgentPool({
  tasks: [{ content: query }],
  tools: [searchTool, reportTool],
  systemPrompt: RESEARCH_PROMPT,
  terminalTool: 'report',  // Matches ReportTool.name
});
```

The pool enforces a work-before-report rule: if the tool registry contains non-terminal tools, the first tool call must be non-terminal. This prevents agents from immediately reporting without doing any research. Reporter sub-pools (which only have the report tool) skip this check.

## Error handling

Return error objects instead of throwing. Thrown errors terminate the agent; returned errors are prefilled as tool results and the agent can adapt:

```typescript
*execute(args: { pattern: string }): Operation<unknown> {
  if (!args.pattern?.trim())
    return { error: 'pattern must not be empty' };

  try {
    const re = new RegExp(args.pattern, 'gi');
    // ... search logic
  } catch {
    return { error: `Invalid regex: ${args.pattern}` };
  }
}
```

Zero-result responses should include guidance:

```typescript
if (matches.length === 0) {
  return {
    totalMatches: 0,
    matches: [],
    note: 'Zero matches. Try search() for semantic matching or a broader regex.',
  };
}
```

This helps the model adjust its strategy on the next turn rather than repeating the same failing call.

## ToolContext

`execute()` receives an optional `ToolContext` with the calling agent's branch and a progress callback:

```typescript
interface ToolContext {
  agentId: number;
  branch?: Branch;                                         // calling agent's branch
  onProgress?: (p: { filled: number; total: number }) => void;
  scorer?: EntailmentScorer;                               // entailment scorer for semantic coherence
  explore?: boolean;                                       // true = explore mode, false = exploit
  pressurePercentAvailable?: number;                       // KV % available at DISPATCH time
}
```

### Recursive forking

If your tool spawns sub-agents, pass `context.branch` as `parent` for [warm path forking](/reference/prefix-sharing#warm-path-fork-from-parent). Sub-agents inherit the calling agent's full attention state:

```typescript
*execute(args: { questions: string[] }, context?: ToolContext): Operation<unknown> {
  return yield* createAgentPool({
    tasks: args.questions.map(q => ({ content: q })),
    tools: [searchTool, reportTool],
    systemPrompt: RESEARCH_PROMPT,
    terminalTool: 'report',
    parent: context?.branch,
  });
}
```

### Progress reporting

Report progress for long-running operations:

```typescript
*execute(args: { query: string }, ctx?: ToolContext): Operation<unknown> {
  const chunks = this._chunks;
  for (let i = 0; i < chunks.length; i++) {
    ctx?.onProgress?.({ filled: i + 1, total: chunks.length });
    // ... process chunk
  }
  return results;
}
```

Progress events surface as `agent:tool_progress` in the event channel, enabling TUI updates during slow operations.

## Reasoning probes

Override the `probe` getter to prefill text after the tool result settles. The probe is injected between the tool result and the grammar reset, giving the model unconstrained prose space to reason about the result before deciding the next action.

```typescript
class SearchTool extends Tool<{ query: string }> {
  readonly name = 'search';
  // ...

  get probe() { return 'Wait, '; }
}
```

The pool handles the lifecycle:
1. Tool result prefilled into agent's KV cache
2. Probe text prefilled immediately after
3. Lazy grammar reset — model generates freely until a trigger fires
4. Model reasons about the result, then produces the next tool call or reports

Probes only fire after real tool dispatches. Nudges, settle rejects, and terminal tool interceptions are unaffected. When `probe` returns `null` (the default), no extra prefill happens — zero cost.

<Note>
Most tools don't need probes. Use them when a tool returns dense results that benefit from explicit intermediate reasoning — semantic search results, large document reads, complex API responses where the model should synthesize before acting.
</Note>
