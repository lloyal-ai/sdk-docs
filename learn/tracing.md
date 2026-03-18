# Tracing

When an agent pool drops agents unexpectedly, a tool returns surprising results, or a scratchpad extraction fails silently, the trace file tells you exactly what happened. lloyal emits structured trace events covering every prompt format, branch lifecycle, tool dispatch, pressure snapshot, and agent termination. This page covers how to enable tracing, what the events mean, and how to diagnose common issues.

## Enabling traces

Pass a `JsonlTraceWriter` to `initAgents`:

```typescript
import * as fs from 'node:fs';
import { JsonlTraceWriter } from '@lloyal-labs/lloyal-agents';

const fd = fs.openSync('trace.jsonl', 'w');
const { session, events } = yield* initAgents(ctx, {
  traceWriter: new JsonlTraceWriter(fd),
});
```

The reference pipeline supports a `--trace` flag that creates the writer automatically:

```bash
npx tsx examples/deep-research-web/main.ts ./models/Qwen3-4B-Q4_K_M.gguf \
  --query "What are the health effects of intermittent fasting?" \
  --trace
```

This produces a `trace.jsonl` file in the working directory -- one JSON object per line.

When tracing is not enabled, `initAgents` sets a `NullTraceWriter` that returns 0 from `nextId()` and no-ops every `write()` call. There is zero runtime cost when tracing is off.

### How the writer works

`JsonlTraceWriter` takes an open file descriptor and buffers up to 64 events before flushing with `fs.writeSync`. Flush also occurs at every `scope:close` event, guaranteeing that scope boundaries are always persisted even if the process crashes mid-pipeline. Write failures are silently swallowed -- tracing must never crash the runtime.

```typescript
class JsonlTraceWriter implements TraceWriter {
  private _fd: number;
  private _nextId = 1;
  private _buffer: string[] = [];

  nextId(): TraceId { return this._nextId++; }

  write(event: TraceEvent): void {
    this._buffer.push(JSON.stringify(event));
    if (this._buffer.length >= 64) this.flush();
  }

  flush(): void {
    if (this._buffer.length === 0) return;
    const data = this._buffer.join('\n') + '\n';
    this._buffer.length = 0;
    try { require('node:fs').writeSync(this._fd, data); }
    catch { /* non-fatal */ }
  }
}
```

## What the trace captures

Every trace event carries a `traceId` (monotonically increasing), a `parentTraceId` (for building the trace tree), a `ts` (millisecond timestamp from `performance.now()`), and a `type` discriminant. The full event union covers the following categories.

### Scope lifecycle

Scopes bracket every major operation -- agent pools, tool dispatches, shared-root regions, generation passes. Each scope emits an open/close pair:

- **`scope:open`** -- `{ name, meta? }`. The `name` is a human-readable label like `"pool"`, `"tool:search"`, `"sharedRoot"`. Optional `meta` carries key-value context.
- **`scope:close`** -- `{ name, durationMs }`. Elapsed time since the matching `scope:open`.

The `parentTraceId` on `scope:open` points to the enclosing scope, building a tree you can walk to understand nesting.

### Prompt formatting

- **`prompt:format`** -- emitted every time a prompt is assembled. Contains the full `promptText` (the formatted chat template output), `tokenCount`, the raw `messages` JSON, optional `tools` and `grammar` strings, and a `role` discriminating the context:
  - `'sharedRoot'` -- the prefix decoded into a shared root branch
  - `'agentSuffix'` -- the per-agent unique portion (user message + generation prompt)
  - `'generate'` -- a standalone `generate()` call (e.g., eval, bridge)
  - `'diverge'` -- the prompt used for diverge attempts
  - `'toolResultDelta'` -- tool result injected into an agent's branch

This is the most useful event for debugging prompt issues -- you can see the exact text the model received.

### Branch events

- **`branch:create`** -- `{ branchHandle, parentHandle, position, role }`. Roles: `'root'`, `'sharedRoot'`, `'agentFork'`, `'scratchpad'`, `'divergeAttempt'`.
- **`branch:prefill`** -- `{ branchHandle, tokenCount, role }`. Roles: `'sharedPrefix'`, `'agentSuffix'`, `'toolResult'`, `'warmDelta'`, `'scratchpad'`.
- **`branch:prune`** -- `{ branchHandle, position }`. The branch's KV cells are freed.

### Generation events

- **`generate:start`** -- `{ branchHandle, hasGrammar, hasParent, role }`. `hasParent` is true for scratchpad extractions (forked from an existing branch). `role` matches the generation context.
- **`generate:end`** -- `{ branchHandle, tokenCount, output, parsed? }`. The raw `output` string and the `parsed` result (if grammar-constrained generation with a parse function).

### Agent pool events

- **`pool:open`** -- `{ agentCount, taskSuffixTokens, pressure: { remaining, softLimit, headroom } }`. Emitted at pool creation with the initial pressure snapshot and per-task suffix token counts.
- **`pool:close`** -- `{ agents: [{ agentId, tokenCount, toolCallCount, findings, ppl }], totalTokens, steps, durationMs }`. Final accounting for every agent in the pool.
- **`pool:tick`** -- `{ phase, activeAgents, pressure: { remaining, cellsUsed, nCtx, headroom } }`. Emitted at every phase transition. The pressure snapshot tells you exactly how much KV budget was available at each point.
- **`pool:agentDrop`** -- `{ agentId, reason }`. Why an agent was terminated. Reasons:
  - `'pressure_init'` -- dropped at pool setup because suffix didn't fit
  - `'pressure_critical'` -- killed by the hard-cut during PRODUCE (remaining < hardLimit)
  - `'pressure_softcut'` -- non-terminal tool call denied because headroom was negative
  - `'pressure_settle_reject'` -- tool result exceeded headroom during SETTLE
  - `'maxTurns'` -- hit the turn limit without calling the terminal tool
  - `'stop_token'` -- model produced a stop token without a parseable tool call

### Agent turn output

- **`agent:turn`** -- `{ agentId, turn, rawOutput, parsedContent, parsedToolCalls: [{ name, arguments }] }`. Emitted after each generation round. Shows exactly what the model produced and how it was parsed.

### Tool events

- **`tool:dispatch`** -- `{ agentId, tool, toolIndex, toolkitSize, args, callId }`. The `toolIndex` is the position of this tool in the toolkit array -- this matters because tool ordering affects model preference.
- **`tool:result`** -- `{ agentId, tool, result, prefillTokenCount, durationMs }`. How large the result was in tokens, how long the tool took.
- **`tool:error`** -- `{ agentId, tool, error }`. Tool execution threw an exception.

### Diverge events

- **`diverge:start`** -- `{ attempts, prefixLength }`. How many branches are being forked.
- **`diverge:end`** -- `{ bestIdx, ppls, outputs, totalTokens }`. Per-branch perplexity scores and the winning index.

### Reranker events

- **`rerank:start`** -- `{ query, chunkCount }`. The query being scored against and how many chunks are being evaluated.
- **`rerank:end`** -- `{ topResults: [{ file, heading, score }], selectedPassageCount, totalChars, durationMs }`. The top-scored results with their relevance scores.

### Source events

- **`source:bind`** -- `{ sourceName }`. Emitted when a source completes late-binding.
- **`source:research`** -- `{ sourceName, questions }`. Research phase starting for a source.
- **`source:chunks`** -- `{ sourceName, chunkCount }`. Chunks collected after research completes.

## Reading a trace

The trace file is newline-delimited JSON. Python one-liners for common queries:

**Load all events:**

```bash
python3 -c "
import json
events = [json.loads(l) for l in open('trace.jsonl')]
print(f'{len(events)} events')
"
```

**Agent trajectory -- what tools did agent 0 call and in what order:**

```bash
python3 -c "
import json
events = [json.loads(l) for l in open('trace.jsonl')]
for e in events:
  if e['type'] == 'tool:dispatch' and e['agentId'] == 0:
    print(f\"  turn {e['toolIndex']}: {e['tool']}({list(e['args'].keys())})\")
  if e['type'] == 'tool:result' and e['agentId'] == 0:
    print(f\"    -> {e['prefillTokenCount']} tokens, {e['durationMs']:.0f}ms\")
"
```

**Prefill budget -- how much KV was used at each pool tick:**

```bash
python3 -c "
import json
events = [json.loads(l) for l in open('trace.jsonl')]
for e in events:
  if e['type'] == 'pool:tick' and e['phase'] == 'PRODUCE':
    p = e['pressure']
    pct = 100 * p['cellsUsed'] / p['nCtx']
    print(f\"  agents={e['activeAgents']} cells={p['cellsUsed']}/{p['nCtx']} ({pct:.0f}%) headroom={p['headroom']}\")
"
```

**Pressure timeline -- watch headroom shrink as tool results accumulate:**

```bash
python3 -c "
import json
events = [json.loads(l) for l in open('trace.jsonl')]
for e in events:
  if e['type'] == 'pool:tick':
    p = e['pressure']
    bar = '#' * max(0, int(40 * p['remaining'] / p['nCtx']))
    print(f\"  {e['phase']:8s} [{bar:40s}] {p['remaining']:5d} remaining\")
"
```

**Drop reasons -- why were agents terminated:**

```bash
python3 -c "
import json
events = [json.loads(l) for l in open('trace.jsonl')]
drops = [e for e in events if e['type'] == 'pool:agentDrop']
for d in drops:
  print(f\"  agent {d['agentId']}: {d['reason']}\")
if not drops:
  print('  no drops')
"
```

**Reranker scoring -- what passages were selected and their scores:**

```bash
python3 -c "
import json
events = [json.loads(l) for l in open('trace.jsonl')]
for e in events:
  if e['type'] == 'rerank:end':
    print(f\"  {e['selectedPassageCount']} passages, {e['totalChars']} chars, {e['durationMs']:.0f}ms\")
    for r in e['topResults'][:5]:
      print(f\"    {r['score']:.3f} {r['heading']} ({r['file']})\")
"
```

## Diagnosing issues

### Tool ordering

The `toolIndex` field in `tool:dispatch` tells you where a tool sits in the toolkit array. Models prefer tools that appear earlier in the system prompt's tool listing. If an agent never calls a tool you expected it to use, check its position:

```bash
python3 -c "
import json
events = [json.loads(l) for l in open('trace.jsonl')]
for e in events:
  if e['type'] == 'tool:dispatch':
    print(f\"  agent {e['agentId']}: {e['tool']} (index {e['toolIndex']}/{e['toolkitSize']})\")
"
```

If `search` is at index 3 and `grep` is at index 0, agents will prefer grep even when semantic search would be more appropriate. The `CorpusSource.bind()` method explicitly unshifts the search tool to position 0 for this reason:

```typescript
this._tools.unshift(new SearchTool(this._chunks, ctx.reranker));
```

### KV pressure drops

When agents are dropped by pressure, the trace shows exactly where the budget ran out. Look for `pool:agentDrop` events and correlate with the preceding `pool:tick` pressure snapshots:

```bash
python3 -c "
import json
events = [json.loads(l) for l in open('trace.jsonl')]
for i, e in enumerate(events):
  if e['type'] == 'pool:agentDrop':
    # Find the most recent tick before this drop
    for j in range(i-1, -1, -1):
      if events[j]['type'] == 'pool:tick':
        p = events[j]['pressure']
        print(f\"  agent {e['agentId']} dropped ({e['reason']})\")
        print(f\"    at: {p['cellsUsed']}/{p['nCtx']} cells, headroom={p['headroom']}\")
        break
"
```

Common patterns:
- **`pressure_init` drops** -- too many agents for the available KV. Reduce agent count or increase `nCtx`.
- **`pressure_settle_reject`** -- a large tool result (web page, long file) consumed the remaining budget. The `tool:result` event immediately before the drop shows `prefillTokenCount` -- that is the tool result that triggered the drop.
- **`pressure_critical`** -- the hard-cut safety net fired. This means `cellsUsed` grew past `nCtx - hardLimit` (default 128). Check for tools that produce very large results without scratchpad extraction.

### Scratchpad extraction failures

Scratchpad extractions (e.g., BufferingFetchPage extracting a summary from a web page) use `generate()` with a `parent` branch. When they work, you see a `generate:start` with `hasParent: true` followed by `generate:end` with a parsed result. When they fail silently, you see the `generate:start` but the corresponding `generate:end` either has no `parsed` field or the `output` is malformed.

```bash
python3 -c "
import json
events = [json.loads(l) for l in open('trace.jsonl')]
for e in events:
  if e['type'] == 'generate:start' and e.get('hasParent'):
    print(f\"  scratchpad start: branch={e['branchHandle']} grammar={e['hasGrammar']}\")
  if e['type'] == 'generate:end' and e.get('parsed') is None:
    # Potential scratchpad failure -- check if this was a parent generation
    print(f\"  generate:end with no parsed result: branch={e['branchHandle']} tokens={e['tokenCount']}\")
    print(f\"    output preview: {e['output'][:120]}...\")
"
```

If scratchpad extractions consistently fail, check:
- Is the parent branch (`ScratchpadParent`) still alive? A disposed parent means the shared root was pruned before the tool finished.
- Is the grammar correct? The `generate:start` event shows `hasGrammar: true` but does not include the grammar itself. Check the preceding `prompt:format` event with `role: 'generate'` for the full grammar string.

### Prompt debugging

When an agent behaves unexpectedly, the `prompt:format` event shows exactly what it received. Filter by role to find the relevant prompt:

```bash
python3 -c "
import json
events = [json.loads(l) for l in open('trace.jsonl')]
for e in events:
  if e['type'] == 'prompt:format' and e['role'] == 'agentSuffix':
    print(f\"  agent suffix ({e['tokenCount']} tokens):\")
    print(f\"    {e.get('taskContent', '')[:200]}...\")
    print()
"
```

The `promptText` field contains the full chat-template-formatted string -- system prompt, tool schemas, user message, and generation header. This is the exact sequence of tokens the model sees.

### Scope tree reconstruction

Every trace event carries `traceId` and `parentTraceId`. You can reconstruct the full scope tree to understand nesting -- which tool dispatch happened inside which agent pool, which pool was spawned by which tool call:

```bash
python3 -c "
import json
events = [json.loads(l) for l in open('trace.jsonl')]
scopes = {}
for e in events:
  if e['type'] == 'scope:open':
    scopes[e['traceId']] = e
    depth = 0
    pid = e.get('parentTraceId')
    while pid and pid in scopes:
      depth += 1
      pid = scopes[pid].get('parentTraceId')
    print(f\"{'  ' * depth}{e['name']} (id={e['traceId']})\")
  if e['type'] == 'scope:close':
    print(f\"  closed: {e['name']} ({e['durationMs']:.0f}ms)\")
"
```

This is particularly useful for understanding recursive tool calls. A `web_research` tool spawns an inner agent pool, which dispatches `fetch_page`, which runs a scratchpad extraction. The scope tree shows the full nesting.

## Next steps

> [Debugging Guide](../guides/debugging.md) covers common failure modes, KV cache sizing, and strategies for reducing pressure in deep pipelines.
