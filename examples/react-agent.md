---
title: "React Agent"
description: "Single agent with corpus tools — the simplest example of ReAct (Reason + Act) with lloyal-agents."
---

The simplest example: one agent with corpus tools answers a question using the ReAct (Reason + Act) pattern. The agent searches a local knowledge base, reads matching sections, and reports findings with evidence.

**Source**: `examples/react-agent/`

## Prerequisites

- A GGUF instruction-tuned model with tool calling support (e.g. [Qwen3-4B-Instruct](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF))
- A GGUF reranker model for semantic search (e.g. [Qwen3-Reranker-0.6B](https://huggingface.co/QuantFactory/Qwen3-Reranker-0.6B-GGUF))
- A text corpus (directory of `.md`/`.txt` files, or a single file)

## Run it

```bash
npx tsx examples/react-agent/main.ts ./models/Qwen3-4B-Q4_K_M.gguf \
  --corpus ./my-docs/ \
  --query "What are the main architectural decisions?"
```

Without `--query`, the agent drops into an interactive REPL where you can ask multiple questions. Other options:

- `--reranker <path>` -- custom reranker model path
- `--verbose` -- show stderr from the inference backend
- `--trace` -- emit raw prompt/completion traces
- `--jsonl` -- machine-readable output (run once, no REPL)

## Code walkthrough

### `main.ts` -- CLI entry point

The entry point handles argument parsing, model loading, and the REPL loop.

**Load the corpus and build tools:**

```typescript
const resources = loadResources(corpusDir!);
const chunks = chunkResources(resources);
```

`loadResources` reads files from the corpus path into memory. `chunkResources` splits them into paragraph-level chunks that the reranker can score individually. These are pure data transforms -- no model involvement yet.

**Load models:**

```typescript
const ctx: SessionContext = yield* call(() =>
  createContext({
    modelPath,
    nCtx: 16384,
    nSeqMax: 16,
    typeK: "q4_0",
    typeV: "q4_0",
  }),
);

const reranker = yield* call(() =>
  createReranker(rerankModelPath, { nSeqMax: 8, nCtx: 4096 }),
);
yield* call(() => reranker.tokenizeChunks(chunks));
```

`createContext` loads the main model and allocates a KV cache with Q4 quantization. `createReranker` loads a separate small model for scoring chunk relevance. `tokenizeChunks` pre-tokenizes every chunk through the reranker so scoring is fast at query time.

**Build tools and initialize agents:**

```typescript
const { toolMap, toolsJson } = createTools({ resources, chunks, reranker });
const { session, events } = yield* initAgents<WorkflowEvent>(ctx);
```

`createTools` builds four tools from the loaded resources:
- `search` -- semantic search via the reranker (returns ranked chunks)
- `grep` -- regex pattern matching across all files
- `read_file` -- read specific line ranges from a file
- `report` -- submit final findings (the terminal tool)

`initAgents` sets up the agent runtime and returns a typed event channel. The TUI subscribes to this channel to render progress:

```typescript
const view = createView({ model, reranker, chunkCount });
yield* spawn(function* () {
  yield* view.subscribe(events);
});
```

All presentation is decoupled from pipeline logic through the `Channel<WorkflowEvent>` boundary.

**REPL loop:**

```typescript
for (const input of yield* each(inputSignal)) {
  if (!input || input === "/quit") break;
  yield* handleQuery(input, harnessOpts);
  yield* each.next();
  rl.prompt();
}
```

Each query is dispatched to `handleQuery` in the harness. The Effection signal bridges Node's readline into structured concurrency scope.

### `harness.ts` -- the workflow

The harness is where the agent actually runs. It is short -- about 45 lines of pipeline code.

**Task loading:**

```typescript
function loadTask(name: string): { system: string; user: string } {
  const raw = fs.readFileSync(path.resolve(__dirname, `tasks/${name}.md`), 'utf8').trim();
  const sep = raw.indexOf('\n---\n');
  if (sep === -1) return { system: raw, user: '' };
  return { system: raw.slice(0, sep).trim(), user: raw.slice(sep + 5).trim() };
}

const RESEARCH = loadTask('research');
```

Task prompts live in `tasks/research.md`. The convention is system prompt above `---`, user content below. This separates prompt engineering from pipeline code -- modify the markdown file to change agent behavior without touching TypeScript.

**`handleQuery` -- the core pipeline:**

```typescript
export function* handleQuery(query: string, opts: HarnessOpts): Operation<void> {
  const { result: pool } = yield* withSharedRoot(
    { systemPrompt: RESEARCH.system, tools: opts.toolsJson },
    function*(root) {
      const pool = yield* useAgentPool({
        tasks: [{
          systemPrompt: RESEARCH.system,
          content: query,
          tools: opts.toolsJson,
          parent: root,
        }],
        tools: opts.toolMap,
        maxTurns: opts.maxTurns,
        terminalTool: 'report',
        trace: opts.trace,
        pressure: { softLimit: 2048 },
        extractionPrompt: REPORT,    // scratchpad extraction for hard-cut agents
        pruneOnReport: true,     // free KV immediately when agent reports
      });

      return { result: pool };
    },
  );
}
```

Two framework primitives do all the work:

1. `withSharedRoot` creates a shared KV prefix containing the system prompt and tool schemas. This prefix is computed once and shared across all agents that fork from `root`. When the callback exits, all branches under this root are pruned automatically.

2. `useAgentPool` runs one agent (a single-element `tasks` array) that generates tokens, calls tools, and reports findings. The agent runs inside the four-phase tick loop (produce, commit, settle, dispatch) until it calls `report` (the `terminalTool`) or exhausts `maxTurns`.

The `pressure: { softLimit: 2048 }` tells the pool to start considering context pressure when fewer than 2048 KV cells remain.

**`extractionPrompt` -- hard-cut recovery:**

If the agent exhausts `maxTurns` or is killed by KV pressure without calling `report`, the pool automatically recovers partial findings. It forks from the agent's branch, runs a grammar-constrained extraction (`{ result: string }`), and records the result with `scratchpad` provenance. A confabulation guard skips agents with fewer than 100 tokens or 2 tool calls — insufficient context would produce hallucinated findings.

### `tasks/research.md` -- the system prompt

```markdown
You are a research assistant analyzing a knowledge base. Your tools:
- **search**: semantic relevance ranking -- discover related content by meaning
- **grep**: regex pattern matching -- use for precise, exhaustive retrieval
- **read_file**: read specific line ranges -- verify and get full context
- **report**: submit your final findings with evidence

Research process:
1. Start with search to discover relevant content broadly.
2. Use grep with specific patterns to find precise references.
3. Read matching sections with read_file to verify in full context.
4. If gaps remain, search or grep with different terms.
5. When you have sufficient evidence, call report with your findings.
```

The prompt guides the agent through a specific research workflow: broad search first, then targeted grep, then read for context, then report. Each tool serves a distinct purpose -- search finds semantically related content, grep finds exact patterns, read_file provides full context for verification.

## What the output looks like

```
  ReAct Agent -- Single Agent with Tools

  * Loading Qwen3-4B-Instruct-2507 (2.5 GB, KV: Q4_0)
  * Loading qwen3-reranker-0.6b (396 MB, reranker)
    Corpus: my-docs/ -- 12 files -> 847 chunks

  Query
  What are the main architectural decisions?

  * Research  1 agent
    [A0] search("architectural decisions")     -> 8 results
    [A0] read_file("design.md", 1, 45)         -> 1.2k chars
    [A0] grep("pattern: 'decision|chose'")     -> 3 matches
    [A0] report(...)

  Answer (147 tokens, 4 tools, 3.2s, ctx: 34%)
  The main architectural decisions are...
```

## Customization

**Change the system prompt**: Edit `examples/react-agent/tasks/research.md`. No code changes needed.

**Add more agents**: Add more elements to the `tasks` array in `useAgentPool`. Each agent gets its own branch forking from the shared root.

**Increase tool turn budget**: Change `MAX_TOOL_TURNS` in `main.ts` (default: 20).

**Use a different corpus**: Pass `--corpus <path>` pointing to any directory of text/markdown files, or a single file.

**Adjust context size**: Set the `LLAMA_CTX_SIZE` environment variable (default: 16384).

## Related pages

- [Quick Start](/learn/quick-start) -- minimal standalone version of this example
- [Tools](/learn/tools) -- how tools work under the hood
- [Concurrency Model](/reference/concurrency) -- the four-phase tick loop that drives agent execution
- [Prefix Sharing](/reference/prefix-sharing) -- how `withSharedRoot` shares KV prefix across agents
- [KV Pressure](/reference/kv-pressure) -- how `pressure` settings affect agent lifecycle
