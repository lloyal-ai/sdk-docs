---
title: "React Agent"
description: "Single agent with corpus tools — the simplest example of ReAct (Reason + Act) with the lloyal HDK."
---

The simplest example: one agent with corpus tools answers a question using the ReAct (Reason + Act) pattern. The agent searches a local knowledge base, reads matching sections, and reports findings with evidence. No pools, no DAGs — just `useAgent` running a single research loop.

**Source**: `examples/react-agent/`

## Prerequisites

- A GGUF instruction-tuned model with native tool calling (e.g. [Qwen3-4B-Instruct](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF))
- A GGUF reranker model (e.g. [Qwen3-Reranker-0.6B](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF))
- A text corpus (directory of `.md`/`.txt` files, or a single file)

## Run it

```bash
npx tsx examples/react-agent/main.ts ./models/Qwen3-4B-Q4_K_M.gguf \
  --corpus ./my-docs/ \
  --query "What are the main architectural decisions?"
```

Without `--query`, the agent drops into an interactive REPL. Other flags:

- `--reranker <path>` — custom reranker model path
- `--verbose` — show stderr from the inference backend
- `--trace` — emit raw prompt/completion traces
- `--jsonl` — machine-readable output (run once, no REPL)

## Code walkthrough

### `main.ts` — bootstrap

The entry point parses args, loads the model + reranker, builds corpus tools, and runs the REPL loop. The same shape as any HDK harness:

```typescript
const ctx = yield* call(() => createContext({ modelPath, nCtx: 16384, ... }));
const reranker = yield* call(() => createReranker(rerankModelPath, { ... }));
yield* call(() => reranker.tokenizeChunks(chunks));

const { toolMap, toolsJson } = createTools({ resources, chunks, reranker });
const { session, events } = yield* initAgents<WorkflowEvent>(ctx);
```

`createTools` builds four tools from the loaded corpus:

- `search` — semantic search via the reranker (returns ranked chunks)
- `grep` — regex pattern matching across all files
- `read_file` — read specific line ranges from a file
- `report` — submit final findings (the terminal tool)

`initAgents` returns the `Session` and a typed event channel. The TUI subscribes to the channel; pipeline logic emits events. Presentation and orchestration are decoupled.

### `harness.ts` — one `useAgent` call

The whole harness is ~45 lines. The core is a single `useAgent`:

```typescript
const agent = yield* useAgent({
  systemPrompt: RESEARCH.system,
  task: query,
  tools: [...opts.tools, reportTool],
  terminalTool: 'report',
  maxTurns: opts.maxTurns,
  trace: opts.trace,
  policy: new DefaultAgentPolicy({
    budget: { context: { softLimit: 2048 } },
  }),
});
```

`useAgent` handles everything: it creates a branch, formats the prompt through the model's chat template, prefills it, runs the [five-phase tick loop](/reference/concurrency#tick-loop) until the agent calls `report` (the terminal tool) or exhausts `maxTurns`, then prunes the branch on scope exit.

The `softLimit: 2048` tells the policy to nudge the agent when fewer than 2048 KV cells remain. At the default `hardLimit`, `shouldExit` kills the agent and `recoverInline` extracts whatever findings are in attention via grammar-constrained extraction.

### `tasks/research.md` — the system prompt

```markdown
You are a research assistant analyzing a knowledge base. Your tools:
- **search**: semantic relevance ranking — discover related content by meaning
- **grep**: regex pattern matching — use for precise, exhaustive retrieval
- **read_file**: read specific line ranges — verify and get full context
- **report**: submit your final findings with evidence

Research process:
1. Start with search to discover relevant content broadly.
2. Use grep with specific patterns to find precise references.
3. Read matching sections with read_file to verify in full context.
4. If gaps remain, search or grep with different terms.
5. When you have sufficient evidence, call report with your findings.
```

The prompt encodes a workflow: broad search → targeted grep → context read → report. Each tool serves a distinct purpose. Edit this markdown file to change agent behavior — no code changes needed.

## What the output looks like

```
ReAct Agent · Qwen3-4B-Instruct (KV: Q4_0) · qwen3-reranker-0.6b
Corpus: my-docs/ · 12 files · 847 chunks

> What are the main architectural decisions?

Research · 1 agent
  search("architectural decisions")        → 8 results
  read_file("design.md", 1, 45)            → 1.2k chars
  grep("decision|chose")                   → 3 matches
  report(...)

Answer · 147 tokens · 4 tools · 3.2s · ctx 34%
The main architectural decisions are...
```

## Customization

- **Change the system prompt** — edit `examples/react-agent/tasks/research.md`. No code changes.
- **Add agents** — swap `useAgent` for `agentPool({ orchestrate: parallel([...]) })` to run multiple agents on shared KV.
- **Increase tool turn budget** — change `MAX_TOOL_TURNS` in `main.ts` (default: 20).
- **Different corpus** — pass `--corpus <path>` pointing to any directory of text/markdown files, or a single file.
- **Adjust context size** — `--n-ctx <size>` or `LLAMA_CTX_SIZE` env var (default: 16384).

## Related pages

- [Quick Start](/learn/quick-start) — minimal standalone version of this example
- [Tools](/learn/tools) — how tools work under the hood
- [Concurrency Model](/reference/concurrency) — the five-phase tick loop that drives agent execution
- [KV Pressure](/reference/kv-pressure) — how budget settings affect agent lifecycle
- [Reflection](/examples/reflection) — same single-agent research, then continues with manual branch lifecycle for self-critique
