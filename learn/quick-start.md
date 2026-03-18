# Quick Start

Build a research agent that investigates a local document using search, grep, and file reading — in ~25 lines of pipeline code.

## Prerequisites

You need three files:

1. **A GGUF model** — any instruction-tuned model with tool calling support. Recommended: [Qwen3-4B-Instruct](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-Q4_K_M-GGUF) (~2.5 GB)
2. **A GGUF reranker** — for semantic search over your corpus. Recommended: [Qwen3-Reranker-0.6B](https://huggingface.co/llamaindex/qwen3-reranker-0.6b-gguf) (~396 MB)
3. **A text file to research** — any `.md` or `.txt` file with content worth investigating

## Install

```bash
npm i @lloyal-labs/lloyal-agents @lloyal-labs/rig @lloyal-labs/lloyal.node effection
```

Three packages:
- `@lloyal-labs/lloyal-agents` — agent runtime (pools, branches, tools, structured concurrency)
- `@lloyal-labs/rig` — research tools (search, grep, read_file), corpus loading, reranker
- `@lloyal-labs/lloyal.node` — native inference backend (prebuilt for macOS/Linux/Windows)

## The code

Create `research.ts`:

```typescript
import { main, call, ensure } from "effection";
import { createContext } from "@lloyal-labs/lloyal.node";
import { initAgents, withSharedRoot, useAgentPool } from "@lloyal-labs/lloyal-agents";
import { loadResources, chunkResources, createReranker, createTools } from "@lloyal-labs/rig";

const MODEL = process.argv[2] || "model.gguf";
const RERANKER = process.argv[3] || "reranker.gguf";
const CORPUS = process.argv[4] || "./docs/";
const QUERY = process.argv[5] || "What are the main topics in this document?";

await main(function* () {
  // 1. Load model and reranker
  const ctx = yield* call(() =>
    createContext({
      modelPath: MODEL,
      nCtx: 16384,
      nSeqMax: 8,
      typeK: "q4_0",
      typeV: "q4_0",
    }),
  );
  const reranker = yield* call(() => createReranker(RERANKER));
  yield* ensure(() => reranker.dispose());

  // 2. Load corpus and build tools
  const resources = loadResources(CORPUS);
  const chunks = chunkResources(resources);
  yield* call(() => reranker.tokenizeChunks(chunks));
  const { toolMap, toolsJson } = createTools({ resources, chunks, reranker });

  // 3. Initialize the agent runtime
  const { session } = yield* initAgents(ctx);

  // 4. Run a research agent
  const SYSTEM = `You are a research assistant. Your tools:
- **search**: find relevant sections by meaning
- **grep**: find exact patterns across files
- **read_file**: read specific line ranges
- **report**: submit your findings with evidence

Search first, then read matching sections, then report with line numbers and quotes.`;

  const pool = yield* withSharedRoot(
    { systemPrompt: SYSTEM, tools: toolsJson },
    function* (root) {
      return yield* useAgentPool({
        tasks: [
          {
            systemPrompt: SYSTEM,
            content: QUERY,
            tools: toolsJson,
            parent: root,
          },
        ],
        tools: toolMap,
        terminalTool: "report",
        maxTurns: 10,
      });
    },
  );

  // 5. Print findings
  const findings = pool.agents[0]?.findings;
  if (findings) {
    console.log("\n" + findings);
  } else {
    console.log("No findings produced.");
  }
});
```

## Run it

```bash
npx tsx research.ts ./models/Qwen3-4B-Q4_K_M.gguf ./models/qwen3-reranker-0.6b.gguf ./my-document.md "What does this document argue?"
```

The agent will:
1. Search the corpus for relevant sections
2. Read matching sections to get full context
3. Grep for specific patterns to verify claims
4. Report findings with line numbers and evidence

## What just happened

```
createContext        → Loaded the model into GPU memory
createReranker       → Loaded a small scoring model for semantic search
loadResources        → Read your file(s) into memory
chunkResources       → Split into paragraph-level chunks
createTools          → Built search, grep, read_file, and report tools
initAgents           → Set up the agent runtime (KV cache, event channel)
withSharedRoot       → Created a shared KV prefix with the system prompt + tool schemas
useAgentPool         → Ran one agent that generates, calls tools, and reports
```

The agent ran inside a four-phase tick loop: generate a token → commit to KV → settle tool results → dispatch tool calls. Each tool result was prefilled directly into the agent's KV cache before the next generation step — the model saw the complete result and made a fresh decision about what to do next.

## What's different from cloud APIs

If you've used LangChain or CrewAI, this looks similar on the surface. The difference is underneath:

- **No API calls.** The model runs locally on your GPU. No network latency, no rate limits, no API keys.
- **Tool results are in KV, not re-sent.** Each tool result is prefilled into the model's attention cache. The model doesn't re-read the system prompt and conversation history on every turn — it continues from where it left off.
- **Grammar-constrained tool calls.** The model can only produce valid tool call JSON. Malformed calls are structurally impossible while the grammar is active.
- **Automatic cleanup.** When `withSharedRoot` exits, all branches are pruned automatically. No leaked GPU state.

## Next: Your First Pipeline

The quick start ran a single agent. Real research benefits from multiple agents investigating different angles in parallel, with planning, synthesis, and verification.

→ [Your First Pipeline](your-first-pipeline.md) builds a plan → research → synthesize pipeline using the reference implementation.

## Next: Thinking in lloyal

If the generator syntax (`function*`, `yield*`) feels unfamiliar, read the mental model guide first.

→ [Thinking in lloyal](thinking-in-lloyal.md) explains why generators, how they map to async/await, and what structured concurrency buys you.
