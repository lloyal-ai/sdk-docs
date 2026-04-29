---
title: "Deployment"
description: "Model selection, context sizing, KV quantization, and runtime configuration."
---

This guide covers model selection, context sizing, KV quantization, and runtime configuration for deploying a lloyal-agents pipeline.

## Model selection

lloyal-agents requires an instruction-tuned model with native tool-calling support. The runtime uses `formatChat()` to embed tool schemas into the system prompt and `parseChatOutput()` to extract structured tool calls from generation output. Models without tool support will fail at agent setup with:

```
Error: Model does not support tool calling. Please use a model with native tool support
```

Confirmed compatible model families:

| Family | Notes |
|--------|-------|
| Qwen 3 | Recommended. Strong tool-call compliance, good depth-first behavior |
| Llama 3.x | Works well with instruction-tuned variants |
| Mistral | Tool-calling variants supported |

The reference pipeline uses `Qwen3-4B-Instruct-2507` quantized to Q4_K_M. Larger models improve reasoning depth (more hypothesis-driven tool calls, better synthesis) but require proportionally more KV memory.

## Context size (nCtx)

`nCtx` is the total KV cache budget in tokens. All agents, shared roots, tool results, and intermediate branches share this budget.

```typescript
const nCtx = parseInt(process.env.LLAMA_CTX_SIZE || '16384', 10);
const ctx = yield* call(() =>
  createContext({
    modelPath,
    nCtx,
    // ...
  }),
);
```

### Sizing guidelines

| Pipeline | Recommended nCtx | Reasoning |
|----------|-----------------|-----------|
| Single agent + tools | 8192 | One branch, moderate tool results |
| Simple research (1-2 sources) | 16384 | Default. 3 agents + synthesis + eval |
| Deep research (multi-source) | 32768 | Bridge stages, multiple source rounds |
| Large corpus research | 32768-65536 | Long documents inflate search/read results |

The budget divides across concurrent branches. With 3 research agents sharing a root:

```
Shared root:     ~200 tokens (system prompt + tools)
Agent suffix:    ~300 tokens each (user message + generation prompt)
Tool results:    ~500-2000 tokens per call (varies by tool)
Generation:      ~200-500 tokens per turn
```

At `nCtx=16384` with 3 agents doing 4 tool calls each, rough budget:
- Shared root: 200
- 3 agents x (300 suffix + 4 x 1000 tool results + 4 x 300 generation): ~15,900
- Remaining for synthesis + eval: ~300

This is tight. If agents hit large tool results (web pages, long search results), context pressure kicks in and starts cutting agents. For comfortable headroom, use 32768.

### LLAMA_CTX_SIZE environment variable

Examples read `LLAMA_CTX_SIZE` from the environment when no explicit `--n-ctx` flag is passed:

```bash
LLAMA_CTX_SIZE=32768 npx tsx examples/react-agent/main.ts model.gguf --corpus ./docs
```

If unset, defaults to 16384.

## Sequence count (nSeqMax)

`nSeqMax` is the maximum number of concurrent branches (sequences) in the KV cache. Each active agent, shared root, scratchpad fork, and diverge attempt needs its own sequence.

```typescript
const ctx = yield* call(() =>
  createContext({
    modelPath,
    nCtx,
    nSeqMax: Math.max(AGENT_COUNT, VERIFY_COUNT) * 4 + 3,
    // ...
  }),
);
```

### Calculating minimum nSeqMax

`nSeqMax` sets the number of sequence slots allocated in the KV cache. Each live branch occupies one slot. You need enough slots for the maximum concurrent branches at any point during execution. The hard ceiling is llama.cpp's limit of 264 sequences.

Count the peak concurrent branches:

| Component | Branches needed |
|-----------|-----------------|
| Research shared root | 1 |
| Research agents | AGENT_COUNT |
| Sub-agent shared root (research tool) | 1 per active research call |
| Sub-agents | Up to AGENT_COUNT per research call |
| Scratchpad forks (BufferingFetchPage, BufferingWebSearch) | 1 per active extraction |
| Bridge shared root + agent | 2 |
| Synthesis shared root + agent | 2 |
| Verify/diverge attempts | VERIFY_COUNT |
| Reporter sub-agents (hard-cut recovery) | Up to AGENT_COUNT |

The formula `max(AGENT_COUNT, VERIFY_COUNT) * 4 + 3` provides margin for nested pools. With `AGENT_COUNT=3` and `VERIFY_COUNT=3`:

```
3 * 4 + 3 = 15 sequences (minimum)
```

This covers: 3 research agents + 1 root + up to 3 sub-agents + 1 sub-root + scratchpad forks + margin.

Setting `nSeqMax` below the peak concurrent branch count causes "no memory slot for batch" errors at runtime. Setting it higher than needed wastes a small amount of memory on per-sequence metadata (recurrent state for GDN models, KV cell tags for attention models). Err on the side of higher — the cost per unused slot is minimal compared to the cost of running out mid-pipeline.

## KV quantization

KV cache quantization reduces memory per token at the cost of some precision. The two options:

| Type | Memory per token | Quality | Use case |
|------|-----------------|---------|----------|
| `q8_0` | ~2 bytes | Near-lossless | Small contexts, quality-critical |
| `q4_0` | ~1 byte | Slight degradation | Large contexts, recommended default |

```typescript
const ctx = yield* call(() =>
  createContext({
    modelPath,
    nCtx: 32768,
    nSeqMax: 15,
    typeK: 'q4_0',
    typeV: 'q4_0',
  }),
);
```

The reference pipeline uses `q4_0` for both K and V. At `nCtx=32768` with a 4B parameter model:

- **q8_0**: ~64MB KV cache
- **q4_0**: ~32MB KV cache

The quality difference is negligible for research pipelines where the primary bottleneck is tool-call quality, not raw generation precision. Use `q8_0` if you observe degraded output quality with long contexts.

You can mix K and V quantization (`typeK: 'q8_0', typeV: 'q4_0'`), but this is rarely needed. Both `q4_0` is the practical default.

## Reranker selection

The reranker scores chunks against queries for post-research passage selection. It runs on a separate model context with its own `nCtx` and `nSeqMax`:

```typescript
import { createReranker } from '@lloyal-labs/rig/node';

const reranker = yield* call(() =>
  createReranker(rerankModelPath, { nSeqMax: 8, nCtx: 4096 }),
);
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `nSeqMax` | 8 | Parallel scoring slots. Higher = faster scoring of many chunks |
| `nCtx` | 4096 | Max tokens per chunk+query pair. 4096 is sufficient for most passages |

The reference pipeline uses `qwen3-reranker-0.6b` at Q4_K_M quantization (~400MB). The reranker is called at two points:
1. After each source's research phase (score source chunks against query)
2. Between sources (bridge reranking for discovery extraction)

Reranker memory is separate from the main model's KV cache. On a system with 16GB, the typical budget: ~4GB main model weights + ~2GB KV cache + ~400MB reranker weights + ~200MB reranker KV.

## Agent count and turn limits

```typescript
const AGENT_COUNT = 3;     // Parallel research agents per source
const VERIFY_COUNT = 3;    // Diverge attempts for eval
const MAX_TOOL_TURNS = 20; // Max tool calls per agent
```

**AGENT_COUNT** controls research parallelism. More agents cover more sub-questions but consume more KV. With `nCtx=16384`, 3 agents is the practical limit. At `nCtx=32768`, you can run 5-6 agents comfortably.

**VERIFY_COUNT** controls eval confidence. Each verify attempt is a separate branch that generates independently from the same prompt. More attempts give better convergence detection but cost tokens. 3 is a reasonable default.

**MAX_TOOL_TURNS** controls research depth. At 2 turns, agents do single-shot retrieval. At 6 turns, agents iterate 3-4 times. At 20 turns, agents go deep -- following citation chains, cross-referencing, building evidence maps. The reference pipeline uses 20 as the ceiling; most agents naturally report after 4-8 turns. When a single question goes through (passthrough), the harness doubles the turn limit (`effectiveMaxTurns = questions.length === 1 ? maxTurns * 2 : maxTurns`).

## Context pressure tuning

Context pressure controls when agents are shut down as KV fills. Two thresholds:

```typescript
const researchPolicy = new DefaultAgentPolicy({
  budget: { context: { softLimit: 1024, hardLimit: 128 } },
  recovery: { prompt: REPORT },
});
```

**softLimit** (default 1024) -- remaining KV floor for new work. When remaining tokens drop below this:
- SETTLE rejects tool results (primary enforcement point)
- PRODUCE nudges agents to report (terminal tools like `report` still pass)
- INIT drops agents that don't fit during setup

Set this to account for downstream work. If synthesis needs ~1000 tokens of headroom, set research pool softLimit to at least 1024.

**hardLimit** (default 128) -- crash-prevention floor. When remaining drops below this, `shouldExit` kills agents immediately before `produceSync()`. This is a safety net, not a tuning knob.

For reporter sub-pools that only make one terminal tool call, use the default policy (softLimit 1024, hardLimit 128).

## Putting it together

A complete deployment configuration:

```typescript
import { createContext } from '@lloyal-labs/lloyal.node';
import { initAgents } from '@lloyal-labs/lloyal-agents';
import { TavilyProvider } from '@lloyal-labs/rig';
import { createReranker, WebSource, CorpusSource,
         loadResources, chunkResources } from '@lloyal-labs/rig/node';

const nCtx = parseInt(process.env.LLAMA_CTX_SIZE || '32768', 10);
const AGENT_COUNT = 3;
const VERIFY_COUNT = 3;

const ctx = yield* call(() =>
  createContext({
    modelPath: '/path/to/Qwen3-4B-Instruct-Q4_K_M.gguf',
    nCtx,
    nSeqMax: Math.max(AGENT_COUNT, VERIFY_COUNT) * 4 + 3,
    typeK: 'q4_0',
    typeV: 'q4_0',
  }),
);

const reranker = yield* call(() =>
  createReranker('/path/to/reranker-q4_k_m.gguf', {
    nSeqMax: 8,
    nCtx: 4096,
  }),
);

const { session, events } = yield* initAgents<WorkflowEvent>(ctx);
```

## Hardware considerations

The main memory consumers:

| Component | Typical size (4B model) |
|-----------|------------------------|
| Model weights (Q4_K_M) | ~2.5 GB |
| KV cache (q4_0, nCtx=32K) | ~1-2 GB |
| Reranker weights | ~400 MB |
| Reranker KV | ~200 MB |
| Runtime overhead | ~500 MB |

Total: ~5-6 GB for a 4B model with 32K context. This fits comfortably on a machine with 8GB available memory.

For larger models (7B, 14B), scale KV cache proportionally with model hidden dimension. A 14B model at Q4_K_M with 32K context needs roughly 15-18 GB total.
