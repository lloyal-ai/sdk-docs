# Multi-Agent Research

A single agent can search, read, and report. But a hard question benefits from multiple agents investigating different angles in parallel and converging on an answer. This page covers how lloyal runs multiple agents on shared GPU compute — from the API you write to the physical KV cache operations underneath.

## Why multiple agents

A research question like *"How does the DOJ connect Apple's iPod-era success to its current monopoly practices?"* has multiple threads: the historical iPod narrative, the legal claims, the connecting mechanism between past success and present behavior. A single agent explores them serially — search, read, search again. Three agents explore them in parallel, each building its own hypothesis from a different entry point, all sharing the same GPU.

The value is not just speed. Agents that start from different sub-questions discover different connections. One reads the historical narrative and discovers the Microsoft antitrust backdrop. Another reads the legal claims backward and tests whether "iPod-era" appears as a phrase. A third systematically covers the document's argumentative arc. Their findings overlap and reinforce — the convergence is itself a signal about the answer's reliability.

## `withSharedRoot` — the shared KV prefix

Every agent in a pool needs the same system prompt and tool schemas in its context. Without sharing, each agent would independently decode those tokens into its own KV cells — hundreds of tokens of redundant GPU work. `withSharedRoot` eliminates this by creating a single root branch that all agents fork from.

```typescript
const pool = yield* withSharedRoot(
  { systemPrompt: RESEARCH_PROMPT, tools: toolsJson },
  function*(root, prefixLen) {
    // root is a Branch with the system prompt already in KV
    // prefixLen is the number of tokens decoded (e.g. 866)
    return yield* runAgents({
      tasks: questions.map(q => ({
        systemPrompt: RESEARCH_PROMPT,
        content: q,
        tools: toolsJson,
        parent: root,
      })),
      tools: toolMap,
      maxTurns: 6,
    });
  },
);
```

What happens inside:

1. `Branch.create(ctx, 0, params)` creates a root branch at position 0
2. The system prompt and tool schemas are tokenized and prefilled into the root's KV cells
3. The `body` function receives the root branch and the shared prefix length
4. When the body returns (or throws, or is cancelled), `withSharedRoot` prunes the root and all its descendants via `try/finally`

The root branch exists only as a prefix donor. It never generates tokens. Its KV cells are shared by every agent that forks from it.

## `useAgentPool` — fork, run, collect

`useAgentPool` takes an array of task specs and runs them as concurrent agents on shared compute. Each task specifies a system prompt, user message, tool schemas, and a parent branch to fork from.

Here is how the deep-research harness uses it:

```typescript
function* research(
  questions: string[],
  opts: WorkflowOpts,
): Operation<{ pool: AgentPoolResult; sharedPrefixLength: number; timeMs: number }> {
  const { result: pool, prefixLen: sharedPrefixLength } = yield* withSharedRoot(
    { systemPrompt: RESEARCH.system, tools: fullToolkit.toolsJson },
    function*(root, prefixLen) {
      const pool = yield* useAgentPool({
        tasks: questions.map(q => ({
          systemPrompt: RESEARCH.system,
          content: q,
          tools: fullToolkit.toolsJson,
          parent: root,
        })),
        tools: fullToolkit.toolMap,
        maxTurns: opts.maxTurns,
        terminalTool: 'report',
        pressure: { softLimit: 2048 },
      });

      // Agents that got hard-cut by pressure get a reporter pass
      yield* reportPass(pool, opts);
      return { result: pool, prefixLen };
    },
  );

  return { pool, sharedPrefixLength, timeMs: performance.now() - t };
}
```

Key options:

- **`tasks`** — one `AgentTaskSpec` per agent. Each has `systemPrompt`, `content` (the sub-question), `tools` (JSON schema), and `parent` (the branch to fork from).
- **`tools`** — the execution registry (`Map<string, Tool>`). This is what actually runs when an agent calls a tool.
- **`terminalTool`** — a tool name (e.g. `'report'`) that extracts findings and marks the agent done. Agents must call at least one non-terminal tool before reporting.
- **`maxTurns`** — maximum tool call rounds per agent. At 6, agents do 3-4 rounds of iterative refinement.
- **`pressure`** — KV budget thresholds. `softLimit` reserves headroom for downstream work.

`useAgentPool` is a resource — it suspends after all agents complete, keeping branches alive so the caller can fork from them (e.g. for verification or reporter passes). When the resource's scope exits, every branch is pruned automatically via the `ensure()` callbacks registered during setup.

## How agents share GPU

Two mechanisms make multi-agent efficient:

### Prefix sharing via O(1) fork

When `setupAgent` calls `parent.forkSync()`, the underlying `llama_kv_self_seq_cp()` does not copy KV tensors. It tags existing KV cells (the shared root's key/value vectors) with a new sequence ID. Multiple agents read the same physical memory at positions 0 through N-1 while writing their own unique cells at positions N and beyond.

```
KV cache layout after 3 forks:

Position:  0 ────────────── N ──────── N+M0
           | shared root    | agent 0  |
           | (seq 0,1,2,3)  | (seq 1)  |
           +────────────────+          |
                            | agent 1  | N+M1
                            | (seq 2)  |
                            +──────────+
                            | agent 2  | N+M2
                            | (seq 3)  |
                            +──────────+
```

The system prompt + tool schemas (~866 tokens for a 5-tool corpus toolkit) are decoded once. Each agent pays only for its unique suffix — the user message and generation prompt (~30 tokens). The savings are significant:

```
Without sharing: 3 agents x 896 tokens = 2,688 GPU decode ops
With sharing:    866 shared + 3 x 30 unique = 956 GPU decode ops
Savings:         64% fewer decode ops at setup
```

Over a full pipeline run with multi-turn tool use, prefix sharing plus KV-resident history produces a measured 78% reduction in total decode operations compared to a prompt-rebuilding approach. See [Prefix Sharing](../concepts/prefix-sharing.md) for the full accounting and comparison against cloud API token economics.

### Batched decode

All active agents advance through a single `store.commit(entries)` call per tick. The `BranchStore` batches their tokens into one `llama_decode` dispatch. Three agents generating one token each costs one GPU call, not three.

Similarly, tool results for multiple agents are batch-prefilled through a single `store.prefill(pairs)` call during the SETTLE phase. The GPU processes all result injections in one dispatch.

## The four-phase tick loop

The agent pool runs a tight loop with four phases per tick:

```
PRODUCE  -- sample one token per active agent, collect tool calls
COMMIT   -- single GPU call: batch-decode all produced tokens
SETTLE   -- drain tool results from previous tick, batch-prefill into agents
DISPATCH -- execute collected tool calls sequentially
```

**PRODUCE** calls `produceSync()` on each active agent. Each call samples one token. When a stop token is produced, the accumulated output is parsed for tool calls. Terminal tools (e.g. `report`) are intercepted immediately — findings extracted, agent marked done. Non-terminal tool calls are collected for DISPATCH.

**COMMIT** batch-decodes all produced tokens in a single GPU dispatch via `store.commit(entries)`.

**SETTLE** drains the `settledBuffer` — tool results from the previous tick's DISPATCH. Each result is pressure-gated: if it exceeds headroom, the result is rejected and the agent is dropped. Surviving results are batch-prefilled, then agents transition back to generating with grammar state reset.

**DISPATCH** executes collected tool calls sequentially. Each tool runs inside `scoped()` + `call()`, creating an error boundary. If the tool throws, the agent is marked done with an error finding — other agents survive. Sequential dispatch ensures exclusive `llama_context` access, which is a correctness requirement for the underlying C++ runtime.

The sequential DISPATCH is what creates the decision boundary — the clean pause between "here is what the tool returned" and "what should I do next" — that enables multi-hop reasoning. See [Concurrency](../concepts/concurrency.md) for the full mechanism and traced tool trajectories showing emergent hypothesis refinement.

## Pressure management

The KV cache is finite. `ContextPressure` snapshots `remaining = nCtx - cellsUsed` on every tick and enforces headroom at four points:

**Prefill gate** — at pool setup, the pool computes the total suffix cost for all agents. If it exceeds headroom, agents are dropped from the end until the remaining suffixes fit. This is the primary mechanism limiting agent count in nested pools.

```typescript
const initPressure = new ContextPressure(ctx, pressureOpts);
const totalSuffix = prefillSetup.reduce((s, [, t]) => s + t.length, 0);
if (!initPressure.canFit(totalSuffix)) {
  // Drop agents from the end until it fits
  while (prefillSetup.length > 0) {
    const needed = prefillSetup.reduce((s, [, t]) => s + t.length, 0);
    if (initPressure.canFit(needed)) break;
    prefillSetup.pop();
    const dropped = agents.pop()!;
    dropped.state = 'done';
  }
}
```

**PRODUCE hard-cut** — agents are killed before `produceSync()` when remaining drops below `hardLimit` (default 128). Pure safety net to prevent llama_decode crashes.

**PRODUCE soft-cut** — non-terminal tool calls are denied when headroom is negative. Terminal tools always pass — an agent that has done research and wants to report should never be blocked by pressure.

**SETTLE rejection** — tool results that would cross the `softLimit` floor are rejected and the agent is dropped. This prevents large tool results (e.g. a 6KB web page) from consuming the remaining budget and starving other agents.

The `softLimit` (default 1024) is the budget knob. Set it higher when your tools spawn inner agent pools that need their own headroom. The deep-research harness uses `softLimit: 2048` for the research phase to reserve room for the `research` tool's inner pool.

## `runAgents` — convenience wrapper

`runAgents` wraps `useAgentPool` in `scoped()` — agent branches are pruned when the scope exits, before the operation returns. Use it when you don't need to fork from agent branches after the pool completes:

```typescript
const pool = yield* withSharedRoot(
  { systemPrompt, tools: toolsJson },
  function*(root) {
    return yield* runAgents({
      tasks: questions.map(q => ({
        systemPrompt,
        content: q,
        tools: toolsJson,
        parent: root,
      })),
      tools: toolMap,
      maxTurns: 6,
    });
  },
);
// Branches are already pruned — pool.agents[*].branch is disposed
```

For multi-level tree topology — where you need to fork from agent branches for verification, reporter passes, or follow-up research — use `useAgentPool` directly within your own scope management. The deep-research harness does this: it runs `useAgentPool`, then forks reporter agents from hard-cut agents' still-live branches before letting the scope close.

## The emergent 4-beat pattern

When agents have corpus tools (search, grep, read_file, report), the sequential DISPATCH model produces a characteristic investigation pattern:

```
grep(/narrow pattern/)        --> 0 matches
search("broad query")         --> finds relevant section
read_file(L83-100)            --> reads section, identifies a connection
grep(/broader|OR|pattern/)    --> 20 matches across 7 lines
```

Narrow grep, broad search, read for context, hypothesis grep. The fourth step is the signature behavior: the agent read a section, identified a structural connection the original query didn't mention, and constructed a grep pattern targeting that specific connection. This is multi-hop reasoning — not "search and report" but "search, form hypothesis, search for confirmation."

This pattern is not prompted. It emerges from the decision boundary created by sequential DISPATCH: the agent gets a clean context break after each tool result, and the model's next-token prediction naturally refines the investigation strategy based on what it found. See [Concurrency](../concepts/concurrency.md) for full traced trajectories from real pipeline runs.

## Next steps

- [Prefix Sharing](../concepts/prefix-sharing.md) — the full KV cache mechanics, measured savings, and comparison with cloud API token economics
- [Concurrency](../concepts/concurrency.md) — the four-phase tick loop in depth, the decision boundary mechanism, recursive agent pools, and traced tool trajectories
