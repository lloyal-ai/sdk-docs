---
title: "Scratchpad Extraction"
description: "Extracting findings from killed agents and fork-attend-extract-prune for tool result compression."
---

"Scratchpad" refers to two related patterns that use temporary branches for grammar-constrained generation: **recovery extraction** (extracting findings from agents killed by KV pressure) and **tool result compression** (summarizing large tool results via the fork-attend-extract-prune pattern). Both use eager grammar to force structured output from context the model has already attended to.

## Recovery extraction

When agents are killed by KV pressure without reporting, the pool extracts their accumulated findings before closing. This is the primary use of scratchpad-style extraction.

Recovery uses the agent's **own branch** -- no fork. The agent's full KV context (system prompt, tool results, reasoning from prior turns) is already in the cache. An extraction prompt is prefilled directly into the branch, eager grammar constrains output to `{"result": "..."}`, and a batched produce/commit loop generates the report.

See [KV Pressure: Recovery extraction](/reference/kv-pressure#recovery-extraction) for the full protocol.

Key properties:
- **No fork**: Uses the agent's own branch. Eliminates RESTRICT prune conflicts and redundant KV.
- **Eager grammar**: `setGrammar()` constrains from token 0. No tool calls possible, no model choice.
- **Parallel**: All recovering agents generate in the same produce/commit loop -- one GPU call per step.
- **One-shot**: Fires at most once per pool run via a guard flag.
- **Pressure-gated**: Skipped if remaining KV can't fit the extraction prompts.

### Policy configuration

The policy's `onRecovery()` method decides per-agent whether to extract:

```typescript
const policy = new DefaultAgentPolicy({
  recovery: {
    prompt: {
      system: "You are a research reporter. Call the report tool with all findings.",
      user: "Report your findings with direct quotes and evidence.",
    },
    minTokens: 100,    // skip agents with fewer than 100 generated tokens
    minToolCalls: 2,    // skip agents with fewer than 2 tool calls
  },
});
```

The `minTokens` and `minToolCalls` guards prevent extraction from agents with insufficient context -- an agent that generated 20 tokens and made 1 tool call would produce hallucinated findings.

### Result provenance

Recovered findings are stored with `resultSource: 'scratchpad'`. This is metadata only -- no downstream code branches on the source. The harness reads `.result` and ignores `.resultSource`. However, the provenance is visible in trace output for debugging.

## Fork-attend-extract-prune pattern

For tool result compression (independent of recovery), the fork-attend-extract-prune pattern uses a temporary branch:

```typescript
const extracted = yield* generate({
  prompt: contentToSummarize,
  grammar: extractionGrammar,
  params: { temperature: 0.3 },
  parse: output => JSON.parse(output),
  parent: scratchpadParent,
});
// extracted.parsed contains the compact summary
// The fork has been pruned -- parent's KV is unchanged
```

Inside `generate`, the flow is:

1. **Fork** from `parent` via `parent.fork()`. The child inherits position and KV prefix.
2. **Set grammar** on the fork (`branch.setGrammar(grammar)` -- eager mode, active from token 0).
3. **Tokenize** the prompt and prepend the turn separator from `ctx.getTurnSeparator()`.
4. **Prefill** the combined tokens into the fork. The fork now attends to the parent's context plus the new content.
5. **Generate** to completion under grammar constraint.
6. **Prune** the fork in the `finally` block. KV cells unique to the fork are freed.

The parent branch is never modified. The net KV cost is transient -- cells exist only during extraction and are reclaimed immediately.

### `ScratchpadParent` context

Tools that perform scratchpad extraction need a branch to fork from. The `ScratchpadParent` Effection context provides this:

```typescript
import { ScratchpadParent } from '@lloyal-labs/lloyal-agents';

// Inside a tool's execute() method:
let parent: Branch | undefined;
try {
  parent = yield* ScratchpadParent.expect();
} catch {
  // No scratchpad parent available -- fall back to raw result
}
```

`ScratchpadParent` is set by `withSharedRoot` when `enableScratchpad: true`:

```typescript
yield* withSharedRoot(
  { systemPrompt, tools, enableScratchpad: true },
  function* (root) { /* tools in this scope can fork from root */ }
);
```

When pools are nested, each inner scope sets its own `ScratchpadParent`. Effection's context scoping ensures each level sees the correct parent.

## Reranker-based chunk selection

`FetchPageTool` implements a variant of scratchpad extraction: when a reranker and query are provided, fetched pages are structurally chunked on heading boundaries and scored against the query. Only the top-K most relevant chunks within a token budget are returned:

```typescript
const tool = new FetchPageTool({ maxChars: 6000, topK: 5, tokenBudget: 2048 });
tool.setReranker(reranker);

// Agent calls: fetch_page({ url: "...", query: "what about X?" })
// Returns: top 5 chunks ranked by relevance to "what about X?"
```

This reduces KV pressure without lossy summarization -- the agent gets the most relevant sections rather than a generic summary. Without a reranker or query, the tool falls back to returning the full content truncated to `maxChars`.

## When to use each pattern

| Pattern | Use case | Branch | Grammar |
|---------|----------|--------|---------|
| Recovery extraction | Agents killed by pressure | Agent's own branch (no fork) | Eager JSON |
| Fork-attend-extract-prune | Tool result compression | Temporary fork from parent | Eager, task-specific |
| Reranker chunk selection | Large web pages | No branch (reranker-only) | None |

Recovery extraction is automatic -- configure the policy's `recovery` option and the pool handles it. Fork-attend-extract-prune requires explicit use of `generate({ parent })` in tool implementations. Reranker selection requires injecting a `Reranker` into `FetchPageTool`.
