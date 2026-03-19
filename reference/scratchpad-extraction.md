---
title: "Scratchpad Extraction"
description: "Fork-attend-extract-prune pattern for compressing large tool results with zero net KV cost."
---

Tool results are often large -- a fetched web page can be 6,000+ tokens, a search result set 2,000+. Prefilling the full result into an agent's branch inflates `cellsUsed` and accelerates KV pressure, causing downstream agents to be dropped. Scratchpad extraction addresses this: fork from the shared root, attend to the full content in a temporary branch, grammar-constrain a compact summary, prune the fork. The agent receives the compact summary instead of the raw content. Zero net KV cost after the fork is pruned.

## The pattern

Scratchpad extraction uses `generate({ parent })` -- when a parent branch is provided, `generate` forks from it instead of creating a new root. The fork inherits the parent's KV prefix (the shared system prompt and tool schemas), attends to the new content via prefill, generates a structured summary under grammar constraint, and is pruned in a `finally` block.

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
4. **Prefill** the combined tokens into the fork. The fork now attends to the parent's context (system prompt, tool schemas) plus the new content.
5. **Generate** to completion under grammar constraint. The output conforms to the grammar schema.
6. **Prune** the fork in the `finally` block. KV cells unique to the fork are freed.

The parent branch is never modified. The fork's cells above the fork point are freed on prune. The net KV cost is transient -- the cells exist only during the extraction and are reclaimed immediately after.

## `ScratchpadParent` context

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

`ScratchpadParent` is set by `withSharedRoot` to the current root branch:

```typescript
// Inside withSharedRoot:
yield* ScratchpadParent.set(root);
return yield* body(root, sharedTokens.length);
```

This means every tool executing within a `withSharedRoot` scope has access to the shared root as a scratchpad parent. The fork inherits the root's KV prefix (system prompt + tool schemas), which gives the extraction model enough context to understand the domain without re-decoding the prompt.

When pools are nested (a tool spawns an inner pool via `withSharedRoot`), the inner scope sets its own `ScratchpadParent`. Tools within the inner pool fork from the inner root, not the outer root. Effection's context scoping ensures each level sees the correct parent.

If no `withSharedRoot` scope is active (e.g., the tool is called outside a pool), `ScratchpadParent.expect()` throws and the tool falls back to returning the raw result.

## How `generate` handles the parent fork path

The `parent` option in `generate` activates a different code path from cold-start generation:

```typescript
// generate.ts
let branch: Branch;
if (opts.parent) {
  branch = yield* call(() => opts.parent!.fork());
} else {
  branch = Branch.create(ctx, 0, samplerParams, undefined, opts.grammar);
}

try {
  if (opts.parent) {
    if (opts.grammar) branch.setGrammar(opts.grammar);
    const sep = ctx.getTurnSeparator();
    const delta = yield* call(() => ctx.tokenize(opts.prompt, false));
    const tokens = [...sep, ...delta];
    yield* call(() => branch.prefill(tokens));
  } else {
    const tokens = ctx.tokenizeSync(opts.prompt);
    yield* call(() => branch.prefill(tokens));
  }

  // Generate to completion via async iterator
  const { output, tokenCount } = yield* call(async () => {
    let output = '';
    let tokenCount = 0;
    for await (const { text } of branch) {
      output += text;
      tokenCount++;
    }
    return { output, tokenCount };
  });

  const parsed = opts.parse ? opts.parse(output) : undefined;
  return { output, tokenCount, parsed };
} finally {
  if (!branch.disposed) branch.pruneSync();
}
```

Key differences from cold-start:

- **Fork instead of create.** The branch starts at the parent's position, inheriting its full KV prefix.
- **Grammar set after fork.** Eager grammar is applied post-fork because the grammar state from the parent (which may be a lazy tool-call grammar) is irrelevant to the extraction task.
- **Turn separator prepended.** The extraction content is wrapped with the model's turn separator so the model perceives it as a new conversational turn, not a continuation of the parent's generation.
- **Prune in finally.** The fork is always cleaned up, whether extraction succeeds, fails, or is cancelled. The parent's KV is untouched.

## Concrete implementations

### `BufferingFetchPage`

Wraps `FetchPageTool` to intercept successful page fetches. The full page content is buffered for post-research reranking. Then an extraction scratchpad summarizes the page into `{ summary, links }`:

```typescript
// Buffer full content for reranking
this._buffer.push({ url, title, text: content });

// Fork from scratchpad parent, extract compact summary
const parent = yield* ScratchpadParent.expect();
const schema = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    links: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'links'],
};
const grammar = yield* call(() => ctx.jsonSchemaToGrammar(JSON.stringify(schema)));

const extracted = yield* generate({
  prompt: extractPrompt,  // system + page content formatted as chat
  grammar,
  params: { temperature: 0.3 },
  parse: o => JSON.parse(o),
  parent,
});

return {
  url, title,
  summary: extracted.parsed.summary,
  links: extracted.parsed.links,
};
```

The calling agent receives a compact `{ url, title, summary, links }` object instead of the full page text. The summary might be 200 tokens; the full page would have been 4,000-6,000 tokens. The difference in KV cost to the agent is substantial -- especially when an agent fetches multiple pages across several turns.

The full content is not lost. It sits in the `FetchedPage` buffer, available for post-research reranking via `getChunks()`. The agent reasons from the summary; the reranker scores from the full text.

### `BufferingWebSearch`

Same pattern for search results. Raw search results (title, URL, snippet per result) are distilled into `{ urls, summary }`:

```typescript
const parent = yield* ScratchpadParent.expect();
const schema = {
  type: 'object',
  properties: {
    urls: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['urls', 'summary'],
};
const grammar = yield* call(() => ctx.jsonSchemaToGrammar(JSON.stringify(schema)));

const extracted = yield* generate({
  prompt: extractPrompt,  // system + formatted search results
  grammar,
  params: { temperature: 0.3 },
  parse: o => JSON.parse(o),
  parent,
});

return {
  urls: extracted.parsed.urls,
  summary: extracted.parsed.summary,
  resultCount: results.length,
};
```

The agent receives a list of promising URLs and a summary instead of the full search result array. It uses the URLs to decide which pages to fetch and the summary to understand what the search found -- without paying the KV cost of 10-20 individual search result entries.

## KV budget impact

Without scratchpad extraction, every tool result is prefilled directly into the agent's branch. A research agent that calls `web_search` (2,000 tokens) then `fetch_page` three times (5,000 tokens each) consumes 17,000 tokens of KV on tool results alone. With `nCtx: 16384`, that leaves almost nothing for other agents.

With scratchpad extraction, the same sequence might consume 200 + 3 x 300 = 1,100 tokens of KV on the agent's branch. The extractions happen on temporary forks that are pruned immediately -- their KV cost is transient, not cumulative.

The tradeoff is GPU compute. Each extraction runs a short generation (50-150 tokens of output, constrained by grammar) on a temporary fork. The fork must be prefilled with the content plus turn separator. For a 5,000-token page, the extraction prefill is ~5,000 tokens of GPU decode. This is real compute -- but it happens once per page, and the fork is pruned immediately. The alternative (prefilling 5,000 tokens into the agent's branch) is the same GPU cost but the KV cells persist for the agent's entire lifetime.

## When to use scratchpad extraction

Scratchpad extraction is useful when:

- **Tool results are large and variable.** Web pages, search results, and document sections vary from hundreds to thousands of tokens. The agent needs the information but not the full text.
- **Agents run multiple turns.** KV accumulates across turns. A 6-turn agent with large tool results per turn will hit pressure limits quickly without extraction.
- **Nested pools share KV.** Inner pool agents compete for the same KV budget as outer agents. Reducing per-tool-result KV cost directly increases the number of agents that can survive pressure enforcement.
- **Post-research reranking needs full content.** Extraction lets the agent reason from summaries while the reranker scores from the full text. Both needs are served from a single fetch.

Scratchpad extraction is unnecessary when:

- **Tool results are already compact.** A grep result returning 10 matches in 500 tokens does not benefit from extraction.
- **The agent needs verbatim content.** If the downstream task requires exact quotes or precise data from the tool result, extraction would lose information.
- **KV pressure is not a concern.** Single-agent workflows with large `nCtx` may not need the KV savings.

## Fallback behavior

Both `BufferingFetchPage` and `BufferingWebSearch` handle failure gracefully. If `ScratchpadParent.expect()` throws (no active `withSharedRoot` scope), or if the parent branch is disposed, or if the extraction generation fails, the tool returns the raw unextracted result:

```typescript
let parent: Branch | undefined;
try { parent = yield* ScratchpadParent.expect(); } catch { /* no parent */ }
if (!parent || parent.disposed) return result;

try {
  const extracted = yield* generate({ /* ... */ parent });
  return { summary: extracted.parsed.summary, /* ... */ };
} catch {
  return result;  // fallback to full result
}
```

The agent gets larger tool results (consuming more KV), but the pipeline does not fail. Scratchpad extraction is an optimization, not a requirement.
