# Sessions

A `Session` manages the ongoing conversation state between queries. It holds a trunk branch — the physical KV cache state representing the conversation so far — and provides methods to promote new state, inject user turns, and inject tool results.

## The Session API

`Session` is defined in `@lloyal-labs/sdk`:

```typescript
import { Session } from '@lloyal-labs/sdk';

const session = new Session({ ctx, store });

// After the first query, promote findings as the trunk
await session.promote(resultBranch);

// For follow-up queries, inject a user turn onto the trunk
await session.prefillUser('Tell me more about X');

// The trunk now has the prior conversation + new question in KV
// Generate from session.trunk to continue
```

### `session.trunk`

The trunk is a `Branch` — a sequence in the KV cache with a position, sampler state, and logits snapshot. When `trunk` is `null`, there's no prior conversation (cold start). When it's set, the model has physical memory of everything that came before.

```typescript
if (session.trunk) {
  // Warm path — prior conversation in KV
  // Agents fork from trunk, inheriting conversation context
} else {
  // Cold path — fresh start
  // Create branches from position 0
}
```

### `session.promote(winner)`

After a query produces findings, `promote()` crowns a branch as the new trunk:

```typescript
// Create a branch with the query + response formatted as a conversation
const trunk = Branch.create(ctx, 0, {});
const messages = [
  { role: 'user', content: query },
  { role: 'assistant', content: findings },
];
const { prompt } = ctx.formatChatSync(JSON.stringify(messages), { enableThinking: false });
const tokens = ctx.tokenizeSync(prompt, false);
await trunk.prefill(tokens);

// Promote: retainOnly(winner) clears all other branches, then sets trunk
await session.promote(trunk);
```

`promote()` calls `store.retainOnly(winner)` under the hood — this prunes every other branch in the store, freeing their KV cells. Then it assigns the winner as the new trunk. After promote, the session has a clean KV state containing only the conversation history.

### `session.prefillUser(content)`

For follow-up queries, `prefillUser()` injects a user turn as a delta on the existing trunk:

```typescript
// The trunk already has: [user: "original query", assistant: "findings"]
// prefillUser adds: [turn_separator, user: "follow-up question"]
await session.prefillUser('What about the specific mechanisms?');

// The trunk now physically contains the full conversation in KV
// Position has advanced by the delta token count
```

Internally, `prefillUser` calls `buildUserDelta(ctx, content)` which:
1. Gets the turn separator tokens (`getTurnSeparator()`)
2. Formats the user message (`formatChatSync`)
3. Tokenizes the formatted text
4. Concatenates: `[...separator, ...messageTokens]`
5. Prefills the token array onto the trunk branch

### `session.prefillToolResult(resultStr, callId)`

Injects a tool result as a delta — used when building multi-turn tool-calling conversations on the trunk:

```typescript
await session.prefillToolResult(JSON.stringify({ data: 'result' }), 'call_001');
```

## Cold vs warm queries

The distinction between cold and warm is physical, not modal — it's determined by whether `session.trunk` exists:

```typescript
function* handleQuery(query: string, opts: WorkflowOpts): Operation<QueryResult> {
  const warm = !!opts.session.trunk;

  if (warm) {
    // Agents fork from the trunk — they see prior conversation in KV
    yield* warmResearch(questions, query, opts);
  } else {
    // Agents fork from a fresh shared root at position 0
    yield* research(questions, query, opts);
  }
}
```

**Cold (first query):** No trunk exists. `withSharedRoot` creates a root branch at position 0, agents fork from it. After research and synthesis, the findings are formatted as a conversation and promoted to trunk via `session.promote()`.

**Warm (follow-up):** Trunk exists with prior conversation. Agents fork from the trunk (or from a root that includes trunk context). After research, findings are appended to the trunk via `session.prefillUser()`.

## What the model remembers

The model doesn't "remember" in an abstract sense — prior conversation tokens are physically present in the trunk's KV cache. When an agent forks from the trunk, it inherits those KV cells at O(1) cost. The model attends to all prior tokens (questions, answers, tool calls, tool results) because they occupy real positions in the attention cache.

This is fundamentally different from cloud APIs where "memory" means re-sending the conversation history as text on every request. In lloyal, the conversation state is a physical branch in GPU memory. Follow-up queries pay only the delta cost of the new turn — the prior conversation is already in KV.

The tradeoff is context window: each follow-up query increases the trunk's position. At some point the trunk approaches the model's `nCtx` limit. Monitor trunk position or check `ctx._storeKvPressure()` to track how much context has been consumed.

## The Session lifecycle

```
1. initAgents(ctx)              → session created, trunk = null
2. First query (cold)           → agents run, findings produced
3. session.promote(trunk)       → findings become the trunk
4. Follow-up query (warm)       → session.prefillUser(question)
                                  → agents fork from trunk
                                  → findings appended to trunk
5. Another follow-up            → repeat step 4
6. session.dispose()            → trunk pruned, KV freed
```

The reference implementation in `examples/deep-research-web/harness.ts` implements this lifecycle in `handleQuery()`, `promoteTrunk()`, and `appendTurn()`. See the [deep research example](../examples/deep-research.md) for the full production pattern.

## Next: Tracing

→ [Tracing](tracing.md) shows how to capture the full inference graph — every prompt, tool call, and agent turn — for debugging and analysis.
