---
title: "Grammar & Tool Ordering"
description: "GBNF grammar constraining, lazy vs eager activation, trigger types, and how tool ordering affects agent behavior."
---

When agents make tool calls, the output must be structurally valid -- a malformed XML tag or broken JSON means a parse failure and a wasted turn. Grammar constraining solves this at the token level: a GBNF grammar masks invalid tokens from the sampler before they can be produced, making malformed tool calls structurally impossible while the grammar is active.

## How grammar gets generated

`formatChatSync` is the entry point. It takes a JSON-serialized message array and options (including tool schemas), applies the model's chat template, and returns a format descriptor:

```typescript
const messages = [
  { role: 'system', content: systemPrompt },
  { role: 'user', content: userContent },
];
const fmtOpts = { enableThinking: false, tools: toolsJson };

const fmt = ctx.formatChatSync(JSON.stringify(messages), fmtOpts);
// fmt.prompt       — formatted prompt string
// fmt.grammar      — GBNF grammar string (if tools present)
// fmt.grammarLazy  — whether grammar should be lazy-activated
// fmt.grammarTriggers — trigger conditions for lazy activation
// fmt.parser       — parser identifier for output extraction
// fmt.format       — chat format enum (template-specific)
```

When tools are provided, `formatChatSync` generates a GBNF grammar from the tool schemas. The grammar encodes the valid tool call structure for the model's specific chat template -- Qwen3 uses `<tool_call>{"name":...}</tool_call>`, Llama 3.x uses `<|python_tag|>`, Mistral uses `[TOOL_CALLS]`. The grammar is template-aware: the same tool schemas produce different GBNF rules depending on the model.

The grammar covers the complete output space: free text (for reasoning) and structured tool calls (for action). The model can generate arbitrary text until it decides to make a tool call, at which point the grammar constrains the tool call structure.

## Lazy vs eager grammar

There are two activation modes:

**Eager grammar** activates immediately. Every token from the first one is constrained by the grammar. Use this for structured generation where the entire output must conform to a schema -- for example, `agent()` with a JSON schema:

```typescript
const result = yield* agent({
  systemPrompt: "You analyze research queries. Output JSON only.",
  task: query,
  schema: planSchema,  // compiled to GBNF grammar, eager: active from token 0
});
const plan = JSON.parse(result.rawOutput);
```

**Lazy grammar** lets the model generate freely until a trigger fires, then activates the grammar. This is the mode used for tool-calling agents. The model reasons in free text (unconstrained), then when it starts a tool call (trigger detected), the grammar activates and constrains the tool call structure.

`formatChatSync` returns `grammarLazy: true` and a `grammarTriggers` array when the chat template supports lazy activation. The triggers are template-specific patterns that indicate the model is beginning a tool call.

## Trigger types

Grammar triggers come in four types, matching what different chat templates use to signal tool calls:

| Type | Meaning | Example |
|---|---|---|
| `WORD` | Literal string match | `<tool_call>` |
| `PATTERN` | Regex pattern match | `\{"name"\s*:` |
| `PATTERN_FULL` | Full-match regex (anchored) | `^<\|python_tag\|>$` |
| `TOKEN` | Specific token ID | Token ID for `[TOOL_CALLS]` |

`setGrammarLazy` on the branch installs the grammar and its triggers. Generation proceeds unconstrained. When the model produces tokens that match a trigger (the beginning of a tool call), the grammar activates and constrains all subsequent tokens until the tool call structure is complete.

```typescript
branch.setGrammarLazy(fmt.grammar, fmt.grammarTriggers);
```

## `applyLazyGrammar` in the tick loop

The agent pool applies lazy grammar at two moments: once during setup (after all agents are prefilled), and again in SETTLE after each tool result is prefilled. The `applyLazyGrammar` function handles both:

```typescript
const applyLazyGrammar = (a: Agent): void => {
  if (a.fmt.grammar && a.fmt.grammarLazy && a.fmt.grammarTriggers.length > 0) {
    const triggers = a.fmt.grammarTriggers.map(t => {
      if (t.type === GrammarTriggerType.WORD) {
        const nlIdx = t.value.indexOf('\n');
        if (nlIdx >= 0 && nlIdx < t.value.length - 1) {
          return { ...t, value: t.value.slice(0, nlIdx + 1) };
        }
      }
      return t;
    });
    a.branch.setGrammarLazy(a.fmt.grammar, triggers);
  }
};
```

The newline truncation for `WORD` triggers is a compatibility fix: some chat templates include multi-line trigger strings where only the first line (up to and including the newline) is the actual trigger prefix. The remainder is part of the grammar's constrained output.

### Initial application

After batch prefill of all agent suffixes, lazy grammar is applied to every agent:

```typescript
for (const a of agents) applyLazyGrammar(a);
```

Each agent can now generate freely. When it starts a tool call, the grammar activates.

### Reset after SETTLE

When a tool result is prefilled into an agent's branch, the grammar state must be reset. The tool result contains text that the grammar was not tracking -- injecting it without resetting would leave the grammar in an inconsistent state. SETTLE handles this:

```typescript
// After prefill succeeds:
for (const a of settledAgents) {
  a.state = 'generating';
  a.rawOutput = '';
  applyLazyGrammar(a);  // fresh grammar state
}
```

The reset is essential. Without it, the grammar would either reject valid tokens (because its internal parser is misaligned with the branch's actual content) or allow invalid tokens (because the parser was tracking the tool call that already completed). Resetting gives the agent a clean slate: free text generation until the next tool call trigger fires.

## Grammar on `Branch`

The `Branch` class exposes three grammar methods:

**`setGrammar(grammarStr)`** -- Eager mode. Activates immediately. Every subsequent token is constrained. Pass an empty string to remove the grammar.

```typescript
branch.setGrammar(jsonGrammar);  // constrain from now
branch.setGrammar('');           // remove constraint
```

**`setGrammarLazy(grammar, triggers)`** -- Lazy mode. Free generation until a trigger fires, then constrained. Used by agent pools for tool-calling agents.

```typescript
branch.setGrammarLazy(grammar, [
  { type: GrammarTriggerType.WORD, value: '<tool_call>' },
]);
```

**Grammar state cloning on fork.** When a branch is forked, the grammar state is cloned. Sibling branches diverge independently -- one can be in constrained mode while another is still generating free text. This is a property of `forkSync()`, not a separate API call.

## Tool ordering sensitivity

The GBNF grammar generated by `formatChatSync` is order-agnostic -- it enumerates valid tool call alternatives without biasing toward any particular tool. However, the tool schemas are also serialized into the system prompt as a JSON array. LLMs exhibit recency bias over the system prompt content: the last tool in the array occupies the most salient position in attention.

This creates a practical interaction between grammar constraining and tool ordering. The grammar guarantees structural validity (any emitted tool call will parse correctly), but the model's choice of which tool to call is influenced by prompt position. When a terminal tool like `report` is placed last in the toolkit array:

```typescript
// report last -- model biased toward early termination
tools: [search, read_file, grep, research, report];
```

Agents terminate after 3-4 tool calls instead of the expected 6-8. The grammar correctly constrains whichever tool call the model chooses, but the model chooses the terminal tool more often because it has higher salience from its position.

The fix is tool ordering, not grammar tuning:

```typescript
// report before its prefix-sharing sibling -- correct ordering
tools: [search, read_file, grep, report, research];
```

When two tools share a token prefix (`report` and `research` both start with `r`), the model's logits must resolve the ambiguity after generating the shared prefix. Recency bias from system prompt position shifts the distribution. Placing the terminal tool before its prefix-sharing sibling counteracts this effect.

This is a model-level effect, not a grammar bug. The grammar does its job perfectly -- ensuring valid structure. The model's tool selection is a separate concern governed by attention over prompt content.

## Direct grammar use with `agent()`

`agent` with a `schema` option compiles the JSON Schema to a GBNF grammar and sets it as an eager constraint:

```typescript
const result = yield* agent({
  systemPrompt: "Extract a summary and confidence score.",
  task: extractionPrompt,
  schema: {
    type: 'object',
    properties: {
      summary: { type: 'string' },
      confidence: { type: 'number' },
    },
    required: ['summary', 'confidence'],
  },
});
const parsed = JSON.parse(result.rawOutput);
// parsed is guaranteed to match the schema -- grammar enforced every token
```

`jsonSchemaToGrammar` (called internally by `agent`) converts a JSON Schema into a GBNF grammar string. The grammar constrains generation from the first token, so the output is guaranteed to be valid JSON conforming to the schema. No retry logic, no post-hoc validation.

This is the mechanism behind scratchpad extraction (see [Scratchpad Extraction](/reference/scratchpad-extraction)) -- fork a branch, attend to content, grammar-constrain a compact summary, prune the fork.
