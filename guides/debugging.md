---
title: "Debugging"
description: "Common failure modes, trace event analysis, and systematic debugging workflow."
---

This guide covers common failure modes in lloyal-agents pipelines and how to diagnose them using trace events. Enable tracing with `--trace` to write structured JSONL trace files, or inspect runtime events via the `events` channel.

## Enable tracing

From the CLI:

```bash
npx tsx examples/deep-research-web/main.ts model.gguf --trace
```

This creates a `trace-{timestamp}.jsonl` file with every branch creation, prompt format, tool dispatch, agent turn, and pressure event.

Programmatically:

```typescript
import { JsonlTraceWriter } from '@lloyal-labs/lloyal-agents';
import * as fs from 'node:fs';

const traceWriter = new JsonlTraceWriter(fs.openSync('trace.jsonl', 'w'));
const { session, events } = yield* initAgents<WorkflowEvent>(ctx, { traceWriter });
```

## Trace event reference

Key event types from `packages/agents/src/trace-types.ts`:

| Event | When | Key fields |
|-------|------|------------|
| `prompt:format` | System prompt or agent suffix tokenized | `promptText`, `taskContent`, `tokenCount`, `tools`, `role` |
| `branch:create` | Branch forked or created | `branchHandle`, `parentHandle`, `role` |
| `branch:prefill` | Tokens injected into a branch | `branchHandle`, `tokenCount`, `role` |
| `pool:open` | Agent pool starts | `agentCount`, `taskSuffixTokens`, `pressure` |
| `pool:agentDrop` | Agent killed by pressure or turn limit | `agentId`, `reason` |
| `pool:close` | Agent pool ends | `agents[]` (each with `tokenCount`, `toolCallCount`, `findings`) |
| `agent:turn` | Agent completes a generation segment | `rawOutput`, `parsedToolCalls` |
| `tool:dispatch` | Tool call starts | `tool`, `toolIndex`, `toolkitSize`, `args` |
| `tool:result` | Tool call returns | `result`, `prefillTokenCount`, `durationMs` |
| `generate:start` | Single-branch generation begins | `hasGrammar`, `hasParent`, `role` |
| `generate:end` | Single-branch generation ends | `output`, `tokenCount`, `parsed` |

## Problem: Agents not using tools

**Symptom:** Agents generate text but never call any tools. They produce raw text output and are marked done without tool calls.

**Diagnosis steps:**

1. Check `prompt:format` events with `role: 'agentSuffix'`. The `tools` field must contain the JSON-serialized tool schemas. If `tools` is missing or empty, the model has no awareness of available tools.

```jsonl
{"type":"prompt:format","role":"agentSuffix","tools":"[{\"type\":\"function\",...}]",...}
```

2. Check `agent:turn` events. If `parsedToolCalls` is an empty array, the model generated text but no tool-call syntax:

```jsonl
{"type":"agent:turn","agentId":5,"parsedContent":"The answer is...","parsedToolCalls":[]}
```

3. Verify the model supports tool calling. If the model's chat template does not handle tools, `formatChatSync` will produce `CHAT_FORMAT_CONTENT_ONLY` or `CHAT_FORMAT_GENERIC`, and agent setup will throw. If you catch/suppress this error, agents will run without tool awareness.

**Common causes:**
- Missing `tools` in the task spec: `tasks: [{ systemPrompt, content, parent: root }]` -- forgot `tools: toolsJson`
- Wrong model: base model instead of instruction-tuned, or instruction-tuned without tool-call training
- Empty toolkit: `createToolkit([])` produces an empty `toolsJson` string

## Problem: Early termination (agents report too soon)

**Symptom:** Agents call 0-1 non-terminal tools, then immediately call `report()` with shallow findings.

**Diagnosis steps:**

1. Check `tool:dispatch` events for each agent. Look at the `toolIndex` and `toolkitSize` fields:

```jsonl
{"type":"tool:dispatch","agentId":5,"tool":"report","toolIndex":4,"toolkitSize":5,...}
```

If `toolIndex` equals `toolkitSize - 1` (the tool is last in the array), check whether it is the terminal tool. Terminal tools at the end of the toolkit array trigger LLM recency bias.

2. Inspect the `pool:open` event's `pressure` field. If `headroom` is very low at pool start, agents may be immediately pressure-cut after their first tool call:

```jsonl
{"type":"pool:open","agentCount":3,"pressure":{"remaining":2048,"softLimit":1024,"headroom":1024}}
```

3. Check `pool:agentDrop` events. If `reason` is `pressure_softcut`, the agent wanted a non-terminal tool but headroom was negative:

```jsonl
{"type":"pool:agentDrop","agentId":5,"reason":"pressure_softcut"}
```

**Common causes:**
- **Tool ordering:** `report` or terminal tool is last in `createToolkit()` array. Fix by placing it before any tool that shares a name prefix. See [Custom Tool](./custom-tool.md) for the ordering rule.
- **Prompt quality:** System prompt does not motivate thorough investigation. Agents will call `report()` early if the prompt says "answer the question" rather than "investigate thoroughly using available tools."
- **Large prior tool results:** Earlier agents consumed most of the KV budget, leaving later agents pressure-constrained from their first turn.

## Problem: Agents killed by context pressure

**Symptom:** `pool:agentDrop` events appear in the trace. Some agents produce no findings.

**Diagnosis steps:**

1. Collect all `pool:agentDrop` events and group by `reason`:

| Reason | Meaning | When |
|--------|---------|------|
| `pressure_init` | Agent didn't fit during pool setup | Before any generation |
| `pressure_critical` | Remaining KV below `hardLimit` | During PRODUCE phase |
| `pressure_softcut` | Agent wanted a non-terminal tool but headroom was negative | At stop token during PRODUCE |
| `pressure_settle_reject` | Tool result tokens would cross softLimit floor | During SETTLE phase |
| `maxTurns` | Agent exhausted its turn budget | At stop token during PRODUCE |

2. Check `pool:open` for the initial pressure state:

```jsonl
{"type":"pool:open","pressure":{"remaining":8000,"softLimit":1024,"headroom":6976}}
```

If `headroom` is tight at pool open, subsequent tool results will exhaust it quickly.

3. Look at `tool:result` events for `prefillTokenCount`. Large tool results (web pages at 2000+ tokens, search results at 500+ tokens) are the primary KV consumers:

```jsonl
{"type":"tool:result","tool":"fetch_page","prefillTokenCount":1847,"durationMs":342}
{"type":"tool:result","tool":"web_search","prefillTokenCount":623,"durationMs":1205}
```

**Solutions:**
- Increase `nCtx` (see [Deployment](./deployment.md))
- Reduce `AGENT_COUNT` to give each agent more KV budget
- Truncate tool results in tool implementations (cap response size)
- Lower `softLimit` in pressure options (trades downstream headroom for agent longevity)
- Use `q4_0` KV quantization to fit more tokens in the same memory

## Problem: Scratchpad extraction fails

**Symptom:** `generate:start` events with `hasParent: true` appear but `generate:end` shows empty or malformed output. Hard-cut agents have null findings despite accumulated tool results.

**Diagnosis steps:**

1. Find `generate:start` events where `hasParent: true`:

```jsonl
{"type":"generate:start","branchHandle":12,"hasGrammar":true,"hasParent":true,"role":"scratchpad"}
```

2. Check the corresponding `generate:end`:

```jsonl
{"type":"generate:end","branchHandle":12,"tokenCount":0,"output":"","parsed":null}
```

Zero tokens or failed parse means the scratchpad generation could not produce valid output.

3. Check if the parent branch is still alive. Look for `branch:prune` events with the parent's handle before the scratchpad `generate:start`:

```jsonl
{"type":"branch:prune","branchHandle":8,"position":0}      // parent pruned
{"type":"generate:start","hasParent":true,"branchHandle":12} // scratchpad fork from pruned parent
```

If the parent was pruned before the fork, the scratchpad has no context to attend to.

**Common causes:**
- Parent branch was disposed between the pool ending and the scratchpad extraction attempt
- Grammar schema mismatch -- the extraction grammar does not match what the model can produce given the context
- Insufficient KV remaining for the scratchpad prompt itself

The hard-cut recovery in `ResearchTool` and `WebResearchTool` handles this gracefully -- extraction failures are caught and treated as non-fatal:

```typescript
try {
  const result = yield* generate<{ findings: string }>({
    prompt, grammar,
    parse: (o: string) => JSON.parse(o),
    parent: a.branch,
  });
  if (result.parsed?.findings) a.findings = result.parsed.findings;
} catch { /* extraction failure non-fatal */ }
```

## Problem: Synthesis ignores research findings

**Symptom:** The synthesis output does not reference research findings. It produces generic answers or hallucinates claims not present in the research.

**Diagnosis steps:**

1. Check the `prompt:format` event for the synthesis agent (look for the synthesis system prompt). The `taskContent` field should contain the full research findings:

```jsonl
{"type":"prompt:format","role":"agentSuffix","taskContent":"## corpus research\n\n### Agent 1\n..."}
```

If `taskContent` is empty or contains `(none)`, the research phase produced no findings.

2. Check the synthesis task content template. The harness fills `{{agentFindings}}` and `{{sourcePassages}}`:

```typescript
const content = SYNTHESIZE.user
  .replace('{{agentFindings}}', agentFindings || '(none)')
  .replace('{{sourcePassages}}', sourcePassages || '(none)')
  .replace('{{query}}', query);
```

If all research agents were dropped by pressure, both `agentFindings` and `sourcePassages` will be `(none)`.

3. Check if the synthesis agent has grounding tools. Look at `tool:dispatch` events from the synthesis pool. If the synthesis pool has only the `report` tool, the agent cannot independently verify claims:

```jsonl
{"type":"pool:open","agentCount":1}
{"type":"tool:dispatch","tool":"report","toolIndex":0,"toolkitSize":1}
```

A synthesis agent with grounding tools (search, read_file, grep, query) can make 4-6 verification calls before reporting. One with only `report` must rely entirely on the content in its prompt.

**Solutions:**
- Ensure research agents produced findings: check earlier `pool:close` events for non-null `findings` in the agents array
- Give synthesis grounding tools: `createToolkit([...groundingTools, reportTool])`
- Increase reranker passage count for more source material in the synthesis prompt

## Problem: Plan produces poor sub-questions

**Symptom:** Research agents investigate tangential topics. Sub-questions are too broad or too keyword-dense, leading to scattered rather than focused investigation.

**Diagnosis steps:**

1. Check the plan output. Find the `generate:end` event from the planning pass:

```jsonl
{"type":"generate:end","output":"{\"questions\":[{\"text\":\"...\",\"intent\":\"research\"},...]"}
```

2. Compare sub-question quality: Good sub-questions are focused and decompose the query into independent investigable units. Poor sub-questions restate the original query with different keywords or are overly broad.

3. Check whether the plan produced an empty array (passthrough). An empty array means the query goes to research as-is, which is correct for focused queries but wrong for complex multi-part queries.

**Solutions:**
- Revise `tasks/plan.md` to be more specific about decomposition criteria
- Adjust `maxQuestions` to constrain the planner
- Add examples to the plan prompt showing good vs bad decompositions

## Debugging workflow

A systematic approach for investigating unexpected behavior:

1. **Run with --trace** to produce a JSONL trace file

2. **Check pool:open events** for initial pressure state and agent count

3. **Check pool:agentDrop events** for premature agent kills

4. **Check agent:turn events** for tool call patterns -- are agents using tools? Are they alternating between search/read/grep (depth-first) or repeating the same tool (scatter)?

5. **Check tool:result events** for `prefillTokenCount` -- are tool results unusually large?

6. **Check prompt:format events** for correct `taskContent` and `tools` -- is the model seeing the right context?

7. **Check pool:close events** for aggregate statistics -- how many agents produced findings? What was the total token count?

For comparing two runs (e.g., before and after a change), extract the tool call sequences and compare:

```bash
# Extract tool dispatch sequence per agent
grep '"type":"tool:dispatch"' trace.jsonl | \
  jq -r '[.agentId, .tool, (.toolIndex|tostring)] | join("\t")'
```

This shows whether agents are following the expected investigation pattern (grep -> search -> read -> return-to-grep -> report) or degenerating into shallow patterns (grep -> report).
