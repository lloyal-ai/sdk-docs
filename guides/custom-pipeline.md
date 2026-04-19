---
title: "Build a Custom Pipeline"
description: "Fork the reference harness, add/remove stages, modify routing, and configure sources."
---

This guide walks through forking the reference deep-research harness and customizing it. The harness in `examples/deep-research-web/harness.ts` implements a five-stage pipeline:

```
Plan -> Research -> [Bridge ->] Synthesize -> Eval
```

Each stage is a generator function that yields Effection operations. You compose stages by calling them in sequence from the entry point (`handleQuery`). Adding, removing, or reordering stages is a matter of editing that sequence.

## Fork the reference harness

Copy the harness and its task prompts:

```bash
cp -r examples/deep-research-web/ examples/my-pipeline/
```

The key files:

| File | Purpose |
|------|---------|
| `harness.ts` | Stage functions, routing logic, entry point |
| `main.ts` | CLI entry, model loading, source configuration |
| `tui.ts` | Terminal UI event renderer |
| `tasks/*.md` | Prompt files for each stage |

The harness imports sources as a typed array and is source-agnostic. You configure which sources to use in `main.ts`.

## Stage anatomy

Every stage is a generator function with the signature:

```typescript
function* stageName(
  inputs...,
  opts: WorkflowOpts,
): Operation<StageResult> {
  // Emit start event
  yield* opts.events.send({ type: 'stage:start', ... });
  const t = performance.now();

  // Do work (create branches, run agents, generate)
  const result = yield* withSharedRoot(
    { systemPrompt: PROMPT.system, tools: toolkit.toolsJson },
    function*(root) {
      const pool = yield* useAgentPool({ ... });
      return pool;
    },
  );

  // Emit done event with timing
  const timeMs = performance.now() - t;
  yield* opts.events.send({ type: 'stage:done', timeMs, ... });

  return result;
}
```

The pattern: emit an event, do work inside `withSharedRoot`, emit a completion event. The `withSharedRoot` scope guarantees the shared root branch is pruned when the stage ends, freeing KV for the next stage.

## Add a new stage

Suppose you want a "fact-check" stage that runs after synthesis but before eval. The stage takes the synthesis output and runs agents to verify individual claims.

### 1. Create the task prompt

Create `tasks/factcheck.md`:

```markdown
You are a fact-checking agent. Verify each claim against the source material.
For each claim, determine: supported, unsupported, or partially supported.
---
Claims to verify:
{{claims}}

Source passages:
{{sourcePassages}}
```

The `---` separator splits system prompt (above) from user content (below). The `loadTask()` helper parses this format.

### 2. Load the prompt

In `harness.ts`, add:

```typescript
const FACTCHECK = loadTask('factcheck');
```

### 3. Write the stage function

```typescript
function* factCheck(
  claims: string,
  sourcePassages: string,
  opts: WorkflowOpts,
): Operation<{ verified: string; timeMs: number }> {
  const content = FACTCHECK.user
    .replace('{{claims}}', claims)
    .replace('{{sourcePassages}}', sourcePassages);

  yield* opts.events.send({ type: 'factcheck:start' });
  const t = performance.now();

  const pool = yield* agentPool({
    tasks: [{ content }],
    tools: [...opts.sources.flatMap(s => s.tools), reportTool],
    systemPrompt: FACTCHECK.system,
    terminalTool: 'report',
    maxTurns: opts.maxTurns,
    trace: opts.trace,
  });

  const timeMs = performance.now() - t;
  yield* opts.events.send({ type: 'factcheck:done', timeMs });

  return { verified: pool.agents[0]?.result || '', timeMs };
}
```

### 4. Wire into the entry point

In `handleQuery`, insert the call between synthesis and eval:

```typescript
// Research -> Synthesize -> Fact-check -> Eval -> Finalize
const res = yield* research(r.questions, query, opts, r.maxTurns);
const s = yield* synthesize(res.agentFindings, res.sourcePassages, query, opts);

// New stage
const fc = yield* factCheck(
  s.pool.agents[0]?.result || '',
  res.sourcePassages,
  opts,
);

// Use fact-checked output for eval and trunk promotion
const findings = fc.verified || s.pool.agents[0]?.result || '';
```

### 5. Add event types

Extend `WorkflowEvent` in `tui.ts` with the new events:

```typescript
export type StepEvent =
  | { type: 'factcheck:start' }
  | { type: 'factcheck:done'; timeMs: number }
  // ... existing events
```

## Remove a stage

To skip the eval stage, remove the `diverge` + `evaluate` calls from `handleQuery` and return the synthesis result directly:

```typescript
const s = yield* synthesize(res.agentFindings, res.sourcePassages, query, opts);
const findings = s.pool.agents[0]?.result || '';

// Skip eval, go straight to finalize
if (warm) {
  yield* appendTurn(query, findings, opts);
} else if (findings) {
  yield* promoteTrunk(query, findings, opts);
}
```

Each stage is independent. Removing one does not affect others as long as the data flow connects.

## Modify routing logic

The `route()` function maps plan output to pipeline behavior:

```typescript
function route(plan: PlanResult, query: string, maxTurns: number): Route {
  const research = plan.questions.filter(q => q.intent === 'research');
  const clarify = plan.questions.filter(q => q.intent === 'clarify');

  if (research.length === 0 && clarify.length > 0)
    return { type: 'clarify', questions: clarify.map(q => q.text) };

  const questions = research.length > 0 ? research.map(q => q.text) : [query];
  const effectiveMaxTurns = questions.length === 1 ? maxTurns * 2 : maxTurns;
  return { type: 'research', questions, maxTurns: effectiveMaxTurns };
}
```

The logic: all-clarify returns to user, empty array passes the original query through, research questions get dispatched. Mixed intent drops clarify questions.

To add a new route type (e.g., "direct answer" for simple questions):

```typescript
type Route =
  | { type: 'clarify'; questions: string[] }
  | { type: 'research'; questions: string[]; maxTurns: number }
  | { type: 'direct' };

function route(plan: PlanResult, query: string, maxTurns: number): Route {
  // Add intent classification in PlanTool grammar
  if (plan.questions.length === 0 && plan.directAnswer)
    return { type: 'direct' };

  // ... existing logic
}
```

Then handle the new route in `handleQuery`:

```typescript
if (r.type === 'direct') {
  // Single agent, no pool
  const result = yield* agent({
    systemPrompt: DIRECT_PROMPT,
    task: query,
    schema: answerSchema,
  });
  yield* call(() => session.commitTurn(query, result.rawOutput));
  return { type: 'done' };
}
```

## Change plan prompts

`PlanTool` uses a grammar-constrained generation pass. The grammar enforces the output shape (`{ questions: [{ text, intent }] }`), while the prompt controls the decomposition strategy.

Edit `tasks/plan.md` to change how queries are broken down:

```markdown
You decompose research queries into focused sub-questions.
Each sub-question should target a single concept or claim.
Classify as "research" (answerable via search) or "clarify" (needs user input).
Return an empty array if the query is focused enough to investigate directly.
---
Analyze this query and produce up to {{count}} sub-questions.

Query: {{query}}
```

The `{{count}}` placeholder is replaced with `opts.agentCount` at runtime. Keep sub-question count aligned with agent count to avoid idle agents.

To change the maximum number of sub-questions, adjust `agentCount` in `main.ts`:

```typescript
const AGENT_COUNT = 5;  // More agents = more parallel questions
```

## Configure budget and recovery

Context pressure, time limits, and scratchpad recovery are configured on the policy's `budget` and `recovery` options. The defaults (`softLimit: 1024`, `hardLimit: 128`) work for most pipelines.

```typescript
const researchPolicy = new DefaultAgentPolicy({
  budget: {
    context: { softLimit: 2048 },  // Reserve more KV for downstream work
    time: { softLimit: 480_000, hardLimit: 600_000 },  // nudge 8min, kill 10min
  },
  recovery: { prompt: REPORT },
});

const pool = yield* agentPool({
  // ...
  policy: researchPolicy,
});
```

Higher `softLimit` means agents are nudged earlier, preserving room for synthesis and eval. Lower `softLimit` lets agents run longer but risks synthesis running out of KV.

For reporter sub-pools (bridge agents), use the default policy — they only need one `report()` call and don't need recovery or time limits.

The general rule: research pools get a policy with budget + recovery; reporter/synthesis pools use the default.

## Configure sources

Sources are configured in `main.ts` and passed to the harness as an array:

```typescript
const sources: Source<SourceContext, Chunk>[] = [];

if (corpusDir) {
  const resources = loadResources(corpusDir);
  const chunks = chunkResources(resources);
  sources.push(new CorpusSource(resources, chunks));
}

if (process.env.TAVILY_API_KEY) {
  sources.push(new WebSource(new TavilyProvider()));
}
```

Sources run sequentially during research. After source N completes, its branches are pruned and KV is freed for source N+1. Between sources, a bridge structures discoveries as durable context for the next source.

Source order matters: put the source with the strongest grounding first. The bridge carries structured discoveries forward, so later sources build on earlier findings rather than re-covering the same ground.

## Warm vs cold paths

The harness handles both cold (first query, no prior context) and warm (follow-up query, trunk exists) paths:

```typescript
const warm = !!opts.session.trunk;
const res = warm
  ? yield* warmResearch(r.questions, query, opts, r.maxTurns)
  : yield* research(r.questions, query, opts, r.maxTurns);
```

The cold path (`research`) wraps everything in `withSharedRoot`. The warm path (`warmResearch`) skips the outer root since it builds on the existing trunk. After completion, cold promotes a new trunk via `promoteTrunk`; warm appends to the existing trunk via `appendTurn`.

This branching is not a mode switch to eliminate -- it reflects the physical difference between starting from position 0 (no prior tokens in KV) and continuing from an existing position (prior conversation tokens present in KV).
