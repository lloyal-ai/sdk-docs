---
title: "Build a Custom Pipeline"
description: "Fork the compare harness, reshape the DAG, swap sources, and customize budget and recovery."
---

This guide walks through forking the [compare example](/examples/compare) and customizing it. Compare is a 6-node DAG that researches two subjects in parallel, compares them across three axes, and synthesizes — and it demonstrates the [skill catalog](/reference/skill-catalog) convention for mixed-role pools. It's a good fork target because the topology is non-trivial (multi-parent edges, multi-child convergence) but small enough to read end-to-end.

For the architectural patterns behind these customizations, see the [RIG Pipeline reference](/reference/rig/pipeline).

## Fork the harness

```bash
cp -r examples/compare/ examples/my-harness/
```

The key files:

| File | Purpose |
|---|---|
| `harness.ts` | DAG topology, orchestrator, pool setup |
| `main.ts` | CLI entry, model loading, source configuration |
| `prompts/*.eta` | Per-skill system prompts + skill catalog |
| `tui/` | Ink TUI (optional — drop this if you don't need a renderer) |

The harness is **source-agnostic** — sources are loaded in `main.ts` and passed to `handleCompare`. The DAG topology is hard-coded in `harness.ts` and easy to reshape.

## Reshape the DAG

The DAG is just an array of `DAGNode` objects:

```typescript
const nodes: DAGNode[] = [
  { id: "research_web_X",   task: { ... }, userContent: `Research findings on ${x}:` },
  { id: "research_corp_Y",  task: { ... }, userContent: `Research findings on ${y}:` },
  ...axes.map<DAGNode>((axis, i) => ({
    id: `compare_axis_${i + 1}`,
    dependsOn: ["research_web_X", "research_corp_Y"],
    task: { ... },
    userContent: `Comparison along axis "${axis}":`,
  })),
  { id: "synthesize", dependsOn: ["compare_axis_1", "compare_axis_2", "compare_axis_3"], task: { ... } },
];
```

To customize:

- **Add a node** — append a new entry. List its dependencies in `dependsOn`. The orchestrator awaits each declared dep's task before spawning.
- **Remove a node** — delete the entry, drop its id from any downstream `dependsOn` lists.
- **Reshape edges** — change `dependsOn`. The orchestrator will handle whatever topology you declare (acyclic only — cycles aren't checked but will deadlock).

`userContent` controls what gets prefilled onto the spine when the node completes. If you don't want a node's result to flow downstream as conversation history, omit `userContent`.

## Add a fact-check stage

Suppose you want a fact-check stage between `compare_axis_*` and `synthesize`. Verify each axis's claims against the source material before synthesizing:

```typescript
...axes.map<DAGNode>((axis, i) => ({
  id: `factcheck_axis_${i + 1}`,
  dependsOn: [`compare_axis_${i + 1}`],
  task: {
    content: `Verify claims in the comparison along axis "${axis}".`,
    systemPrompt: renderTemplate(FACTCHECK, { axis }),
    seed: 4000 + i,
  },
  userContent: `Fact-check on axis "${axis}":`,
})),
{
  id: "synthesize",
  dependsOn: ["factcheck_axis_1", "factcheck_axis_2", "factcheck_axis_3"],   // ← updated
  task: { ... },
},
```

Add a `FACTCHECK` template under `prompts/` and load it like the other templates. If the fact-check skill needs additional tools beyond the existing palette, list them in the skill catalog so the model knows they exist.

## Change skills

Each per-spec `systemPrompt` starts with `Apply the **<skill>** skill.` The skills are described in `prompts/skill-catalog.eta`, which is what gets prefilled onto `queryRoot`:

```eta
The agent system message will tell you which skill to apply. Use only that skill's tools.

## Skills

### web_research
- Tools: web_search, fetch_page, report
- Use to investigate topics on the live web.

### compare
- Tools: report
- Use when comparing two researched subjects along an axis.

### synthesize
- Tools: report
- Use when integrating multiple comparisons into a final report.
```

To add a new skill:

1. Document it in `skill-catalog.eta` with its tool subset and intended use.
2. Add the per-skill system prompt template to `prompts/`.
3. Load and render it in `harness.ts`.
4. The new skill's nodes prepend `Apply the **<your_skill>** skill.` in their system prompt.

The schemas for the new skill's tools must already be in the toolkit at `withSharedRoot({ toolsJson })` — only role-level skill selection happens via prompt; tool registration happens in `main.ts` when sources are bound.

## Configure sources

Sources are configured in `main.ts` and passed as a typed array. Compare's flow is `web + corpus`, but the `Source` contract is uniform — swap in any combination:

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
// Add custom sources here — vector stores, databases, JIRA, anything that
// implements Source<TCtx, TChunk>. See /learn/sources.
```

If you only want one research lane, drop the corresponding research node from the DAG and remove the `dependsOn` on it from descendants. The orchestrator scales to whatever topology you declare.

## Choose an orchestrator

Compare inlines a custom `dagWithEvents` orchestrator that emits per-node lifecycle events for the Ink TUI. If you don't need a TUI, use the framework's stock `dag()`:

```typescript
import { dag } from "@lloyal-labs/lloyal-agents";

const pool = yield* withSharedRoot({ ... }, function* (queryRoot) {
  return yield* agentPool({
    orchestrate: dag(nodes),       // ← stock orchestrator, no TUI hooks
    tools, parent: queryRoot, terminalTool: "report", maxTurns,
  });
});
```

Other shapes if your pipeline isn't a DAG:

- `chain([spec1, spec2, ...], (spec, i) => { ... })` — sequential stages with `extendRoot` between them
- `parallel([spec1, spec2, ...])` — independent agents on shared KV
- `fanout(landscapeSpec, [domainSpec1, ...])` — landscape pass that informs N parallel domain agents

See [Concurrency](/reference/concurrency) for the full orchestrator catalog.

## Configure budget and recovery

Context pressure, time limits, and scratchpad recovery are configured on the policy:

```typescript
const policy = new DefaultAgentPolicy({
  budget: {
    context: { softLimit: 2048 },                            // nudge with 2k cells left
    time: { softLimit: 480_000, hardLimit: 600_000 },        // nudge 8min, kill 10min
  },
  recovery: { prompt: REPORT },                              // grammar-constrained extraction
});

const pool = yield* agentPool({
  // ...
  policy,
});
```

Higher `softLimit` means agents are nudged earlier, preserving room for downstream stages. Lower `softLimit` lets agents run longer but risks downstream stages running out of KV.

The general rule: research pools (broad investigation, longer running) get a policy with budget + recovery. Reporter or synthesis pools that just need one `report()` call use the default policy.

## Warm vs cold

If your harness is reused across queries (e.g. a REPL), wire `Session.commitTurn` to extend the trunk:

```typescript
const result = yield* handleCompare(session, sources, reranker, opts);
if (result.answer) {
  yield* call(() => session.commitTurn(query, result.answer));
}
```

Tomorrow's query forks from `session.trunk` and reads yesterday's Q&A through KV attention. See [Sessions](/learn/sessions) for the multi-turn lifecycle.

## Customize the TUI

If you want to keep the Ink TUI but render a different topology, the renderer is parameterized by `dag:topology` and `dag:node:spawn` events emitted by `dagWithEvents`. Adjust the layout in `tui/DagCanvas.tsx` to handle your new node shapes. If you don't need a TUI, delete `tui/` and drop the `emitDagEvent` plumbing from `harness.ts`.

## See also

- [Compare example](/examples/compare) — the canonical fork target
- [RIG Pipeline reference](/reference/rig/pipeline) — architectural patterns for retrieval-interleaved harnesses
- [Skill Catalog](/reference/skill-catalog) — the convention for mixed-role pools
- [Concurrency](/reference/concurrency) — orchestrator catalog and the five-phase tick loop
- [Sources](/learn/sources) — implementing `Source<TCtx, TChunk>` for custom backends
