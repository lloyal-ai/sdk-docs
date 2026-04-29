---
title: "Compare"
description: "DAG primer — research two subjects in parallel, compare across three axes concurrently, synthesize. Multi-parent dependencies, skill catalog, spine extension."
---

A 6-node DAG that researches two subjects on different sources, compares them across three axes simultaneously, and synthesizes the result. The smallest topology that genuinely needs `dag()` instead of `chain` or `fanout` — three sibling nodes each depend on **both** research roots, and the synthesizer depends on all three siblings.

```
  research_web_X ──┐                          ┌──▶ compare_axis_1 ──┐
  (WebSource)      │                          │                     │
                   ├──────────────────────────┼──▶ compare_axis_2 ──┼──▶ synthesize
  research_corp_Y ─┘                          │                     │
  (CorpusSource)                              └──▶ compare_axis_3 ──┘

       roots                       fan-in / fan-out                  sink
   (parallel, no deps)          (3 siblings sharing deps)
```

This example is the framework primer for `dag(...)` — and the canonical demonstration of the [skill catalog](/reference/skill-catalog) convention applied to mixed-role agents.

**Source**: `examples/compare/`

## Prerequisites

- A GGUF instruction-tuned model with native tool calling
- A GGUF reranker model
- A directory of text/markdown files for the corpus side
- A Tavily API key for the web side (free tier at [tavily.com](https://tavily.com))

## Run it

```bash
export TAVILY_API_KEY=tvly-…

npx tsx examples/compare/main.ts \
  --x "Rust's ownership model" \
  --y "Swift's automatic reference counting" \
  --corpus ~/Documents/swift-docs \
  --reranker ~/.cache/lloyal/models/Qwen3-Reranker-0.6B-Q8_0.gguf \
  --axes "memory safety,runtime cost,ergonomics" \
  ~/.cache/lloyal/models/Qwen3.5-4B-Q4_K_M.gguf
```

Or via the workspace script: `npm run examples:compare -- --x "…" --y "…" …`

Required flags: `--x`, `--y`, `--corpus`, `--reranker`, `TAVILY_API_KEY`. Optional: `--axes` (default `accuracy,performance,complexity` — must be exactly three), `--max-turns`, `--n-ctx`, `--trace`, `--jsonl`.

## What it teaches

- **Multi-parent DAG dependencies** — each `compare_axis_*` node depends on **two** research nodes simultaneously. `chain` and `fanout` cannot express this.
- **Sibling parallelism with shared deps** — the three compare nodes fire the moment both research nodes complete, then run concurrently.
- **Multi-child convergence** — `synthesize` waits on all three siblings before spawning.
- **Spine extension is causal, not just sequential** — each node's `userContent` is prefilled onto the shared root via `ctx.extendRoot`. The compare nodes don't merely *follow* the research nodes — they *attend to* them. The edge in the diagram is the spine.
- **Skill catalog over mixed roles** — researcher, comparer, synthesizer all draw from one tool palette. Tool schemas and role descriptions live at the root once; per-spec systemPrompts say `Apply the **<skill>** skill`.

## Code walkthrough

### `harness.ts` — DAG topology

The 6 nodes are declared as a flat array of `DAGNode`. Dependencies are explicit `dependsOn` strings:

```typescript
const nodes: DAGNode[] = [
  {
    id: "research_web_X",
    task: { content: `Research subject: ${x}`, systemPrompt: renderTemplate(RESEARCH_WEB, ...), seed: 1001 },
    userContent: `Research findings on ${x}:`,
  },
  {
    id: "research_corp_Y",
    task: { content: `Research subject: ${y}`, systemPrompt: renderTemplate(RESEARCH_CORPUS, ...), seed: 1002 },
    userContent: `Research findings on ${y}:`,
  },
  ...axes.map<DAGNode>((axis, i) => ({
    id: `compare_axis_${i + 1}`,
    dependsOn: ["research_web_X", "research_corp_Y"],   // ← multi-parent edge
    task: { content: `Compare ${x} vs ${y} on: ${axis}`, systemPrompt: renderTemplate(COMPARE, { x, y, axis }), seed: 2000 + i },
    userContent: `Comparison along axis "${axis}":`,
  })),
  {
    id: "synthesize",
    dependsOn: ["compare_axis_1", "compare_axis_2", "compare_axis_3"],
    task: { content: `Write the final compare-and-contrast report on ${x} vs ${y}.`, systemPrompt: renderTemplate(SYNTHESIZE, ...), seed: 3000 },
  },
];
```

Each `userContent` field is the curated turn that gets prefilled onto the spine when the node completes. Dependent nodes attend over those prefills as if they were prior conversation turns — the model sees `Research findings on X: <agent's report>` in its KV at the position where the edge fires.

### Pool setup — skill catalog at queryRoot

The catalog (system prompt + tool schemas) is amortized at the harness's `queryRoot` once. All six agents fork from there and inherit it via prefix-share:

```typescript
const toolkit = createToolkit(tools);
const pool = yield* withSharedRoot(
  {
    parent: session.trunk ?? undefined,
    systemPrompt: SKILL_CATALOG,             // ← skill catalog at root
    toolsJson: toolkit.toolsJson,            // ← tool schemas at root
  },
  function* (queryRoot) {
    return yield* agentPool({
      orchestrate: dagWithEvents(nodes, emit),
      tools, parent: queryRoot,
      terminalTool: "report",
      maxTurns,
      pruneOnReport: true,
      scorer: primaryScorer,
      trace,
    });
  },
);
```

Per-spec system prompts inside the eta templates start with `Apply the **<skill>** skill.` — the model selects which set of tools to use without re-emitting schemas per agent. See [Skill Catalog](/reference/skill-catalog) for the full convention.

### `dagWithEvents` — orchestrator with TUI hooks

The example inlines a custom orchestrator that mirrors the framework's `dag()` (after the Task-as-Future refactor) but emits per-node lifecycle events so the Ink TUI can map agent ids back to DAG node ids:

```typescript
function dagWithEvents(nodes: DAGNode[], emit: (ev: DagEvent) => void): Orchestrator {
  return function* (ctx) {
    emit({ type: 'dag:topology', nodes: nodes.map(...) });

    const tasks = new Map<string, Task<void>>();

    function* runNode(n: DAGNode): Operation<void> {
      // Each declared dep awaited as a Task<T> — Task extends Future extends Operation,
      // so awaiting another task IS the cross-task rendezvous primitive.
      for (const depId of n.dependsOn ?? []) {
        yield* tasks.get(depId)!;
      }
      const agent = yield* ctx.spawn({ ...n.task, parent: n.task.parent ?? ctx.root });
      emit({ type: 'dag:node:spawn', id: n.id, agentId: agent.id });
      yield* ctx.waitFor(agent);
      if (agent.result && n.userContent) {
        yield* ctx.extendRoot(n.userContent, agent.result);   // ← spine extension
      }
    }

    for (const n of nodes) tasks.set(n.id, yield* spawn(() => runNode(n)));
    for (const t of tasks.values()) yield* t;
  };
}
```

This is a 25-line illustration of the canonical Effection DAG pattern: each node runs as a child Task; "A depends on B" becomes `yield* tasks.get(B)!` inside A's body. No mutable Sets. No race window. Failure in any node halts the rest via structured concurrency. For the framework's stock `dag()` orchestrator (without the event hooks), see [Concurrency — DAG](/reference/concurrency#dag).

### `prompts/` — eta templates per skill

- `skill-catalog.eta` — the catalog rendered onto `queryRoot`. Lists `web_research`, `corpus_research`, `compare`, `synthesize` skills with their tool subsets.
- `research-web.eta`, `research-corpus.eta` — per-skill researcher prompts; both prepend `Apply the **<skill>** skill`.
- `compare.eta` — per-axis comparison prompt.
- `synthesize.eta` — final synthesis prompt.

### TUI

`tui/` mounts an Ink renderer in TTY mode. Cards are laid out in topological layers connected by orthogonal box-drawing edges; pending cards show a dotted background; active cards stream tokens live; completed cards collapse to a one-line summary.

```
╭ DAG · Rust ownership vs Swift ARC · 0:32 ─────────────────────────╮
│ 1840 tok · 18 tools                                               │
╰───────────────────────────────────────────────────────────────────╯

[research_web_X]  ──┐
  searching…       │     [compare_axis_1] memory safety
                   ├───→ comparing…
[research_corp_Y] ─┘     [compare_axis_2] runtime cost
  reading…               [compare_axis_3] ergonomics
                              │
                              ↓
                         [synthesize] (pending)
```

In non-TTY mode (`--jsonl` or piped output), it falls back to one-line stderr events and a plain stdout final answer for scripting.

## What's unique vs the other examples

| Aspect | react-agent | reflection | compare |
|---|---|---|---|
| Topology | single agent | linear pipeline | DAG (multi-parent + multi-child) |
| Sources | corpus | corpus | web + corpus |
| Pool primitive | `useAgent` | `useAgent` + manual branches | `agentPool({ orchestrate: dag(...) })` |
| Spine | single branch | branch chain (fork forward) | `extendRoot` per node |
| Catalog | none (single role) | none (single role) | skill catalog (mixed roles) |

Compare is the example to study when you need multi-parent dependencies, want to see the skill catalog in action, or want a worked DAG harness to fork.

## Customization

- **Different axes** — `--axes "axis1,axis2,axis3"` (must be exactly three).
- **Different research sides** — both research nodes use the same source contract; swap `WebSource` for another implementation in `main.ts` if you want corpus-vs-corpus or web-vs-web.
- **More compare siblings** — adjust the axes array length and the `dependsOn` of `synthesize`. The DAG structure scales arbitrarily.
- **Replace `dagWithEvents`** with the framework's `dag()` if you don't need the TUI's per-node hooks.

## Related pages

- [Skill Catalog](/reference/skill-catalog) — the convention this example uses for mixed-role pools
- [Continuous Context Spine](/reference/continuous-context-spine) — what `extendRoot` writes and how forks attend to it
- [Concurrency Model — DAG](/reference/concurrency) — the framework's `dag()` orchestrator and Task-as-Future pattern
- [RIG Pipeline](/reference/rig/pipeline) — reference architecture for retrieval-interleaved harnesses
