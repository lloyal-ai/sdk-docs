# Thinking in lloyal

This guide bridges the gap between conventional async/await programming and the structured concurrency model that lloyal uses. If you've built agents with LangChain, CrewAI, or the Anthropic SDK, you already know how to work with tools, prompts, and multi-step pipelines. The new concept is how concurrent operations are managed — and why it matters for local inference.

## Why not async/await?

Cloud API-based agent frameworks use async/await because each step is an independent HTTP request. The LLM call returns a response, you parse it, call the next tool, send the next request. Each step is stateless — if your process crashes between steps, no GPU state is lost because there is no GPU state. The cloud provider manages everything.

Local inference is different. You have a model loaded in GPU memory. You have a KV cache with token-level state per branch. You have multiple agents sharing that cache through forked sequences. When something goes wrong — a tool throws, an agent hits the context limit, the user cancels — you need to clean up GPU resources (branches, KV cells, sampler state) in the correct order, or you leak memory that accumulates until the process dies.

Async/await doesn't guarantee cleanup. A `Promise.race()` between two operations leaves the loser running in the background. An `await` that throws skips cleanup code unless you manually wrap everything in try/finally. A detached `Promise` from a tool call can outlive the agent that spawned it. These are manageable annoyances with cloud APIs (the worst case is a wasted HTTP request). With local GPU state, they're memory leaks and crashes.

## Structured concurrency in one sentence

**Every concurrent operation is bound to the scope that created it and cannot outlive it.**

This is the same guarantee that synchronous code provides for local variables — when a function returns, its locals are freed. Structured concurrency extends this to asynchronous operations — when a scope exits, all operations it spawned are halted and cleaned up, guaranteed.

## The mental model: scopes as lifetimes

In lloyal, everything runs inside a scope. Scopes nest like function calls:

```
main()
 └─ initAgents()                         ← sets up Ctx, Store, Events
     └─ withSharedRoot()                 ← creates root branch, prefills system prompt
         └─ useAgentPool()               ← spawns N agents, runs tick loop
             ├─ agent 0 (generating)
             ├─ agent 1 (awaiting tool)
             │   └─ scoped(tool.execute)  ← tool runs in child scope
             │       └─ withSharedRoot()  ← inner pool for recursive tool
             │           └─ useAgentPool()
             └─ agent 2 (generating)
```

When a scope exits — normally, via error, or via cancellation — every child scope is halted, and cleanup runs in reverse order. The inner `useAgentPool` scope prunes inner branches. The inner `withSharedRoot` scope prunes the inner root. The tool's `scoped()` boundary catches errors without crashing the outer pool. This happens automatically, every time, regardless of how the scope exits.

## Generators instead of async/await

lloyal uses generator functions (`function*`) with `yield*` instead of `async` functions with `await`. The reason is control — the framework can interrupt a generator between yields, which enables cancellation and cleanup. An `async` function that's awaiting a Promise can't be interrupted until the Promise settles.

The syntax is similar:

```typescript
// async/await (cloud API pattern)
async function research(query: string) {
  const plan = await llm.chat({ messages, tools });
  const results = await Promise.all(
    plan.questions.map((q) => llm.chat({ messages: [q], tools })),
  );
  return synthesize(results);
}

// lloyal (structured concurrency pattern)
function* research(query: string): Operation<string> {
  const plan = yield* generate({ prompt, grammar });
  const pool = yield* useAgentPool({
    tasks: plan.questions.map((q) => ({
      systemPrompt,
      content: q,
      tools,
      parent: root,
    })),
    tools: toolMap,
  });
  return pool.agents.map((a) => a.findings);
}
```

The key differences:

- `yield*` instead of `await` — runs an Operation within the current scope
- `Operation<T>` instead of `Promise<T>` — describes what to do, doesn't execute immediately
- `useAgentPool` instead of `Promise.all` — concurrent agents with shared GPU resources and automatic cleanup

## The five Effection primitives lloyal uses

### 1. `call()` — bridge to Promises

Wraps a Promise-returning function so it can be used inside a generator. This is how lloyal interacts with async APIs (tokenization, prefill, model loading):

```typescript
// Wrap an async operation
const tokens = yield * call(() => ctx.tokenize(prompt, false));
yield * call(() => branch.prefill(tokens));
```

`call()` also handles plain values and Operations uniformly — you don't need to know what a tool's `execute()` returns.

### 2. `ensure()` — guaranteed cleanup

Registers a cleanup function that runs when the current scope exits, regardless of how it exits:

```typescript
const branch = parent.forkSync();
yield *
  ensure(() => {
    if (!branch.disposed) branch.pruneSync();
  });
// branch is ALWAYS pruned when this scope exits — normal return, error, or cancellation
```

This is how every agent branch is cleaned up in `setupAgent`. The developer never manually prunes branches.

### 3. `spawn()` — concurrent work within a scope

Launches a concurrent operation bound to the current scope:

```typescript
yield *
  spawn(function* () {
    for (const ev of yield* each(events)) {
      renderToUI(ev);
      yield* each.next();
    }
  });
// The event subscriber runs concurrently but is halted when the parent scope exits
```

### 4. `scoped()` — error boundaries

Creates a child scope that catches errors without crashing the parent:

```typescript
try {
  const result =
    yield *
    scoped(function* () {
      return yield* call(() => tool.execute(args));
    });
} catch (err) {
  agent.findings = `Tool error: ${err.message}`;
  // Outer pool continues — other agents survive
}
```

This is how the agent pool's DISPATCH phase runs tools — each tool executes in its own `scoped()` boundary.

### 5. `createContext()` — scoped dependency injection

Creates a typed value that's available to all child operations without argument passing:

```typescript
// Define
export const Ctx = createContext<SessionContext>("lloyal.ctx");

// Set (in parent scope)
yield * Ctx.set(sessionContext);

// Read (in any child scope)
const ctx = yield * Ctx.expect();
```

lloyal uses six contexts: `Ctx` (model/tokenizer), `Store` (batch decode), `Events` (event channel), `Trace` (trace writer), `TraceParent` (trace tree linkage), and `ScratchpadParent` (scratchpad fork point). They're set once in `initAgents()` or `withSharedRoot()` and inherited by every nested operation automatically.

## Mapping to familiar concepts

If you know these patterns, lloyal's structured concurrency is the same idea applied to GPU-backed inference:

| Concept            | Swift                                    | Kotlin                      | lloyal                                         |
| ------------------ | ---------------------------------------- | --------------------------- | ---------------------------------------------- |
| Scoped concurrency | `withTaskGroup`                          | `coroutineScope`            | `withSharedRoot` + `useAgentPool`              |
| Spawn child task   | `group.addTask`                          | `launch`                    | `spawn()`                                      |
| Structured cleanup | Task cancellation propagates to children | `Job` hierarchy + `finally` | Scope exit halts all children, `ensure()` runs |
| Scoped values      | `TaskLocal`                              | `CoroutineContext`          | `createContext()`                              |
| Error boundary     | `withThrowingTaskGroup`                  | `supervisorScope`           | `scoped()`                                     |
| Async bridge       | `withCheckedContinuation`                | `suspendCoroutine`          | `call()`                                       |

If you've never used any of these, the core insight is: **a scope owns its children, and cleanup is automatic**. You don't manage branch lifecycles, agent cleanup, or KV pressure recovery manually. The scope tree does it.

## What this means in practice

### You never manually prune branches

```typescript
// This is wrong — manual lifecycle management
const branch = parent.forkSync();
try {
  // ... use branch ...
} finally {
  branch.pruneSync(); // What if you forget? What if an inner call throws first?
}

// This is how lloyal does it — scope-managed
const branch = parent.forkSync();
yield *
  ensure(() => {
    if (!branch.disposed) branch.pruneSync();
  });
// Cleanup is registered once. It WILL run. You can't forget.
```

### Tool errors don't crash the pipeline

A tool that throws inside `scoped()` is caught at the dispatch level. The agent that called the tool is marked done with an error finding. Other agents in the pool continue. The outer pool is unaffected. No try/catch boilerplate needed in tool implementations — the scope tree handles it.

### Cancellation propagates correctly

When a user hits Ctrl-C during a deep research run with nested pools (research → web_research → sub-agents), the halt propagates from `main()` down through every scope. Inner agents are halted, inner branches are pruned, inner roots are pruned, outer agents are halted, outer branches are pruned, the model is disposed. All cleanup runs in the correct order. No orphaned GPU state. No leaked KV cells.

### Context replaces argument drilling

Instead of passing `sessionContext`, `branchStore`, `traceWriter`, and `scratchpadParent` through every function call, they're set in the scope tree and read where needed:

```typescript
// In initAgents (once):
yield * Ctx.set(ctx);
yield * Trace.set(traceWriter);

// In any tool, anywhere in the tree:
const ctx = yield * Ctx.expect(); // SessionContext
const tw = yield * Trace.expect(); // TraceWriter
```

A tool doesn't need to know how many layers of agent pools it's nested inside. It reads from context and gets the right value for its scope.

## Getting started

The fastest path to working with lloyal:

1. **Read `examples/deep-research/main.ts`** — the simplest entry point. Model loading, `initAgents`, event subscription, `handleQuery`.

2. **Read `examples/deep-research/harness.ts`** — the pipeline. `plan()` → `research()` → `synthesize()` → `evaluate()`. Each stage is a generator function that composes framework primitives.

3. **Run with `--trace`** — produces a JSONL file showing every prompt, tool call, agent turn, and branch lifecycle. This is the best way to understand what the pipeline does at runtime.

4. **Modify a task prompt** — change `examples/deep-research/tasks/research.md` and re-run. The pipeline structure stays the same; the model's behavior changes. This is where most iteration happens.

5. **Add a tool** — extend `Tool`, implement `execute()` as a generator, add it to `createToolkit()`. The tool automatically gets scope management, error boundaries, and trace integration.

The generator syntax (`function*`, `yield*`) looks unfamiliar for about an hour. After that, it reads like sequential code with automatic cleanup — which is exactly what it is.
