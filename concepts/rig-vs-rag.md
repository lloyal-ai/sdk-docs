# RIG vs RAG

RAG retrieves first, then generates. A retrieval step runs upfront — query the vector store, get top-k passages, inject them into the prompt, call the model once. The model sees static context. Retrieval and generation are separate phases.

RIG interleaves retrieval and generation inside the decode loop. Agents generate reasoning, decide to search, process results, reason further, fetch a page, form hypotheses from the content, search again with refined queries. Retrieval decisions emerge from ongoing generation — each search query is informed by everything the agent has already discovered.

## The observable difference

The difference is in tool call inputs. A RAG system constructs search queries from the original user question. A RIG agent constructs queries from hypotheses formed during generation:

```
grep(/memory leak/)            → 3 matches in 2 files
read_file(pool.ts L40-80)      → reads allocation logic, spots missing cleanup
search("resource cleanup on connection close")  → finds teardown handler
read_file(server.ts L120-155)  → discovers close handler never calls pool.drain()
grep(/drain|dispose|cleanup/)  → 8 matches, confirms drain exists but is unused
search("pool drain connection lifecycle interaction")  → targets the gap
report(findings)
```

The last search — `"pool drain connection lifecycle interaction"` — is the signature behavior. The agent read the allocation logic, discovered the drain method existed but was never called on connection close, and constructed a search specifically targeting that interaction. This is multi-hop reasoning: not "search and report" but "search, form hypothesis, search for confirmation."

## Why it's emergent

This behavior is not prompted or engineered. It emerges from the concurrency semantics of the agent runtime.

The four-phase tick loop creates a clean decision boundary between each tool call and the next generation step:

1. Agent generates tokens, hits stop token, tool call extracted
2. Tool executes to completion — agent is suspended
3. Tool result fully prefilled into the agent's KV cache
4. Grammar state resets — clean slate for next decision
5. Agent resumes generating with the complete result as the last thing in context

Step 5 is the critical moment. The model's next-token prediction operates on a context where the tool result is fully present and the grammar is clean. The model makes a fresh decision: call another tool, call the same tool with different arguments, or report findings. This decision is informed by everything the agent has seen — all prior tool results are physically present in the branch's KV cache.

An agent that greps with a narrow pattern and gets 0 matches will broaden the pattern on its next grep — not because it's prompted to retry, but because the 0-match result is in context and the model naturally adjusts. An agent that reads a section and discovers an unexpected connection will construct a search query targeting that specific connection — the read result is in context, and the model forms a hypothesis from it.

## The decision boundary matters

Under a concurrent dispatch model where tool results arrive mid-generation, the agent is already producing tokens when results land. The result gets incorporated, but there's no clean pause for hypothesis formation. The observable effect: sequential dispatch produces progressively more specific queries; concurrent dispatch produces variations on the original question.

This is why tool results are prefilled into the KV cache during the SETTLE phase rather than streamed in during PRODUCE. The full result must be present before the model generates another token, so the decision at step 5 is informed by complete information.

## Depth scales with `maxTurns`

At `maxTurns: 2`, agents do single-shot retrieval. At `maxTurns: 6`, agents do 3-4 rounds of iterative refinement. At `maxTurns: 20`, agents go deep — following citation chains, cross-referencing claims, building evidence maps. The quality difference is in the later tool call inputs.

For more on the concurrency model that produces this behavior, see [Concurrency](concurrency.md). For measured real-world tool trajectories, see the [Receipts](concurrency.md#receipts) section.
