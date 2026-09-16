# LLM Token Optimizer

Cuts LLM token spend three ways — prefix caching, context pruning, model routing —
and measures each against a real workload rather than estimating it.

The workload is [ops-copilot](https://github.com/umars28/ops-copilot): its ten
MCP tool schemas and system prompt, and a recorded eight-turn investigation.

## Results

| optimisation | saving | what it costs |
| --- | --- | --- |
| prefix caching | **21%** | nothing; the model sees identical input |
| context pruning | **37%** of input tokens | changes what the model sees — pair with an eval |
| model routing | **64%** | only valid next to an accuracy measure on routed traffic |

The prefix is **2,543 tokens resent on every turn** — ten tool schemas and a
system prompt that never change within a run. That is what caching removes.

```
token-optimizer measure workloads/ops-copilot.json --routing 24
token-optimizer audit path/to/system_prompt.txt
```

## Measured in tokens, computed in dollars

Every saving above is a ratio between two token counts taken with the same
tokenizer, so the tokenizer's systematic error cancels out of the ratio. **"21%
fewer tokens" survives the approximation.**

Dollar figures do not. They are computed from published list prices and a proxy
BPE count, because Anthropic's tokenizer is not published. They are labelled
computed, never billed, and `count_tokens` against the real API is how to settle
an absolute number when one is needed.

A proxy tokenizer is used rather than characters-divided-by-four, because the two
disagree most on exactly the content this optimises: JSON, code, repeated
structure and long identifiers.

## Caching is a property of your prefix, not of the cache

A cache is a prefix match. Any byte that changes anywhere before the breakpoint
invalidates everything after it — so a single timestamp in a system prompt drops
the hit rate to zero while every line of caching code still looks correct.

`audit` finds those: timestamps, UUIDs, epoch seconds, request ids.

```
$ token-optimizer audit prompt.txt
not cacheable  prefix contains 1 per-request value(s); the cache would never hit
  timestamp  2026-09-16T04:12:00
      changes every request, invalidating everything after it
```

Two more things that make caching look broken when it is working as specified:

**A prefix below ~1024 tokens is never cached.** A breakpoint there is silently
ignored, and nothing in the response says so.

**A cache write costs 1.25x fresh input and a read 0.1x**, so a prefix used once
is *more* expensive cached than not. The breakeven for Opus 5 is **2 calls**,
which the tool reports rather than assuming.

## The three savings are not interchangeable

**Caching is free.** The model sees byte-identical input; only the billing
changes. Adopt it whenever the prefix is stable and above breakeven.

**Pruning changes what the model sees.** Removing repeated tool results is safe
— an agent that queries the same thing twice pays twice and learns nothing new.
Summarising superseded observations is not obviously safe, and the 37% here is
the saving, not a claim that quality held. That pairing is what
[llm-eval-harness](https://github.com/umars28/llm-eval-harness) exists for.

**Routing is the biggest number and the easiest to overstate.** 64% assumes the
cheap model handles the routine traffic correctly, and this repository does not
measure whether it does. The router here is keyword-based and says so; a real one
would classify the task, and that classifier would need its own evaluation before
its decisions counted.

Reporting 64% without that caveat would be the kind of number that survives a
README and fails in production.

## Where it fits

One of a set that share a spine:
[ops-copilot](https://github.com/umars28/ops-copilot) is the agent whose prompt
this measures, [llm-eval-harness](https://github.com/umars28/llm-eval-harness)
is what proves a pruning change did not cost quality, and
[runbook-rag](https://github.com/umars28/runbook-rag),
[agent-guardrail](https://github.com/umars28/agent-guardrail) and
[pr-reviewer](https://github.com/umars28/pr-reviewer) complete the set.

## Licence

MIT
