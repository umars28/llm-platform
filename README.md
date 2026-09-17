# llm-platform

Seven components for building and operating LLM systems — retrieval, evaluation,
tracing, cost routing, prompt-injection defence — and three agents built on top
of them. No orchestration framework: the Anthropic SDK, the MCP protocol,
OpenTelemetry and a vector store, wired directly.

The organising constraint is that **every claim here is a number produced by a
command you can run**. Each component ships its own labelled corpus, and the
corpora live in `benchmarks/` so one harness scores all of them.

## Results

| component | measured | result |
| --- | --- | --- |
| [runbook-rag](apps/runbook-rag) | 70 labelled queries over 381 K8s doc pages | recall@5 **0.526 → 0.638**; hit@5 0.757 → 0.871 |
| [ops-copilot](apps/ops-copilot) | 30 reproducible incident scenarios | root cause **96.7%** (29/30); strictly correct 93.3% |
| [agent-guardrail](packages/agent-guardrail) | 40 attacks / 40 hard negatives, cross-validated | **68% detection at 0% false positives** (shippable layer) |
| [pr-reviewer](apps/pr-reviewer) | 55 diffs, defects from real git history | an 8B local model finds **0 of 15** — see below |
| [llm-token-optimizer](packages/llm-token-optimizer) | a 2,543-token resent prefix | **21%** caching, 37% pruning, 64% routing |
| [llm-eval-harness](packages/llm-eval-harness) | judge scored against known-answer probes | per-criterion, not a single flattering average |
| [llm-tracing](packages/llm-tracing) | — | crash-safe spans, redaction before export |

Three of these numbers are worth reading twice.

**pr-reviewer reports 0% recall.** That is a finding about `qwen3:8b`, not about
the harness: at 8B the reviewer names none of the 15 real defects while flagging
20% of clean diffs, which makes it net-negative to ship. The harness is what
turns that from an argument into a measurement.

**agent-guardrail ships its weakest detector.** The learned classifier scores
higher (88% detection, held-out AUC 0.890) than the pattern layer (68%, AUC
0.841) — and is the wrong choice, because its 20–30% false positive rate gets it
switched off inside a day, and a detector that is switched off detects nothing.

**runbook-rag's reranker costs ×200 latency** for +21% recall@5, and drops
recall@10 by 2%. Whether that trade is worth taking depends on what reads the
result, which is why the table reports all six metrics instead of the one that
improved most.

## Layout

```
packages/     the platform layer
  llm-tracing           OpenTelemetry spans, redaction, token + cost attributes
  llm-eval-harness      deterministic assertions, versioned judge, cost as a gate
  llm-token-optimizer   prefix caching, context pruning, model routing
  agent-guardrail       provenance-gated capabilities against prompt injection

apps/         agents built on that layer
  ops-copilot           MCP incident diagnosis with a human approval gate
  pr-reviewer           LLM code review, recall and false-positive rate measured
  runbook-rag           hybrid retrieval and reranking over operational docs

benchmarks/   the labelled corpora every result above is scored against
```

## Why one repository

These began as seven. They were merged because the seams were already leaking:
`ops-copilot`'s incident fixtures had been hand-copied into `llm-eval-harness`
and `llm-token-optimizer`, and `ops-copilot` declared an optional dependency on
`runbook-rag` pointing at a GitHub URL. The evaluation harness is a dependency of
everything that claims to be measured, and keeping seven copies of it in sync was
the thing that would eventually make the numbers wrong.

## Running it

```
uv sync
uv run runbook-rag eval
uv run ops-copilot eval
uv run guardrail evaluate
```

Everything except `ops-copilot` runs locally with no API key. Python 3.11+.

## Licence

MIT.
