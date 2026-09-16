# LLM Tracing

OpenTelemetry tracing for LLM applications. Spans that survive a crash, redact
secrets before they leave the process, and carry the attributes an LLM trace is
actually opened for.

```
+ investigate SC-001                            91ms
  + chat claude-opus-5    15ms  $0.0426  3,443in/380out
  + tool list_services     8ms
  + tool get_recent_deploys 8ms
  + chat claude-opus-5    15ms  $0.0325  4,343in/380out
  ...
  x tool request_remediation  0ms
      PermissionError: change request queued, awaiting human approval

11 spans  $0.1121  1 error
```

Try it: `llm-tracing demo && llm-tracing show traces/demo.jsonl`

## Three decisions that differ from a general-purpose tracer

**A span is written when it ends, even if it ends badly.** The traces worth
having are of runs that crashed, and a buffered exporter that flushes on clean
shutdown loses exactly those. Spans append to disk as they close. A test raises
inside a span and asserts the span is on disk with `status: error`.

**Prompts and completions are redacted before export, and off by default.** A
trace of an LLM call contains whatever the user typed and whatever the model
read — on an ops agent that is production logs and Kubernetes state. Shipping it
to a tracing backend moves a security boundary quietly, so content is opt-in via
`LLM_TRACING_CONTENT=1` and API keys, JWTs, labelled secrets, emails and IPs are
stripped on the way out regardless.

**Cost is computed when the span closes.** A trace whose cost depends on a query
someone has to write correctly is a trace whose cost nobody knows. Cache reads
are priced separately from fresh input, at a tenth, so a caching change is
visible in the trace rather than inferred from the bill.

## Attributes worth having

Generic tracing answers "what was slow". An LLM trace is opened for different
questions, and they are all attributes:

| attribute | question it answers |
| --- | --- |
| `gen_ai.usage.input_tokens` / `output_tokens` | how much context did this turn carry |
| `gen_ai.usage.cache_read_input_tokens` | did the cache hit, or is it silently missing |
| `gen_ai.usage.cost_usd` | what did this run cost |
| `gen_ai.request.model` | which model answered |
| `gen_ai.tool.name` | how many tools before it committed |

The names follow OpenTelemetry's `gen_ai.*` semantic conventions, so an OTLP
backend groups them without per-field configuration.

## Export, and proving it works

Spans convert to OTLP-JSON and POST to any OTLP-HTTP endpoint — Langfuse,
Jaeger, Grafana Tempo.

```bash
llm-tracing export traces/demo.jsonl --to http://localhost:3000/api/public/otel \
    --header "Authorization: Basic $LANGFUSE_AUTH"
```

"It exports correctly" is a claim, and a claim about a wire format needs
something on the other end. `LocalReceiver` is a few dozen lines of HTTP server
that accepts OTLP-JSON and records what it got, so the export path is covered by
a test rather than by a screenshot of a dashboard. Tests assert that parent
links, event names, integer-typed token counts and the error status code all
survive the round trip.

Conversion is written out by hand rather than delegated to the OpenTelemetry SDK.
The SDK is the right dependency for a production service and the wrong one for
showing what the protocol is — a reader who wants to check whether attributes map
correctly can read twenty lines instead of installing a tree.

**Not verified against a running Langfuse instance.** That needs Docker, which is
not installed here, so the claim made is the one that is tested: spans serialise
to OTLP-JSON and arrive intact at an OTLP-HTTP endpoint. Whether Langfuse renders
them the way you want is between you and Langfuse.

## Exporter failures raise

An exporter that swallows its own errors leaves you believing you have
observability that you do not, which is worse than having none and knowing it.
An unreachable endpoint raises; a non-200 raises with the body.

## Where it fits

One of a set that share a spine.
[ops-copilot](https://github.com/umars28/ops-copilot) is the agent this traces —
the demo reproduces one of its investigations, approval gate refusal included.
[llm-token-optimizer](https://github.com/umars28/llm-token-optimizer) is where
the cache-read attribute becomes a saving, and
[llm-eval-harness](https://github.com/umars28/llm-eval-harness) is what says
whether any of it stayed correct.

## Licence

MIT
