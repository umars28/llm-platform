# Ops Copilot

An incident diagnosis agent built on the Model Context Protocol. Claude Opus 5
investigates a production alert through ten MCP tools over logs, metrics,
Kubernetes state, deploy history and runbooks, commits to a causal diagnosis,
and queues any remediation behind a human approval gate. It is evaluated against
thirty reproducible incident scenarios.

The interesting part is not that an LLM can read logs. It is that the whole
thing is measurable: the same thirty incidents replay identically on every run,
so a prompt change produces a number that moved rather than an anecdote.

```
alert ──> agent (Claude Opus 5, tool_runner)
            │  stdio
            ▼
     MCP server ── 10 tools ──> scenario fixture (frozen incident)
            │
            ├─ propose_remediation ──> scored against ground truth
            └─ request_remediation ──> pending change request + audit log
                                        (never applies anything)
```

## Why the approval gate is the design, not a feature

An agent that can restart pods is a demo. An agent that can restart pods
*unattended* is an incident waiting to happen. `request_remediation` cannot
apply a change under any circumstance — it appends a `pending` record to an
append-only JSONL log and returns the request id. Approval is a separate human
action that appends an `approved` event rather than rewriting the pending one,
so the trail shows who approved what and when, and re-approving is rejected.

There is deliberately no tool that applies a change, and the tool description
says so, because a model that believes an escape hatch exists will spend turns
looking for it.

## The corpus

Thirty scenarios across ten failure families. Each is a frozen snapshot — alert,
logs, metrics, Kubernetes objects, deploy history, runbooks — with timestamps
authored as offsets from the alert, so a scenario replays identically whenever
it is run.

| family | n | examples |
| --- | --- | --- |
| resource exhaustion | 6 | connection pool, unbounded cache, fd leak, node disk, event-loop blocking |
| dependency failure | 4 | expired TLS cert, breaking API change, Kafka rebalance loop, provider quota |
| capacity | 3 | autoscaler ceiling, autoscaler on the wrong metric, requests/limits overcommit |
| config error | 3 | cross-environment SMTP host, dropped Helm value, health check path |
| data layer | 3 | missing index, migration lock, replica lag |
| network | 3 | NetworkPolicy drop, DNS search domains, node packet loss |
| cascading failure | 3 | retry amplification, cache stampede, stuck circuit breaker |
| bad release | 2 | N+1 behind a flag, mutable image tag |
| false alarm | 2 | relabelled metric, self-healed spot reclaim |
| misleading signal | 1 | clock skew on one node |

Three properties make the corpus worth scoring against rather than just
demonstrating on:

**Every scenario has at least one red herring**, recorded explicitly in its
ground truth. The failure mode worth measuring is the plausible-but-wrong
diagnosis, not the absence of one. In SC-001 Postgres connection counts are
rising but nowhere near their limit; in SC-018 a service at 100% CPU looks
exactly like a capacity shortfall and is actually a retry storm.

**The alerting service is frequently the victim.** SC-004 alerts on a search
service evicted by a disk another service filled. SC-030 alerts on a ledger
service whose neighbour was given a 24Gi memory limit against a 2Gi request.
An investigation that stays inside the alerting service gets these wrong.

**Two scenarios have `no_action_required` as the correct answer.** A corpus made
only of real incidents cannot catch an agent that always finds something to
change, and proposing a production change against a healthy system is its own
failure mode. `over_reach` counts it separately.

## Scoring

Deterministic and keyword-based, not LLM-judged. A judge that varies between
runs makes it impossible to tell whether a score moved because the agent changed
or because the judge did — and the judge would then need its own evaluation. An
LLM judge is the right tool one layer up, against a harness whose numbers are
already stable.

Three signals stay separate rather than collapsing into one accuracy figure:

- **root cause** — every required concept group appears in the committed
  diagnosis. Groups are lists of alternatives, so wording does not decide the
  score.
- **action** — the proposed action id is one the scenario accepts.
- **clean** — the diagnosis asserts none of the scenario's misleading claims.

Substring matching genuinely cannot tell a claim from a mention. That limit is
reported rather than hidden: a run that reaches the right conclusion while
naming a red herring shows up as `root_cause_hit` true and `clean` false, which
is a case worth reading, not a number worth burying.

## Results

Run `ops-copilot eval` to populate this section. Each run writes
`runs/<timestamp>/results.json` and a `results.md` with the tables below filled
in, per category and per scenario.

| metric | value |
| --- | --- |
| root cause identified | _not yet run_ |
| action matched | _not yet run_ |
| strictly correct (cause + action + clean) | _not yet run_ |
| unwarranted change requests | _not yet run_ |
| mean investigation tool calls | _not yet run_ |
| mean cost per incident | _not yet run_ |

No numbers are quoted here until a full sweep has been run and committed.

## Running it

Requires Python 3.11+ and credentials for the Anthropic API.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

### Credentials

The SDK resolves credentials in this order, first match wins, and this project
does nothing special on top of it:

1. `ANTHROPIC_API_KEY`
2. `ANTHROPIC_AUTH_TOKEN`
3. the OAuth profile written by `ant auth login`
4. workload identity federation

So there are three ways to run it:

```bash
# A static key
export ANTHROPIC_API_KEY=sk-ant-...

# Or an OAuth profile, with no key to manage. The SDK finds it on its own.
brew install anthropics/tap/ant
ant auth login
ant auth status          # shows which source won

# Or a gateway that speaks the Anthropic Messages API, e.g. OpenRouter
unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL=https://openrouter.ai/api
export ANTHROPIC_AUTH_TOKEN=sk-or-v1-...
export OPS_COPILOT_MODEL=anthropic/claude-opus-5
```

`unset` the API key rather than blanking it — an empty `ANTHROPIC_API_KEY=""`
still occupies its precedence slot, and the SDK then sends both credentials and
the request is rejected.

`OPS_COPILOT_MAX_TOKENS` matters on gateways: they reserve budget up front as
`max_tokens x output price`, so a low spend limit rejects a large request even
when the real response would be cheap.

`OPS_COPILOT_MODEL` exists because gateways namespace their model ids. Adaptive
thinking and `effort` are Anthropic-specific, so they are sent only when the
model id resolves to a Claude model; `OPS_COPILOT_NATIVE_PARAMS=0` or `1` forces
the decision if the inference is wrong for your gateway.

A set `ANTHROPIC_API_KEY` silently shadows an OAuth profile, including an empty
one — `unset` it rather than blanking it if you mean to use the profile.

The harness refuses to start when it can resolve no credentials at all, because
a sweep that fails at authentication reports 0% and reads as a failing agent.
A custom `ANTHROPIC_BASE_URL` counts as configured, and
`OPS_COPILOT_SKIP_AUTH_CHECK=1` overrides the preflight entirely.

```bash

ops-copilot list                  # the corpus
ops-copilot run SC-001            # one incident, streaming the tool trace
ops-copilot eval                  # all thirty, scored, written to runs/
ops-copilot eval SC-001 SC-018 --effort medium --label cheap-sweep
ops-copilot approvals             # change requests awaiting a human
ops-copilot approve CR-1a2b3c4d
```

The test suite runs without an API key and covers the corpus structure, the
world query layer and the scoring rules:

```bash
pytest
```

## The MCP server

`ops_copilot.server` speaks stdio and is usable by any MCP client, not just this
agent. `OPS_COPILOT_SCENARIO` selects which incident it serves.

| tool | purpose |
| --- | --- |
| `list_services` | inventory with tiers, owners, dependencies |
| `get_recent_deploys` | releases in a window, newest last |
| `query_logs` | regex and level search; repeated lines carry a `count` |
| `list_metrics` / `get_metrics` | series with min/max/latest precomputed |
| `describe_k8s_resource` | spec, status and events for any object |
| `search_runbook` | operating limits telemetry cannot tell you |
| `list_remediation_actions` | the 25-action catalogue with risk levels |
| `propose_remediation` | the committed diagnosis — the scored output |
| `request_remediation` | queues a change for approval; applies nothing |

Read tools answer a wrong service or metric name with the valid options rather
than an exception, so the model corrects itself in one turn instead of spending
several guessing.

## Design notes

**Timestamps are relative.** Scenarios store offsets in minutes from the alert
and resolve them at query time. The fixtures do not rot, and `since_minutes`
means the same thing to the model in every scenario.

**Each run is isolated.** Every scenario gets its own MCP subprocess and its own
audit directory, so a change request queued by one run cannot leak into
another's trail, and the sweep can run concurrently.

**The action catalogue is shared and wide.** Twenty-five actions in
`scenarios/_common.yaml`, not four per scenario. Choosing correctly out of
twenty-five is a real decision.

**Traces are kept, not just verdicts.** Every run records its ordered tool
calls, turn count, token usage and wall time. How much work the agent did to
reach an answer is as much a quality signal as whether the answer was right.

## Runbook retrieval

`search_runbook` returns two separately labelled sources.

`results` are the environment's own runbooks -- its connection budgets, its
pre-approved actions, its conventions. They are authoritative here and are
always searched.

`reference` is real published Kubernetes documentation, retrieved by
[runbook-rag](https://github.com/umars28/runbook-rag) using the hybrid
configuration that project measured. It is background knowledge and says nothing
about this cluster's limits. The two are never merged into one ranked list,
because doing so invites the specific wrong inference of reading a capacity
figure out of upstream documentation as though it were this environment's.

Reference results below a cosine similarity of 0.70 are dropped rather than
shown. The corpus genuinely does not cover application concerns like connection
pools or retry policy, and returning its best guess anyway spends the agent's
context and invites a wrong turn. The floor comes from the measured separation:
relevant passages score 0.78-0.84, and the best available match for an
out-of-domain question scores 0.63-0.64.

Both upgrades degrade rather than fail. Semantic search over the environment
runbooks needs only the embedding model and no database; reference lookup needs
the corpus indexed. Without either, the tool falls back to keyword overlap and
says which backend is missing, so a retrieval problem never ends an
investigation.

```bash
pip install -e ".[retrieval]"   # optional; see the runbook-rag repository
```

## Where this goes next

The traces and the deterministic scores are the substrate an LLM-judged eval
suite and a cost-optimisation pass need. Neither is in scope here.

## Licence

MIT
