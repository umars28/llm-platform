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

30 scenarios, 30 completed, no errors.

| metric | value |
| --- | --- |
| root cause identified | **96.7%** (29/30) |
| action matched | **96.7%** |
| free of misleading claims | **100%** |
| strictly correct (all three) | **93.3%** (28/30) |
| unwarranted change requests | 1 |
| mean investigation | 27.5 tool calls |

Model: a free-tier gateway model, so the run cost nothing. Reproduce with
`ops-copilot eval`.

| category | n | strictly correct |
| --- | --- | --- |
| resource exhaustion | 6 | 6/6 |
| dependency failure | 4 | 4/4 |
| cascading failure | 3 | 3/3 |
| config error | 3 | 3/3 |
| data layer | 3 | 3/3 |
| network | 3 | 3/3 |
| capacity | 3 | 2/3 |
| bad release | 2 | 2/2 |
| misleading signal | 1 | 1/1 |
| false alarm | 2 | 1/2 |

### The two it got wrong

Both are the hard end of the corpus, and they fail differently.

**SC-027** (a relabelled metric making a healthy service read as dead) got the
mechanism exactly right — the Prometheus relabel, the alert rule still selecting
the old label, the series going unqueryable — and chose the right action. What it
never said is that **the service was fine**. No "artefact", no "still serving",
no "no user impact". Explaining why a metric is zero without stating that nothing
is wrong leaves an on-call engineer uncertain, and an uncertain engineer changes
something.

**SC-030** (a memory limit raised without the matching request, starving the
node) reached a config-revert action rather than naming the requests-and-limits
mismatch as the cause.

Notably the *other* false-alarm scenario, SC-028, was answered correctly with
`no_action_required` — the agent declined to change anything about a spot reclaim
that had already resolved before its own alert fired.

### A paired comparison, and what it cost

Seven scenarios were also run on Claude Opus 5 before the budget ran out. On
those same seven:

| | Opus 5 | free model |
| --- | --- | --- |
| strictly correct | 6/7 | 7/7 |
| mean tool calls | **14.4** | 36.4 |
| cost per incident | $0.58 | $0.00 |

The free model reached the same answers using two and a half times as many
steps. That matters more than it looks: every turn resends the whole
conversation, so token spend grows with the square of the turn count — which is
why a full Opus 5 sweep costs about $17 rather than the $6 first estimated.

Two runs of the same suite is the more useful artefact. A harness that only ever
sees one model tells you about that model; one that sees two tells you what your
choice of model is buying.

## Scoring, and a correction

The scorer flags a diagnosis that asserts one of the scenario's misleading
claims. The first version of that list contained single words — `rollback`,
`capacity`, `postgres`, the name of a service — and it scored three correct
answers as wrong.

Each was penalised for a sentence like *"it avoids a restart, rollback, timeout
relaxation, or dependency-side change"*: the agent considered an alternative,
ruled it out with a reason, and was marked down for naming it. That is better
reasoning being punished, and it punished exactly the behaviour the system prompt
asks for.

The forbidden phrases are now specific claims rather than bare words, and the
thirty runs were rescored from their saved traces in one consistent pass. Three
scenarios moved from incorrect to correct; none moved the other way. **The
numbers above are the rescored ones.** Substring matching still cannot tell a
claim from a mention, which is why `clean` is reported separately from
`root_cause_hit` rather than folded into a single figure.

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
