# LLM Gateway

Infrastructure for running LLM workloads: per-tenant quota, provider failover,
cost attribution and SLO alerting. Not an AI product — the thing a platform team
owns once an organisation calls model APIs from more than one service.

Deployed to Kubernetes by Terraform, and **verified running** rather than
described.

```
$ kubectl get pod -n llm-platform
gw-llm-gateway-5d7b9f69f9-r5wlm         1/1  Running
gw-llm-gateway-5d7b9f69f9-xxzkw         1/1  Running
gw-llm-gateway-redis-557765fcfb-fn75t   1/1  Running

$ kubectl exec -n llm-platform deploy/gw-llm-gateway -- wget -qO- localhost:8080/readyz
{"ready":true,"healthy_providers":["anthropic","openrouter"],
 "quota_store":{"shared":true,"healthy":true}, ...}
```

## Measured, not claimed

Three properties that are easy to assert and were instead tested against the
running cluster. Two of them failed the first time.

| property | measurement | result |
| --- | --- | --- |
| Quota holds across replicas | 40 concurrent requests, tenant burst 10 | **exactly 10 admitted, 30 refused** |
| Rolling update drops nothing | ~670 requests during a rollout | **0 failed** (was 7 of 535) |
| Credentials stay out of state | grep the state file | **0 plaintext, 0 base64** |

Each number came from a command, and each command is in this README or the
[runbook](../../deploy/RUNBOOK.md).

## Three bugs the measurements found

**Per-replica budgets.** Quota counters lived in process memory, so each pod
enforced the limit against its own arithmetic. A tenant with a $200 budget could
spend $200 *per pod*, and the overspend scaled linearly with replica count — on
a deployment that autoscales to ten. This was already deployed when it was
found. Counters now live in Redis, with check-and-increment done in Lua so the
read and the write are one round trip; splitting them would reintroduce the same
race in a smaller and less visible window.

**A drain that guarded nothing.** The service tracked in-flight requests and
waited for them during lifespan shutdown. It worked perfectly and protected
nothing, because uvicorn stops accepting connections the moment it receives
SIGTERM and lifespan shutdown runs *after* that — five seconds spent waiting
politely while requests were refused. Unit tests passed; a rolling restart under
load dropped 7 of 535. The fix is a `preStop` hook, which delays SIGTERM itself.
The in-app drain still covers the other half: requests already being served.

**A data source is not a safe way to read a secret.** Provider credentials were
moved from a `kubernetes_secret` resource to a `data` block, on the assumption
that reading a secret is safer than managing one. It is not: a data source
persists what it reads, and grepping the state file found the credential in
plain text *and* base64. The secret is now referenced by name only and never
read. `sensitive = true` hides a value from console output, not from state.

## The questions it answers

**What happens when a provider returns 500s at 3am?** Failover to the next
provider, with a circuit breaker so one dead provider does not become a latency
incident for every tenant paying the full timeout. The breaker probes on a timer
rather than waiting for a restart — a breaker that only closes on redeploy turns
a twelve-minute outage into a three-hour one.

**Which team spent the money?** `/v1/usage` reports settled spend, reserved
spend and budget utilisation per tenant. Every Prometheus series carries a
`tenant` label, because the first question in an incident is "is this everyone or
one team", and a metric that cannot answer it sends someone to the logs.

**How do you know before a tenant calls?** Multi-window burn-rate alerts against
a 99.5% availability SLO. 14.4× burn pages (budget gone in ~2 days); 6× opens a
ticket (~5 days). A fixed "error rate above 1%" threshold fires on a
thirty-second blip and misses a slow bleed that exhausts the month.

Six alerts, **three page**. A provider's breaker opening deliberately does not —
failover means it may have no user-visible effect, and a rotation woken by
everything learns to ignore the pager.

## Decisions worth defending

**Denials use the status code that is true.** Rate limit is `429` with
`retry-after`, because retrying works. An exhausted budget is `402`, because it
does not — `429` invites the client to retry into a wall and its backoff hides
the problem from whoever can raise the limit. A model outside a tenant's
allow-list is `403` naming the allowed models, rather than a silent downgrade
that produces a quality regression nobody can trace.

**Budget is reserved at admission and settled at completion.** An admitted
request has already committed money; counting only completed requests lets a
burst overshoot together. Settlement replaces the estimate with billed usage, so
estimate error does not accumulate — without it a tenant is eventually throttled
by the gateway's arithmetic rather than its own spending. It runs in a `finally`,
because a leaked reservation looks like a tenant slowly losing quota for no
reason.

**Only provider-side failures fail over.** A 500 or a timeout is theirs. A 400 is
malformed at every provider, so failing over turns one clear error into three
slow ones. A 401 is our credentials, not their health — failing over on it marks
a healthy provider broken because someone rotated a key. That asymmetry has an
operational consequence worth knowing: a bad credential presents as *every
request failing while every breaker reads closed*, which is in the runbook.

**Failure policy is asymmetric on purpose.** Budgets fail **closed** — a budget
that lifts itself when its store is unreachable is not a budget. Rate limits fail
**open** — refusing all traffic because Redis blinked is an outage we caused.

**Liveness and readiness ask different questions.** `/healthz` never touches an
upstream: restarting the gateway does not fix a provider outage, it adds a cold
start to one. `/readyz` covers provider health, quota store health, and whether
the pod is draining.

**Memory request equals its limit; CPU has no limit.** Burstable, not Guaranteed,
and intended. Equal memory stops the pod being killed for growth the scheduler
never accounted for. No CPU limit avoids throttling a latency-sensitive proxy at
a number someone guessed months ago — CPU is compressible and memory is not.

*(An earlier chart comment claimed Guaranteed QoS while setting unequal CPU
values. `kubectl` said Burstable; the comment was wrong and the configuration is
now what it describes.)*

## Guards that refuse bad configuration

Enforced at plan or render time, before anything reaches a cluster:

- Terraform refuses `image_tag = "latest"` — a mutable tag turns a node
  reschedule into an unannounced rollout.
- Terraform refuses `replicas < 2` — no rolling update, no useful disruption
  budget.
- Helm refuses to render more than one replica without a shared quota store,
  with a message explaining the consequence.
- `terminationGracePeriodSeconds` is computed from the drain and grace windows,
  so it cannot silently fall below them.
- `helm_release` is `atomic` — this rolled back a release whose image was missing
  the `redis` dependency, rather than replacing a working one with crashloops.

## Running it

```bash
colima start --cpu 4 --memory 6 --kubernetes
docker build -t llm-gateway:0.4.0 apps/llm-gateway

export OPENROUTER_API_KEY=sk-or-v1-...
./deploy/create-provider-secret.sh          # out of band, never via Terraform

cd deploy/terraform && tofu init && tofu apply
```

`pytest` in `apps/llm-gateway` runs 134 tests — none need a cluster or an API
key. Twelve of them assert deployment properties as data, because a comment
claiming a property the configuration does not have is the failure mode this
project kept hitting.

`promtool check rules` validates the alert PromQL; a rule file with a syntax
error loads silently and never fires.

## What it still cannot do

Stated here rather than discovered at 3am, and repeated in the runbook:

- **Quota counters are not persistent.** A Redis restart resets the month's
  committed spend. Fine for one Redis; wrong where billing accuracy matters.
- **Redis is a single point of failure for budgets.** One replica, and budgets
  fail closed, so losing it stops all traffic. Production wants Sentinel or a
  managed instance.
- **No request queueing.** A tenant over its rate limit is rejected, not queued.
- **No streaming.** Responses are buffered.
- **Terraform state is local.** Credentials are kept out of it entirely, but
  everything else it records still belongs in an encrypted remote backend —
  `backend.tf` has the configuration, commented.
- **No ingress, deliberately.** A service holding provider credentials should not
  be reachable from outside the cluster without a decision about who may reach
  it.
