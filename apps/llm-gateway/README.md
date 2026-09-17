# LLM Gateway

Infrastructure for running LLM workloads: per-tenant quota, provider failover,
cost attribution and SLO metrics. Not an AI product — the thing a platform team
owns once an organisation starts calling model APIs from more than one service.

Deployed to Kubernetes by Terraform, and verified running rather than described.

```
$ kubectl get all -n llm-platform
deployment.apps/gw-llm-gateway   2/2     Running
horizontalpodautoscaler/gw-llm-gateway   cpu: 2%/70%   min 2  max 10

$ kubectl run smoke --rm -i -n llm-platform --image=curlimages/curl -- \
    curl -s http://gw-llm-gateway:8080/readyz
{"ready":true,"healthy_providers":["anthropic","openrouter"], ...}
```

## The questions it answers

Not "which model is best". These:

**What happens when a provider returns 500s at 3am?** Traffic fails over to the
next provider, and a circuit breaker stops sending requests to the failing one —
so one dead provider does not turn into a latency incident for every tenant
paying the full timeout. The breaker **probes on a timer** rather than waiting
for a restart, because a breaker that only closes when someone redeploys turns a
twelve-minute outage into a three-hour one.

**Which team spent the money?** `/v1/usage` reports settled spend, reserved
spend and budget utilisation per tenant. Every Prometheus series is labelled by
tenant, because the first question in an incident is "is this everyone or one
team", and a metric that cannot answer it sends someone to the logs.

**What stops one team's retry loop costing ten thousand dollars overnight?** A
budget, enforced before the call rather than reported after it.

**Can I review a change before it reaches the cluster?** Terraform plan. The
config also refuses to apply an image tagged `latest` or a single replica.

## Three decisions worth defending

**Budget is reserved at admission, settled at completion.** A request that has
been admitted but not yet answered has already committed money. A gateway that
counts only completed requests lets a burst of concurrent calls blow through the
limit together. Settlement then replaces the estimate with the billed usage, so
estimate error does not accumulate — without it a tenant is eventually throttled
by the gateway's arithmetic rather than by its own spending. The settle runs in
a `finally`, because a leaked reservation looks like a tenant slowly losing quota
for no reason.

**Denials use the status code that is true.** Rate limit is `429` with
`retry-after`, because retrying works. An exhausted budget is **`402`**, because
retrying does not — returning `429` invites the client to retry into a wall, and
its backoff hides the real problem from whoever needs to raise the limit. A model
outside a tenant's allow-list is `403` and says which models are allowed, rather
than silently downgrading, which produces a quality regression nobody can trace.

**Only provider-side failures fail over.** A 500, a timeout or a refused
connection is the provider's problem. A 400 is malformed at every provider, so
failing over turns one clear error into three slow ones. A 401 is our
credentials, not their health — failing over on it would mark a healthy provider
broken because someone rotated a key. Get this wrong and one bad client deploy
opens every breaker in the fleet.

## Kubernetes specifics

**Liveness and readiness are different checks.** `/healthz` asks whether the
process is alive and never touches an upstream — restarting the gateway does not
fix a provider outage, it adds a cold start to it. `/readyz` asks whether traffic
can be served, and is `503` while every breaker is open. Wiring both to the same
check makes a rollout either hang or send traffic into a service that cannot
serve it.

**Memory request equals its limit; CPU has a request and no limit.** That is
Burstable, not Guaranteed, and it is the intended trade. Equal memory stops the
pod being killed for growth the scheduler never accounted for. No CPU limit
avoids throttling a latency-sensitive proxy at a number someone guessed months
ago — CPU is compressible and memory is not, so a CPU limit does not protect the
node the way a memory limit does.

*(An earlier version of this chart claimed Guaranteed QoS in a comment while
setting unequal CPU values. `kubectl` said Burstable and the comment was wrong;
the configuration is now what the comment describes.)*

**A config change rolls the pods.** The deployment carries a checksum of the
config file, so editing it restarts the fleet. Without that, a ConfigMap edit
applies to new pods only and the fleet silently runs two different policies.

Also: non-root with a read-only root filesystem and all capabilities dropped, a
PodDisruptionBudget so a node drain cannot take every replica, and topology
spread so two replicas do not sit on one node.

## Running it

```bash
# local cluster
colima start --cpu 4 --memory 6 --kubernetes

# build; colima's k3s uses cri-dockerd, so a locally built image needs no import
docker build -t llm-gateway:0.1.0 apps/llm-gateway

cd deploy/terraform
tofu init && tofu apply
```

Tests: `pytest` in `apps/llm-gateway` — 69, none needing a cluster or an API key.

## What it does not do

No streaming, no request queueing, no persistence: budgets reset when the
process restarts, which is fine for a single replica and wrong for a fleet — a
real deployment puts the counters in Redis. There is no ingress by design; a
service holding provider credentials should not be reachable from outside the
cluster without a deliberate decision about who may reach it.

Terraform state is local. It holds the provider secret in plain text, so a real
deployment needs an encrypted remote backend before anyone runs it for real.
