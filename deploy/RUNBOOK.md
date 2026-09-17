# LLM Gateway runbook

Written for someone who has just been paged, does not have the code open, and
wants the shortest path to "is this still getting worse".

Every alert links here by anchor. If an alert has no section here, that alert
should not exist.

## Orient

```bash
NS=llm-platform

kubectl get pod -n $NS
kubectl exec -n $NS deploy/gw-llm-gateway -- \
  python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8080/readyz').read().decode())" | jq
```

`/readyz` answers most of the first questions at once: which providers are
healthy, whether each breaker is open, whether the quota store is reachable, and
whether the pod is draining.

```bash
# What is failing, with request ids
kubectl logs -n $NS -l app.kubernetes.io/name=llm-gateway --tail=200 \
  | jq -c 'select(.level == "error")'

# One tenant's traffic
kubectl logs -n $NS -l app.kubernetes.io/name=llm-gateway --tail=500 \
  | jq -c 'select(.tenant == "platform")'

# Spend and budget, per tenant
kubectl exec -n $NS deploy/gw-llm-gateway -- \
  python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8080/v1/usage').read().decode())" | jq
```

---

## error-budget-burning

**Symptom:** a rising share of requests are not 2xx.

**First:** find out whether it is one tenant or everyone. Every metric and log
line is labelled by tenant precisely for this.

```bash
kubectl logs -n $NS -l app.kubernetes.io/name=llm-gateway --tail=500 \
  | jq -r 'select(.status >= 400) | "\(.tenant) \(.status)"' | sort | uniq -c | sort -rn
```

| what you see | what it means | what to do |
|---|---|---|
| One tenant, all `429` | They are over their rate limit | Nothing is broken. Raise `requests_per_s` in the ConfigMap if the increase is legitimate |
| One tenant, all `402` | They exhausted their budget | Working as designed. Raise `monthly_budget_usd` or let them wait for the window |
| One tenant, all `403` | They are requesting a model outside their allow-list | Their deploy changed, not ours |
| Everyone, `503` | Upstream — see [all-providers-failing](#all-providers-failing) |
| Everyone, `402` | The quota store is unreachable and budgets are failing closed — see [quota-store-down](#quota-store-down) |
| Everyone, `500` | Ours. Get a request id from the logs and read the traceback |

**A 402 storm across every tenant is not a budget problem.** Budgets fail closed
by design, so an unreachable Redis presents as every tenant being out of money
at the same instant. If several tenants hit 402 simultaneously, check the store
before you check anyone's spending.

---

## latency-high

**Symptom:** p95 above the 30s objective.

Almost always an upstream provider slowing rather than the gateway. Confirm
before touching anything:

```bash
kubectl exec -n $NS deploy/gw-llm-gateway -- \
  python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8080/readyz').read().decode())" | jq '.providers'
```

`mean_latency_ms` is per provider. If one is slow and the other is not, lower its
priority in the ConfigMap so traffic prefers the healthy one:

```bash
kubectl edit configmap -n $NS gw-llm-gateway-config   # raise the slow provider's priority number
kubectl rollout restart deploy/gw-llm-gateway -n $NS
```

If both are slow, it is upstream and there is nothing to fix here. Say so on the
incident channel rather than restarting the gateway — a restart adds a cold start
to an upstream problem and changes nothing else.

**Do not scale up to fix latency caused upstream.** More replicas send more
concurrent requests to a provider that is already struggling, which is the retry
amplification pattern with extra steps.

---

## all-providers-failing

**Symptom:** `gateway_upstream_failures_total` rising. Failover has run out of
providers.

```bash
kubectl logs -n $NS -l app.kubernetes.io/name=llm-gateway --tail=100 \
  | jq -c 'select(.msg == "upstream unavailable") | .attempts'
```

That prints what each provider actually said. Read it before concluding they are
down:

**`HTTP 401` on every attempt is a credential problem, not an outage.** A 401 is
classified as our fault rather than the provider's, so it does not open a
breaker — which means every request fails while every breaker reads `closed`.
That combination is the signature.

```bash
kubectl get secret -n $NS llm-gateway-providers -o jsonpath='{.data}' | jq 'keys'
# rotate, then
kubectl rollout restart deploy/gw-llm-gateway -n $NS
```

**`HTTP 5xx` or timeouts on every attempt** means the providers really are
failing. The breakers will be open; they probe every 30s on their own, so the
gateway recovers without intervention once upstream does. There is nothing to
restart.

**One provider open, one closed, still failing** means the healthy one does not
serve that model. Check the `models` map in the ConfigMap — a model that only one
provider serves has no failover, whatever the chart says.

---

## quota-store-down

**Symptom:** pods not ready; `/readyz` shows `quota_store.healthy: false`.

Budgets fail closed, so this is a total outage from the first request onward.
The rate limiter fails open, so that half keeps working.

```bash
kubectl get pod -n $NS -l app.kubernetes.io/name=llm-gateway-redis
kubectl logs -n $NS -l app.kubernetes.io/name=llm-gateway-redis --tail=50
kubectl exec -n $NS deploy/gw-llm-gateway-redis -- redis-cli ping
```

Redis is deployed with `Recreate` and no persistence on purpose. Restarting it
loses the counters, which resets everyone's committed spend to zero — tenants get
a fresh budget window early. That is a billing inaccuracy, not an outage, and it
is the right trade when the alternative is refusing all traffic.

```bash
kubectl rollout restart deploy/gw-llm-gateway-redis -n $NS
```

**Do not "fix" this by disabling the quota store.** Running more than one replica
without shared state lets every tenant spend its budget once per pod; the chart
refuses to render that configuration for this reason.

---

## tenant-near-budget

Not an outage. One team is at 90% of its monthly budget and will start getting
402s before the window resets.

```bash
kubectl exec -n $NS deploy/gw-llm-gateway -- \
  python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8080/v1/usage').read().decode())" | jq '.tenants'
```

Tell them before it fails, not after. If the increase is expected, raise
`monthly_budget_usd` in the ConfigMap; the deployment rolls automatically because
the pod template carries a checksum of the config.

---

## Things that look like incidents and are not

**Pods `Terminating` for ~45 seconds during a rollout.** That is
`terminationGracePeriodSeconds`, covering the preStop delay plus the drain
window. Without it a rolling restart dropped 7 of 535 requests; with it, 0 of
670. Shortening it to make rollouts feel faster re-breaks that.

**A new pod in the namespace cannot reach Redis.** That is the network policy,
not a fault. Only the gateway may open 6379; anything else gets connection
refused. If the *gateway* cannot reach it, that is
[quota-store-down](#quota-store-down), not this.

**A breaker showing `half_open`.** It is probing a provider that failed earlier.
This is recovery in progress, not a fault.

**`reserved_usd` above `settled_usd`.** Requests are in flight. Spend is reserved
at admission and settled at completion, so the gap is normal under load and
closes on its own.

---

## What this deployment still cannot do

State the limits before someone discovers them at 3am:

- **Quota counters are not persistent.** A Redis restart resets the month's
  committed spend. Acceptable for a single Redis; wrong for anything where
  billing accuracy matters.
- **Redis is a single point of failure for budgets.** It has one replica, and
  budgets fail closed, so losing it stops all traffic. A production deployment
  wants Sentinel or a managed instance.
- **No request queueing.** A tenant over its rate limit is rejected, not queued.
- **Terraform state is local and holds the provider secret in plain text.** It
  belongs in an encrypted remote backend before anyone relies on it.
