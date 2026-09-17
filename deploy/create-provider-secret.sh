#!/usr/bin/env bash
# Creates the provider credential secret, out of band from Terraform.
#
# Terraform writes every managed attribute to state, `sensitive` or not, so a
# credential managed by Terraform is a credential written to disk in plain text.
# Creating it here keeps it out of state entirely; Terraform only reads its name.
#
# Keys must match the api_key_env values in the gateway config.
set -euo pipefail

NAMESPACE="${NAMESPACE:-llm-platform}"
SECRET="${SECRET:-llm-gateway-providers}"

if [[ -z "${ANTHROPIC_API_KEY:-}${OPENROUTER_API_KEY:-}" ]]; then
  cat >&2 <<'MSG'
No credentials in the environment.

  export ANTHROPIC_API_KEY=sk-ant-...
  export OPENROUTER_API_KEY=sk-or-v1-...
  ./deploy/create-provider-secret.sh

At least one is required. The gateway starts without any, and every request
fails at the upstream with 401 -- which does not open a breaker, because a 401
is our fault rather than the provider's.
MSG
  exit 1
fi

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f - >/dev/null

# --dry-run | apply rather than `create`, so re-running rotates rather than failing.
kubectl create secret generic "$SECRET" \
  --namespace "$NAMESPACE" \
  ${ANTHROPIC_API_KEY:+--from-literal=ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"} \
  ${OPENROUTER_API_KEY:+--from-literal=OPENROUTER_API_KEY="$OPENROUTER_API_KEY"} \
  --dry-run=client -o yaml | kubectl apply -f -

echo
echo "Secret $SECRET is in place. Terraform reads its name and never its contents."
echo "After rotating, restart the pods so they pick up the new value:"
echo "  kubectl rollout restart deploy/gw-llm-gateway -n $NAMESPACE"
