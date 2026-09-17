{{- define "llm-gateway.guard" -}}
{{- $replicas := ternary .Values.autoscaling.minReplicas .Values.replicaCount .Values.autoscaling.enabled -}}
{{- if and (gt (int $replicas) 1) (not .Values.redis.enabled) (not .Values.redis.url) -}}
{{- fail "More than one replica without a shared quota store: each pod would enforce the budget separately, so a tenant could spend its limit once per replica. Set redis.enabled=true, point redis.url at an external instance, or run a single replica." -}}
{{- end -}}
{{- end -}}
