output "namespace" {
  value = kubernetes_namespace.platform.metadata[0].name
}

output "service_url" {
  description = "In-cluster address. There is no ingress by design: a gateway holding provider credentials should not be reachable from outside the cluster without a deliberate decision about who may reach it."
  value       = "http://${var.release_name}-llm-gateway.${var.namespace}.svc.cluster.local:8080"
}

output "smoke_test" {
  description = "One command that proves the deployment serves traffic."
  value       = "kubectl run smoke --rm -i --restart=Never -n ${var.namespace} --image=curlimages/curl:8.11.1 -- curl -s http://${var.release_name}-llm-gateway:8080/readyz"
}
