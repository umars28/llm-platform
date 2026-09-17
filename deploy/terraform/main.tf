# The gateway, declared. `helm install` by hand works once; it does not tell you
# what is deployed six months later, and it cannot be reviewed before it runs.

resource "kubernetes_namespace" "platform" {
  metadata {
    name = var.namespace
    labels = {
      "app.kubernetes.io/part-of" = "llm-platform"
    }
  }
}

resource "kubernetes_secret" "providers" {
  metadata {
    name      = "llm-gateway-providers"
    namespace = kubernetes_namespace.platform.metadata[0].name
  }

  # Keys arrive from the environment at apply time. Putting them in a tfvars
  # file in the repository is the mistake this variable exists to avoid --
  # though note they are still stored in plain text in Terraform state, which is
  # why state belongs in an encrypted backend rather than on a laptop.
  data = var.provider_api_keys
  type = "Opaque"
}

resource "helm_release" "gateway" {
  name      = var.release_name
  namespace = kubernetes_namespace.platform.metadata[0].name
  chart     = "${path.module}/../helm/llm-gateway"

  # Wait for readiness rather than reporting success at the moment the objects
  # are accepted. An apply that goes green while the pods crashloop is worse
  # than one that fails.
  wait          = true
  atomic        = true
  timeout       = 300
  recreate_pods = false

  set {
    name  = "image.tag"
    value = var.image_tag
  }

  set {
    name  = "autoscaling.minReplicas"
    value = var.replicas
  }

  set {
    name  = "providerSecret.name"
    value = kubernetes_secret.providers.metadata[0].name
  }

  depends_on = [kubernetes_secret.providers]
}
