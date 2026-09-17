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

# Provider credentials are referenced by name and never read.
#
# Terraform writes every managed attribute to state, and `sensitive` hides a
# value from console output rather than from the state file. Encrypting the
# backend narrows who can read state; it does not stop the secret being written
# down.
#
# A `data "kubernetes_secret"` block does not help either, which is worth
# stating because it looks like it should: a data source persists what it reads,
# so reading the secret to check it exists put the credential in state in both
# plain text and base64. Verified by grepping the state file, not assumed.
#
# So the name is a string and nothing here ever touches the value. The cost is
# that a missing secret is not caught at plan time -- the pods fail to start and
# `atomic = true` rolls the release back, which is a worse error message for a
# better property.

resource "helm_release" "gateway" {
  name      = var.release_name
  namespace = kubernetes_namespace.platform.metadata[0].name
  chart     = "${path.module}/../helm/llm-gateway"

  # Wait for readiness rather than reporting success at the moment the objects
  # are accepted. An apply that goes green while the pods crashloop is worse
  # than one that fails -- this caught a missing dependency in the image and
  # rolled back before it replaced a working release.
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
    value = var.provider_secret_name
  }
}
