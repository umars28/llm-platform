variable "kubeconfig" {
  description = "Path to the kubeconfig to act against."
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "Context within the kubeconfig. Named explicitly so an apply cannot land in the wrong cluster because someone switched context in another terminal."
  type        = string
  default     = "colima"
}

variable "namespace" {
  description = "Namespace to deploy into."
  type        = string
  default     = "llm-platform"
}

variable "release_name" {
  type    = string
  default = "gw"
}

variable "provider_secret_name" {
  description = "Secret holding upstream credentials. Created out of band so its contents never enter Terraform state; see the README."
  type        = string
  default     = "llm-gateway-providers"
}

variable "image_tag" {
  description = "Image tag to deploy. Never 'latest': a mutable tag makes a node reschedule into an unannounced rollout."
  type        = string
  default     = "0.4.0"

  validation {
    condition     = var.image_tag != "latest"
    error_message = "Refusing 'latest'. Pin a tag or a digest so a reschedule cannot roll out an unreleased build."
  }
}

variable "replicas" {
  type    = number
  default = 2

  validation {
    condition     = var.replicas >= 2
    error_message = "A single replica has no rolling update and no disruption budget worth having."
  }
}
