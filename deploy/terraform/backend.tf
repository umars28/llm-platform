# State backend.
#
# Local by default, because this repository deploys to a laptop cluster and a
# remote backend nobody can reach is worse than an honest local file.
#
# For anything shared, uncomment the block below. Three properties matter, and
# only the third is about secrets:
#
#   encrypt        state at rest, since it records every managed attribute
#   use_lockfile   so two applies cannot interleave and corrupt state
#   versioning     on the bucket, so a bad apply can be rolled back
#
# Note what this does NOT solve: provider credentials are kept out of state
# entirely (see main.tf) rather than encrypted inside it. A remote backend
# protects the state you cannot avoid writing, not the secrets you can.

terraform {
  # backend "s3" {
  #   bucket       = "llm-platform-tfstate"
  #   key          = "gateway/terraform.tfstate"
  #   region       = "ap-southeast-2"
  #   encrypt      = true
  #   use_lockfile = true
  # }
}
