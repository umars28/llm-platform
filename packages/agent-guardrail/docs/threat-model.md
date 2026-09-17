# Threat model

## What this defends

A tool-using agent whose context is assembled from sources of unequal
trustworthiness. The operator writes the system prompt. A human writes the task.
Everything after that — log lines, Kubernetes annotations, HTTP response bodies,
retrieved documentation, MCP tool descriptions — arrives from somewhere the
operator does not control, and arrives as text in the same context window as the
instructions.

The model has no reliable way to tell an instruction it should follow from a
sentence that merely looks like one. That is the whole problem: **a language
model reads its context, not its provenance.**

### The attack that matters here

Direct prompt injection — a user typing "ignore previous instructions" — is the
well-known case and the least interesting one. The user is already allowed to
ask the agent to do things, so the ceiling is whatever the user could have asked
for anyway.

**Indirect injection is different.** The attacker never talks to the agent. They
write into something the agent will later read:

| Channel | What an attacker controls | Realistic in |
|---|---|---|
| Log lines | Anything reflected into a log — a User-Agent, a username, an error string | Any system that logs request data |
| Kubernetes objects | Annotations, labels, ConfigMap values, event messages | Any cluster with more than one team |
| Retrieved documents | A page in the corpus, a wiki edit, a runbook PR | Any RAG system |
| MCP tool descriptions | The description field of a tool the agent loads | Any agent loading third-party MCP servers |
| Upstream HTTP responses | A response body the agent fetches and reads | Any agent that reads the web |

The attacker's goal is not usually to make the agent say something. It is to
make the agent **call a privileged tool it would not otherwise have called**, or
to exfiltrate what is already in its context into somewhere the attacker can
read.

This is the surface that [ops-copilot](https://github.com/umars28/ops-copilot)
and [runbook-rag](https://github.com/umars28/runbook-rag) create by existing.
An agent that reads logs and retrieves documents in order to diagnose incidents
is, by construction, an agent that reads attacker-influenced text.

## The design position

**Detection is a filter, not a boundary.** A classifier that catches 95% of
injections still lets one in twenty through, and an attacker only needs one.
Anything whose safety depends on the classifier being right is designed wrong.

So detection is used to make a *trust* decision, and the trust decision drives a
*capability* decision:

1. Every piece of content entering the context is labelled by provenance —
   operator, user, or untrusted.
2. Untrusted content is scanned. A detection raises the alarm and marks the
   context tainted; it does not, by itself, block anything.
3. Tools are gated on the taint state of the context, not on the score. Once
   untrusted content has been read, privileged tools require human approval
   regardless of how confident the classifier was.

Point 3 is the part that holds when detection fails. An injection that evades
every scanner still cannot reach a privileged tool, because the gate is on
provenance rather than on content.

## Threat actors in scope

- **An insider with write access to one system** — can edit a runbook, add a
  Kubernetes annotation, or influence a log line, but has no cluster admin.
- **An external party who can influence logged data** — sets a header, a
  username, a filename that ends up in a log the agent reads.
- **A compromised or hostile third-party MCP server** — serves tool
  descriptions and tool results the agent trusts by default.

## Explicitly out of scope

- **An attacker who controls the system prompt or the model weights.** Nothing
  downstream can help.
- **A malicious operator.** This defends the operator's intent; it does not
  constrain it.
- **Jailbreaks aimed at content policy.** Getting a model to write something it
  should not is a different problem with different defences. This is about
  actions and data, not words.
- **Classical application security of the tools themselves.** If a tool has an
  SQL injection, that is the tool's bug. The guardrail governs whether the tool
  is reachable, not whether it is correctly written.

## What "success" means

Three numbers, and the third is the one that is usually omitted:

- **Detection rate** per attack family, on held-out payloads.
- **False positive rate** on benign operational text. An alert on every
  stack trace is the same as no alerting at all, because it gets turned off.
- **Latency overhead** per call. A guardrail that doubles agent latency does not
  ship, and one that does not ship defends nothing.

A fourth, unmeasurable but decisive: **what an attacker achieves when detection
fails**. That is what the capability gate is for, and it is asserted by tests
rather than scored.
