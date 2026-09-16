"""The capability gate: what a tool call is allowed to do, given what has been read.

This is the part that holds when detection fails, and it is the reason detection
is allowed to be imperfect.

A classifier that catches 95% of injections lets one in twenty through, and an
attacker needs one. So nothing here is gated on a detection score. Tools are
gated on **provenance**: once untrusted content has entered the context, the
privileged tools require a human, whatever the scanner concluded. An injection
that evades every layer still cannot reach `apply_remediation`, because the gate
never asked the scanner's opinion.

Detections still matter -- they explain, they alert, and they escalate a
borderline call from allowed to needs-approval. They just are not load-bearing.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterable

from .detect import Finding
from .trust import Context, Trust


class Capability(enum.Enum):
    """What a tool can do, which is what the gate actually reasons about."""

    READ = "read"  # observe state; no side effects
    WRITE = "write"  # change state in this system
    PRIVILEGED = "privileged"  # change production, or touch credentials
    EGRESS = "egress"  # send data somewhere the operator does not control


class Decision(enum.Enum):
    ALLOW = "allow"
    NEEDS_APPROVAL = "needs_approval"
    DENY = "deny"


# A default catalogue covering the ops-copilot tool surface. Callers register
# their own; an unregistered tool is treated as PRIVILEGED, so adding a tool
# and forgetting to classify it fails closed rather than open.
DEFAULT_CAPABILITIES: dict[str, Capability] = {
    "list_services": Capability.READ,
    "get_recent_deploys": Capability.READ,
    "query_logs": Capability.READ,
    "list_metrics": Capability.READ,
    "get_metrics": Capability.READ,
    "describe_k8s_resource": Capability.READ,
    "search_runbook": Capability.READ,
    "list_remediation_actions": Capability.READ,
    "propose_remediation": Capability.WRITE,
    "request_remediation": Capability.PRIVILEGED,
    "apply_remediation": Capability.PRIVILEGED,
    "http_request": Capability.EGRESS,
    "send_webhook": Capability.EGRESS,
}


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str
    capability: Capability
    tainted: bool
    findings: tuple[Finding, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW

    @property
    def blocked(self) -> bool:
        return self.decision is Decision.DENY


@dataclass
class Policy:
    """Gates tool calls on the provenance of what the context has absorbed."""

    capabilities: dict[str, Capability] = field(
        default_factory=lambda: dict(DEFAULT_CAPABILITIES)
    )
    # Reading untrusted content does not restrict reading more of it. The point
    # is to stop an injection reaching a lever, not to stop investigation.
    tainted_allows: frozenset[Capability] = frozenset({Capability.READ})
    # Egress is denied outright once tainted: the whole exfiltration family
    # depends on it, and unlike a remediation there is no approval that makes
    # sending the context to an attacker acceptable.
    tainted_denies: frozenset[Capability] = frozenset({Capability.EGRESS})

    def capability_of(self, tool: str) -> Capability:
        """Unknown tools are privileged, so forgetting to classify fails closed."""
        return self.capabilities.get(tool, Capability.PRIVILEGED)

    def register(self, tool: str, capability: Capability) -> None:
        self.capabilities[tool] = capability

    def evaluate(
        self,
        tool: str,
        context: Context,
        findings: Iterable[Finding] = (),
    ) -> Verdict:
        capability = self.capability_of(tool)
        findings = tuple(f for f in findings if f.triggered)
        tainted = context.tainted

        if not tainted:
            # Nothing untrusted has been read. Privileged tools still need a
            # human, because that is a separate policy about production changes
            # and not a reaction to any attack.
            decision = (
                Decision.NEEDS_APPROVAL
                if capability is Capability.PRIVILEGED
                else Decision.ALLOW
            )
            reason = (
                "context is clean; privileged tools still require approval"
                if decision is Decision.NEEDS_APPROVAL
                else "context is clean"
            )
            return Verdict(decision, reason, capability, tainted, findings)

        sources = ", ".join(context.untrusted_sources[:3]) or "untrusted content"

        if capability in self.tainted_denies:
            return Verdict(
                Decision.DENY,
                f"context has absorbed untrusted content ({sources}); "
                f"{capability.value} is denied because exfiltration cannot be "
                "made safe by approving it",
                capability, tainted, findings,
            )

        if capability in self.tainted_allows:
            return Verdict(
                Decision.ALLOW,
                f"{capability.value} is unaffected by taint; investigation continues",
                capability, tainted, findings,
            )

        detail = (
            f" A scanner also flagged: {findings[0].reasons[0]}."
            if findings and findings[0].reasons
            else ""
        )
        return Verdict(
            Decision.NEEDS_APPROVAL,
            f"context has absorbed untrusted content ({sources}), so "
            f"{capability.value} requires a human regardless of scanner "
            f"confidence.{detail}",
            capability, tainted, findings,
        )


def explain(verdict: Verdict) -> str:
    """One line for a log or an audit trail."""
    return f"{verdict.decision.value} [{verdict.capability.value}] {verdict.reason}"
