"""SafeAgent Guard components — rule-based enforcement for SafeAgentBench.

Mirrors the moral_machine guard pair. Both are deliberately symbolic (no LLM):
the Guard exists precisely because LLM judgment is unreliable at the
enforcement step, so the enforcement must not itself be an LLM.

Coverage: DMM disobedience of the binding requirements, in both directions
(executing against a refuse-obligation, refusing against an execute-obligation).
Not covered: anything upstream — theory gaps, grounding misses, ontology
omissions. Those produce wrong requirements, which the Guard will faithfully
enforce.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from grace.symbolic_reasoning.horty import NLReason

if TYPE_CHECKING:
    from safeagent_environment import Action, Observation


def _is_execute_obligation(conclusion: str) -> bool:
    """True = execute-obligation, False = refuse-obligation.

    Same polarity convention as SafeAgentConflictChecker: all conclusions are
    'execute plan p' or 'do not execute plan p' (with substitutions), so
    polarity is the whole classification — no group/target dimension.
    """
    return not conclusion.strip().lower().startswith("do not")


class SafeAgentConformanceAuditor:
    """Rule-based conformance check for execute/refuse decisions.

    For each binding rule, classify its conclusion's polarity and compare with
    the chosen action. A rule is violated if:
      - it is a refuse-obligation and the action is 'execute', or
      - it is an execute-obligation and the action is 'refuse'.

    Satisfies ConformanceAuditor[Observation, list[NLReason], Action].
    Declares `set_log_sink`, so the Guard wires its audit log in at construction.
    """

    def __init__(self) -> None:
        self._log_event: Callable[..., None] = lambda **kw: None

    def set_log_sink(self, sink: Callable[..., None]) -> None:
        self._log_event = sink

    def conformance_audit(
        self, action: "Action", rules: list[NLReason], obs: "Observation"
    ) -> list[NLReason]:
        violated: list[NLReason] = []
        for rule in rules:
            wants_execute = _is_execute_obligation(rule.conclusion)
            self._log_event(
                type="classification",
                source="conformance_auditor",
                conclusion=rule.conclusion,
                polarity="execute" if wants_execute else "do_not_execute",
                action=action,
            )
            if wants_execute and action == "refuse":
                violated.append(rule)
            elif not wants_execute and action == "execute":
                violated.append(rule)
        return violated


class SafeAgentDefaultSelector:
    """Rule-based fallback: pick the action that satisfies the violated rules.

    Any violated refuse-obligation forces 'refuse' (safety wins if the DMM
    ignored a prohibition). Otherwise a violated execute-obligation yields
    'execute'. With no classifiable violation, fail SAFE: 'refuse'.

    Satisfies DefaultActionSelector[Observation, list[NLReason], Action].
    """

    def select_default(
        self, action: "Action", violated_rules: list[NLReason], obs: "Observation"
    ) -> "Action":
        for rule in violated_rules:
            if not _is_execute_obligation(rule.conclusion):
                return "refuse"
        for rule in violated_rules:
            if _is_execute_obligation(rule.conclusion):
                return "execute"
        return "refuse"
