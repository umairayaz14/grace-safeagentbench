"""RuleFollowingDMM — a deterministic, LLM-free DMM.

Implements the _DMMInner protocol: act(rules, observation) -> Action. The
decision IS the binding requirements: any "do not execute" requirement means
refuse; otherwise execute. This makes the symbolic layer the sole decision
authority — appropriate for interactive control, where an LLM adds latency
and noise but no information (the requirements already contain the verdict).

Also exposes the last-seen binding rules (LAST["rules"]) so the caller can
print the derived requirement as the refusal reason.
"""

from typing import Callable

from grace.symbolic_reasoning.horty import NLReason

LAST: dict = {"rules": []}


def _is_prohibition(conclusion: str) -> bool:
    return conclusion.strip().lower().startswith("do not")


class RuleFollowingDMM:
    def __init__(self, action_type=None) -> None:
        self._log_event: Callable[..., None] = lambda **kw: None

    def set_log_sink(self, sink: Callable[..., None]) -> None:
        self._log_event = sink

    def act(self, rules: list[NLReason], observation) -> str:
        LAST["rules"] = list(rules)
        prohibited = [r for r in rules if _is_prohibition(r.conclusion)]
        action = "refuse" if prohibited else "execute"
        self._log_event(
            source="rule_following_dmm",
            kind="decision",
            action=action,
            binding=[(r.premise, r.conclusion) for r in rules],
        )
        return action

    def rethink(self, violated: list[NLReason], rejected_action: str,
                observation) -> str:
        """Retrigger protocol: told which rules the rejected action violated,
        a rule-follower's revision is determined — take the conforming action.
        Any violated prohibition -> refuse; otherwise the violation was an
        execute-obligation -> execute."""
        action = ("refuse" if any(_is_prohibition(r.conclusion) for r in violated)
                  else "execute")
        self._log_event(
            source="rule_following_dmm",
            kind="rethink",
            rejected=rejected_action,
            action=action,
            violated=[(r.premise, r.conclusion) for r in violated],
        )
        return action
