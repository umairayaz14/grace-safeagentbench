"""SafeAgentConflictChecker — implements grace.ConflictChecker for SafeAgentBench tasks."""

from grace.symbolic_reasoning.horty import ConflictChecker, NLReason as Reason


def _classify(conclusion: str) -> bool:
    """True = pro-execution, False = anti-execution.

    Simpler than the moral-machine classify: all reasons in this domain concern
    the same single plan p, so there is no group dimension — only polarity.
    """
    return not conclusion.strip().lower().startswith("do not")


class SafeAgentConflictChecker(ConflictChecker):
    """ConflictChecker for embodied task execution.

    Every reason's conclusion is either "execute plan p" or "do not execute
    plan p" (possibly with substituted names). Since all reasons target the one
    plan under consideration, the only conflict type is polarity: a pro-execution
    reason clashes with any anti-execution reason. Two reasons of the same
    polarity never conflict (multiple hazards all pushing "do not execute"
    reinforce, not contradict, each other).
    """

    def conflicted(self, a: Reason, b: Reason) -> bool:
        return _classify(a.conclusion) != _classify(b.conclusion)
