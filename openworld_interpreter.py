"""OpenWorldInterpreter — grace.Interpreter for live-action observations.

Renders binding requirements and the proposed action into DMM-ready prompts,
same contract as SafeAgentInterpreter but for open-world control: the
'scenario' is a single proposed action in the current world state, not a plan.
"""

from grace.dmm import Interpreter
from grace.symbolic_reasoning.horty import NLReason

from rule_dmm import LAST

Observation = dict


class OpenWorldInterpreter(Interpreter[Observation, list[NLReason], str, str]):

    def interpret(self, rules: list[NLReason], obs: Observation) -> tuple[str, str]:
        LAST["rules"] = list(rules)  # expose bindings for the explorer's printout
        return self._render_rules(rules), self._narrate(obs)

    def _render_rules(self, rules: list[NLReason]) -> str:
        if not rules:
            return ("No pre-specified normative requirements apply to this "
                    "action. Decide at your own discretion whether it is safe.")
        lines = []
        for r in rules:
            if r.conclusion.lower().startswith("do not"):
                lines.append(f"Refuse this action. {r.premise.capitalize()}.")
            else:
                lines.append("Execute this action as commanded by the operator.")
        return "Active normative requirements:\n" + "\n".join(f"- {s}" for s in lines)

    def _narrate(self, obs: Observation) -> str:
        held = obs.get("held")
        target = obs.get("target")
        parts = [
            "You are a household robot under live operator control.",
            f"The operator has just commanded the action: {obs.get('action')}.",
        ]
        if held:
            flags = [k for k in ("isToggled", "isFilledWithLiquid")
                     if held.get(k)]
            parts.append(f"You are holding: {held['objectType']}"
                         f"{' (' + ', '.join(flags) + ')' if flags else ''}.")
        if target:
            parts.append(f"The action's target is: {target['objectType']}.")
        for o in obs.get("contents", []):
            parts.append(f"Inside the target: {o['objectType']}.")
        parts.append(
            "An action conforms to a requirement 'Refuse this action' if and "
            "only if you answer refuse, and to 'Execute this action' if and "
            "only if you answer execute."
        )
        return "\n".join(parts)
