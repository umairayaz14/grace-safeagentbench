"""SafeAgentGrounding — implements grace.SymbolicGrounding for SafeAgentBench tasks.

Mirrors MoralMachineGrounding's pattern: a lookup from generic
(premise, conclusion) pairs to theory reason ids, and get_facts() that detects
which hazard configurations hold in the concrete plan and instantiates the
generic reasons with concrete object names. Reasons whose text is not in the
loaded theory are silently skipped, exactly as in the moral-machine example.

Unlike the moral-machine domain, the possible-reason space here is open (any
object can appear), so premises are instantiated dynamically from plan analysis
rather than enumerated in a fixed table — the situation the moral-machine
grounding's docstring anticipates.

Observation: the SafeAgentBench task dict, with keys
    "instruction": str, "step": list[str], plus optional scene metadata.
"""

from grace.symbolic_reasoning import SymbolicGrounding
from grace.symbolic_reasoning.horty import HortyTheory, NLGroundedReason as GroundedReason

from safety_ontology import categories_of

Observation = dict  # SafeAgentBench task dict


# ---------------------------------------------------------------------------
# Plan analysis (self-contained; same logic as plan_grounding.py)
# ---------------------------------------------------------------------------

_ACTION_ALIASES = {
    "turn on": "toggle_on", "turn_on": "toggle_on",
    "turn off": "toggle_off", "turn_off": "toggle_off",
    "pick up": "pickup", "pick_up": "pickup", "pick": "pickup",
    "put": "put_in", "drop": "drop", "throw": "throw", "find": "find",
    "open": "open", "close": "close", "slice": "slice", "break": "break",
    "dirty": "dirty", "fill": "fill", "fillliquid": "fill",
    "empty": "empty", "pour": "pour",
    "cook": "cook", "clean": "clean",
}


def _parse_step(step: str):
    s = step.strip()
    for phrase in sorted(_ACTION_ALIASES, key=len, reverse=True):
        if s.lower().startswith(phrase):
            rest = s[len(phrase):].strip()
            return _ACTION_ALIASES[phrase], (rest or None)
    return s.lower(), None


def _analyze_plan(steps: list[str]) -> dict:
    """Walk the plan, tracking held/active objects; return hazard configurations."""
    held = None
    active: dict[str, int] = {}
    opened: dict[str, int] = {}
    last_found = None
    has_liquid = False
    released_active: set[str] = set()
    placed_in: set[tuple[str, str]] = set()
    broken: set[str] = set()
    soiled: set[str] = set()
    poured_on: set[str] = set()

    for raw in steps:
        action, obj = _parse_step(raw)
        target = obj or held
        if action == "toggle_on" and target:
            active[target] = 1
        elif action == "toggle_off" and target:
            active.pop(target, None)
        elif action == "open" and obj:
            opened[obj] = 1
        elif action == "close" and obj:
            opened.pop(obj, None)
        elif action == "pickup" and obj:
            held = obj
        elif action == "find" and obj:
            last_found = obj
        elif action == "break" and target:
            broken.add(target)
        elif action == "dirty" and target:
            soiled.add(target)
        elif action == "fill":
            has_liquid = True
        elif action == "pour":
            if has_liquid and last_found:
                poured_on.add(last_found)
            has_liquid = False
        elif action in ("drop", "throw", "put_in"):
            if held:
                container = obj if (action == "put_in" and obj) else last_found
                if container:
                    placed_in.add((held, container))
                if held in active:
                    released_active.add(held)
            held = None

    heated = {
        obj for (obj, container) in placed_in
        if "heat_appliance" in categories_of(container) and container in active
    }
    return {
        "released_active": released_active,
        "placed_in": placed_in,
        "heated": heated,
        "ends_active": set(active),
        "ends_open": set(opened),
        "broken": broken,
        "soiled": soiled,
        "poured_on": poured_on,
    }


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------

class SafeAgentGrounding(SymbolicGrounding[Observation, frozenset[GroundedReason]]):
    """Symbolic grounding for SafeAgentBench task plans."""

    def __init__(self, theory: HortyTheory) -> None:
        self._lookup: dict[tuple[str, str], str] = {
            (r.premise, r.conclusion): r.id for r in theory.reasons
        }

    def get_facts(self, observation: Observation) -> frozenset[GroundedReason]:
        grounded: set[GroundedReason] = set()
        steps = observation["step"]
        cfg = _analyze_plan(steps)

        # Base default: the instruction itself is a reason to execute.
        self._add(grounded,
            "the user instructed the agent to carry out plan p",
            "execute plan p",
            {})

        # Fire: activated ignition source released mid-plan.
        for obj in cfg["released_active"]:
            if "ignition_source" in categories_of(obj):
                self._add(grounded,
                    "plan p releases object o while o is an active ignition source",
                    "do not execute plan p",
                    {"object o": f"object {obj}", " o ": f" {obj} "})

        # Fire/damage: heat-sensitive or flammable object heated in an appliance.
        for obj in cfg["heated"]:
            cats = categories_of(obj)
            if "heat_sensitive" in cats or "flammable" in cats:
                self._add(grounded,
                    "plan p heats object o and o is damaged or dangerous when heated",
                    "do not execute plan p",
                    {"object o": f"object {obj}", " o ": f" {obj} "})

        # Water: water-sensitive object into a water source.
        for (obj, container) in cfg["placed_in"]:
            if ("water_sensitive" in categories_of(obj)
                    and "water_source" in categories_of(container)):
                self._add(grounded,
                    "plan p places water-sensitive object o into water source w",
                    "do not execute plan p",
                    {"object o": f"object {obj}",
                     "water source w": f"water source {container}"})

        # Fire: heat appliance left running at plan end.
        for obj in cfg["ends_active"]:
            if "heat_appliance" in categories_of(obj):
                self._add(grounded,
                    "plan p leaves heat appliance o switched on at the end of the plan",
                    "do not execute plan p",
                    {"heat appliance o": f"heat appliance {obj}"})

        # Damage: deliberate breakage of any object.
        for obj in cfg["broken"]:
            self._add(grounded,
                "plan p deliberately breaks object o",
                "do not execute plan p",
                {"object o": f"object {obj}"})

        # Damage: deliberate soiling of furniture or objects.
        for obj in cfg["soiled"]:
            self._add(grounded,
                "plan p deliberately dirties object o",
                "do not execute plan p",
                {"object o": f"object {obj}"})

        # Electrical/damage: liquid poured onto a water-sensitive object.
        for obj in cfg["poured_on"]:
            if "water_sensitive" in categories_of(obj):
                self._add(grounded,
                    "plan p pours liquid onto water-sensitive object o",
                    "do not execute plan p",
                    {"object o": f"object {obj}"})

        # Slip: liquid poured onto a walking surface.
        for obj in cfg["poured_on"]:
            if "walking_surface" in categories_of(obj):
                self._add(grounded,
                    "plan p pours liquid onto walking surface o",
                    "do not execute plan p",
                    {"walking surface o": f"walking surface {obj}"})

        # Spoilage/energy: cold appliance left open at plan end.
        for obj in cfg["ends_open"]:
            if "cold_appliance" in categories_of(obj):
                self._add(grounded,
                    "plan p leaves cold appliance o open at the end of the plan",
                    "do not execute plan p",
                    {"cold appliance o": f"cold appliance {obj}"})

        return frozenset(grounded)

    def _add(
        self,
        grounded: set[GroundedReason],
        generic_premise: str,
        generic_conclusion: str,
        substitutions: dict[str, str],
    ) -> None:
        """Instantiate a generic reason with concrete names; skip if absent
        from the theory — same contract as MoralMachineGrounding._add."""
        rule_id = self._lookup.get((generic_premise, generic_conclusion))
        if rule_id is None:
            return
        premise, conclusion = generic_premise, generic_conclusion
        for generic, concrete in substitutions.items():
            premise = premise.replace(generic, concrete)
            conclusion = conclusion.replace(generic, concrete)
        grounded.add(GroundedReason(reason_id=rule_id, rule=(premise, conclusion)))
