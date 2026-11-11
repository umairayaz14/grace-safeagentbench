"""GraceStateGrounding — grace.SymbolicGrounding over live simulator state.

The open-world counterpart of SafeAgentGrounding: instead of parsing a plan,
get_facts reads an observation assembled from the live event at the moment of
a proposed action, and instantiates whichever generic reasons from the loaded
theory apply. Same lookup/_add contract as the moral-machine grounding: reasons
absent from the theory are silently skipped, so ONE grounding class serves both
the prohibition theory (action observations) and the conduct theory (passive
observations) — each SR instance only ever activates its own theory's reasons.

Observation dict (assembled by the explorer):
    action:   str | None    proposed action name ("drop","throw","toggle","put");
                            None for a passive conduct query
    held:     dict | None   metadata dict of the currently held object
    target:   dict | None   metadata dict of the action's target (toggle/put)
    contents: list[dict]    objects inside the target (toggle on containers)
    near_heat: list[tuple]  (objectType, distance) of active heat appliances
                            within proximity
"""

from grace.symbolic_reasoning import SymbolicGrounding
from grace.symbolic_reasoning.horty import HortyTheory, NLGroundedReason as GroundedReason

from safety_ontology import categories_of

Observation = dict


class GraceStateGrounding(SymbolicGrounding[Observation, frozenset[GroundedReason]]):

    def __init__(self, theory: HortyTheory) -> None:
        self._lookup: dict[tuple[str, str], str] = {
            (r.premise, r.conclusion): r.id for r in theory.reasons
        }

    def get_facts(self, obs: Observation) -> frozenset[GroundedReason]:
        grounded: set[GroundedReason] = set()
        action = obs.get("action")
        held = obs.get("held")
        target = obs.get("target")

        # ---- prohibition-theory premises (only ground for real actions) ----
        if action is not None:
            self._add(grounded,
                      "the operator commanded action a",
                      "execute action a",
                      {"action a": f"action {action}"})

            hcats = categories_of(held["objectType"]) if held else set()

            if action in ("drop", "throw") and held:
                if held.get("isToggled") and "ignition_source" in hcats:
                    self._add(grounded,
                              "action a releases object o while o is an active ignition source",
                              "do not execute action a",
                              {"action a": f"action {action}",
                               "object o": f"object {held['objectType']}",
                               " o ": f" {held['objectType']} "})
                if action == "throw" and "fragile" in hcats:
                    self._add(grounded,
                              "action a throws fragile object o",
                              "do not execute action a",
                              {"action a": f"action {action}",
                               "object o": f"object {held['objectType']}"})

            if action == "toggle" and target and not target.get("isToggled"):
                if "heat_appliance" in categories_of(target["objectType"]):
                    for o in obs.get("contents", []):
                        icats = categories_of(o["objectType"])
                        if "heat_sensitive" in icats or "flammable" in icats:
                            self._add(grounded,
                                      "action a activates heat appliance h with object o inside that is damaged or dangerous when heated",
                                      "do not execute action a",
                                      {"action a": f"action {action}",
                                       "heat appliance h": f"heat appliance {target['objectType']}",
                                       "object o": f"object {o['objectType']}"})

            if action == "put" and held and target:
                tcats = categories_of(target["objectType"])
                if "water_sensitive" in hcats and "water_source" in tcats:
                    self._add(grounded,
                              "action a places water-sensitive object o into water source w",
                              "do not execute action a",
                              {"action a": f"action {action}",
                               "object o": f"object {held['objectType']}",
                               "water source w": f"water source {target['objectType']}"})
                if held.get("isToggled") and "ignition_source" in hcats:
                    self._add(grounded,
                              "action a places object o while o is a lit ignition source",
                              "do not execute action a",
                              {"action a": f"action {action}",
                               "object o": f"object {held['objectType']}"})

        # ---- conduct-theory premises (ground from state, action-independent) --
        if held:
            hcats = categories_of(held["objectType"])
            if "fragile" in hcats:
                self._add(grounded,
                          "the agent is holding fragile object o",
                          "move slowly while holding o",
                          {"object o": f"object {held['objectType']}",
                           " o": f" {held['objectType']}"})
            if held.get("isFilledWithLiquid"):
                self._add(grounded,
                          "the agent is holding liquid-filled object o",
                          "move slowly and avoid sharp turns while holding o",
                          {"object o": f"object {held['objectType']}",
                           " o": f" {held['objectType']}"})
        for (t, d) in obs.get("near_heat", []):
            self._add(grounded,
                      "an active heat source o is within 1.5 meters",
                      "keep movements small near o",
                      {"source o": f"source {t}", "near o": f"near {t}"})

        return frozenset(grounded)

    def _add(self, grounded, generic_premise, generic_conclusion, subs) -> None:
        rule_id = self._lookup.get((generic_premise, generic_conclusion))
        if rule_id is None:
            return
        premise, conclusion = generic_premise, generic_conclusion
        for g, s in subs.items():
            premise = premise.replace(g, s)
            conclusion = conclusion.replace(g, s)
        grounded.add(GroundedReason(reason_id=rule_id, rule=(premise, conclusion)))
