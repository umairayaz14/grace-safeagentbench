"""State grounding for conduct norms.

get_state_facts(event)  — pure function: sim metadata + ontology -> fact tuples.
derive_mats(facts)      — instantiate conduct-theory premises against the facts
                          and return the binding conduct requirements (MATs).

v1 note: conduct obligations here never conflict, so MATs are derived by direct
premise matching rather than running the full Horty engine. The theory file
keeps the identical schema, so upgrading to the SDK ReasoningModule (needed the
moment two conduct norms can clash and priorities matter) is a drop-in change.
"""

import json
from pathlib import Path

from safety_ontology import categories_of

_THEORY = json.loads(
    (Path(__file__).parent / "data" / "conduct_theory.json").read_text()
)["reasons"]

HEAT_PROXIMITY_M = 1.5


def get_state_facts(event) -> frozenset[tuple]:
    """Read the live event metadata, join with the ontology, emit facts."""
    facts = set()
    for o in event.metadata["objects"]:
        t = o["objectType"]
        cats = categories_of(t)

        if o.get("isPickedUp"):
            facts.add(("holding", t))
            for cat in cats:
                facts.add(("holding_category", cat, t))
            if o.get("isFilledWithLiquid"):
                facts.add(("holding_category", "liquid_filled", t))

        if o.get("isToggled") and "heat_appliance" in cats:
            d = o.get("distance")
            if d is not None and d <= HEAT_PROXIMITY_M:
                facts.add(("near_active_heat", t, round(d, 2)))

    return frozenset(facts)


def derive_mats(facts: frozenset) -> list[dict]:
    """Match conduct premises against facts; return active MATs.

    Each MAT: {"name", "requirement", "effect"} where effect is the
    deterministic execution profile the ConductExecutor understands.
    """
    mats = []
    for f in facts:
        if f[0] == "holding_category" and f[1] == "fragile":
            mats.append(_mat("careful_with_fragile", f[2],
                             effect={"move_magnitude": 0.06, "rotate_step": 15,
                                     "pause": 0.35}))
        elif f[0] == "holding_category" and f[1] == "liquid_filled":
            mats.append(_mat("careful_with_liquid", f[2],
                             effect={"move_magnitude": 0.06, "rotate_step": 10,
                                     "pause": 0.45}))
        elif f[0] == "near_active_heat":
            mats.append(_mat("careful_near_heat", f[1],
                             effect={"move_magnitude": 0.10, "rotate_step": 15,
                                     "pause": 0.25}))
    return mats


def _mat(name: str, obj: str, effect: dict) -> dict:
    reason = next(r for r in _THEORY if r["name"] == name)
    requirement = reason["conclusion"].replace(" o", f" {obj}") \
        .replace("near o", f"near {obj}")
    return {"name": name, "object": obj,
            "requirement": requirement, "effect": effect}


def merge_effects(mats: list[dict]) -> dict | None:
    """Most-cautious-wins when several MATs are active."""
    if not mats:
        return None
    return {
        "move_magnitude": min(m["effect"]["move_magnitude"] for m in mats),
        "rotate_step": min(m["effect"]["rotate_step"] for m in mats),
        "pause": max(m["effect"]["pause"] for m in mats),
    }
