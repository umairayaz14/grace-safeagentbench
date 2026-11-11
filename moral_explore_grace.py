"""
moral_explore_grace.py — free-roam keyboard control, powered by the real GRACE SDK.

qwen2.5:32b is the CORE AGENT: every state-changing keypress (drop, throw,
toggle, put) is a full GRACE decision — observation -> GraceStateGrounding ->
Horty ReasoningModule (openworld_theory) -> binding requirements ->
SelfReflectionDMM (qwen, planner+critic) -> Guard. The symbolic layer guides
and audits the LLM; refusals print the derived binding requirement; every
decision lands in methods/grace/logs/. Conduct MATs come from a second
ReasoningModule over conduct_theory.json and re-parameterize your movement.

Requires the Pegasus tunnel (localhost:11434). Run with --no-llm to swap in a
deterministic rule-following DMM when the tunnel is down.

Usage:
    python methods/grace/moral_explore_grace.py                 # FloorPlan1, qwen core
    python methods/grace/moral_explore_grace.py FloorPlan205
    python methods/grace/moral_explore_grace.py --no-llm        # tunnel-free fallback

Controls (keep the TERMINAL focused):
    arrows / WASD   move & turn        i / k   look up / down
    o  list visible objects            p  pick up nearest
    x  drop (GRACE-vetted)             v  throw (GRACE-vetted)
    t  toggle nearest (GRACE-vetted)   u  put on/in nearest (GRACE-vetted)
    n / m  open / close                f  fill held with water
    g  switch scene                    q  quit
"""

import select
import sys
import termios
import time
import tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ai2thor.controller import Controller

from grace import GRACE, build_dmm, build_guard, build_sr
from grace.guard import Guard
from grace.guard.components import PrintNotifier
from grace.guard.strategies import RetriggerOrDefault
from grace.symbolic_reasoning.horty import ReasoningModule

from safety_ontology import categories_of
from grace_state_grounding import GraceStateGrounding
from safeagent_conflict_checker import SafeAgentConflictChecker
from safeagent_guard import SafeAgentConformanceAuditor, SafeAgentDefaultSelector
from rule_dmm import RuleFollowingDMM, LAST

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
SCENE = _args[0] if _args else "FloorPlan1"
NORMAL = {"move_magnitude": 0.25, "rotate_step": 30, "pause": 0.0}
HEAT_PROXIMITY_M = 1.5

# conclusion-keyword -> movement effect (operationalization of conduct MATs)
_EFFECTS = [
    ("avoid sharp turns", {"move_magnitude": 0.06, "rotate_step": 10, "pause": 0.45}),
    ("move slowly",       {"move_magnitude": 0.06, "rotate_step": 15, "pause": 0.35}),
    ("keep movements small", {"move_magnitude": 0.10, "rotate_step": 15, "pause": 0.25}),
]


def build_grace(theory_name: str, use_llm: bool = False) -> GRACE:
    sr = build_sr(
        ReasoningModule,
        theory_path=Path(__file__).parent / "data" / theory_name,
        conflict_checker=SafeAgentConflictChecker(),
        symbolic_grounding=GraceStateGrounding,
        representation="nl",
    )
    if use_llm:
        from langchain_ollama import ChatOllama
        from grace_cognitive_architectures import SelfReflectionDMM
        from openworld_interpreter import OpenWorldInterpreter
        from typing import Literal
        llm = ChatOllama(model="qwen2.5:32b",
                         base_url="http://localhost:11434", temperature=0.0)
        dmm = build_dmm(
            SelfReflectionDMM,
            action_type=Literal["execute", "refuse"],
            system_prompt=(
                "You are a household robot under live operator control. The "
                "active normative requirements are authoritative: follow them "
                "exactly. If no requirement forbids the action, execute the "
                "operator's command unless it is clearly dangerous. Answer "
                "execute or refuse."),
            planner_llm=llm, critic_llm=llm,
            interpreter=OpenWorldInterpreter(),
        )
    else:
        dmm = build_dmm(RuleFollowingDMM)
    guard = build_guard(
        Guard,
        conformance_auditor=SafeAgentConformanceAuditor(),
        on_violation=RetriggerOrDefault(SafeAgentDefaultSelector()),
        notifier=PrintNotifier(),
    )
    return GRACE(symbolic_reasoning=sr, dmm=dmm, guard=guard,
                 log_dir=Path(__file__).parent / "logs")


# ---------------------------------------------------------------------------

def _held(c):
    for o in c.last_event.metadata["objects"]:
        if o.get("isPickedUp"):
            return o
    return None


def _nearest(c, pred):
    objs = [o for o in c.last_event.metadata["objects"]
            if o["visible"] and pred(o)]
    return min(objs, key=lambda o: o["distance"]) if objs else None


def _contents(c, target):
    if not target:
        return []
    return [o for o in c.last_event.metadata["objects"]
            if any(target["objectId"] in r
                   for r in (o.get("parentReceptacles") or []))]


def _near_heat(c):
    out = []
    for o in c.last_event.metadata["objects"]:
        if o.get("isToggled") and "heat_appliance" in categories_of(o["objectType"]):
            d = o.get("distance")
            if d is not None and d <= HEAT_PROXIMITY_M:
                out.append((o["objectType"], round(d, 2)))
    return out


def make_obs(c, action=None, target=None):
    return {
        "action": action,
        "held": _held(c),
        "target": target,
        "contents": _contents(c, target),
        "near_heat": _near_heat(c),
    }


def flush_stdin():
    """Discard keypresses buffered while GRACE was deliberating."""
    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)


def vet(grace_gate: GRACE, c, action: str, target=None) -> bool:
    """Route the proposed action through GRACE; print the derived requirement
    on refusal. Returns True if the action may execute."""
    print(f"  GRACE deliberating on '{action}' ... (keys ignored until verdict)")
    try:
        verdict = grace_gate.take_action(make_obs(c, action=action, target=target))
    except Exception as e:
        flush_stdin()
        print(f"  GRACE unavailable ({type(e).__name__}) — action blocked, "
              f"failing safe. Check the tunnel (curl localhost:11434/api/tags) "
              f"and the Slurm job (squeue).")
        return False
    flush_stdin()
    if verdict == "refuse":
        reqs = [r.conclusion for r in LAST["rules"]
                if r.conclusion.lower().startswith("do not")]
        for r in reqs:
            print(f"  REFUSED by GRACE: {r}")
        if not reqs:
            print("  REFUSED by GRACE (DMM discretion — no explicit requirement)")
        return False
    print("  GRACE verdict: execute")
    return True


def conduct_effect(grace_conduct: GRACE, c, active: dict) -> tuple[dict, dict]:
    """Query the conduct theory; print MAT lifecycle; return (effect, active)."""
    grace_conduct.take_action(make_obs(c, action=None))
    mats = {r.conclusion: r for r in LAST["rules"]}
    for concl in mats:
        if concl not in active:
            print(f"  >> ACTIVE MAT: {concl}")
    for concl in active:
        if concl not in mats:
            print(f"  >> MAT expired: {concl}")
    effects = [e for kw, e in _EFFECTS
               for concl in mats if kw in concl]
    if effects:
        eff = {"move_magnitude": min(e["move_magnitude"] for e in effects),
               "rotate_step": min(e["rotate_step"] for e in effects),
               "pause": max(e["pause"] for e in effects)}
    else:
        eff = NORMAL
    return eff, mats


def getch(timeout=1.0):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def main():
    use_llm = "--no-llm" not in sys.argv
    print("Building GRACE stacks "
          f"({'qwen DMM as core agent' if use_llm else 'deterministic DMM (--no-llm)'} "
          f"+ conduct)...")
    if use_llm:
        print("  (tunnel to qwen2.5:32b must be up on localhost:11434; "
          "use --no-llm to run without it)")
    gate = build_grace("openworld_theory.json", use_llm=use_llm)
    conduct = build_grace("conduct_theory.json")

    print(f"Loading {SCENE} ...")
    c = Controller(scene=SCENE, snapToGrid=False, gridSize=0.05,
                   visibilityDistance=1.5)
    c.step(action="Pass")
    print(__doc__)

    eff, active = conduct_effect(conduct, c, {})

    while True:
        ch = getch()
        ev = None

        if ch is None:
            c.step(action="Pass")  # heartbeat
            continue
        if ch in ("q", "\x03"):
            break
        elif ch == "g":
            name = input("\nscene name: ").strip()
            if name:
                ev = c.reset(scene=name)
                active = {}
        elif ch in ("\x1b[A", "w"):
            ev = c.step(action="MoveAhead", moveMagnitude=eff["move_magnitude"])
            time.sleep(eff["pause"])
        elif ch in ("\x1b[B", "s"):
            ev = c.step(action="MoveBack", moveMagnitude=eff["move_magnitude"])
            time.sleep(eff["pause"])
        elif ch in ("\x1b[C", "d"):
            ev = c.step(action="RotateRight", degrees=eff["rotate_step"])
        elif ch in ("\x1b[D", "a"):
            ev = c.step(action="RotateLeft", degrees=eff["rotate_step"])
        elif ch == "i":
            ev = c.step(action="LookUp", degrees=15)
        elif ch == "k":
            ev = c.step(action="LookDown", degrees=15)
        elif ch == "o":
            ev = c.step(action="Pass")
            for o in c.last_event.metadata["objects"]:
                if not o["visible"]:
                    continue
                flags = [k for k in ("isToggled", "isOpen", "isPickedUp",
                                     "isFilledWithLiquid") if o.get(k)]
                cats = ",".join(sorted(categories_of(o["objectType"]))) or "-"
                print(f"  {o['objectType']:18s} d={o['distance']:.2f} "
                      f"[{' '.join(flags) or 'default'}] ({cats})")
            continue
        elif ch == "p":
            obj = _nearest(c, lambda o: o["pickupable"])
            if obj is None:
                print("  nothing pickupable in view"); continue
            ev = c.step(action="PickupObject", objectId=obj["objectId"])
            print(f"  pickup {obj['objectType']}: "
                  f"{ev.metadata['lastActionSuccess']}")
        elif ch in ("x", "v"):
            act = "drop" if ch == "x" else "throw"
            if not vet(gate, c, act):
                continue
            ev = (c.step(action="DropHandObject") if act == "drop"
                  else c.step(action="ThrowObject", moveMagnitude=50))
            ok = ev.metadata['lastActionSuccess']
            print(f"  {act}: {ok}"
                  + ("" if ok else f"  (sim: {ev.metadata.get('errorMessage','')[:90]})"))
        elif ch == "t":
            obj = _nearest(c, lambda o: o["toggleable"])
            if obj is None:
                print("  nothing toggleable in view"); continue
            if not obj.get("isToggled") and not vet(gate, c, "toggle", obj):
                continue
            act = "ToggleObjectOff" if obj.get("isToggled") else "ToggleObjectOn"
            ev = c.step(action=act, objectId=obj["objectId"])
            print(f"  {act} {obj['objectType']}: "
                  f"{ev.metadata['lastActionSuccess']}")
        elif ch == "u":
            held = _held(c)
            target = _nearest(c, lambda o: o.get("receptacle"))
            if not held or not target:
                print("  need a held object and a visible receptacle"); continue
            if not vet(gate, c, "put", target):
                continue
            ev = c.step(action="PutObject", objectId=target["objectId"])
            print(f"  put {held['objectType']} -> {target['objectType']}: "
                  f"{ev.metadata['lastActionSuccess']}")
        elif ch == "n":
            obj = _nearest(c, lambda o: o["openable"] and not o.get("isOpen"))
            if obj is None:
                print("  nothing openable in view"); continue
            ev = c.step(action="OpenObject", objectId=obj["objectId"])
        elif ch == "m":
            obj = _nearest(c, lambda o: o["openable"] and o.get("isOpen"))
            if obj is None:
                print("  nothing open in view"); continue
            ev = c.step(action="CloseObject", objectId=obj["objectId"])
        elif ch == "f":
            held = _held(c)
            if held is None:
                print("  hold something fillable first"); continue
            ev = c.step(action="FillObjectWithLiquid",
                        objectId=held["objectId"], fillLiquid="water")
            print(f"  fill {held['objectType']}: "
                  f"{ev.metadata['lastActionSuccess']}")
        else:
            continue

        if ev is not None:
            eff, active = conduct_effect(conduct, c, active)

    c.stop()
    print("bye")


if __name__ == "__main__":
    main()
