"""
conduct_demo.py — conduct norms (MATs) governing HOW an approved plan executes.

Per step: execute -> re-ground live state -> derive MATs -> print activations/
expirations -> next step runs under the merged effect. With --no-conduct the
same plan runs at normal stride for the side-by-side contrast.

This demo runs WITHOUT the LLM/tunnel (the plan-level gate is a separate,
already-demonstrated layer); it assumes the task is approved and shows the
conduct layer in isolation.

Usage (from SafeAgentBench root, safeagent-env active):
    python methods/grace/conduct_demo.py safe 0
    python methods/grace/conduct_demo.py safe 0 --no-conduct
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "low_level_controller"))

from state_grounding import get_state_facts, derive_mats, merge_effects
from conduct_executor import ConductExecutor

BANNER = "=" * 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["safe", "unsafe"])
    ap.add_argument("index", type=int)
    ap.add_argument("--no-conduct", action="store_true",
                    help="run the same plan without conduct norms (contrast)")
    args = ap.parse_args()

    bench_root = Path(__file__).resolve().parents[2]
    path = bench_root / "dataset" / f"{args.dataset}_detailed_1009.jsonl"
    tasks = [json.loads(l) for l in open(path) if l.strip()]
    task = tasks[args.index]

    print(BANNER)
    print(f" {task['instruction']}")
    print(f" plan: {task['step']}")
    print(f" conduct norms: {'OFF (contrast run)' if args.no_conduct else 'ON'}")
    print(BANNER)

    from ai2thor.controller import Controller
    from low_level_controller import LowLevelPlanner

    c = Controller(scene=task["scene_name"], snapToGrid=False,
                   gridSize=0.05, visibilityDistance=1.5)
    planner = LowLevelPlanner(c)
    planner.restore_scene()
    ex = ConductExecutor(c, planner)

    active: dict[str, dict] = {}

    for i, inst in enumerate(task["step"], 1):
        effect = None if args.no_conduct else merge_effects(list(active.values()))
        mode = f"under: {', '.join(active)} " if (active and not args.no_conduct) else ""
        print(f" STEP {i}: {inst!r} {mode}")

        ret = ex.run_step(inst, effect)
        print(f"    -> success={ret.get('success')}")

        # re-ground and diff MATs
        tag = "  (derived, NOT enforced)" if args.no_conduct else ""
        mats = {m["name"] + ":" + m["object"]: m
                for m in derive_mats(get_state_facts(c.last_event))}
        for key, m in mats.items():
            if key not in active:
                print(f"    ACTIVE MAT: {m['requirement']}{tag}")
        for key in list(active):
            if key not in mats:
                print(f"    MAT expired: {active[key]['requirement']}")
        active = mats
        time.sleep(0.5)

    print(f"\n done.")
    # ground truth: where did the task's objects actually end up?
    print(" final object states:")
    for o in c.last_event.metadata["objects"]:
        names = {n.lower() for n in (task.get("objects") or [])}
        step_targets = {s.split()[-1].lower() for s in task["step"] if len(s.split()) > 1}
        if o["objectType"].lower() in names | step_targets:
            recs = o.get("parentReceptacles") or []
            where = recs[0].split("|")[0] if recs else (
                "in hand" if o.get("isPickedUp") else "?")
            flags = [k for k in ("isToggled", "isOpen", "isFilledWithLiquid",
                                 "isBroken") if o.get(k)]
            print(f"   {o['objectType']}: on/in {where}"
                  f"{'  [' + ' '.join(flags) + ']' if flags else ''}")
    input(" Press Enter to close the simulator...")
    c.stop()


if __name__ == "__main__":
    main()
