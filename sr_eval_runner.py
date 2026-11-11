"""
sr_eval_runner.py — evaluate the SR module on states along executed plans.

For each task: obtain a plan (dataset reference steps, or the LLM planner
with dataset fallback), execute it fully in AI2-THOR, and after EVERY step
run two SR instances on the resulting state — the LLM-grounded SR under
test and the hand-coded oracle SR — logging both. Nothing is blocked.

Usage (from SafeAgentBench root, safeagent-env active, tunnel up):
  python methods/grace/sr_eval_runner.py --theory specific --n 10
  python methods/grace/sr_eval_runner.py --theory abstract --n 10 \
      --mode per_premise --render raw --bias conservative
  python methods/grace/sr_eval_runner.py --theory specific --n 10 --planner qwen

Output: methods/grace/sr_logs/<theory>_<mode>_<render>_<bias>[_planner].jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "low_level_controller"))

from langchain_ollama import ChatOllama

from grace import build_sr
from grace.symbolic_reasoning.horty import ReasoningModule

from safeagent_conflict_checker import SafeAgentConflictChecker
from sr_state_render import snapshot
from neural_grounding import NeuralGrounding, CAPTURE
from oracle_grounding import OracleGrounding, CAPTURE_ORACLE
from sr_planner import get_plan

HERE = Path(__file__).parent
BROKEN = {"safe": {80, 101}, "unsafe": {233, 280, 286}}  # scene/object mismatches


def load_tasks(bench_root, split):
    path = bench_root / "dataset" / f"{split}_detailed_1009.jsonl"
    return [json.loads(l) for l in open(path) if l.strip()]


def build(theory_file, grounding_cls):
    return build_sr(
        ReasoningModule,
        theory_path=HERE / "data" / theory_file,
        conflict_checker=SafeAgentConflictChecker(),
        symbolic_grounding=grounding_cls,
        representation="nl",
    )


_SPELLINGS = {"turn_on": "turn on", "turn_off": "turn off", "turnon": "turn on",
              "turnoff": "turn off", "toggle_on": "turn on", "toggle_off": "turn off",
              "pickup": "pick", "put_on": "put", "putin": "put", "put_in": "put"}


def _normalize(inst: str) -> str:
    """Map dataset spellings onto the executor's case-sensitive skill prefixes."""
    parts = inst.strip().split()
    if not parts:
        return inst
    head = parts[0].lower()
    if head in _SPELLINGS:
        return " ".join([_SPELLINGS[head]] + parts[1:])
    if head in ("fillliquid", "emptyliquid"):
        head = "fillLiquid" if head == "fillliquid" else "emptyLiquid"
        return " ".join([head] + parts[1:])
    return " ".join([head] + parts[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", choices=["specific", "abstract"], required=True)
    ap.add_argument("--n", type=int, default=10, help="tasks per split")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--splits", default="unsafe,safe")
    ap.add_argument("--mode", choices=["batch", "per_premise"], default="batch")
    ap.add_argument("--render", choices=["nl", "raw"], default="nl")
    ap.add_argument("--bias", choices=["neutral", "conservative"], default="neutral")
    ap.add_argument("--planner", choices=["dataset", "qwen"], default="dataset")
    ap.add_argument("--model", default="qwen2.5:32b")
    ap.add_argument("--append", action="store_true",
                    help="append to an existing log (resume) instead of overwriting")
    args = ap.parse_args()

    bench_root = HERE.parents[1]
    llm = ChatOllama(model=args.model, base_url="http://127.0.0.1:11434",
                     temperature=0.0)
    NeuralGrounding.configure(llm, mode=args.mode, render=args.render,
                              bias=args.bias, role="llm")
    theory_file = f"{args.theory}_theory.json"
    sr_llm = build(theory_file, NeuralGrounding)
    sr_oracle = build(theory_file, OracleGrounding)

    tag = f"{args.theory}_{args.mode}_{args.render}_{args.bias}"
    if args.planner != "dataset":
        tag += f"_{args.planner}"
    out_dir = HERE / "sr_logs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{tag}.jsonl"
    out = open(out_path, "a" if args.append else "w")

    from ai2thor.controller import Controller
    from low_level_controller import LowLevelPlanner
    controller = None

    for split in args.splits.split(","):
        tasks = load_tasks(bench_root, split)
        for idx in range(args.start, min(args.start + args.n, len(tasks))):
            if idx in BROKEN[split]:
                continue
            task = tasks[idx]
            # scene first: the two-stage planner reads the scene's object list
            if controller is None:
                controller = Controller(scene=task["scene_name"])
            else:
                controller.reset(scene=task["scene_name"])
            planner = LowLevelPlanner(controller)
            planner.restore_scene()

            plan, plan_source = get_plan(task, args.planner, llm, controller)
            print(f"\n[{split} {idx}] {task['instruction']}  (plan: {plan_source})")

            # state 0: before any step
            states = [("<start>", None, controller.last_event)]
            for inst in plan:
                norm = _normalize(inst)
                try:
                    ret = planner.llm_skill_interact(norm)
                    ok = bool(ret.get("success"))
                except AssertionError:
                    print(f"  (executor does not support step {inst!r}; skipping)")
                    ok = None
                except Exception as e:
                    print(f"  (executor error on {inst!r}: {type(e).__name__}; skipping)")
                    ok = None
                states.append((inst, ok, controller.last_event))

            for step_i, (inst, ok, ev) in enumerate(states):
                snap = snapshot(ev)
                obs = {"snapshot": snap}
                t0 = time.time()
                rules_llm = sr_llm.infer(obs)
                dt = time.time() - t0
                rules_oracle = sr_oracle.infer(obs)
                rec = {
                    "split": split, "task": idx, "instruction": task["instruction"],
                    "risk_category": task.get("risk_category"),
                    "plan_source": plan_source, "step": step_i, "action": inst,
                    "action_success": ok, "held": snap["held"]["type"] if snap["held"] else None,
                    "llm": CAPTURE.get("llm", {}),
                    "llm_rules": [(r.premise, r.conclusion) for r in rules_llm],
                    "oracle_triggered": CAPTURE_ORACLE.get("triggered", []),
                    "oracle_rules": [(r.premise, r.conclusion) for r in rules_oracle],
                    "llm_seconds": round(dt, 2),
                    "snapshot": snap,
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                lt, ot = set(rec["llm"].get("triggered", [])), set(rec["oracle_triggered"])
                mark = "  " if lt == ot else "!!"
                print(f"  {mark} step {step_i:2d} {inst:28s} llm={sorted(lt)} oracle={sorted(ot)}")

    if controller is not None:
        controller.stop()
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
