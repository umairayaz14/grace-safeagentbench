"""
Run a single SafeAgentBench task end-to-end in AI2-THOR.

Usage (from the SafeAgentBench repo root, with safeagent-env active):
    python run_one_task.py              # runs task 0 (candle -> sink, Fire Hazard)
    python run_one_task.py 3            # runs task index 3 instead
    python run_one_task.py 0 safe       # runs task 0 from the SAFE detailed set
"""

import json
import sys

sys.path.insert(0, "low_level_controller")

from ai2thor.controller import Controller
from low_level_controller import LowLevelPlanner

# ---- pick task ----
task_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
which_set = sys.argv[2] if len(sys.argv) > 2 else "unsafe"

dataset_file = {
    "unsafe": "dataset/unsafe_detailed_1009.jsonl",
    "safe": "dataset/safe_detailed_1009.jsonl",
}[which_set]

with open(dataset_file) as f:
    tasks = [json.loads(line) for line in f if line.strip()]

task = tasks[task_index]

print("=" * 60)
print("DATASET:     ", dataset_file)
print("TASK INDEX:  ", task_index)
print("INSTRUCTION: ", task["instruction"])
print("RISK:        ", task.get("risk_category"))
print("SCENE:       ", task["scene_name"])
print("PLAN:        ", task["step"])
print("=" * 60)

# ---- boot simulator in the task's scene ----
env = Controller(scene=task["scene_name"])
planner = LowLevelPlanner(env)
planner.restore_scene()

# ---- execute plan step by step ----
results = []
for inst in task["step"]:
    ret = planner.llm_skill_interact(inst)
    results.append((inst, ret))
    print(f"STEP: {inst!r:30s} -> {ret}")

print("=" * 60)
print("DONE. Steps executed:", len(results))

env.stop()
