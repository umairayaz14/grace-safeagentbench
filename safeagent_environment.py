"""SafeAgentEnvironment — SafeAgentBench tasks behind the MoralMachineEnvironment interface.

Single-step episodes, same as the trolley env: each task is one episode.
reset() returns the next task as an Observation; step(action) either executes
the plan through SafeAgentBench's LowLevelPlanner ("execute") or records a
refusal ("refuse") and returns a StepResult.

By default runs dry (no simulator) so verdict sweeps over hundreds of tasks
are instant; pass execute_in_sim=True to actually drive AI2-THOR. The Unity
controller is booted once and re-used across episodes via controller.reset().
"""

import json
from pathlib import Path
from typing import Literal, TypedDict


class Observation(TypedDict):
    id: str
    instruction: str
    step: list[str]
    objects: list[str]
    risk_category: str | None
    scene_name: str
    dataset: str


Action = Literal["execute", "refuse"]


class StepResult(TypedDict):
    status: str                 # "executed" | "refused" | "executed_dry"
    step_results: list[dict]    # per-step controller returns (empty if dry/refused)


class SafeAgentEnvironment:
    """Serves SafeAgentBench tasks as one-shot episodes."""

    def __init__(
        self,
        bench_root: str | Path,
        datasets: tuple[str, ...] = ("unsafe_detailed_1009", "safe_detailed_1009"),
        execute_in_sim: bool = False,
    ) -> None:
        self.bench_root = Path(bench_root)
        self.execute_in_sim = execute_in_sim
        self._tasks: list[Observation] = []
        for name in datasets:
            path = self.bench_root / "dataset" / f"{name}.jsonl"
            with open(path) as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    t = json.loads(line)
                    self._tasks.append(Observation(
                        id=f"{name}:{i}",
                        instruction=t["instruction"],
                        step=t["step"],
                        objects=t.get("objects", []),
                        risk_category=t.get("risk_category"),
                        scene_name=t["scene_name"],
                        dataset=name,
                    ))
        self._cursor = 0
        self._current: Observation | None = None
        self._controller = None
        self._planner = None

    # -- episode protocol ---------------------------------------------------

    def reset(self, seed: int | None = None, index: int | None = None) -> Observation:
        """Return the next task (or a specific one via ``index``)."""
        if index is not None:
            self._cursor = index
        if self._cursor >= len(self._tasks):
            raise IndexError("no tasks left; environment exhausted")
        self._current = self._tasks[self._cursor]
        self._cursor += 1
        return self._current

    def step(self, action: Action) -> StepResult:
        if self._current is None:
            raise RuntimeError("Call reset() before step()")
        task = self._current

        if action == "refuse":
            return StepResult(status="refused", step_results=[])

        if not self.execute_in_sim:
            return StepResult(status="executed_dry", step_results=[])

        planner = self._get_planner(task["scene_name"])
        results = [planner.llm_skill_interact(inst) for inst in task["step"]]
        return StepResult(status="executed", step_results=results)

    def close(self) -> None:
        if self._controller is not None:
            self._controller.stop()
            self._controller = None

    def __len__(self) -> int:
        return len(self._tasks)

    # -- simulator plumbing -------------------------------------------------

    def _get_planner(self, scene: str):
        import sys
        sys.path.insert(0, str(self.bench_root / "low_level_controller"))
        from ai2thor.controller import Controller
        from low_level_controller import LowLevelPlanner

        if self._controller is None:
            self._controller = Controller(scene=scene)
        else:
            self._controller.reset(scene=scene)
        self._planner = LowLevelPlanner(self._controller)
        self._planner.restore_scene()
        return self._planner
