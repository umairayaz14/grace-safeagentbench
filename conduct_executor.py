"""ConductExecutor — executes a plan under active conduct MATs.

Movement is done with our own primitives so it is parameterizable (their
`find` skill is atomic and exposes no movement knobs): rotate toward the
target in `rotate_step` increments, advance in `move_magnitude` strides with
`pause` beats — under a MAT this is visibly slow, careful walking; with no MAT
active it delegates `find` to the fast built-in skill. Manipulation (pick, put,
toggle, open, close, slice, fill, pour...) always delegates to SafeAgentBench's
LowLevelPlanner. After a manual approach, a sync `find` is issued so the
planner's internal state matches reality; by then the agent is already at the
target, so any snap is small.
"""

import math
import time


class ConductExecutor:
    def __init__(self, controller, planner, log=print):
        self.c = controller
        self.p = planner
        self.log = log

    # ---- public -------------------------------------------------------

    def run_step(self, inst: str, effect: dict | None) -> dict:
        """Execute one plan step under the given merged MAT effect (or None)."""
        action = inst.split()[0].lower()
        target = inst.split()[-1] if len(inst.split()) > 1 else None

        if action == "find" and target and effect is not None:
            ok = self._careful_approach(target, effect)
            if ok:
                # arrived under our own power; their find would TeleportFull
                # redundantly (and can crash on camera horizon at close range).
                # Manipulation skills look objects up by name, so no sync needed.
                return {"action": inst, "success": True, "careful_walk": True,
                        "message": "", "errorMessage": ""}
            ret = self.p.llm_skill_interact(inst)
            ret["careful_walk"] = False
            return ret

        return self.p.llm_skill_interact(inst)

    # ---- careful navigation --------------------------------------------

    def _find_object(self, name_fragment):
        for o in self.c.last_event.metadata["objects"]:
            if o["objectType"].lower() == name_fragment.lower():
                return o
        for o in self.c.last_event.metadata["objects"]:
            if name_fragment.lower() in o["objectType"].lower():
                return o
        return None

    def _get_path(self, obj) -> list | None:
        """Ask the sim's navmesh for an obstacle-free path to the object."""
        ev = self.c.step(action="GetShortestPath",
                         objectId=obj["objectId"], allowedError=0.5)
        if ev.metadata["lastActionSuccess"]:
            corners = (ev.metadata.get("actionReturn") or {}).get("corners")
            if corners:
                return corners
        # fallback: path to the object's position instead of its id
        ev = self.c.step(action="GetShortestPathToPoint",
                         position=obj["position"], allowedError=0.5)
        if ev.metadata["lastActionSuccess"]:
            corners = (ev.metadata.get("actionReturn") or {}).get("corners")
            if corners:
                return corners
        return None

    def _careful_approach(self, target: str, effect: dict,
                          stop_distance: float = 1.2) -> bool:
        obj = self._find_object(target)
        if obj is None:
            self.log(f"    [conduct] {target}: not in metadata, "
                     f"falling back to built-in find")
            return False

        mag, rot, pause = (effect["move_magnitude"],
                           effect["rotate_step"], effect["pause"])

        path = self._get_path(obj)
        if path is None:
            self.log(f"    [conduct] no navmesh path to {obj['objectType']}, "
                     f"falling back to built-in find")
            return False

        agent = self.c.last_event.metadata["agent"]
        dist0 = math.hypot(obj["position"]["x"] - agent["position"]["x"],
                           obj["position"]["z"] - agent["position"]["z"])
        self.log(f"    [conduct] approaching {obj['objectType']} carefully "
                 f"({dist0:.1f}m, {len(path)} waypoints; stride={mag}m, "
                 f"turn<={rot}deg, pause={pause}s)")

        obj_type = obj["objectType"]
        for w, corner in enumerate(path, 1):
            wx, wz = corner["x"], corner["z"]
            budget = 400  # per-waypoint safety cap
            while budget:
                budget -= 1
                # arrived at the target itself? done early.
                obj = self._find_object(obj_type) or obj
                if obj.get("visible") or obj.get("distance", 99) <= stop_distance:
                    self.log(f"    [conduct] arrived "
                             f"({obj.get('distance', 0):.2f}m, "
                             f"visible={bool(obj.get('visible'))})")
                    return True

                agent = self.c.last_event.metadata["agent"]
                ax, az = agent["position"]["x"], agent["position"]["z"]
                if math.hypot(wx - ax, wz - az) <= 0.20:
                    break  # waypoint reached, next corner

                desired = math.degrees(math.atan2(wx - ax, wz - az)) % 360
                current = agent["rotation"]["y"] % 360
                diff = (desired - current + 540) % 360 - 180
                if abs(diff) > 5:
                    step = max(-rot, min(rot, diff))
                    act = "RotateRight" if step > 0 else "RotateLeft"
                    self.c.step(action=act, degrees=abs(step))
                    time.sleep(pause / 2)
                    continue

                ev = self.c.step(action="MoveAhead", moveMagnitude=mag)
                if not ev.metadata["lastActionSuccess"]:
                    # on a navmesh path this should be rare; nudge sideways once
                    self.c.step(action="MoveRight", moveMagnitude=mag)
                time.sleep(pause)
            else:
                self.log(f"    [conduct] waypoint {w} unreachable, "
                         f"falling back to built-in find")
                return False

        # walked the whole navmesh path — now close the final gap: the path
        # ends in the object's neighborhood, but manipulation needs the object
        # visible / in interaction range. Face it, then creep straight in
        # until it's visible or we bump its edge.
        agent = self.c.last_event.metadata["agent"]
        ax, az = agent["position"]["x"], agent["position"]["z"]
        desired = math.degrees(math.atan2(obj["position"]["x"] - ax,
                                          obj["position"]["z"] - az)) % 360
        current = agent["rotation"]["y"] % 360
        diff = (desired - current + 540) % 360 - 180
        while abs(diff) > 5:
            step = max(-rot, min(rot, diff))
            self.c.step(action="RotateRight" if step > 0 else "RotateLeft",
                        degrees=abs(step))
            time.sleep(pause / 2)
            current = self.c.last_event.metadata["agent"]["rotation"]["y"] % 360
            diff = (desired - current + 540) % 360 - 180

        for _ in range(60):  # up to ~3.6m of closing distance
            obj = self._find_object(obj_type) or obj
            if obj.get("visible"):
                self.log(f"    [conduct] arrived — target visible "
                         f"({obj.get('distance', 0):.2f}m)")
                return True
            ev = self.c.step(action="MoveAhead", moveMagnitude=mag)
            if not ev.metadata["lastActionSuccess"]:
                obj = self._find_object(obj_type) or obj
                self.log(f"    [conduct] arrived — at target's edge "
                         f"({obj.get('distance', 0):.2f}m, "
                         f"visible={bool(obj.get('visible'))})")
                return True
            time.sleep(pause)

        obj = self._find_object(obj_type) or obj
        self.log(f"    [conduct] could not close final gap "
                 f"({obj.get('distance', 0):.2f}m) — falling back")
        return False
