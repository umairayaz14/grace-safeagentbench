"""State rendering for SR evaluation.

snapshot(event)      -> compact dict of the normatively relevant state
render_raw(snap)     -> JSON string (raw metadata variant)
render_nl(snap)      -> natural-language description (NL variant)

The snapshot keeps: the held object with its flags, every object within
NEAR_M of the agent with its flags and container, and pairwise adjacency
between lit objects and nearby objects (for 'next to' premises).
"""

import json
import math

NEAR_M = 2.0
ADJACENT_M = 0.6
BIG = {"Floor", "CounterTop", "Cabinet", "Drawer", "Shelf", "DiningTable",
       "SideTable", "Desk", "Bed", "Sofa", "Wall", "Window", "Curtains"}
FLAGS = ("isToggled", "isOpen", "isPickedUp", "isFilledWithLiquid",
         "isBroken", "isDirty", "isSliced", "isCooked")


def _pos(o):
    p = o["position"]
    return p["x"], p["z"]


def snapshot(event) -> dict:
    objs = event.metadata["objects"]
    ax, az = (event.metadata["agent"]["position"]["x"],
              event.metadata["agent"]["position"]["z"])
    held, nearby = None, []
    for o in objs:
        entry = {
            "type": o["objectType"],
            "flags": [f for f in FLAGS if o.get(f)],
            "inside": [r.split("|")[0] for r in (o.get("parentReceptacles") or [])],
            "distance": round(o.get("distance", 99), 2),
            "toggleable": bool(o.get("toggleable")),
            "openable": bool(o.get("openable")),
        }
        if o.get("isPickedUp"):
            held = entry
        elif o.get("distance", 99) <= NEAR_M:
            nearby.append(entry)

    # proximity between any two nearby objects (not only switched-on ones)
    adjacent = []
    near_objs = [o for o in objs if o.get("distance", 99) <= NEAR_M and not o.get("isPickedUp")]
    held_obj = next((o for o in objs if o.get("isPickedUp")), None)
    candidates = near_objs + ([held_obj] if held_obj else [])
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if a["objectType"] in BIG or b["objectType"] in BIG:
                continue
            d = math.hypot(*(x - y for x, y in zip(_pos(a), _pos(b))))
            if d <= ADJACENT_M:
                adjacent.append((a["objectType"], b["objectType"], round(d, 2)))

    return {"held": held, "nearby": sorted(nearby, key=lambda e: e["distance"]),
            "adjacent": adjacent}


def render_raw(snap: dict) -> str:
    return json.dumps(snap, indent=None)


def _flags_nl(flags, toggleable=False, openable=False):
    m = {"isToggled": "switched on", "isOpen": "open", "isFilledWithLiquid":
         "filled with liquid", "isBroken": "broken", "isDirty": "dirty",
         "isSliced": "sliced", "isCooked": "cooked"}
    parts = [m[f] for f in flags if f in m]
    # explicit negatives: an unlit candle / off stove must read as OFF,
    # a closed microwave as CLOSED, never as unstated
    if toggleable and "isToggled" not in flags:
        parts.append("switched off")
    if openable and "isOpen" not in flags:
        parts.append("closed")
    return ", ".join(parts)


def render_nl(snap: dict) -> str:
    lines = []
    h = snap["held"]
    if h:
        fl = _flags_nl(h["flags"], h.get("toggleable"), h.get("openable"))
        lines.append(f"The agent is holding a {h['type']}"
                     + (f" ({fl})." if fl else "."))
    else:
        lines.append("The agent is not holding anything.")
    for e in snap["nearby"]:
        fl = _flags_nl(e["flags"], e.get("toggleable"), e.get("openable"))
        where = f", inside/on {e['inside'][0]}" if e["inside"] else ""
        lines.append(f"Nearby ({e['distance']} m): {e['type']}"
                     + (f" ({fl})" if fl else "") + where + ".")
    if snap["adjacent"]:
        for a, b, d in snap["adjacent"]:
            lines.append(f"The {a} is right next to the {b} ({d} m).")
    else:
        lines.append("No two objects are right next to each other.")
    return "\n".join(lines)
