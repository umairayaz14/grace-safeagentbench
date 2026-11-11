"""
Interactive AI2-THOR explorer — drive the agent with your keyboard.

Usage (from SafeAgentBench repo root, safeagent-env active):
    python explore.py                  # default scene FloorPlan401 (the candle bathroom)
    python explore.py FloorPlan1       # any scene name, e.g. kitchens FloorPlan1-30,
                                       # living rooms 201-230, bedrooms 301-330, bathrooms 401-430

Controls (press keys in the TERMINAL window, not the Unity window):
    arrow keys / WASD   move forward-back, turn left-right
    i / k               look up / down
    e / r               crouch / stand
    o                   list all currently visible objects
    p                   pick up nearest visible pickupable object
    x                   drop held object
    t                   toggle nearest visible on/off-able object (lamp, candle...)
    n                   open nearest openable object
    m                   close nearest openable object
    q                   quit
"""

import sys
import termios
import tty

from ai2thor.controller import Controller

SCENE = sys.argv[1] if len(sys.argv) > 1 else "FloorPlan401"


def getch():
    """Read a single keypress (handles arrow-key escape sequences)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # escape sequence (arrow keys)
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def visible_objects(event):
    return [o for o in event.metadata["objects"] if o["visible"]]


def nearest(event, predicate):
    objs = [o for o in visible_objects(event) if predicate(o)]
    if not objs:
        return None
    return min(objs, key=lambda o: o["distance"])


def report(event, note=""):
    objs = visible_objects(event)
    names = ", ".join(sorted({o["objectType"] for o in objs})) or "(nothing)"
    ok = event.metadata["lastActionSuccess"]
    err = event.metadata.get("errorMessage", "")
    status = "ok" if ok else f"FAILED {err}"
    print(f"[{status}] {note}  visible: {names}")


def main():
    print(f"Loading {SCENE} ... (Unity window will open; keep THIS terminal focused)")
    c = Controller(scene=SCENE, gridSize=0.25, visibilityDistance=1.5)
    event = c.step(action="Pass")
    report(event, "ready")
    print(__doc__)

    while True:
        ch = getch()

        if ch in ("q", "\x03"):  # q or Ctrl-C
            break
        elif ch in ("\x1b[A", "w"):
            event = c.step(action="MoveAhead")
        elif ch in ("\x1b[B", "s"):
            event = c.step(action="MoveBack")
        elif ch in ("\x1b[C", "d"):
            event = c.step(action="RotateRight", degrees=30)
        elif ch in ("\x1b[D", "a"):
            event = c.step(action="RotateLeft", degrees=30)
        elif ch == "i":
            event = c.step(action="LookUp", degrees=15)
        elif ch == "k":
            event = c.step(action="LookDown", degrees=15)
        elif ch == "e":
            event = c.step(action="Crouch")
        elif ch == "r":
            event = c.step(action="Stand")
        elif ch == "o":
            event = c.step(action="Pass")
            print("--- visible objects ---")
            for o in visible_objects(event):
                flags = []
                if o["pickupable"]:
                    flags.append("pickupable")
                if o["openable"]:
                    flags.append("openable")
                if o["toggleable"]:
                    flags.append("toggleable")
                if o.get("isToggled"):
                    flags.append("ON")
                if o.get("isOpen"):
                    flags.append("open")
                print(f"  {o['objectType']:20s} d={o['distance']:.2f}  {' '.join(flags)}")
            continue
        elif ch == "p":
            obj = nearest(c.last_event, lambda o: o["pickupable"])
            if obj is None:
                print("nothing pickupable in view")
                continue
            event = c.step(action="PickupObject", objectId=obj["objectId"])
            report(event, f"pickup {obj['objectType']}")
            continue
        elif ch == "x":
            event = c.step(action="DropHandObject")
            report(event, "drop")
            continue
        elif ch == "t":
            obj = nearest(c.last_event, lambda o: o["toggleable"])
            if obj is None:
                print("nothing toggleable in view")
                continue
            action = "ToggleObjectOff" if obj.get("isToggled") else "ToggleObjectOn"
            event = c.step(action=action, objectId=obj["objectId"])
            report(event, f"{action} {obj['objectType']}")
            continue
        elif ch == "n":
            obj = nearest(c.last_event, lambda o: o["openable"] and not o.get("isOpen"))
            if obj is None:
                print("nothing openable in view")
                continue
            event = c.step(action="OpenObject", objectId=obj["objectId"])
            report(event, f"open {obj['objectType']}")
            continue
        elif ch == "m":
            obj = nearest(c.last_event, lambda o: o["openable"] and o.get("isOpen"))
            if obj is None:
                print("nothing open in view")
                continue
            event = c.step(action="CloseObject", objectId=obj["objectId"])
            report(event, f"close {obj['objectType']}")
            continue
        else:
            continue

        report(event)

    c.stop()
    print("bye")


if __name__ == "__main__":
    main()
