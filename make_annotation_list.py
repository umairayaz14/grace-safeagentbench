from pathlib import Path
import csv
from ai2thor.controller import Controller

types = set()
scenes = [f"FloorPlan{i}" for i in list(range(1, 31)) + list(range(201, 231))
          + list(range(301, 331)) + list(range(401, 431))]

c = Controller(scene=scenes[0])
for s in scenes:
    c.reset(scene=s)
    for x in c.last_event.metadata["objects"]:
        types.add(x["objectType"])
c.stop()

types |= {"AppleSliced", "BreadSliced", "EggCracked", "LettuceSliced", "PotatoSliced", "TomatoSliced"}

path = "methods/grace/data/annotation.csv"
cols = ["object_type", "dangerous_to_hold", "active_heat_source_when_on",
        "fragile", "heat_sensitive", "electronic", "notes"]

existing = {}
with open(path) as f:
    for row in csv.DictReader(f):
        existing[row["object_type"]] = row

with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for t in sorted(types):
        old = existing.get(t, {})
        new = {c: old.get(c, "") for c in cols}
        new["object_type"] = t
        w.writerow(new)

print(len(types), "types written")