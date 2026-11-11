"""
sr_evidence.py — print the LLM's per-premise judgments and evidence for
chosen states, to understand false alarms and misses.

Usage (from SafeAgentBench root):
  python methods/grace/sr_evidence.py                 # the known suspicious states
  python methods/grace/sr_evidence.py --all-diffs     # every state where LLM != oracle
  python methods/grace/sr_evidence.py --task unsafe 0 # every state of one task
"""

import argparse
import json
from pathlib import Path

LOGS = Path(__file__).parent / "sr_logs"
FILES = ["specific_batch_nl_neutral.jsonl", "abstract_batch_nl_neutral.jsonl"]

SUSPICIOUS = {  # (split, task, step)
    ("safe", 0, 0),      # cabinet scene start state: false alarm in both theories
    ("unsafe", 2, 6),    # egg in closed microwave: miss in abstract theory
}


def show(rec, only_holds=True):
    L = set(rec["llm"].get("triggered", []))
    O = set(rec["oracle_triggered"])
    print(f"\n[{rec['_file']}] {rec['split']} {rec['task']} step {rec['step']} "
          f"{rec['action']!r}   llm={sorted(L)} oracle={sorted(O)}")
    for j in rec["llm"].get("judgments", []):
        if only_holds and not j["holds"] and j["name"] not in O:
            continue
        flag = "" if (j["holds"] == (j["name"] in O)) else "   <-- disagrees with oracle"
        print(f"   {j['name']}: holds={j['holds']}{flag}")
        print(f"      instantiation: {j['instantiation'][:110]}")
        print(f"      evidence:      {j['evidence'][:140]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-diffs", action="store_true")
    ap.add_argument("--task", nargs=2, metavar=("SPLIT", "IDX"))
    args = ap.parse_args()

    for f in FILES:
        p = LOGS / f
        if not p.exists():
            continue
        latest = {}
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                r["_file"] = f.split("_")[0]
                latest[(r["split"], r["task"], r["step"])] = r
        for key, r in sorted(latest.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            L = set(r["llm"].get("triggered", []))
            O = set(r["oracle_triggered"])
            if args.task:
                if key[0] == args.task[0] and key[1] == int(args.task[1]):
                    show(r, only_holds=False)
            elif args.all_diffs:
                if L != O:
                    show(r)
            elif key in SUSPICIOUS:
                show(r, only_holds=False)


if __name__ == "__main__":
    main()
