"""
sr_rescore_oracle.py — recompute the oracle's answers on stored snapshots.

After changing oracle_grounding.py (or the ontology / annotation), run this
instead of re-running the simulator and the LLM. Only records that carry a
"snapshot" field can be re-scored; older logs without it are left untouched.

Usage:
  python methods/grace/sr_rescore_oracle.py --theory abstract methods/grace/sr_logs/abstract_batch_nl_neutral.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from grace import build_sr
from grace.symbolic_reasoning.horty import ReasoningModule

from safeagent_conflict_checker import SafeAgentConflictChecker
from oracle_grounding import OracleGrounding, CAPTURE_ORACLE

HERE = Path(__file__).parent


def build_oracle(theory):
    return build_sr(
        ReasoningModule,
        theory_path=HERE / "data" / f"{theory}_theory.json",
        conflict_checker=SafeAgentConflictChecker(),
        symbolic_grounding=OracleGrounding,
        representation="nl",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--theory", choices=["specific", "abstract"], required=True)
    args = ap.parse_args()

    sr = build_oracle(args.theory)

    # keep the latest record per state
    latest = {}
    for line in open(args.log):
        if line.strip():
            r = json.loads(line)
            latest[(r["split"], r["task"], r["step"])] = r

    rescored = skipped = 0
    for r in latest.values():
        if "snapshot" not in r:
            skipped += 1
            continue
        rules = sr.infer({"snapshot": r["snapshot"]})
        r["oracle_triggered"] = CAPTURE_ORACLE.get("triggered", [])
        r["oracle_rules"] = [(x.premise, x.conclusion) for x in rules]
        rescored += 1

    with open(args.log, "w") as f:
        for key in sorted(latest):
            f.write(json.dumps(latest[key]) + "\n")

    print(f"re-scored {rescored} states, skipped {skipped} without snapshot, wrote {args.log}")


if __name__ == "__main__":
    main()