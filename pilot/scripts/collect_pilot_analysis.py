"""Compare the collection-pilot graphs (run_collect_pilot.sh) on dated phenopacket cases.

Only the 50 pilot diseases received extra articles, so the collection effect is read on cases whose
gold is a pilot disease; the remaining cases show the side effect on everyone else (the enriched
diseases also compete for their rankings). Splitting/relinking changes every disease.

    python collect_pilot_analysis.py --out pilot/cache/v2/collect_pretest/analysis.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from temporal_analysis import hit, paired

C = "pilot/cache/v2/collect_pretest"
CONTRASTS = [("base", "pilot"), ("base_ot1", "pilot_ot1"), ("base", "base_split"), ("base", "pilot_split"),
             ("base_ot1", "pilot_split_ot1"), ("base", "base_ot1"), ("pilot", "pilot_tables"),
             ("base", "pilot_tables_split"), ("base_ot1", "pilot_tables_split_ot1")]


def macro(rows, k) -> float:
    by = defaultdict(list)
    for r in rows:
        by[tuple(r["omim"])].append(hit(r, k))
    return round(100 * sum(sum(v) / len(v) for v in by.values()) / max(1, len(by)), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pilot = set(json.load(open(f"{C}/pilot_omims.json")))
    runs = {}
    for g in ("base", "pilot", "base_split", "pilot_split", "pilot_tables", "pilot_tables_split"):
        for s in ("", "_ot1"):
            try:
                runs[g + s] = json.load(open(f"{C}/eval_{g}{s}.json"))["rows"]
            except FileNotFoundError:
                pass
    res = {"runs": {}, "contrasts": []}
    for g, rows in runs.items():
        res["runs"][g] = {}
        for grp, sel in (("pilot_gold", [r for r in rows if set(r["omim"]) & pilot]),
                         ("other", [r for r in rows if not set(r["omim"]) & pilot]), ("all", rows)):
            res["runs"][g][grp] = {"n": len(sel), "diseases": len({tuple(r["omim"]) for r in sel}),
                                   "@1": round(100 * sum(hit(r, 1) for r in sel) / max(1, len(sel)), 1),
                                   "@10": round(100 * sum(hit(r, 10) for r in sel) / max(1, len(sel)), 1),
                                   "macro@10": macro(sel, 10)}
    for x, y in CONTRASTS:
        if x in runs and y in runs:
            for grp in ("pilot_gold", "other", "all"):
                f = (lambda r: set(r["omim"]) & pilot) if grp == "pilot_gold" else \
                    (lambda r: not set(r["omim"]) & pilot) if grp == "other" else (lambda r: True)
                for k in (1, 10):
                    res["contrasts"].append({"sys_a": x, "sys_b": y, "cases": grp, "k": k,
                                             **paired([r for r in runs[x] if f(r)], [r for r in runs[y] if f(r)], k)})
    json.dump(res, open(a.out, "w"), indent=1)
    for g, v in res["runs"].items():
        print(g, {k: (x["@1"], x["@10"], x["macro@10"], x["n"]) for k, x in v.items()})
    for c in res["contrasts"]:
        print(c["sys_a"], "->", c["sys_b"], c["cases"], f"@{c['k']}", c.get("diff_B_minus_A"), c.get("ci"))


if __name__ == "__main__":
    main()
