"""Accuracy of the HPOA graph as IE-like wrong edges are added (run_noise_sensitivity.sh).

    python noise_analysis.py --out pilot/cache/v2/noise/analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json

from temporal_analysis import hit, hpoa_citations, paired

BASE = "pilot/cache/v2/temporal/hpoa_v2024-12-12.json"
TAG = "v2024-12-12"


def summary(rows) -> dict:
    n = max(1, len(rows))
    return {"n": len(rows), "@1": round(100 * sum(hit(r, 1) for r in rows) / n, 1),
            "@10": round(100 * sum(hit(r, 10) for r in rows) / n, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cite = hpoa_citations(TAG)
    runs = {"r0": json.load(open(BASE))["rows"]}
    for f in sorted(glob.glob("pilot/cache/v2/noise/[rk]*_s*.json")):
        if not f.endswith("_summary.json"):
            runs[f.rsplit("/", 1)[1][:-5]] = json.load(open(f))["rows"]
    res = {}
    for g, rows in runs.items():
        for r in rows:
            r["cited"] = any(r["pmid"] in cite.get(o, ()) for o in r["omim"])
        res[g] = {"all": summary(rows),
                  "past_cited": summary([r for r in rows if r["age_days"] < 0 and r["cited"]]),
                  "past_uncited": summary([r for r in rows if r["age_days"] < 0 and not r["cited"]]),
                  "future": summary([r for r in rows if r["age_days"] >= 0])}
        if g != "r0":
            res[g]["vs_r0@10"] = paired(runs["r0"], rows, 10)
            res[g]["vs_r0@1"] = paired(runs["r0"], rows, 1)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
