"""Accuracy of the HPOA graph as IE-like wrong edges are added (run_noise_sensitivity.sh).

    python noise_analysis.py --out pilot/cache/v2/noise/analysis.json
    python noise_analysis.py --runs 'pilot/cache/v2/noise/composition/*/[rk]*_s*.json' \
        'pilot/cache/v2/noise/k*_r0_s*.json' --out pilot/cache/v2/noise/composition/analysis.json

Runs are keyed by their path under ROOT. A run whose build summary records true_edges also gets the
realized edge precision.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from temporal_analysis import hit, hpoa_citations, paired

BASE = "pilot/cache/v2/temporal/hpoa_v2024-12-12.json"
ROOT = "pilot/cache/v2/noise"
TAG = "v2024-12-12"


def summary(rows) -> dict:
    n = max(1, len(rows))
    return {"n": len(rows), "@1": round(100 * sum(hit(r, 1) for r in rows) / n, 1),
            "@10": round(100 * sum(hit(r, 10) for r in rows) / n, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+", default=[f"{ROOT}/[rk]*_s*.json"], help="globs of run outputs")
    a = ap.parse_args()
    cite = hpoa_citations(TAG)
    with open(BASE) as fh:
        runs = {"r0": json.load(fh)["rows"]}
    precision = {}
    for pat in a.runs:
        for f in sorted(glob.glob(pat)):
            if f.endswith("_summary.json"):
                continue
            g = os.path.relpath(f, ROOT)[:-5]
            with open(f) as fh:
                runs[g] = json.load(fh)["rows"]
            if os.path.exists(f[:-5] + "_summary.json"):
                with open(f[:-5] + "_summary.json") as fh:
                    sm = json.load(fh)
                if "true_edges" in sm:
                    precision[g] = {"edges": sm["edges"], "true_edges": sm["true_edges"],
                                    "precision": round(100 * sm["true_edges"] / sm["edges"], 1)}
    res = {}
    for g, rows in runs.items():
        for r in rows:
            r["cited"] = any(r["pmid"] in cite.get(o, ()) for o in r["omim"])
        res[g] = {**precision.get(g, {}), "all": summary(rows),
                  "past_cited": summary([r for r in rows if r["age_days"] < 0 and r["cited"]]),
                  "past_uncited": summary([r for r in rows if r["age_days"] < 0 and not r["cited"]]),
                  "future": summary([r for r in rows if r["age_days"] >= 0])}
        if g != "r0":
            res[g]["vs_r0@10"] = paired(runs["r0"], rows, 10)
            res[g]["vs_r0@1"] = paired(runs["r0"], rows, 1)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
