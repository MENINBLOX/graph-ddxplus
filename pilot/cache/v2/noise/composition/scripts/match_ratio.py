"""--noise ratios that give each composition a set realized precision.

Candidates that are already edges are skipped, so the same --noise ratio yields different precision per
composition. Bisection on the ratio (2 decimals) at each keep, seed 0, HPOA v2024-12-12; no Neo4j.
Without --target the pure compositions are matched to the default 0.7 mix at the main grid's ratios
(2.33, 1); with --target every composition is set to that precision.

    .venv/bin/python pilot/cache/v2/noise/composition/scripts/match_ratio.py \
        pilot/cache/v2/noise/composition/matched_ratios.json
    .venv/bin/python pilot/cache/v2/noise/composition/scripts/match_ratio.py \
        pilot/cache/v2/noise/composition/ratios_p20.json --target 20 --keeps 0.3 0.15
"""
from __future__ import annotations

import argparse
import json
import random
import sys

sys.path.insert(0, "pilot/scripts")
import build_reference_graph as B
import curated_agreement as CA
import v2_ids


def precision(edge: dict, ratio: float, sib: float) -> float:
    return 100 * len(edge) / len(B.add_noise(edge, ratio, 0, sib))


def search(edge: dict, sib: float, target: float, lo: float, hi: float) -> dict:
    for _ in range(9):
        mid = (lo + hi) / 2
        if precision(edge, mid, sib) > target:
            lo = mid
        else:
            hi = mid
    cands = sorted({round(lo, 2), round(hi, 2), round(lo - 0.01, 2), round(hi + 0.01, 2)})
    best = min(cands, key=lambda r: abs(precision(edge, r, sib) - target))
    return {"target": round(target, 2), "ratio": best, "precision": round(precision(edge, best, sib), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--target", type=float, default=None, help="realized precision (%%) for every composition")
    ap.add_argument("--keeps", type=float, nargs="+", default=[1.0, 0.3, 0.15])
    ap.add_argument("--sibs", type=float, nargs="+", default=None)
    a = ap.parse_args()
    CA.HPOA = "data/external_kg/hpoa_history/v2024-12-12.hpoa"
    uni = dict(v2_ids.universe())
    with open(f"{B.V2}/exclude_nonhuman_diseases.json") as fh:
        for k in json.load(fh):
            uni.pop(k, None)
    full = B.pairs_hpoa(uni)
    res = {}
    for keep in a.keeps:
        edge = full
        if keep < 1:
            rng = random.Random(0)
            edge = {e: n for e, n in sorted(full.items()) if rng.random() < keep}
        if a.target is None:
            for r0 in (2.33, 1.0):
                target = precision(edge, r0, 0.7)
                for sib in a.sibs or (1.0, 0.0):
                    res[f"k{keep}_r{r0}_sib{sib}"] = search(edge, sib, target, 0.5 * r0, 2.0 * r0)
                    print(keep, r0, sib, res[f"k{keep}_r{r0}_sib{sib}"], flush=True)
        else:
            nominal = 100 / a.target - 1
            for sib in a.sibs or (0.7, 1.0, 0.0):
                res[f"k{keep}_p{a.target:g}_sib{sib}"] = search(edge, sib, a.target, 0.5 * nominal, 3.0 * nominal)
                print(keep, sib, res[f"k{keep}_p{a.target:g}_sib{sib}"], flush=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
