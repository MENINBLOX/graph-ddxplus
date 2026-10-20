"""Recall and precision effects of the noise simulation under each wrong-edge composition.

Reads the runs of run_noise_sensitivity.sh in pilot/cache/v2/noise/composition/sib{0.7,1,0}/ (seed 0,
HPOA v2024-12-12) and the shared noise-free points (precision 100%). Two ways to set the ~36% and ~55%
levels: "nominal" keeps the main grid's --noise ratio (2.33, 1), "matched" uses matched_ratios.json so
the realized precision equals the default mix's at the same keep. "p20" sets every composition to 20%
precision at keep 0.3 and 0.15 (ratios_p20.json). Effects are paired over the same 10,603 cases with a
bootstrap 95% CI (temporal_analysis.paired). "realization" compares the main grid's keep-1 runs with
their reruns here: same seed, but those were built before add_noise sorted its input.

    .venv/bin/python pilot/cache/v2/noise/composition/scripts/composition_effects.py \
        pilot/cache/v2/noise/composition/effects.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "pilot/scripts")
from temporal_analysis import hit, paired

NOISE = "pilot/cache/v2/noise"
COMP = f"{NOISE}/composition"
BASE = "pilot/cache/v2/temporal/hpoa_v2024-12-12.json"
KEEPS = (1.0, 0.3, 0.15)
SIBS = {"sib0.7": 0.7, "sib1": 1.0, "sib0": 0.0}
NOMINAL = {"36": 2.33, "55": 1.0}


def name(keep: float, ratio: float) -> str:
    g = f"r{ratio:g}_s0"
    return g if keep == 1 else f"k{keep:g}_{g}"


def load(path: str) -> list[dict] | None:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)["rows"]


def point(rows: list[dict], summary_path: str | None) -> dict:
    n = len(rows)
    out = {"@1": round(100 * sum(hit(r, 1) for r in rows) / n, 1),
           "@10": round(100 * sum(hit(r, 10) for r in rows) / n, 1)}
    if summary_path and os.path.exists(summary_path):
        with open(summary_path) as fh:
            sm = json.load(fh)
        if "true_edges" in sm:
            out |= {"edges": sm["edges"], "precision": round(100 * sm["true_edges"] / sm["edges"], 1)}
    return out


def effect(a: list[dict], b: list[dict]) -> dict:
    return {"@1": paired(a, b, 1), "@10": paired(a, b, 10)}


def main() -> None:
    with open(f"{COMP}/matched_ratios.json") as fh:
        matched = json.load(fh)
    with open(f"{COMP}/ratios_p20.json") as fh:
        p20 = json.load(fh)
    clean = {1.0: load(BASE), 0.3: load(f"{NOISE}/k0.3_r0_s0.json"), 0.15: load(f"{NOISE}/k0.15_r0_s0.json")}
    res = {"clean": {f"{k:g}": point(rows, None) for k, rows in clean.items()}}
    for comp, sib in SIBS.items():
        for mode in ("nominal", "matched", "p20"):
            if mode == "matched" and sib == 0.7:
                continue
            for level, r0 in ({"20": None} if mode == "p20" else NOMINAL).items():
                runs, pts = {}, {}
                for keep in KEEPS:
                    if mode == "nominal":
                        ratio = r0
                    elif mode == "matched":
                        ratio = matched[f"k{keep}_r{r0}_sib{sib}"]["ratio"]
                    elif f"k{keep}_p20_sib{sib}" in p20:
                        ratio = p20[f"k{keep}_p20_sib{sib}"]["ratio"]
                    else:
                        continue
                    g = f"{COMP}/{comp}/{name(keep, ratio)}"
                    rows = load(f"{g}.json")
                    if rows is None:
                        continue
                    runs[keep] = rows
                    pts[f"{keep:g}"] = {"ratio": ratio, **point(rows, f"{g}_summary.json")}
                eff = {}
                if 0.15 in runs and 0.3 in runs:
                    eff["recall_k0.15_to_k0.3"] = effect(runs[0.15], runs[0.3])
                for keep, rows in runs.items():
                    eff[f"precision_to_100_at_k{keep:g}"] = effect(rows, clean[keep])
                res[f"{comp}/{mode}/{level}"] = {"points": pts, "effects": eff}
                print(comp, mode, level, json.dumps(pts), flush=True)
    res["realization"] = {g: {"main_grid": point(load(f"{NOISE}/{g}.json"), None),
                              "rerun": point(load(f"{COMP}/sib0.7/{g}.json"), None)}
                          for g in ("r2.33_s0", "r1_s0", "k0.3_r2.33_s0", "k0.15_r2.33_s0")}
    with open(sys.argv[1], "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
