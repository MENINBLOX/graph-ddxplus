"""Auxiliary statistics of the design E confirmation (docs/research_design.md E4, not used for judgment).

Gold-disease cluster bootstrap CIs for every judged comparison, at Top-10 and Top-1: patients of one
disease are not independent draws, so this CI resamples whole diseases. Reads the finished run in
pilot/cache/v2/confirmE and needs CONFIRM_E_UNSEAL=1 to rebuild the per-patient gold keys.

    CONFIRM_E_UNSEAL=1 python pilot/scripts/confirm_e_aux.py
"""
from __future__ import annotations

import json
from pathlib import Path

import opendx_eval
import v2_ids
from paired_stats import compare

O = Path("pilot/cache/v2/confirmE")
PAIRS = [("S-D", "S-mix"), ("S-llm", "S-mix"), ("S-dict", "S-mix"), ("S-union", "S-mix"),
         ("S-bm25", "S-mix"), ("S-mix", "S-comb"), ("S-bm25", "S-comb")]


def main() -> None:
    if not (O / "DONE").exists():
        raise SystemExit("confirmation E has not finished")
    opendx_eval.RAW_GOLD = True
    sets = opendx_eval.load_confirm_e()
    clusters = {b: [tuple(sorted(v2_ids.gold_keys(g))) for g, ev in pats if v2_ids.gold_keys(g) and ev]
                for b, pats in sets.items()}
    out = []
    for a, b in PAIRS:
        ra, rb = (json.loads((O / f"{s}.json").read_text()) for s in (a, b))
        for k in (10, 1):
            for r in compare(ra, rb, k, clusters=clusters, require=sorted(sets)):
                if "ci_cluster" not in r:
                    raise SystemExit(f"{a} vs {b} {r['bench']}: {len(clusters[r['bench']])} gold keys for {r['n']} patients")
                out.append({"A": a, "B": b, "k": k, **{x: r[x] for x in ("bench", "n", "diff", "ci", "ci_cluster", "clusters")}})
                print(f"{b} - {a} @{k} {r['bench']:17} diff {r['diff']:+5.1f} patient CI [{r['ci'][0]:+5.1f}, {r['ci'][1]:+5.1f}]"
                      f"  disease CI [{r['ci_cluster'][0]:+5.1f}, {r['ci_cluster'][1]:+5.1f}] ({r['clusters']} diseases)")
    (O / "aux_cluster.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
