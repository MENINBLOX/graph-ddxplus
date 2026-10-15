"""Window vs full scoring of unified-evidence diseases (graph_dx.py --full_unified), paired per benchmark.

Reads the runs of pilot/scripts/run_full_unified.sh and the matching window runs of run_graph_variants.sh.

    python full_unified_compare.py --out pilot/cache/v2/devD/window/summary.json
"""
from __future__ import annotations

import argparse
import json

from paired_stats import compare, sign_test

W = "pilot/cache/v2/devD/window"
V = "pilot/cache/v2/devD/variants"
# name: (A, B), both dev result files
PAIRS = {
    "frozen: window -> full": (f"{V}/dev_base_nm_ot0.0.json", f"{W}/base_nm_uf_ot0.0.json"),
    "v2umls: window -> full": (f"{V}/dev_v2umls_ot0.5.json", f"{W}/v2umls_uf_ot0.5.json"),
    "v2umls_ns: window -> full": (f"{V}/dev_v2umls_ns_ot0.5.json", f"{W}/v2umls_ns_uf_ot0.5.json"),
    "full: v2umls -> v2umls_ns": (f"{W}/v2umls_uf_ot0.5.json", f"{W}/v2umls_ns_uf_ot0.5.json"),
}


def macro(d: dict) -> list[float]:
    r = d["results"]
    return [round(sum(x["lineage@10"] for x in r) / len(r), 2), round(sum(x["lineage@1"] for x in r) / len(r), 2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for name, (fa, fb) in PAIRS.items():
        da, db = json.load(open(fa)), json.load(open(fb))
        row = {"A": fa, "B": fb, "macro_A": macro(da), "macro_B": macro(db)}
        for k in (1, 10):
            rows = compare(da, db, k)
            pos, neg = sum(r["diff"] > 0 for r in rows), sum(r["diff"] < 0 for r in rows)
            row[f"@{k}"] = {"rows": rows, "B_better": pos, "A_better": neg, "sign_p": sign_test(pos, neg)}
        out[name] = row
        print(name, row["macro_A"], "->", row["macro_B"],
              *(f"@{k} {row[f'@{k}']['B_better']}/{row[f'@{k}']['A_better']} p {row[f'@{k}']['sign_p']:.3g}" for k in (1, 10)))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
