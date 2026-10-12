"""E-H3: does PPR favour high-degree (hub) diseases? (research design D10)

For two systems on the same graph: accuracy by gold-disease degree tertile (from saved per-patient
ranks), and the degree of each system's top-1 prediction.

    python hub_analysis.py --graph G.npz --onehop dev_gemma.json --ppr dev_ppr_gemma.json \
        --onehop_cfg tune_gemma.json --ppr_cfg tune_ppr_gemma.json --uri bolt://localhost:7688 --out X.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opendx_eval as O  # noqa: E402
import v2_ids  # noqa: E402
from graph_dx import GraphDx  # noqa: E402
from ppr_dx import PPR  # noqa: E402
from tune_dev import dev_sets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("graph", "onehop", "ppr", "onehop_cfg", "ppr_cfg", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--uri", default="bolt://localhost:7688")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--sample", type=int, default=1000)
    a = ap.parse_args()
    O.RAW_GOLD = True
    g = np.load(a.graph)
    deg = np.bincount(g["hf_d"], minlength=len(g["diseases"]))
    ddeg = dict(zip(g["diseases"], deg.tolist()))
    sets = dev_sets()
    r1 = {r["bench"]: r["ranks_lineage"] for r in json.load(open(a.onehop))["results"]}
    r2 = {r["bench"]: r["ranks_lineage"] for r in json.load(open(a.ppr))["results"]}
    # gold-degree tertiles over all evaluated dev patients
    golds = {}
    for b, pats in sets.items():
        golds[b] = [max(ddeg.get(k, 0) for k in v2_ids.gold_keys(gr)) for gr, ev in pats if v2_ids.gold_keys(gr) and ev]
    alld = sorted(d for b in golds for d in golds[b])
    t1, t2 = alld[len(alld) // 3], alld[2 * len(alld) // 3]
    tert = lambda d: 0 if d <= t1 else (1 if d <= t2 else 2)  # noqa: E731
    acc = {s: [[0, 0] for _ in range(3)] for s in ("onehop", "ppr")}
    for b in golds:
        for d, p1, p2 in zip(golds[b], r1[b], r2[b]):
            t = tert(d)
            for s, p in (("onehop", p1), ("ppr", p2)):
                acc[s][t][0] += p is not None and p <= 10
                acc[s][t][1] += 1
    res = {"gold_degree_tertile_cuts": [t1, t2],
           "top10_by_gold_degree_tertile": {s: [round(100 * x / max(1, n), 2) for x, n in v] for s, v in acc.items()},
           "patients_per_tertile": [n for _, n in acc["onehop"]]}
    # degree of top-1 predictions on a sample per benchmark
    one = GraphDx(a.uri, a.auth, **json.load(open(a.onehop_cfg))["selected"])
    ppr = PPR(a.graph)
    pc = json.load(open(a.ppr_cfg))["selected"]
    top1 = {"onehop": [], "ppr": []}
    for b, pats in sets.items():
        evs = [ev for gr, ev in pats if v2_ids.gold_keys(gr) and ev][: a.sample]
        top1["ppr"] += [ddeg.get(r[0], 0) for r in ppr.rank_batch(evs, pc, k=1) if r]
        top1["onehop"] += [ddeg.get(r[0], 0) for r in (one.rank(ev, 1) for ev in evs) if r]
    res["top1_prediction_degree_median"] = {s: statistics.median(v) for s, v in top1.items()}
    res["disease_degree_median_all_with_findings"] = statistics.median([d for d in deg.tolist() if d > 0])
    res["gold_degree_median"] = statistics.median(alld)
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
