"""Which gold diseases account for Top-1 hits on one benchmark (forensic, test data read-only).

    python top1_forensic.py --bench RareBench-RAMEDIS --cfg pilot/cache/v2/eval_gemma_v21/tune.json --out X.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opendx_eval as O  # noqa: E402
import v2_ids  # noqa: E402
from graph_dx import GraphDx  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="RareBench-RAMEDIS")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    a = ap.parse_args()
    O.RAW_GOLD = True
    pats = O.load_rarebench()[a.bench]
    cfg = json.load(open(a.cfg))["selected"]
    dx = GraphDx("bolt://localhost:7687", a.auth, **cfg)
    uni = v2_ids.universe()
    hit, tot, top = Counter(), Counter(), Counter()
    for gold_raw, ev in pats:
        gold = v2_ids.gold_keys(gold_raw)
        if not gold or not ev:
            continue
        lin = dx.lineage(gold)
        r = dx.rank(ev, 10)
        g = min(gold)
        tot[g] += 1
        hit[g] += bool(r) and r[0] in lin
        if r:
            top[r[0]] += 1
    name = lambda k: uni.get(k, {}).get("name", k)  # noqa: E731
    res = {"bench": a.bench, "cfg": cfg, "patients": sum(tot.values()), "top1_hits": sum(hit.values()),
           "by_gold": [[g, name(g), tot[g], hit[g]] for g, _ in hit.most_common(15) if hit[g]],
           "most_frequent_top1": [[d, name(d), n] for d, n in top.most_common(10)]}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
