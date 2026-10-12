"""Design D configuration selection on the development sets (research design D3).

Coordinate search over the same grid as tune_v2, but the criterion is the macro-average lineage
Top-10 over the nine development test sets (ties: macro Top-1). One configuration for all
benchmarks. The sealed confirmation sets are never loaded here.

    python tune_dev.py --uri bolt://localhost:7687 --out pilot/cache/v2/devD/tune_qwen3.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opendx_eval as O  # noqa: E402
from graph_dx import GraphDx, evaluate  # noqa: E402
from tune_v2 import DEFAULT, GRID  # noqa: E402

DEV = ("ddxplus-test", "rarebench", "pumch", "phenopacket", "symcat")


def dev_sets(ddx_n: int = 3000) -> dict:
    sets = {}
    sets.update(O.load_ddxplus("test", ddx_n))
    sets.update(O.load_rarebench())
    sets.update(O.load_pumch())
    sets.update(O.load_phenopacket())
    sets.update(O.load_symcat(10))
    return sets


def macro(results: list[dict]) -> tuple[float, float]:
    n = len(results)
    return (round(sum(r["lineage@10"] for r in results) / n, 3),
            round(sum(r["lineage@1"] for r in results) / n, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--start", default="", help="JSON config to start from (default: tune_v2 DEFAULT)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    O.RAW_GOLD = True
    sets = dev_sets()
    trail = []

    def score(cfg):
        dx = GraphDx(a.uri, a.auth, **cfg)
        res = [evaluate(dx, k, v) for k, v in sets.items()]
        dx.drv.close()
        m = macro(res)
        trail.append({"cfg": dict(cfg), "macro_top10": m[0], "macro_top1": m[1],
                      "per_bench": {r["bench"]: [r["lineage@1"], r["lineage@10"]] for r in res}})
        print(f"cfg {cfg} -> macro@10 {m[0]} macro@1 {m[1]}", flush=True)
        return m

    best = json.loads(a.start) if a.start else dict(DEFAULT)
    best_s = score(best)
    for _ in range(2):                                   # two coordinate passes
        changed = False
        for key, vals in GRID:
            for v in vals:
                if v == best[key]:
                    continue
                cfg = {**best, key: v}
                s = score(cfg)
                if s > best_s:
                    best, best_s, changed = cfg, s, True
            print(f"after {key}: {best} -> {best_s}", flush=True)
        if not changed:
            break
    json.dump({"criterion": "dev macro lineage Top-10 (ties Top-1), 9 development test sets",
               "selected": best, "selected_score": best_s, "trail": trail}, open(a.out, "w"), indent=1)
    print("SELECTED", best, best_s, flush=True)


if __name__ == "__main__":
    main()
