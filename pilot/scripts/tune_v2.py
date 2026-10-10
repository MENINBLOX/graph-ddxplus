"""Select the scoring configuration on DDXPlus validate only (research design §5, D12).

Coordinate search from a default configuration over a fixed grid, one parameter at a time, keeping
a change only if DDXPlus-validate lineage Top-1 improves (ties -> Top-3). Every evaluated
configuration is logged, so the selection is auditable. Test splits are never read here.

    python tune_v2.py --out pilot/cache/v2/tune_qwen3.json [--ddx_n 3000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opendx_eval as O  # noqa: E402
from graph_dx import GraphDx, evaluate  # noqa: E402

DEFAULT = {"alpha": 0.35, "mu": 2.0, "decay": 0.5, "unified_w": 1.0, "support_beta": 0.0, "offtopic_w": 1.0}
GRID = [("alpha", [0.2, 0.25, 0.35, 0.5]), ("mu", [0.0, 1.0, 2.0, 4.0]),
        ("decay", [0.0, 0.3, 0.5, 0.7]), ("unified_w", [0.0, 0.5, 1.0, 2.0]),
        ("support_beta", [0.0, 0.5, 1.0, 2.0]), ("offtopic_w", [1.0, 0.5, 0.25, 0.0])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ddx_n", type=int, default=3000)
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    a = ap.parse_args()
    O.RAW_GOLD = True
    pats = O.load_ddxplus("validate", a.ddx_n)["DDXPlus-validate"]
    trail = []

    def score(cfg):
        dx = GraphDx("bolt://localhost:7687", a.auth, **cfg)
        r = evaluate(dx, f"validate {cfg}", pats)
        dx.drv.close()
        trail.append({"cfg": dict(cfg), **r})
        return (r["lineage@1"], r["lineage@3"])

    best = dict(DEFAULT)
    best_s = score(best)
    for key, vals in GRID:
        for v in vals:
            if v == best[key]:
                continue
            cfg = {**best, key: v}
            s = score(cfg)
            if s > best_s:
                best, best_s = cfg, s
        print(f"after {key}: {best} -> {best_s}", flush=True)
    json.dump({"criterion": "DDXPlus validate lineage Top-1 (ties Top-3)", "selected": best,
               "selected_score": best_s, "trail": trail}, open(a.out, "w"), indent=1)
    print("SELECTED", best, best_s)


if __name__ == "__main__":
    main()
