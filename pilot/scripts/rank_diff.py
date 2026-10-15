"""Why does one graph lose a Top-k hit where another keeps it? Patient-level contrast of two loaded graphs.

For each patient of one benchmark the gold's lineage position is taken in graph A and graph B (same
frozen gemma 1-hop configuration). For patients A ranks within k and B does not, it records the disease
that displaced the gold (B's highest-ranked Top-k disease that A did not rank above the gold) and, in
both graphs, the per-concept score contributions of that disease and of the gold, their profile sizes,
the length-normalization denominator of the RANK query and the unified-evidence term
(total = sum / denom + unified_w * n_unified, as GraphDx.rank scores).

    python rank_diff.py --bench RareBench-RAMEDIS --a bolt://localhost:7687 --b bolt://localhost:7688 --out X.json [--k 10] [--offtopic_a W] [--offtopic_b W]
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
from tune_dev import dev_sets  # noqa: E402

FROZEN = "pilot/cache/v2/devD/frozen_config_D.json"
CONTRIB = """
UNWIND $ev AS cui
MATCH (c0:Concept {cui: cui})
MATCH p = (c0)-[:IS_A*0..1]->(c:Concept)
WITH c, max(CASE length(p) WHEN 0 THEN 1.0 ELSE $decay END) AS w
WHERE c.idf IS NOT NULL
MATCH (c)<-[e:HAS_FINDING]-(d:Disease {id: $d})
RETURN c.cui AS cui, w * c.idf * e.n_docs ^ $beta * CASE WHEN e.n_topic_docs > 0 THEN 1.0 ELSE $gamma END AS s
"""
UNIFIED_ONE = """
UNWIND $ev AS cui
MATCH (:Concept {cui: cui})-[:IS_DISEASE]->(d0:Disease)
MATCH (d:Disease {id: $d})-[:IS_A*0..1]->(d0)
RETURN count(*) AS n
"""


def contrib(dx: GraphDx, ev: set[str], d: str) -> dict:
    with dx.drv.session() as s:
        rows = list(s.run(CONTRIB, ev=sorted(ev), d=d, decay=dx.decay, beta=dx.beta, gamma=dx.gamma))
        x = s.run("MATCH (x:Disease {id: $d}) RETURN x.n_findings AS nf, x.src_chars AS sc, x.name AS name", d=d).single()
        n_uni = s.run(UNIFIED_ONE, ev=sorted(ev), d=d).single()["n"] if dx.unified_w else 0
    nf, sc = x["nf"] or 0, x["sc"] or 0
    denom = (nf + dx.mu * (1.0 - sc / (sc + dx.smed)) * dx.lbar) ** dx.alpha
    total = sum(r["s"] for r in rows)
    return {"id": d, "name": x["name"], "n_findings": nf, "src_chars": sc, "sum": round(total, 2),
            "denom": round(denom, 2), "score": round(total / denom, 3), "n_unified": n_uni,
            "total": round(total / denom + dx.unified_w * n_uni, 3),
            "concepts": {r["cui"]: round(r["s"], 2) for r in sorted(rows, key=lambda r: -r["s"])}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--offtopic_a", type=float, default=None)
    ap.add_argument("--offtopic_b", type=float, default=None)
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(FROZEN))["gemma_1hop"]
    A, B = (GraphDx(uri, a.auth, **{**cfg, **({"offtopic_w": ot} if ot is not None else {})})
            for uri, ot in ((a.a, a.offtopic_a), (a.b, a.offtopic_b)))
    O.RAW_GOLD = True
    pats = dev_sets()[a.bench]
    depth = max(10, a.k)
    lost, won, top_new = [], 0, Counter()
    for gold_raw, ev in pats:
        gold = v2_ids.gold_keys(gold_raw)
        if not gold or not ev:
            continue
        lin = A.lineage(gold)
        ra, rb = A.rank(ev, depth), B.rank(ev, depth)
        pa = next((i + 1 for i, d in enumerate(ra) if d in lin), None)
        pb = next((i + 1 for i, d in enumerate(rb) if d in lin), None)
        hit_a, hit_b = pa is not None and pa <= a.k, pb is not None and pb <= a.k
        if hit_b and not hit_a:
            won += 1
        if hit_a and not hit_b:
            g = ra[pa - 1]
            top = next((d for d in rb[:a.k] if d not in lin and d not in ra[:pa - 1]), rb[0])
            top_new[top] += 1
            lost.append({"gold": g, "pos_a": pa, "pos_b": pb, "new_top": top, "ev": sorted(ev),
                         "gold_a": contrib(A, ev, g), "gold_b": contrib(B, ev, g),
                         "top_a": contrib(A, ev, top), "top_b": contrib(B, ev, top)})
    names = {x["new_top"]: x["top_b"]["name"] for x in lost}
    out = {"bench": a.bench, "k": a.k, "lost": len(lost), "won": won,
           "new_top_diseases": [(d, names[d], n) for d, n in top_new.most_common(20)], "cases": lost}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"lost {len(lost)}, won {won}")
    for d, nm, n in out["new_top_diseases"]:
        print(f"  {n:4d}  {d}  {nm}")


if __name__ == "__main__":
    main()
