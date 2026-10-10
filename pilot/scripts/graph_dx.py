"""Open-set differential diagnosis on the Neo4j graph (research design D3, D8, D15).

The ranking is one Cypher query per patient: patient concepts are expanded up the concept hierarchy,
joined to diseases through HAS_FINDING, weighted by IDF and by literature support (n_docs ^ beta;
edges backed by more documents agree more often with independent curation, research design §9.16)
and down-weighted by gamma when no supporting article is about the disease (its title does not name it),
and length-normalized with coverage smoothing; a patient concept that IS a disease additionally supports that disease and its
descendants (unified evidence). Configuration is chosen on DDXPlus validate only (D12).

    python graph_dx.py --bench ddxplus-validate [--alpha 0.35 --mu 2.0 --decay 0.5]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opendx_eval as O  # noqa: E402
import v2_ids  # noqa: E402

RANK = """
UNWIND $ev AS cui
MATCH (c0:Concept {cui: cui})
MATCH p = (c0)-[:IS_A*0..1]->(c:Concept)
WITH c, max(CASE length(p) WHEN 0 THEN 1.0 ELSE $decay END) AS w
WHERE c.idf IS NOT NULL
MATCH (c)<-[e:HAS_FINDING]-(d:Disease)
WITH d, sum(w * c.idf * e.n_docs ^ $beta * CASE WHEN e.n_topic_docs > 0 THEN 1.0 ELSE $gamma END) AS s
WITH d, s / ((d.n_findings + $mu * (1.0 - toFloat(d.src_chars) / (d.src_chars + $smed)) * $lbar) ^ $alpha) AS score
RETURN d.id AS id, score
ORDER BY score DESC, id
LIMIT $k
"""

UNIFIED = """
UNWIND $ev AS cui
MATCH (:Concept {cui: cui})-[:IS_DISEASE]->(d0:Disease)
MATCH (d:Disease)-[:IS_A*0..1]->(d0)
RETURN d.id AS id, count(*) AS n
"""

LINEAGE = """
UNWIND $gold AS g
MATCH (d:Disease)-[:IS_A*0..3]->(:Disease {id: g})
RETURN collect(DISTINCT d.id) AS ids
"""


class GraphDx:
    def __init__(self, uri: str, auth: str, alpha: float, mu: float, decay: float, unified_w: float,
                 support_beta: float = 0.0, offtopic_w: float = 1.0):
        self.drv = GraphDatabase.driver(uri, auth=tuple(auth.split("/", 1)))
        self.alpha, self.mu, self.decay, self.unified_w = alpha, mu, decay, unified_w
        self.beta, self.gamma = support_beta, offtopic_w
        with self.drv.session() as s:
            r = s.run("MATCH (d:Disease) WHERE d.n_findings > 0 "
                      "RETURN avg(d.n_findings) AS lbar, percentileDisc(d.src_chars, 0.5) AS smed, "
                      "count(d) AS n").single()
            self.lbar, self.smed, self.n = r["lbar"], max(1, r["smed"]), r["n"]

    def rank(self, ev: set[str], k: int = 50) -> list[str]:
        with self.drv.session() as s:
            sc = {r["id"]: r["score"] for r in s.run(
                RANK, ev=sorted(ev), decay=self.decay, mu=self.mu, smed=self.smed, lbar=self.lbar,
                alpha=self.alpha, beta=self.beta, gamma=self.gamma, k=k * 4)}
            if self.unified_w:
                for r in s.run(UNIFIED, ev=sorted(ev)):
                    sc[r["id"]] = sc.get(r["id"], 0.0) + self.unified_w * r["n"]
        return sorted(sc, key=lambda d: (-sc[d], d))[:k]

    def lineage(self, gold: set[str]) -> set[str]:
        with self.drv.session() as s:
            return set(s.run(LINEAGE, gold=sorted(gold)).single()["ids"]) | gold


def evaluate(dx: GraphDx, name: str, pats, boot: int = 0) -> dict:
    import random
    hits = {r: {k: 0 for k in O.KS} for r in ("strict", "lineage")}
    m = 0
    ranks = {"strict": [], "lineage": []}           # per patient: 1-based position or None
    t0 = time.time()
    for gold_raw, ev in pats:
        gold = v2_ids.gold_keys(gold_raw)
        if not gold or not ev:
            continue
        m += 1
        ranked = dx.rank(ev, max(O.KS))
        lin = dx.lineage(gold)
        for rule, ok in (("strict", gold), ("lineage", lin)):
            pos = next((i + 1 for i, d in enumerate(ranked) if d in ok), None)
            ranks[rule].append(pos)
            for k in O.KS:
                hits[rule][k] += pos is not None and pos <= k
    res = {"bench": name, "patients": len(pats), "evaluated": m, "sec": round(time.time() - t0)}
    for rule in hits:
        for k in O.KS:
            res[f"{rule}@{k}"] = round(100 * hits[rule][k] / max(1, m), 2)
            if boot and m:
                rng = random.Random(0)
                xs = [p is not None and p <= k for p in ranks[rule]]
                dist = sorted(100 * sum(xs[rng.randrange(m)] for _ in range(m)) / m for _ in range(boot))
                res[f"{rule}@{k}_ci"] = [round(dist[int(0.025 * boot)], 1), round(dist[int(0.975 * boot) - 1], 1)]
    print(f"{name:20} n={m:6} | lineage " + " ".join(f"@{k} {res[f'lineage@{k}']:5.1f}" for k in O.KS)
          + " | strict " + " ".join(f"@{k} {res[f'strict@{k}']:5.1f}" for k in O.KS)
          + f" | {res['sec']}s", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="ddxplus-validate")
    ap.add_argument("--ddx_n", type=int, default=3000)
    ap.add_argument("--symcat_ppd", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--mu", type=float, default=2.0)
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--unified_w", type=float, default=1.0)
    ap.add_argument("--support_beta", type=float, default=0.0)
    ap.add_argument("--offtopic_w", type=float, default=1.0)
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--boot", type=int, default=0, help="patient-bootstrap resamples for 95% CI")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    O.RAW_GOLD = True
    dx = GraphDx(a.uri, a.auth, a.alpha, a.mu, a.decay, a.unified_w, a.support_beta, a.offtopic_w)
    print(f"graph: {dx.n} diseases with findings, mean profile {dx.lbar:.1f}, median source chars {dx.smed}",
          flush=True)
    sets = {}
    for b in a.bench.split(","):
        if b.startswith("ddxplus-"):
            sets.update(O.load_ddxplus(b.split("-", 1)[1], a.ddx_n))
        elif b == "rarebench":
            sets.update(O.load_rarebench())
        elif b == "pumch":
            sets.update(O.load_pumch())
        elif b == "phenopacket":
            sets.update(O.load_phenopacket())
        elif b == "symcat":
            sets.update(O.load_symcat(a.symcat_ppd))
    res = [evaluate(dx, k, v, a.boot) for k, v in sets.items()]
    if a.out:
        json.dump({"args": vars(a), "results": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
