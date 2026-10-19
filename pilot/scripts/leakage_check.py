"""Data-driven leakage check for benchmarks built from published cases (research design §9.24).

RareBench MME/LIRICAL/HMS and Phenopacket-store come from published case reports; only
Phenopacket-store lists its source PMIDs (excluded at collection). For every patient, find the
single graph document whose mentions for the gold disease cover the largest share of the patient's
concepts. A document covering >= 50% of a patient's concepts (and >= 5 of them) is flagged as a
possible source case; accuracy is then reported with and without flagged patients.
Thresholds fixed before looking at accuracy.

    python leakage_check.py --ranks pilot/cache/v2/eval_qwen3_v21/ranks/frozen.json --out X.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opendx_eval as O  # noqa: E402
import v2_ids  # noqa: E402

COVER, MIN_CONCEPTS = 0.5, 5
Q = """
UNWIND $gold AS g
MATCH (d:Disease {id: g})<-[:OF]-(m:Mention)-[:MENTIONS]->(c:Concept)
WHERE c.cui IN $ev
MATCH (m)-[:FROM]->(x:Doc)
WITH x, count(DISTINCT c) AS n
RETURN x.id AS doc, x.pmid AS pmid, n ORDER BY n DESC LIMIT 1
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", required=True, help="graph_dx output with ranks_lineage (same patients)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    a = ap.parse_args()
    O.RAW_GOLD = True
    sets = {**O.load_rarebench(), **O.load_phenopacket()}
    ranks = {r["bench"]: r["ranks_lineage"] for r in json.load(open(a.ranks))["results"]}
    drv = GraphDatabase.driver("bolt://localhost:7687", auth=tuple(a.auth.split("/", 1)))
    out = {}
    with drv.session() as s:
        for bench, pats in sets.items():
            flags, i = [], 0
            for gold_raw, ev in pats:
                gold = v2_ids.gold_keys(gold_raw)
                if not gold or not ev:
                    continue
                r = s.run(Q, gold=sorted(gold), ev=sorted(ev)).single()
                n = r["n"] if r else 0
                flags.append(n >= MIN_CONCEPTS and n / len(ev) >= COVER)
                i += 1
            rk = ranks.get(bench, [])
            if len(rk) != len(flags):
                print(f"{bench}: rank list mismatch, skipped", flush=True)
                continue

            def acc(sel, k):
                xs = [p is not None and p <= k for p, f in zip(rk, flags) if sel(f)]
                return round(100 * sum(xs) / max(1, len(xs)), 2), len(xs)

            out[bench] = {"patients": len(flags), "flagged": sum(flags),
                          **{f"all@{k}": acc(lambda f: True, k)[0] for k in (1, 10)},
                          **{f"clean@{k}": acc(lambda f: not f, k)[0] for k in (1, 10)},
                          **{f"flagged@{k}": acc(lambda f: f, k)[0] for k in (1, 10)}}
            print(bench, out[bench], flush=True)
    drv.close()
    json.dump({"cover": COVER, "min_concepts": MIN_CONCEPTS, "results": out}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
