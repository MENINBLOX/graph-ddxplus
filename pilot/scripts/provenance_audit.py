"""Provenance audit sheet (research design RQ3).

Samples HAS_FINDING edges uniformly at random (seeded) and, for each, one supporting Mention with its
sentence and source document. A human marks whether the sentence states that the disease has the
finding (supports / does not support / wrong concept). Also reports provenance coverage: the share
of HAS_FINDING edges backed by at least one Mention (by construction 100%), and the share of
Mentions whose source is a PubMed-indexed article (has a PMID) vs a book chapter.

    python provenance_audit.py --n 200 --out pilot/cache/v2/provenance_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import os

from neo4j import GraphDatabase

SAMPLE = """
MATCH (d:Disease)-[e:HAS_FINDING]->(c:Concept)
WITH d, c, e, rand() AS r ORDER BY r LIMIT $n
MATCH (m:Mention)-[:OF]->(d), (m)-[:MENTIONS]->(c), (m)-[:FROM]->(x:Doc)
WITH d, c, e, m, x ORDER BY m.phrase
WITH d, c, e, head(collect({m: m, x: x})) AS ev
RETURN d.id AS disease_id, d.name AS disease, c.cui AS cui, e.n_docs AS n_docs,
       ev.m.phrase AS phrase, ev.m.sentence AS sentence, ev.x.id AS doc_id, ev.x.pmid AS pmid,
       ev.x.title AS title
"""
COVERAGE = """
MATCH (:Disease)-[e:HAS_FINDING]->() WITH count(e) AS edges
MATCH (m:Mention)-[:FROM]->(x:Doc)
RETURN edges, count(m) AS mentions, sum(CASE WHEN x.pmid <> '' THEN 1 ELSE 0 END) AS with_pmid
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    a = ap.parse_args()
    drv = GraphDatabase.driver("bolt://localhost:7687", auth=tuple(a.auth.split("/", 1)))
    with drv.session() as s:
        s.run("CALL apoc.util.sleep(0)").consume()
        cov = s.run(COVERAGE).single()
        # Cypher rand() is not seedable; sample by a seeded hash of the edge key instead
        rows = list(s.run(SAMPLE.replace("rand() AS r", f"apoc.util.md5([d.id, c.cui, '{a.seed}']) AS r"),
                          n=a.n))
    drv.close()
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["disease_id", "disease", "cui", "n_docs", "phrase", "sentence", "doc_id", "pmid",
                    "title", "judgement (supports / not_supported / wrong_concept)"])
        for r in rows:
            w.writerow([r[k] for k in ("disease_id", "disease", "cui", "n_docs", "phrase", "sentence",
                                       "doc_id", "pmid", "title")] + [""])
    print(f"edges {cov['edges']}  mentions {cov['mentions']}  mentions with PMID "
          f"{100 * cov['with_pmid'] / max(1, cov['mentions']):.1f}%  -> {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
