"""Blind clinician audit sheet for the S-mix graph (design E3, provenance of the main system).

Samples HAS_FINDING edges of the Mention build of S-mix (v3llmd_ns) uniformly at random and, for each,
one supporting Mention with its sentence and source document, for the same three-way judgement as the
v2.1 audit (provenance_audit.py, research design 9.17 and 9.25). The blind sheet keeps that sheet's
columns without n_docs and with a row number. Everything that would tell a judge how an edge was made or
how strongly it is supported (origin of the mention, the LLM span of a re-linked item, n_docs,
n_topic_docs) goes to a separate key file, joined on the row number after judging.

Origin of a Mention: dict_relink when it carries a span (llm_dict_relink.py named it by a UMLS string
found inside the LLM phrase; the phrase is then that string), otherwise llm_table (phrase table link).

Sampling is done in Python on the edges ordered by (disease id, CUI), not with Cypher rand(), so the
same build gives the same rows: a seeded index into that order is mapped to a disease by cumulative
out-degree, then to a CUI by its position among the disease's sorted CUIs.

    python pilot/scripts/smix_audit_sample.py --n 100 --seed 1 \
        --blind docs/audit/smix_audit_blind.csv --key docs/audit/smix_audit_key.csv
"""
from __future__ import annotations

import argparse
import bisect
import csv
import itertools
import json
import math
import os
import random
import sys
from collections import Counter

from neo4j import GraphDatabase

DEGREES = """
MATCH (d:Disease)
RETURN d.id AS id, COUNT { (d)-[:HAS_FINDING]->() } AS n
"""
CUIS = "MATCH (:Disease {id: $d})-[:HAS_FINDING]->(c:Concept) RETURN c.cui AS cui"
EDGE = """
MATCH (d:Disease {id: $d})-[e:HAS_FINDING]->(c:Concept {cui: $c})
OPTIONAL MATCH (d)<-[:OF]-(m:Mention)-[:MENTIONS]->(c)
OPTIONAL MATCH (m)-[:FROM]->(x:Doc)
RETURN d.name AS disease, e.n_docs AS n_docs, e.n_topic_docs AS n_topic_docs, e.n_mentions AS n_mentions,
       collect(CASE WHEN m IS NULL THEN NULL ELSE {phrase: m.phrase, sentence: m.sentence, span: m.span,
               topic: m.topic, doc_id: x.id, pmid: x.pmid, title: x.title} END) AS mentions
"""
POPULATION = "MATCH (m:Mention) RETURN m.span IS NOT NULL AS relinked, count(*) AS n"
BLIND_COLS = ["row", "disease_id", "disease", "cui", "phrase", "sentence", "doc_id", "pmid", "title",
              "judgement (supports / not_supported / wrong_concept)"]
KEY_COLS = ["row", "disease_id", "cui", "mention_origin", "mention_span", "mention_topic", "edge_origin",
            "edge_mentions_llm_table", "edge_mentions_dict_relink", "n_mentions", "n_docs", "n_topic_docs"]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if not n:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def origin(m: dict) -> str:
    return "dict_relink" if m["span"] is not None else "llm_table"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--uri", default="bolt://localhost:7689")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--frozen", default="pilot/cache/v2/devD/frozen_config_E.json",
                    help="the loaded graph must have this file's S-mix edge and disease counts")
    ap.add_argument("--blind", required=True)
    ap.add_argument("--key", required=True)
    a = ap.parse_args()
    with open(a.frozen) as f:
        frozen = json.load(f)["graph_systems"]["S-mix"]

    drv = GraphDatabase.driver(a.uri, auth=tuple(a.auth.split("/", 1)))
    with drv.session() as s:
        deg = sorted((r["id"], r["n"]) for r in s.run(DEGREES))
        total = sum(n for _, n in deg)
        if total != frozen["edges"] or len(deg) != frozen["diseases"]:
            sys.exit(f"{a.uri} holds {total} HAS_FINDING edges / {len(deg)} diseases, frozen S-mix has "
                     f"{frozen['edges']} / {frozen['diseases']}: not the S-mix build")
        starts = list(itertools.accumulate((n for _, n in deg), initial=0))
        picks = random.Random(a.seed).sample(range(total), a.n)
        cuis: dict[str, list[str]] = {}
        rows = []
        for i in picks:
            j = bisect.bisect_right(starts, i) - 1
            d = deg[j][0]
            if d not in cuis:
                cuis[d] = sorted(r["cui"] for r in s.run(CUIS, d=d))
            c = cuis[d][i - starts[j]]
            r = s.run(EDGE, d=d, c=c).single()
            ms = r["mentions"]
            # Provenance invariant of build_graph_v2.py: every scoring edge is backed by its Mentions.
            if (not ms or len(ms) != r["n_mentions"] or any(m["doc_id"] is None for m in ms)
                    or len({m["doc_id"] for m in ms}) != r["n_docs"]):
                sys.exit(f"edge {d} -> {c}: {len(ms)} Mentions from {len({m['doc_id'] for m in ms})} docs, "
                         f"edge says n_mentions {r['n_mentions']}, n_docs {r['n_docs']}")
            ms.sort(key=lambda m: (m["doc_id"], m["phrase"], m["sentence"], m["span"] or ""))
            m = random.Random(f"{a.seed}:{d}:{c}").choice(ms)
            per = Counter(origin(x) for x in ms)
            rows.append({"disease_id": d, "disease": r["disease"], "cui": c, "phrase": m["phrase"],
                         "sentence": m["sentence"], "doc_id": m["doc_id"], "pmid": m["pmid"] or "",
                         "title": m["title"] or "", "mention_origin": origin(m), "mention_span": m["span"] or "",
                         "mention_topic": m["topic"],
                         "edge_origin": "both" if len(per) == 2 else next(iter(per)),
                         "edge_mentions_llm_table": per["llm_table"], "edge_mentions_dict_relink": per["dict_relink"],
                         "n_mentions": r["n_mentions"], "n_docs": r["n_docs"], "n_topic_docs": r["n_topic_docs"]})
        pop = {r["relinked"]: r["n"] for r in s.run(POPULATION)}
    drv.close()

    with open(a.blind, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BLIND_COLS)
        w.writerows([k, *(row[col] for col in BLIND_COLS[1:-1]), ""] for k, row in enumerate(rows, 1))
    with open(a.key, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(KEY_COLS)
        w.writerows([k, *(row[col] for col in KEY_COLS[1:])] for k, row in enumerate(rows, 1))

    n = len(rows)
    print(f"{a.uri}: {total} HAS_FINDING edges, {len(deg)} diseases (= frozen S-mix), "
          f"{sum(pop.values())} Mentions ({pop.get(True, 0)} dict_relink, {pop.get(False, 0)} llm_table)")
    print(f"sample: {n} edges, seed {a.seed}, every edge has >= 1 Mention -> {a.blind}, {a.key}")
    for col in ("mention_origin", "edge_origin"):
        for v, k in sorted(Counter(row[col] for row in rows).items()):
            lo, hi = wilson(k, n)
            print(f"  {col} {v}: {k}/{n} (Wilson 95% CI {100 * lo:.1f}-{100 * hi:.1f})")
    print(f"  n_docs 1: {sum(row['n_docs'] == 1 for row in rows)}, >= 2: {sum(row['n_docs'] >= 2 for row in rows)}; "
          f"n_topic_docs >= 1: {sum(row['n_topic_docs'] >= 1 for row in rows)}, 0: "
          f"{sum(row['n_topic_docs'] == 0 for row in rows)}")


if __name__ == "__main__":
    main()
