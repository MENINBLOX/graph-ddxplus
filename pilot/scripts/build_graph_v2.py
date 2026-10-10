"""Load corpus-v2 extraction into Neo4j (research design D8, RQ3).

Graph model
  (:Disease {id, name, vocab, n_findings, src_chars})          universe key: UMLS CUI or MONDO ID
  (:Concept {cui, tui, idf})                                     finding / evidence concept
  (:Doc {id, pmid, pmcid, kind, title, year, license})           article or book chapter
  (:Mention {phrase, sentence})                                  one extracted, sentence-located item
  (Disease)-[:HAS_FINDING {n_mentions, n_docs}]->(Concept)      aggregated edge used for scoring
  (Mention)-[:OF]->(Disease), (Mention)-[:MENTIONS]->(Concept), (Mention)-[:FROM]->(Doc)
  (Concept)-[:IS_A]->(Concept)                                   UMLS concept hierarchy
  (Disease)-[:IS_A]->(Disease)                                   MONDO + UMLS disease hierarchy
  (Concept)-[:IS_DISEASE]->(Disease)                             unified evidence: a finding that is a disease

Every HAS_FINDING edge is backed by >= 1 Mention, so any diagnosis can be traced to sentences and
the articles they came from.

    python build_graph_v2.py --ie IE.jsonl --table phrase2cui.json --sources DIR [--wipe]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_ids  # noqa: E402

BATCH = 5000


def run_batches(session, query: str, rows: list[dict], label: str) -> None:
    for i in range(0, len(rows), BATCH):
        session.run(query, rows=rows[i:i + BATCH]).consume()
    print(f"  {label}: {len(rows)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--wipe", action="store_true")
    ap.add_argument("--max-doc-diseases", type=int, default=20)
    ap.add_argument("--exclude-diseases", default="pilot/cache/v2/exclude_nonhuman_diseases.json")
    ap.add_argument("--exclude-tuis", default="T058,T059,T060,T061",
                    help="semantic types that are tests/procedures/health-care activities, not findings")
    ap.add_argument("--summary", default="pilot/cache/v2/graph_summary.json")
    a = ap.parse_args()

    uni = dict(v2_ids.universe())
    # v2.1 (provenance audit): diseases under MONDO "non-human animal disease" are not human
    # diagnoses, and a name query retrieves human literature for them (33 of 39 audited edges on
    # such nodes were unsupported). Tests, procedures and health-care activities are not findings.
    excluded_dz = set(json.load(open(a.exclude_diseases))) if a.exclude_diseases else set()
    for k in excluded_dz:
        uni.pop(k, None)
    excluded_tui = set(a.exclude_tuis.split(",")) if a.exclude_tuis else set()
    table = json.load(open(a.table))
    # Source specificity: an article retrieved for many different diseases (a methods paper or a
    # broad review caught by full-text fallback queries) is not evidence about any one of them.
    # Threshold fixed a priori from the corpus alone (no benchmark involved).
    use = defaultdict(int)
    for f in glob.glob(f"{a.sources}/*.json"):
        for x in json.load(open(f))["docs"]:
            use[x["doc_id"]] += 1
    generic = {d for d, n in use.items() if n >= a.max_doc_diseases}
    print(f"generic documents dropped (used by >= {a.max_doc_diseases} diseases): {len(generic)}", flush=True)
    mentions, docs_used = [], set()
    edge = defaultdict(lambda: [0, set()])            # (disease, cui) -> [n_mentions, docs]
    for line in (ln for path in a.ie.split(",") for ln in open(path)):
        r = json.loads(line)
        d = r["id"]
        for it in r.get("items", []):
            t = table.get(it["t"].lower().strip(), {})
            if not t.get("cui") or it["doc"] in generic or d not in uni or t.get("tui") in excluded_tui:
                continue
            c = t["cui"]
            mentions.append({"d": d, "c": c, "doc": it["doc"], "phrase": it["t"], "sent": it["sent"]})
            e = edge[(d, c)]
            e[0] += 1
            e[1].add(it["doc"])
            docs_used.add(it["doc"])
    print(f"mentions {len(mentions)}, edges {len(edge)}, docs {len(docs_used)}", flush=True)

    # document metadata and per-disease collected text size (coverage signal for scoring)
    docs, src_chars = {}, {}
    for f in glob.glob(f"{a.sources}/*.json"):
        rec = json.load(open(f))
        src_chars[rec["id"]] = sum(len(x["text"]) for x in rec["docs"])
        for x in rec["docs"]:
            if x["doc_id"] in docs_used and x["doc_id"] not in docs:
                docs[x["doc_id"]] = {k: x.get(k, "") for k in ("pmid", "pmcid", "kind", "title", "year", "license")}

    prof = defaultdict(set)
    for (d, c) in edge:
        prof[d].add(c)
    n_dz = max(1, len(prof))
    df = defaultdict(int)
    for cs in prof.values():
        for c in cs:
            df[c] += 1
    cuis = set(df)
    parents = json.load(open("pilot/cache/umls_maps/phenotype_parents.json"))
    isa_c = [{"c": c, "p": p} for c in cuis for p in parents.get(c, [])]
    cuis |= {r["p"] for r in isa_c}
    tui = {v["cui"]: v.get("tui", "") for v in table.values() if v.get("cui")}

    drv = GraphDatabase.driver(a.uri, auth=tuple(a.auth.split("/", 1)))
    with drv.session() as s:
        if a.wipe:
            print("wiping graph", flush=True)
            # Relationships first, then nodes, in small batches: a failed apoc batch is only a
            # server-side warning, so the result is checked and the build stops on any failure
            # (v2.1 incident: a memory-pool failure left 2,682 stale nodes and 62,497 stale edges).
            for match, act in (("MATCH ()-[r]->() RETURN r", "DELETE r"),
                               ("MATCH (n) RETURN n", "DETACH DELETE n")):
                res = s.run("CALL apoc.periodic.iterate($m, $a, {batchSize: 2000}) "
                            "YIELD failedBatches, errorMessages RETURN failedBatches, errorMessages",
                            m=match, a=act).single()
                if res["failedBatches"]:
                    sys.exit(f"wipe failed: {res['errorMessages']}")
            left = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
            if left:
                sys.exit(f"wipe incomplete: {left} nodes remain")
        for q in ("CREATE CONSTRAINT dz IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE",
                  "CREATE CONSTRAINT cp IF NOT EXISTS FOR (c:Concept) REQUIRE c.cui IS UNIQUE",
                  "CREATE CONSTRAINT dc IF NOT EXISTS FOR (x:Doc) REQUIRE x.id IS UNIQUE"):
            s.run(q).consume()
        run_batches(s, "UNWIND $rows AS r MERGE (d:Disease {id: r.id}) SET d.name = r.name, "
                       "d.vocab = r.vocab, d.n_findings = r.n, d.src_chars = r.sc",
                    [{"id": k, "name": v["name"], "vocab": v["src"], "n": len(prof.get(k, ())),
                      "sc": src_chars.get(k, 0)} for k, v in uni.items()], "Disease")
        run_batches(s, "UNWIND $rows AS r MERGE (c:Concept {cui: r.cui}) SET c.tui = r.tui, c.idf = r.idf",
                    [{"cui": c, "tui": tui.get(c, ""),
                      "idf": math.log(n_dz / df[c]) if df.get(c) else None} for c in cuis], "Concept")
        run_batches(s, "UNWIND $rows AS r MERGE (x:Doc {id: r.id}) SET x += r.meta",
                    [{"id": k, "meta": v} for k, v in docs.items()], "Doc")
        run_batches(s, "UNWIND $rows AS r MATCH (d:Disease {id: r.d}), (c:Concept {cui: r.c}) "
                       "MERGE (d)-[e:HAS_FINDING]->(c) SET e.n_mentions = r.n, e.n_docs = r.nd",
                    [{"d": d, "c": c, "n": v[0], "nd": len(v[1])} for (d, c), v in edge.items()],
                    "HAS_FINDING")
        run_batches(s, "UNWIND $rows AS r MATCH (d:Disease {id: r.d}), (c:Concept {cui: r.c}), "
                       "(x:Doc {id: r.doc}) CREATE (m:Mention {phrase: r.phrase, sentence: r.sent}) "
                       "CREATE (m)-[:OF]->(d), (m)-[:MENTIONS]->(c), (m)-[:FROM]->(x)",
                    mentions, "Mention")
        run_batches(s, "UNWIND $rows AS r MATCH (c:Concept {cui: r.c}), (p:Concept {cui: r.p}) "
                       "MERGE (c)-[:IS_A]->(p)", isa_c, "Concept IS_A")
        run_batches(s, "UNWIND $rows AS r MATCH (c:Disease {id: r.c}), (p:Disease {id: r.p}) "
                       "MERGE (c)-[:IS_A]->(p)",
                    [{"c": c, "p": p} for c, p in v2_ids.disease_isa() if c in uni and p in uni], "Disease IS_A")
        run_batches(s, "UNWIND $rows AS r MATCH (c:Concept {cui: r.id}), (d:Disease {id: r.id}) "
                       "MERGE (c)-[:IS_DISEASE]->(d)",
                    [{"id": c} for c in cuis if c in uni], "IS_DISEASE")
        # Provenance invariant (module docstring): every scoring edge is backed by a Mention.
        orphan = s.run("MATCH (d:Disease)-[e:HAS_FINDING]->(c) WHERE NOT EXISTS { "
                       "MATCH (m:Mention)-[:OF]->(d) WHERE (m)-[:MENTIONS]->(c) } "
                       "RETURN count(e) AS n").single()["n"]
        n_dz = s.run("MATCH (d:Disease) RETURN count(d) AS n").single()["n"]
        if orphan or n_dz != len(uni):
            sys.exit(f"graph check failed: {orphan} edges without Mention, {n_dz} Disease nodes vs {len(uni)}")
        print("graph check: every HAS_FINDING edge has a Mention; Disease nodes = universe", flush=True)
    drv.close()
    json.dump({"diseases": len(uni), "diseases_with_findings": len(prof), "concepts": len(cuis),
               "edges": len(edge), "mentions": len(mentions), "docs": len(docs)},
              open(a.summary, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
