"""Load an existing knowledge source into the comparator Neo4j with the graph's schema (§9.22).

Reviewers ask whether accuracy comes from the ranker or the knowledge. Each comparator source is
loaded into the same schema (Disease, Concept, HAS_FINDING with support counts, IS_A, IS_DISEASE)
over the same 43,575-disease universe, so `graph_dx.py --uri bolt://localhost:7688` ranks with the
identical Cypher query. There are no Mention nodes: these sources carry no sentence provenance.

  hpoa     HPO annotations (curated from literature; mostly rare diseases)
  primekg  PrimeKG disease_phenotype_positive (integrated curated KG; HPO, Orphanet, ...)
  cooc     MEDLINE MeSH co-occurrence (NLM, 2024 summary): literature-scale, no LLM; support count =
           number of co-indexed citations

    python build_reference_graph.py --source hpoa

--noise R adds R wrong edges per true edge with the error mix the provenance audit found in the
literature graph (§9.25): 70% findings of a related disease (sibling under a shared parent, else a
random disease), 30% concepts from the IE vocabulary that are not findings of the disease.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
from collections import defaultdict

from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curated_agreement as CA  # noqa: E402
import v2_ids  # noqa: E402
from build_graph_v2 import run_batches  # noqa: E402

V2 = "pilot/cache/v2"
# finding/evidence semantic types accepted by normalize_v2, minus tests/procedures (T058-T061)
VALID_TUI = {"T005", "T007", "T019", "T020", "T031", "T033", "T034", "T037", "T046", "T047", "T048",
             "T049", "T050", "T054", "T055", "T056", "T097", "T098", "T109", "T121", "T184", "T190",
             "T191", "T195", "T200", "T204"}
COOC = "data/external_kg/summary_CoOccurs_2024.txt.gz"


def pairs_hpoa(uni) -> dict:
    h2c, par = CA.hpo_to_cui()
    exact, _ = CA.curated_pairs(h2c, par)
    return {(d, c): 1 for d, cs in exact.items() if d in uni for c in cs}


def pairs_primekg(uni) -> dict:
    h2c, _ = CA.hpo_to_cui()
    mondo2key = v2_ids._maps()[0]
    out = {}
    import csv
    for f in csv.reader(open("data/external_kg/primekg.csv")):
        if f[0] != "disease_phenotype_positive":
            continue
        # x = disease (MONDO or MONDO_grouped "id_id_..."), y = HPO phenotype
        x_id, x_src, y_id, y_src = f[3], f[6], f[8], f[11]
        if y_src != "HPO" or not x_src.startswith("MONDO"):
            continue
        cuis = h2c.get(f"HP:{int(y_id):07d}", set())
        for m in x_id.split("_"):
            k = mondo2key.get(f"MONDO:{int(m):07d}")
            if k in uni:
                for c in cuis:
                    out[(k, c)] = 1
    return out


def pairs_cooc(uni) -> dict:
    sty = defaultdict(set)
    for ln in open("pilot/cache/umls_maps/cui_sty.tsv"):
        c, t = ln.split("\t")[:2]
        sty[c].add(t)
    out = defaultdict(int)
    with gzip.open(COOC, "rt") as fh:
        next(fh)
        for ln in fh:
            f = ln.split("|", 5)
            c1, c2, n = f[1], f[3], int(f[4])
            for d, c in ((c1, c2), (c2, c1)):
                if d in uni and c != d and sty.get(c, set()) & VALID_TUI:
                    out[(d, c)] += n
    return dict(out)


def add_noise(edge: dict, ratio: float, seed: int) -> dict:
    rng = random.Random(seed)
    prof = defaultdict(list)
    for d, c in edge:
        prof[d].append(c)
    dz = sorted(prof)
    up, down = defaultdict(list), defaultdict(list)
    for c, p in v2_ids.disease_isa():
        up[c].append(p)
        down[p].append(c)
    sib = {d: sorted({s for p in up[d] for s in down[p] if s != d and s in prof}) for d in dz}
    tab = json.load(open(f"{V2}/phrase2cui_gemma.json"))
    vocab = [v["cui"] for v in tab.values() if v.get("cui") and v.get("tui") in VALID_TUI]
    out = dict(edge)
    for d in dz:
        k = int(ratio * len(prof[d])) + (rng.random() < ratio * len(prof[d]) % 1)
        for _ in range(k):
            if rng.random() < 0.7:
                src = rng.choice(sib[d]) if sib[d] else rng.choice(dz)
                c = rng.choice(prof[src])
            else:
                c = rng.choice(vocab)
            out.setdefault((d, c), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hpoa", "primekg", "cooc"], required=True)
    ap.add_argument("--uri", default="bolt://localhost:7688")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--hpoa", default=CA.HPOA, help="phenotype.hpoa file (e.g. a historical release)")
    ap.add_argument("--summary", default="")
    ap.add_argument("--noise", type=float, default=0.0, help="wrong edges added per true edge")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    CA.HPOA = a.hpoa
    uni = dict(v2_ids.universe())
    for k in json.load(open(f"{V2}/exclude_nonhuman_diseases.json")):
        uni.pop(k, None)
    edge = {"hpoa": pairs_hpoa, "primekg": pairs_primekg, "cooc": pairs_cooc}[a.source](uni)
    if a.noise:
        edge = add_noise(edge, a.noise, a.seed)
    prof = defaultdict(set)
    for d, c in edge:
        prof[d].add(c)
    df = defaultdict(int)
    for cs in prof.values():
        for c in cs:
            df[c] += 1
    n_dz = max(1, len(prof))
    cuis = set(df)
    parents = json.load(open("pilot/cache/umls_maps/phenotype_parents.json"))
    isa_c = [{"c": c, "p": p} for c in cuis for p in parents.get(c, [])]
    cuis |= {r["p"] for r in isa_c}
    print(f"{a.source}: edges {len(edge)}, diseases with findings {len(prof)}, concepts {len(df)}", flush=True)
    drv = GraphDatabase.driver(a.uri, auth=tuple(a.auth.split("/", 1)))
    with drv.session() as s:
        for match, act in (("MATCH ()-[r]->() RETURN r", "DELETE r"), ("MATCH (n) RETURN n", "DETACH DELETE n")):
            r = s.run("CALL apoc.periodic.iterate($m, $a, {batchSize: 2000}) YIELD failedBatches "
                      "RETURN failedBatches", m=match, a=act).single()
            if r["failedBatches"]:
                sys.exit("wipe failed")
        for q in ("CREATE CONSTRAINT dz IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE",
                  "CREATE CONSTRAINT cp IF NOT EXISTS FOR (c:Concept) REQUIRE c.cui IS UNIQUE"):
            s.run(q).consume()
        # src_chars is constant: coverage smoothing then adds the same length to every disease
        run_batches(s, "UNWIND $rows AS r MERGE (d:Disease {id: r.id}) SET d.name = r.name, "
                       "d.n_findings = r.n, d.src_chars = 1",
                    [{"id": k, "name": v["name"], "n": len(prof.get(k, ()))} for k, v in uni.items()], "Disease")
        run_batches(s, "UNWIND $rows AS r MERGE (c:Concept {cui: r.cui}) SET c.idf = r.idf",
                    [{"cui": c, "idf": math.log(n_dz / df[c]) if df.get(c) else None} for c in cuis], "Concept")
        run_batches(s, "UNWIND $rows AS r MATCH (d:Disease {id: r.d}), (c:Concept {cui: r.c}) "
                       "MERGE (d)-[e:HAS_FINDING]->(c) SET e.n_docs = r.n, e.n_topic_docs = r.n",
                    [{"d": d, "c": c, "n": n} for (d, c), n in edge.items()], "HAS_FINDING")
        run_batches(s, "UNWIND $rows AS r MATCH (c:Concept {cui: r.c}), (p:Concept {cui: r.p}) "
                       "MERGE (c)-[:IS_A]->(p)", isa_c, "Concept IS_A")
        run_batches(s, "UNWIND $rows AS r MATCH (c:Disease {id: r.c}), (p:Disease {id: r.p}) "
                       "MERGE (c)-[:IS_A]->(p)",
                    [{"c": c, "p": p} for c, p in v2_ids.disease_isa() if c in uni and p in uni], "Disease IS_A")
        run_batches(s, "UNWIND $rows AS r MATCH (c:Concept {cui: r.id}), (d:Disease {id: r.id}) "
                       "MERGE (c)-[:IS_DISEASE]->(d)", [{"id": c} for c in cuis if c in uni], "IS_DISEASE")
        n = s.run("MATCH (d:Disease) RETURN count(d) AS n").single()["n"]
        if n != len(uni):
            sys.exit(f"graph check failed: {n} Disease nodes vs {len(uni)}")
    drv.close()
    json.dump({"source": a.source, "edges": len(edge), "diseases_with_findings": len(prof),
               "concepts": len(df)}, open(a.summary or f"{V2}/refgraph_{a.source}_summary.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
