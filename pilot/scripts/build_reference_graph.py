"""Load an existing knowledge source into the comparator Neo4j with the graph's schema (§9.22).

Reviewers ask whether accuracy comes from the ranker or the knowledge. Each comparator source is
loaded into the same schema (Disease, Concept, HAS_FINDING with support counts, IS_A, IS_DISEASE)
over the same 43,575-disease universe, so `graph_dx.py --uri bolt://localhost:7688` ranks with the
identical Cypher query. There are no Mention nodes: these sources carry no sentence provenance.

  hpoa     HPO annotations (curated from literature; mostly rare diseases)
  primekg  PrimeKG disease_phenotype_positive (integrated curated KG; HPO, Orphanet, ...)
  cooc     MEDLINE MeSH co-occurrence (NLM, 2024 summary): literature-scale, no LLM; support count =
           number of co-indexed citations
  semmeddb SemMedDB VER43 2024 (SemRep rules over PubMed, no LLM): MANIFESTATION_OF, COEXISTS_WITH and
           ASSOCIATED_WITH between a universe disease and a finding-type concept; support count = distinct
           PMIDs. The orientation follows the predicate: MANIFESTATION_OF has the finding as subject and
           the disease as object; COEXISTS_WITH and ASSOCIATED_WITH state an association without a
           direction, so either end may be the disease (--semmed-orient subject keeps only a disease
           subject for those two). --semmed-generic drop leaves out predications whose subject or object
           SemMedDB flags as generic (novelty 0, e.g. "Disease", "Patients").

    python build_reference_graph.py --source hpoa

--noise R draws R wrong-edge candidates per true edge. --sib-share S sets their composition: a share
S are findings of a related disease (sibling under a shared parent, else a random disease), the rest
concepts from the IE vocabulary. The default 0.7 is the mix of the main grid (§9.26); 1.0 and 0.0 are
the two pure compositions. A candidate that is already an edge is skipped, so the realized precision
(true_edges / edges in the summary) is above 1/(1+R) and depends on S.
--keep P first keeps each true edge with probability P (recall); noise is then added relative to
the kept edges, so --keep and --noise together simulate a graph of given recall and precision.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict

from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import curated_agreement as CA
import v2_ids
from build_graph_v2 import run_batches

V2 = "pilot/cache/v2"
# finding/evidence semantic types accepted by normalize_v2, minus tests/procedures (T058-T061)
VALID_TUI = {"T005", "T007", "T019", "T020", "T031", "T033", "T034", "T037", "T046", "T047", "T048",
             "T049", "T050", "T054", "T055", "T056", "T097", "T098", "T109", "T121", "T184", "T190",
             "T191", "T195", "T200", "T204"}
COOC = "data/external_kg/summary_CoOccurs_2024.txt.gz"
SEMMED = "data/semmeddb/semmedVER43_2024_R_PREDICATION.csv.gz"
SEMMED_PREDS = ("MANIFESTATION_OF", "COEXISTS_WITH", "ASSOCIATED_WITH")


def pairs_hpoa(uni) -> dict:
    h2c, par = CA.hpo_to_cui()
    exact, _ = CA.curated_pairs(h2c, par)
    return {(d, c): 1 for d, cs in exact.items() if d in uni for c in cs}


def pairs_primekg(uni) -> dict:
    h2c, _ = CA.hpo_to_cui()
    mondo2key = v2_ids._maps()[0]
    out = {}
    with open("data/external_kg/primekg.csv") as fh:
        for f in csv.reader(fh):
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


def cui_sty() -> dict[str, set[str]]:
    sty = defaultdict(set)
    with open("pilot/cache/umls_maps/cui_sty.tsv") as fh:
        for ln in fh:
            c, t = ln.split("\t")[:2]
            sty[c].add(t)
    return sty


def pairs_cooc(uni) -> dict:
    sty = cui_sty()
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


def semmed_rows(uni, cache: str) -> list[tuple[str, str, bool, str, bool, int]]:
    """(predicate, subject, subject generic, object, object generic, PMID) for SEMMED_PREDS rows that can
    give a disease-finding pair in either orientation. The full scan takes minutes, so it is cached."""
    if os.path.exists(cache):
        with gzip.open(cache, "rt") as fh:
            return [(p, sc, sg == "1", oc, og == "1", int(pm))
                    for p, sc, sg, oc, og, pm in (ln.rstrip("\n").split("\t") for ln in fh)]
    sty = cui_sty()

    def finding(c: str) -> bool:
        return bool(sty.get(c, set()) & VALID_TUI)

    # grep keeps NEG_ predicates out: the quote before the name must follow the comma
    pat = ',"(' + "|".join(SEMMED_PREDS) + ')",'
    zc = subprocess.Popen(["zcat", SEMMED], stdout=subprocess.PIPE)
    gr = subprocess.Popen(["grep", "-a", "-E", pat], stdin=zc.stdout, stdout=subprocess.PIPE,
                          encoding="utf-8", errors="replace")
    zc.stdout.close()
    rows, n_lines = [], 0
    for f in csv.reader(gr.stdout):
        n_lines += 1
        if len(f) < 12 or f[3] not in SEMMED_PREDS:
            continue
        # gene arguments carry "CUI|EntrezID"; only CUIs can be diseases or findings
        for sc in (x for x in f[4].split("|") if x.startswith("C")):
            for oc in (x for x in f[8].split("|") if x.startswith("C")):
                if sc != oc and ((sc in uni and finding(oc)) or (oc in uni and finding(sc))):
                    rows.append((f[3], sc, f[7] == "0", oc, f[11] == "0", int(f[2])))
    if gr.wait() or zc.wait():
        sys.exit("SemMedDB scan failed")
    print(f"SemMedDB: {n_lines:,} {'/'.join(SEMMED_PREDS)} rows, {len(rows):,} with a disease-finding pair",
          flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with gzip.open(cache + ".tmp", "wt") as fh:
        for p, sc, sg, oc, og, pm in rows:
            fh.write(f"{p}\t{sc}\t{int(sg)}\t{oc}\t{int(og)}\t{pm}\n")
    os.replace(cache + ".tmp", cache)
    return rows


def pairs_semmeddb(uni, cache: str, orient: str, generic: str) -> tuple[dict, dict]:
    """(disease, finding) -> distinct PMIDs, and predication counts by predicate and orientation."""
    sty = cui_sty()
    pmids: dict[tuple[str, str], set[int]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for p, sc, sg, oc, og, pm in semmed_rows(uni, cache):
        if generic == "drop" and (sg or og):
            counts["dropped_generic"] += 1
            continue
        if p == "MANIFESTATION_OF":
            ends = [(oc, sc)]
        elif orient == "semantic":
            ends = [(sc, oc), (oc, sc)]
        else:
            ends = [(sc, oc)]
        for d, c in ends:
            if d in uni and sty.get(c, set()) & VALID_TUI:
                pmids[(d, c)].add(pm)
                counts[f"{p} disease_{'subject' if d == sc else 'object'}"] += 1
    return {e: len(v) for e, v in pmids.items()}, dict(sorted(counts.items()))


def add_noise(edge: dict, ratio: float, seed: int, sib_share: float = 0.7) -> dict:
    rng = random.Random(seed)
    prof = defaultdict(list)
    # sorted: pairs_hpoa builds edges from sets, whose order changes with PYTHONHASHSEED
    for d, c in sorted(edge):
        prof[d].append(c)
    dz = sorted(prof)
    up, down = defaultdict(list), defaultdict(list)
    for c, p in v2_ids.disease_isa():
        up[c].append(p)
        down[p].append(c)
    sib = {d: sorted({s for p in up[d] for s in down[p] if s != d and s in prof}) for d in dz}
    with open(f"{V2}/phrase2cui_gemma.json") as fh:
        tab = json.load(fh)
    vocab = [v["cui"] for v in tab.values() if v.get("cui") and v.get("tui") in VALID_TUI]
    out = dict(edge)
    for d in dz:
        k = int(ratio * len(prof[d])) + (rng.random() < ratio * len(prof[d]) % 1)
        for _ in range(k):
            if rng.random() < sib_share:
                src = rng.choice(sib[d]) if sib[d] else rng.choice(dz)
                c = rng.choice(prof[src])
            else:
                c = rng.choice(vocab)
            out.setdefault((d, c), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hpoa", "primekg", "cooc", "semmeddb"], required=True)
    ap.add_argument("--uri", default="bolt://localhost:7688")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--hpoa", default=CA.HPOA, help="phenotype.hpoa file (e.g. a historical release)")
    ap.add_argument("--summary", default="")
    ap.add_argument("--noise", type=float, default=0.0, help="wrong edges added per true edge")
    ap.add_argument("--keep", type=float, default=1.0, help="share of true edges kept")
    ap.add_argument("--sib-share", type=float, default=0.7,
                    help="share of wrong edges taken from a related disease's findings (rest: IE vocabulary)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--semmed-cache", default="pilot/cache/v3/p0/semmeddb/predications.tsv.gz",
                    help="filtered SemMedDB rows; written on the first semmeddb build")
    ap.add_argument("--semmed-orient", choices=["semantic", "subject"], default="semantic")
    ap.add_argument("--semmed-generic", choices=["drop", "keep"], default="drop")
    a = ap.parse_args()
    CA.HPOA = a.hpoa
    uni = dict(v2_ids.universe())
    with open(f"{V2}/exclude_nonhuman_diseases.json") as fh:
        for k in json.load(fh):
            uni.pop(k, None)
    counts: dict[str, int] = {}
    if a.source == "semmeddb":
        edge, counts = pairs_semmeddb(uni, a.semmed_cache, a.semmed_orient, a.semmed_generic)
    else:
        edge = {"hpoa": pairs_hpoa, "primekg": pairs_primekg, "cooc": pairs_cooc}[a.source](uni)
    if a.keep < 1:
        rng = random.Random(a.seed)
        edge = {e: n for e, n in sorted(edge.items()) if rng.random() < a.keep}
    n_true = len(edge)
    if a.noise:
        edge = add_noise(edge, a.noise, a.seed, a.sib_share)
    prof = defaultdict(set)
    for d, c in edge:
        prof[d].add(c)
    df = defaultdict(int)
    for cs in prof.values():
        for c in cs:
            df[c] += 1
    n_dz = max(1, len(prof))
    cuis = set(df)
    with open("pilot/cache/umls_maps/phenotype_parents.json") as fh:
        parents = json.load(fh)
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
    with open(a.summary or f"{V2}/refgraph_{a.source}_summary.json", "w") as fh:
        info = {"source": a.source, "edges": len(edge), "true_edges": n_true,
                "diseases_with_findings": len(prof), "concepts": len(df)}
        if a.source == "semmeddb":
            info.update(orient=a.semmed_orient, generic=a.semmed_generic, predications=counts,
                        n_docs_total=sum(edge.values()))
        json.dump(info, fh, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
