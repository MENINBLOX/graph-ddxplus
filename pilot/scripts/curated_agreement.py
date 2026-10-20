"""Agreement of literature-extracted graph edges with an independent curated resource (RQ3 support).

A human or LLM judging a sampled edge against its sentence is one reviewer's opinion. This script adds
a reviewer-free check: for diseases that the HPO project annotates (phenotype.hpoa, curated from
literature by HPO biocurators), how many of our HAS_FINDING edges name a phenotype the curators
also attached to that disease, compared with the same edges re-assigned to random annotated diseases.

The curated file is used only here, as a reference; it never enters the graph or the scores.
Curated annotation is incomplete, so agreement is a lower bound on edge correctness, not precision.

Edges are split by provenance signal: supporting documents (n_docs), whether a topic document supports
the edge (n_topic_docs >= 1), and with --compare-uri whether a second graph has the same edge. The
random baseline is reported for every split, because a split can select generic phenotypes that match
by chance more often.

    python curated_agreement.py --out pilot/cache/v2/curated_agreement_qwen3_v21.json
    python curated_agreement.py --uri bolt://localhost:7689 --compare-uri bolt://localhost:7688 \\
        --out pilot/cache/v3/review/curated_agreement_smix.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from array import array
from collections import defaultdict

import numpy as np
from neo4j import READ_ACCESS, GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_ids

HPOA = "data/external_kg/phenotype.hpoa"
HP_JSON = "data/medkg/hp.json"
N_DOCS_BINS = ((1, 1, "1 doc"), (2, 2, "2 docs"), (3, 4, "3-4 docs"), (5, 10**9, ">=5 docs"))
EDGES = ("MATCH (d:Disease)-[e:HAS_FINDING]->(c:Concept) WHERE d.id IN $ids "
         "RETURN d.id AS d, c.cui AS c, e.n_docs AS nd, e.n_topic_docs AS nt")


def hpo_to_cui() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """HPO term -> UMLS CUIs (ontology xrefs), and HPO is_a parents."""
    with open(HP_JSON) as f:
        g = json.load(f)["graphs"][0]

    def short(u: str) -> str:
        return u.rsplit("/", 1)[-1].replace("_", ":")

    h2c: dict[str, set[str]] = {}
    for n in g["nodes"]:
        xs = {x["val"][5:] for x in n.get("meta", {}).get("xrefs", []) if x["val"].startswith("UMLS:")}
        if xs:
            h2c[short(n["id"])] = xs
    par: dict[str, set[str]] = defaultdict(set)
    for e in g["edges"]:
        if e["pred"] == "is_a":
            par[short(e["sub"])].add(short(e["obj"]))
    return h2c, par


def curated_pairs(h2c, par) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """universe disease key -> curated phenotype CUIs (exact), and the same closed upward one is_a step."""
    exact, near = defaultdict(set), defaultdict(set)
    with open(HPOA) as fh:
        for line in fh:
            if line.startswith(("#", "database_id")):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] == "NOT" or f[10] != "P":
                continue
            for key in v2_ids.gold_keys({f[0].replace("ORPHANET:", "ORPHA:")}):
                exact[key] |= h2c.get(f[3], set())
                for p in par.get(f[3], ()):
                    near[key] |= h2c.get(p, set())
    return exact, near


def stream_edges(uri: str, auth: str, dix: dict[str, int], cix: dict[str, int]) -> dict[str, np.ndarray]:
    """Comparable edges of the annotated diseases as int32 columns d, c, nd, nt (nt -1 when the graph has no n_topic_docs).

    Comparable: the concept exists in HPO, so the curators could have annotated it. Rows are streamed
    because a dictionary graph returns millions of edges for these diseases.
    """
    col = {k: array("i") for k in ("d", "c", "nd", "nt")}
    drv = GraphDatabase.driver(uri, auth=tuple(auth.split("/", 1)))
    with drv.session(default_access_mode=READ_ACCESS) as s:
        for r in s.run(EDGES, ids=list(dix)):
            ci = cix.get(r["c"])
            if ci is None:
                continue
            col["d"].append(dix[r["d"]])
            col["c"].append(ci)
            col["nd"].append(r["nd"])
            col["nt"].append(-1 if r["nt"] is None else r["nt"])
    drv.close()
    return {k: np.array(v, dtype=np.int32) for k, v in col.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--compare-uri", default=None, help="second graph; splits edges by whether it has the same edge")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--shuffles", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    h2c, par = hpo_to_cui()
    exact, near = curated_pairs(h2c, par)
    both = {k: exact[k] | near[k] for k in exact}
    names = sorted(exact)
    dix = {d: i for i, d in enumerate(names)}
    cix = {c: i for i, c in enumerate(sorted(set().union(*h2c.values())))}
    nc = len(cix)

    def keyset(ref: dict[str, set[str]]) -> np.ndarray:
        return np.unique(np.fromiter((dix[d] * nc + cix[c] for d, cs in ref.items() for c in cs), dtype=np.int64))

    ref_e, ref_n = keyset(exact), keyset(both)
    e = stream_edges(a.uri, a.auth, dix, cix)
    keys = e["d"].astype(np.int64) * nc + e["c"]
    dz_idx = np.unique(e["d"])
    dz = [names[i] for i in dz_idx]

    groups: dict[tuple[str, ...], np.ndarray] = {(): np.ones(len(keys), dtype=bool)}
    for lo, hi, lab in N_DOCS_BINS:
        groups[("by_n_docs", lab)] = (e["nd"] >= lo) & (e["nd"] <= hi)
    if (e["nt"] >= 0).all():
        topic = {"topic doc >=1": e["nt"] >= 1, "off-topic docs only": e["nt"] == 0}
        for t, m in topic.items():
            groups[("by_topic", t)] = m
            for lo, hi, lab in N_DOCS_BINS:
                groups[("by_n_docs_topic", lab, t)] = m & groups[("by_n_docs", lab)]
    if a.compare_uri:
        other = stream_edges(a.compare_uri, a.auth, dix, cix)
        in_other = np.isin(keys, other["d"].astype(np.int64) * nc + other["c"])
        del other
        groups[("by_other_graph", "also in other graph")] = in_other
        groups[("by_other_graph", "only in this graph")] = ~in_other

    def rate(hit: np.ndarray, m: np.ndarray) -> float:
        return 100 * float(hit[m].sum()) / max(1, int(m.sum()))

    hit_e, hit_n = np.isin(keys, ref_e), np.isin(keys, ref_n)
    # Null: the same edges with diseases permuted among the diseases that have comparable edges.
    rng = random.Random(0)
    null: dict[tuple[str, ...], list[tuple[float, float]]] = defaultdict(list)
    for _ in range(a.shuffles):
        perm = np.arange(len(dix))
        perm[[dix[d] for d in dz]] = [dix[d] for d in rng.sample(dz, len(dz))]
        sh = perm[e["d"]].astype(np.int64) * nc + e["c"]
        se, sn = np.isin(sh, ref_e), np.isin(sh, ref_n)
        for g, m in groups.items():
            null[g].append((rate(se, m), rate(sn, m)))

    def block(g: tuple[str, ...]) -> dict:
        m = groups[g]
        ne, nn = zip(*null[g])
        return {"edges": int(m.sum()), "agree_exact": rate(hit_e, m), "agree_exact_or_parent": rate(hit_n, m),
                "null_exact_mean": sum(ne) / len(ne), "null_exact_or_parent_mean": sum(nn) / len(nn)}

    top = block(())
    res = {"uri": a.uri, "compare_uri": a.compare_uri, "diseases_annotated": len(exact),
           "diseases_with_comparable_edges": len(dz), "comparable_edges": top["edges"],
           **{k: v for k, v in top.items() if k != "edges"}}
    for g in groups:
        if g:
            node = res
            for p in g[:-1]:
                node = node.setdefault(p, {})
            node[g[-1]] = block(g)
    # Recall side: curated pairs (with CUI in graph vocabulary) found in the graph.
    cur = ref_e[np.isin(ref_e // nc, dz_idx)]
    res["curated_pairs_on_these_diseases"] = len(cur)
    res["curated_recall_exact"] = 100 * float(np.isin(cur, keys).sum()) / max(1, len(cur))
    print(json.dumps(res, indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
