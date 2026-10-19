"""Agreement of literature-extracted graph edges with an independent curated resource (RQ3 support).

A human or LLM judging a sampled edge against its sentence is one reviewer's opinion. This script adds
a reviewer-free check: for diseases that the HPO project annotates (phenotype.hpoa, curated from
literature by HPO biocurators), how many of our HAS_FINDING edges name a phenotype the curators
also attached to that disease, compared with the same edges re-assigned to random annotated diseases.

The curated file is used only here, as a reference; it never enters the graph or the scores.
Curated annotation is incomplete, so agreement is a lower bound on edge correctness, not precision.

    python curated_agreement.py --out pilot/cache/v2/curated_agreement_qwen3_v21.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_ids  # noqa: E402

HPOA = "data/external_kg/phenotype.hpoa"
HP_JSON = "data/medkg/hp.json"


def hpo_to_cui() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """HPO term -> UMLS CUIs (ontology xrefs), and HPO is_a parents."""
    g = json.load(open(HP_JSON))["graphs"][0]
    short = lambda u: u.rsplit("/", 1)[-1].replace("_", ":")  # noqa: E731
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
    for line in open(HPOA):
        if line.startswith("#") or line.startswith("database_id"):
            continue
        f = line.rstrip("\n").split("\t")
        if f[2] == "NOT" or f[10] != "P":
            continue
        for key in v2_ids.gold_keys({f[0].replace("ORPHANET:", "ORPHA:")}):
            exact[key] |= h2c.get(f[3], set())
            for p in par.get(f[3], ()):
                near[key] |= h2c.get(p, set())
    return exact, near


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--shuffles", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    h2c, par = hpo_to_cui()
    exact, near = curated_pairs(h2c, par)
    hpo_cuis = set().union(*h2c.values())
    drv = GraphDatabase.driver(a.uri, auth=tuple(a.auth.split("/", 1)))
    with drv.session() as s:
        rows = s.run("MATCH (d:Disease)-[e:HAS_FINDING]->(c:Concept) WHERE d.id IN $ids "
                     "RETURN d.id AS d, c.cui AS c, e.n_docs AS nd", ids=list(exact)).data()
    drv.close()
    # Comparable edges: the concept exists in HPO, so the curators could have annotated it.
    rows = [r for r in rows if r["c"] in hpo_cuis]
    dz = sorted({r["d"] for r in rows})

    def rate(pairs, ref) -> float:
        return 100 * sum(c in ref.get(d, ()) for d, c in pairs) / max(1, len(pairs))

    pairs = [(r["d"], r["c"]) for r in rows]
    res = {"diseases_annotated": len(exact), "diseases_with_comparable_edges": len(dz),
           "comparable_edges": len(pairs),
           "agree_exact": rate(pairs, exact),
           "agree_exact_or_parent": rate(pairs, {k: exact[k] | near[k] for k in exact})}
    rng = random.Random(0)
    null_e, null_n = [], []
    both = {k: exact[k] | near[k] for k in exact}
    for _ in range(a.shuffles):
        perm = dict(zip(dz, rng.sample(dz, len(dz))))
        sh = [(perm[d], c) for d, c in pairs]
        null_e.append(rate(sh, exact))
        null_n.append(rate(sh, both))
    res["null_exact_mean"] = sum(null_e) / len(null_e)
    res["null_exact_or_parent_mean"] = sum(null_n) / len(null_n)
    # Does more literature support make an edge more likely to be curated?
    by = {}
    for lo, hi, lab in ((1, 1, "1 doc"), (2, 2, "2 docs"), (3, 4, "3-4 docs"), (5, 10**9, ">=5 docs")):
        sub = [(r["d"], r["c"]) for r in rows if lo <= r["nd"] <= hi]
        by[lab] = {"edges": len(sub), "agree_exact": rate(sub, exact), "agree_exact_or_parent": rate(sub, both)}
    res["by_n_docs"] = by
    # Recall side: curated pairs (with CUI in graph vocabulary) found in the graph.
    have = defaultdict(set)
    for d, c in pairs:
        have[d].add(c)
    cur = [(d, c) for d in dz for c in exact[d]]
    res["curated_pairs_on_these_diseases"] = len(cur)
    res["curated_recall_exact"] = 100 * sum(c in have[d] for d, c in cur) / max(1, len(cur))
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
