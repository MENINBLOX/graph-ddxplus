"""Multi-turn differential diagnosis on the graph: simulated questioning (research design RQ2).

Loop: rank diseases from the evidence known so far -> ask the finding that best splits the current
top candidates (a graph query over their HAS_FINDING neighbourhoods) -> the simulated patient
answers from their record -> re-rank. After the budget is used, the last ranking is the final
diagnosis. IL = number of questions asked.

Question selection never looks at the benchmark's question list (design §3, D5): any concept in the
candidates' neighbourhood may be asked. A question the record cannot answer is answered "unknown";
it still counts toward IL. Answers:
  DDXPlus            present if in the record; absent if it is a DDXPlus evidence concept not in the
                     record (the questionnaire is complete); unknown otherwise.
  Phenopacket-store  present if observed; absent if explicitly excluded; unknown otherwise.

    python il_sim.py --bench ddxplus-validate --budget 20 [--ddx_n 1000]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opendx_eval as O  # noqa: E402
import v2_ids  # noqa: E402
from graph_dx import GraphDx  # noqa: E402

NEIGH = """
MATCH (d:Disease)-[:HAS_FINDING]->(c:Concept)
WHERE d.id IN $cands AND NOT c.cui IN $asked AND c.idf IS NOT NULL
  AND ($space IS NULL OR c.cui IN $space)
  AND ($tuis IS NULL OR c.tui IN $tuis)
RETURN c.cui AS cui, collect(d.id) AS ds
"""
# Open action space (design D18): only things a patient or an examiner can be asked about -- signs,
# symptoms, findings and conditions -- not tests, procedures or drugs.
ASKABLE_TUI = ["T184", "T033", "T047", "T046", "T048", "T191", "T019", "T020", "T037", "T190"]
HOLDERS = "MATCH (d:Disease)-[:HAS_FINDING]->(:Concept {cui: $cui}) RETURN collect(d.id) AS ds"


def ddxplus_cases(split: str, n: int, seed: int = 0):
    cond = json.load(open("data/ddxplus/release_conditions_en.json"))
    dmap = json.load(open("data/ddxplus/disease_umls_mapping_v2.json"))["mapping"]
    fr2en = {i.get("cond-name-fr") or i.get("condition_name"): en for en, i in cond.items()}
    convert, vocab = O.ddx_crosswalk()
    rows = list(csv.DictReader(open(f"data/ddxplus/release_{split}_patients.csv")))
    rows = random.Random(seed).sample(rows, min(n, len(rows)))
    out = []
    for r in rows:
        gold = dmap.get(fr2en.get(r["PATHOLOGY"]), {}).get("umls_cui")
        ev = convert(ast.literal_eval(r["EVIDENCES"]))
        init = convert([r["INITIAL_EVIDENCE"]])
        if gold and ev and init:
            out.append({"gold": {gold}, "init": init, "pos": ev, "neg": vocab - ev})
    return out, sorted(vocab)


def phenopacket_cases(n: int, seed: int = 0):
    _, hp = O._rb_maps()
    z = zipfile.ZipFile("data/external_benchmarks/phenopacket_store/all_phenopackets.zip")
    names = sorted(x for x in z.namelist() if x.endswith(".json"))
    space = set()
    for nm in names:                      # benchmark action space: every HPO concept it records
        for f in json.loads(z.read(nm)).get("phenotypicFeatures", []):
            c = hp(f["type"]["id"])
            if c:
                space.add(c)
    names = random.Random(seed).sample(names, min(n, len(names)))
    out = []
    for nm in names:
        p = json.loads(z.read(nm))
        gold = {d["term"]["id"] for d in p.get("diseases", [])}
        pos = [hp(f["type"]["id"]) for f in p.get("phenotypicFeatures", []) if not f.get("excluded")]
        neg = {hp(f["type"]["id"]) for f in p.get("phenotypicFeatures", []) if f.get("excluded")}
        pos = [c for c in pos if c]
        if gold and pos:
            out.append({"gold": gold, "init": {pos[0]}, "pos": set(pos), "neg": {c for c in neg if c}})
    return out, sorted(space)


def run_case(dx: GraphDx, case: dict, budget: int, top: int, neg_w: float, holders_cache: dict,
             space=None, tuis=None):
    known_pos, known_neg, asked = set(case["init"]), set(), set(case["init"])

    def ranking():
        r = dx.rank(known_pos, 200)
        if not known_neg:
            return r
        pen = {}
        for c in known_neg:
            if c not in holders_cache:
                with dx.drv.session() as s:
                    holders_cache[c] = set(s.run(HOLDERS, cui=c).single()["ds"])
            for d in holders_cache[c]:
                pen[d] = pen.get(d, 0) + 1
        # rank-preserving penalty: each denied finding the candidate predicts moves it down
        base = {d: i for i, d in enumerate(r)}
        return sorted(r, key=lambda d: (base[d] + neg_w * pen.get(d, 0) * len(r) / 10, d))

    ranked = ranking()
    for _ in range(budget):
        cands = ranked[:top]
        if not cands:
            break
        wt = {d: 1.0 / (i + 1) for i, d in enumerate(cands)}      # rank-based belief
        tot = sum(wt.values())
        with dx.drv.session() as s:
            rows = list(s.run(NEIGH, cands=cands, asked=sorted(asked), space=space, tuis=tuis))
        if not rows:
            break
        # most balanced split of the current belief mass (max binary entropy)
        def gain(ds):
            p = sum(wt[d] for d in ds) / tot
            return -(p * math.log(p) + (1 - p) * math.log(1 - p)) if 0 < p < 1 else 0.0
        best = max(rows, key=lambda r: (gain(r["ds"]), r["cui"]))
        q = best["cui"]
        asked.add(q)
        if q in case["pos"]:
            known_pos.add(q)
        elif q in case["neg"]:
            known_neg.add(q)
        ranked = ranking()
    return ranked, len(asked) - len(case["init"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="ddxplus-validate")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--neg_w", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--mu", type=float, default=2.0)
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--unified_w", type=float, default=1.0)
    ap.add_argument("--support_beta", type=float, default=0.0)
    ap.add_argument("--offtopic_w", type=float, default=1.0)
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--space", choices=["open", "benchmark"], default="open",
                    help="open: any askable concept (sign/symptom/finding/condition); benchmark: only "
                         "concepts the benchmark records (its questionnaire / phenotype vocabulary)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    O.RAW_GOLD = True
    dx = GraphDx("bolt://localhost:7687", a.auth, a.alpha, a.mu, a.decay, a.unified_w, a.support_beta, a.offtopic_w)
    cases, vocab = (ddxplus_cases(a.bench.split("-", 1)[1], a.n) if a.bench.startswith("ddxplus-")
                    else phenopacket_cases(a.n))
    space = vocab if a.space == "benchmark" else None
    tuis = None if a.space == "benchmark" else ASKABLE_TUI
    hits = {k: 0 for k in O.KS}
    il, m, cache = [], 0, {}
    for case in cases:
        gold = v2_ids.gold_keys(case["gold"])
        if not gold:
            continue
        m += 1
        ranked, n_q = run_case(dx, case, a.budget, a.top, a.neg_w, cache, space, tuis)
        lin = dx.lineage(gold)
        pos = next((i + 1 for i, d in enumerate(ranked) if d in lin), None)
        for k in O.KS:
            hits[k] += pos is not None and pos <= k
        il.append(n_q)
    res = {"bench": a.bench, "space": a.space, "n": m, "budget": a.budget, "mean_IL": sum(il) / max(1, len(il)),
           **{f"lineage@{k}": round(100 * hits[k] / max(1, m), 2) for k in O.KS}}
    print(json.dumps(res))
    if a.out:
        json.dump({"args": vars(a), "result": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
