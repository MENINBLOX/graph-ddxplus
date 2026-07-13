#!/usr/bin/env python3
"""Q-aware scoring on cross-benchmarks (SymCat, RareBench).

Same Q-aware logic but adapted to each benchmark's structure.
"""
from __future__ import annotations
import sys, json, math, random, pickle, csv, ast
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph_v4.pkl"
random.seed(42)


def build_d_q(G, dcs_list, Q, hop2_decay=0.5):
    d_q = {}
    for d in dcs_list:
        if d not in G: d_q[d] = {}; continue
        phen_w = {}
        for _, p, edata in G.out_edges(d, data=True):
            if edata.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + edata.get("weight", 0)
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, edata2 in G.out_edges(p_direct, data=True):
                if edata2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + hop2_decay * dw * edata2.get("weight", 0)
        d_q[d] = {p: w for p, w in phen_w.items() if p in Q}
    return d_q


def score(d_q, dcs_list, pcuis):
    out = {}
    for d in dcs_list:
        qp = d_q.get(d, {})
        if not qp: out[d] = -1e6; continue
        pos = sum(w for q, w in qp.items() if q in pcuis)
        total = sum(qp.values())
        out[d] = pos / (math.sqrt(total) or 1)
    return out


def eval_symcat(G):
    sc = json.load(open("data/symcat/symcat_parsed.json"))
    sym_umls = json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    dis_umls = json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]

    candidates = {}
    for dn, info in dis_umls.items():
        cui = info.get("umls_cui")
        if cui and cui in G:
            candidates[dn] = cui
    dcs_list = sorted(set(candidates.values()))

    sym2cui = {sn: info.get("umls_cui") for sn, info in sym_umls.items() if info.get("umls_cui")}
    # Q = all SymCat symptom CUIs ∩ KG phenotypes
    Q = set(sym2cui.values())
    d_q = build_d_q(G, dcs_list, Q)

    pairs = sc["disease_symptom_pairs"]
    n = 0; c1=c3=c5=c10=0; rr_sum=0
    for dn, sym_list in pairs.items():
        if dn not in candidates: continue
        true_cui = candidates[dn]
        if not sym_list: continue
        sym_cuis = [(sym2cui.get(s), f) for s, f in sym_list if sym2cui.get(s)]
        if not sym_cuis: continue
        for _ in range(50):
            pcuis = set()
            for cui, freq in sym_cuis:
                if random.random() * 100 < freq: pcuis.add(cui)
            if not pcuis: continue
            scores = score(d_q, dcs_list, pcuis)
            ranked = sorted(dcs_list, key=lambda d: -scores.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = len(dcs_list)
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank
    print(f"SymCat (Q-aware, n={n}, |candidates|={len(dcs_list)}): @1={100*c1/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


def eval_rarebench(G):
    hpo_map = json.load(open("data/rarebench/hpo_umls_mapping.json"))["mapping"]
    dis_map = json.load(open("data/rarebench/disease_umls_mapping.json"))["mapping"]

    patients = []
    for ds in ["HMS", "LIRICAL", "MME", "RAMEDIS"]:
        p = Path(f"data/rarebench/data/{ds}.jsonl")
        if not p.exists(): continue
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                target_cuis = []
                for t in r["RareDisease"]:
                    info = dis_map.get(t)
                    if info and info.get("umls_cui"):
                        target_cuis.append(info["umls_cui"])
                if not target_cuis: continue
                pcuis = set()
                for hp in r["Phenotype"]:
                    info = hpo_map.get(hp)
                    c = info.get("umls_cui") if isinstance(info, dict) else info if isinstance(info, str) else None
                    if c: pcuis.add(c)
                if pcuis: patients.append((target_cuis, pcuis))

    all_targets = set()
    for ts, _ in patients: all_targets.update(ts)
    candidates = sorted([c for c in all_targets if c in G])

    Q = set()
    for ts, pcuis in patients: Q.update(pcuis)
    d_q = build_d_q(G, candidates, Q)

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    for target_cuis, pcuis in patients:
        target_in = [t for t in target_cuis if t in candidates]
        if not target_in: n += 1; continue
        scores = score(d_q, candidates, pcuis)
        ranked = sorted(candidates, key=lambda d: -scores.get(d, -1e9))
        ranks = []
        for tc in target_in:
            try: ranks.append(ranked.index(tc) + 1)
            except: ranks.append(len(candidates))
        rank = min(ranks)
        n += 1
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr_sum += 1.0/rank
    print(f"RareBench (Q-aware, n={n}, |candidates|={len(candidates)}): @1={100*c1/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


def main():
    with GRAPH.open("rb") as f:
        G = pickle.load(f)
    eval_symcat(G)
    eval_rarebench(G)


if __name__ == "__main__":
    main()
