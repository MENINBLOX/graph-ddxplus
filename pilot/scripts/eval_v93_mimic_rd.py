#!/usr/bin/env python3
"""v93 KG on MIMIC-RD initial-diff-diagnosis annotation set.

Public annotations only (text requires MIMIC-IV credential). Each patient
has HPO-extracted phenotypes and a list of ORPHA disease IDs.

Same cosine + IDF algorithm as v71.
"""
from __future__ import annotations
import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/max/Graph-DDXPlus")
PR_UNIVERSE = ROOT / "pilot" / "data" / "pr_universe.json"
HPO_MAP = ROOT / "data" / "rarebench" / "hpo_umls_mapping.json"
DIS_MAP = ROOT / "data" / "rarebench" / "disease_umls_mapping.json"
MIMIC_RD = (ROOT / "data" / "external_benchmarks" / "mimic_rd"
            / "public_data" / "initial_diff_diagnosis_benchmark.json")
DEFAULT_GRAPH = ROOT / "pilot" / "data" / "onlykg_graph_v93_s3.pkl"


def build_profile(G, dcs, kappa, pr_set):
    profile, all_evs = {}, set()
    allowed = {"patient_reportable", "history", "demographic"}
    for d in dcs:
        if d not in G:
            profile[d] = {}; continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None:
                if p not in pr_set: continue
            elif cat not in allowed:
                continue
            ed_w[p] += ed.get("weight", 0.0)
        prof = {p: w / (w + kappa) for p, w in ed_w.items() if w > 0}
        profile[d] = prof
        all_evs.update(prof.keys())
    return profile, all_evs


def compute_idf(profile, df_threshold):
    n = len(profile)
    df = defaultdict(int)
    for prof in profile.values():
        for e, p in prof.items():
            if p >= df_threshold:
                df[e] += 1
    return {e: math.log((n + 1) / (df_e + 1)) + 1.0 for e, df_e in df.items()}


def reweight(profile, idf, alpha, beta):
    return {d: {e: (p ** alpha) * (idf.get(e, 1.0) ** beta)
                for e, p in prof.items()}
            for d, prof in profile.items()}


def score_cosine(pcuis, profile, idf, beta):
    pat_vec = {e: idf.get(e, 1.0) ** beta for e in pcuis}
    p_norm = math.sqrt(sum(v * v for v in pat_vec.values())) or 1e-9
    scores = {}
    for d, prof in profile.items():
        if not prof: scores[d] = -1e9; continue
        dot = sum(pat_vec[e] * prof[e] for e in pcuis if e in prof)
        d_norm = math.sqrt(sum(v * v for v in prof.values())) or 1e-9
        scores[d] = dot / (p_norm * d_norm)
    return scores


def _normalize_orpha_id(raw: str) -> str:
    """RDMA's IDs are inconsistent: '86886', 'Orpha:98375', 'ORPHA:223735'."""
    r = raw.strip()
    if ":" in r:
        prefix, num = r.split(":", 1)
        return f"ORPHA:{num.strip()}"
    # plain digits
    if r.isdigit():
        return f"ORPHA:{r}"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(DEFAULT_GRAPH))
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--df_thr", type=float, default=0.12)
    args = ap.parse_args()

    print(f"Loading v93 KG: {args.graph}", flush=True)
    G = pickle.load(open(args.graph, "rb"))
    print(f"  nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}",
          flush=True)
    pr_set = set(json.load(open(PR_UNIVERSE)))
    hpo_map = json.load(open(HPO_MAP))["mapping"]
    dis_map = json.load(open(DIS_MAP))["mapping"]

    data = json.load(open(MIMIC_RD))
    print(f"Loaded MIMIC-RD: {len(data)} patients")

    patients = []
    for pid, rec in data.items():
        target_cuis = set()
        for raw in rec.get("orpha_codes", []):
            norm = _normalize_orpha_id(raw)
            info = dis_map.get(norm)
            if info and info.get("umls_cui"):
                target_cuis.add(info["umls_cui"])
        pcuis = set()
        for ph in rec.get("matched_phenotypes", []):
            hp = ph.get("hp_id")
            if not hp: continue
            info = hpo_map.get(hp)
            if isinstance(info, dict) and info.get("umls_cui"):
                pcuis.add(info["umls_cui"])
            elif isinstance(info, str):
                pcuis.add(info)
        patients.append((target_cuis, pcuis, pid))

    n_total = len(patients)
    n_with_target = sum(1 for t, _, _ in patients if t)
    n_with_pcuis = sum(1 for _, p, _ in patients if p)
    print(f"  with target CUI: {n_with_target}, with pcuis: {n_with_pcuis}")

    all_targets = set()
    for t, _, _ in patients:
        all_targets.update(t)
    in_kg = sorted(c for c in all_targets if c in G)
    print(f"  unique target CUIs: {len(all_targets)}; in v93 KG: {len(in_kg)}",
          flush=True)

    base, all_evs = build_profile(G, in_kg, args.kappa, pr_set)
    idf = compute_idf(base, args.df_thr)
    profile = reweight(base, idf, args.alpha, args.beta)
    n_nonempty = sum(1 for d in in_kg if profile[d])
    print(f"  non-empty profiles: {n_nonempty}/{len(in_kg)}; |evs|={len(all_evs)}",
          flush=True)

    n = c1 = c3 = c5 = c10 = 0
    empty = unrank = 0
    rr = 0.0
    for tcuis, pcuis, _ in patients:
        if not tcuis: continue
        n += 1
        target_in_pool = [t for t in tcuis if t in in_kg]
        if not target_in_pool:
            unrank += 1; continue
        pmatch = pcuis & all_evs
        if not pmatch:
            empty += 1; rank = len(in_kg)
        else:
            scores = score_cosine(pmatch, profile, idf, args.beta)
            ranked = sorted(in_kg, key=lambda d: -scores.get(d, -1e9))
            ranks = [ranked.index(t) + 1 if t in ranked else len(in_kg)
                     for t in target_in_pool]
            rank = min(ranks)
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr += 1.0 / rank

    print(f"\n=== v93 KG on MIMIC-RD (n={n}, pool={len(in_kg)}) ===")
    print(f"  Empty-evidence  : {empty} ({100*empty/n:.1f}%)" if n else "n=0")
    print(f"  Unranked (target not in KG): {unrank}")
    print(f"  Random baseline @1: {100/len(in_kg):.3f}%" if in_kg else "n/a")
    if n:
        def p(x): return f"{100*x/n:.2f}%"
        print(f"  GTPA: @1={p(c1)}  @3={p(c3)}  @5={p(c5)}  @10={p(c10)}")
        print(f"  MRR : {rr/n:.4f}")


if __name__ == "__main__":
    main()
