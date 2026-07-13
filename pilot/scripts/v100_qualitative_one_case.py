#!/usr/bin/env python3
"""seed=42 anaphylaxis case로 v71 (binary) vs v100 (value-weighted) 비교."""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from v100_value_weighted import (
    build_profile, compute_idf, reweight, precompute_signal_v71,
    detect_numeric_evidences, load_ddxplus_value_weighted, score_v100
)

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
EV_META = "data/ddxplus/release_evidences.json"


def score_v71_baseline(pos_set, neg_binary, profile_rw, idf, beta, signal, lam):
    """v71: binary patient vector (no value weighting)."""
    pat_vec = {e: idf.get(e, 1.0)**beta for e in pos_set}
    p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
    scores = {}
    for d, prof in profile_rw.items():
        if not prof: scores[d] = -1e9; continue
        d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        dot = sum(pat_vec[e] * prof[e] for e in pos_set if e in prof)
        pos_s = dot / (p_norm * d_norm)
        sig = signal.get(d, {})
        neg_pen = sum(sig.get(ev, 0.0) for ev in neg_binary)
        neg_norm = math.sqrt(len(neg_binary)) or 1e-9
        neg_s = neg_pen / (neg_norm * d_norm)
        scores[d] = pos_s - lam * neg_s
    return scores


def main():
    G = pickle.load(open("pilot/data/onlykg_graph_v95_full_s3.pkl", "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))
    ev_meta = json.load(open(EV_META))
    value_cuis = json.load(open(VALUE_CUIS))
    cond = json.load(open("data/ddxplus/release_conditions_en.json"))
    icd = json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    fr2en = {info.get("cond-name-fr",""): dn for dn, info in cond.items()}

    numeric_evs = detect_numeric_evidences(ev_meta)
    dcs_list, patients, binary_evs = load_ddxplus_value_weighted(10000, numeric_evs)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"]
              for dn,info in cond.items() if dn in icd}
    cui2en = {v: fr2en.get(k, k) for k,v in fr2cui.items()}

    # Pick seed=42 patient (same as docs/ddxplus_case_example_seed42.md)
    rng = random.Random(42)
    pool = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            pool.append(row)
            if len(pool) >= 10000: break
    case_row = rng.choice(pool)
    truth_fr = case_row["PATHOLOGY"]
    truth_cui = fr2cui.get(truth_fr)
    truth_en = fr2en.get(truth_fr, truth_fr)

    # Find this patient in the loaded patients (match by true_cui and evidence count)
    # Easiest: rebuild this single patient via load
    case_idx = pool.index(case_row)
    # We need to re-extract just this patient's pos_weighted dict
    # Re-do extraction from row directly
    evs = ast.literal_eval(case_row["EVIDENCES"])
    pos_weighted = {}
    answered = set()
    pos_binary = set()  # for v71 (no weight)
    for ev in evs:
        if "_@_" in ev:
            base, val = ev.split("_@_", 1)
            m = value_cuis.get(base, {})
            if base in numeric_evs:
                try:
                    v_int = int(val)
                    weight = v_int / numeric_evs[base]
                except: weight = 1.0
            else:
                weight = 1.0
            cui_set = set()
            for k in ("_question", val):
                v = m.get(k, [])
                if isinstance(v, list): cui_set.update(v)
            for c in cui_set:
                if weight > pos_weighted.get(c, 0):
                    pos_weighted[c] = weight
                pos_binary.add(c)
        else:
            if ev in binary_evs: answered.add(ev)
            for c in value_cuis.get(ev, {}).get("_question", []):
                pos_weighted[c] = 1.0
                pos_binary.add(c)
    neg_binary = binary_evs - answered

    # Profile + idf + scoring
    base_prof, all_evs = build_profile(G, dcs_list, 20.0, pr, top_k=999)
    idf = compute_idf(base_prof, 0.12)
    profile_rw = reweight(base_prof, idf, 1.0, 0.75)
    signal = precompute_signal_v71(profile_rw, value_cuis, binary_evs, idf, 1.5, 0.5)

    # v71 baseline
    pos_v71 = pos_binary & all_evs
    s_v71 = score_v71_baseline(pos_v71, neg_binary, profile_rw, idf, 0.75, signal, 0.4)
    rank_v71 = sorted(profile_rw.keys(), key=lambda d: -s_v71[d])

    # v100 value-weighted
    pos_v100 = {c: w for c, w in pos_weighted.items() if c in all_evs and w > 0}
    s_v100 = score_v100(pos_v100, neg_binary, profile_rw, idf, 0.75, signal, 0.4)
    rank_v100 = sorted(profile_rw.keys(), key=lambda d: -s_v100[d])

    print(f"=== seed=42 patient comparison ===")
    print(f"Truth: {truth_en} ({truth_cui})\n")

    print(f"--- Patient vector comparison ---")
    print(f"v71 binary | v100 weighted | CUI | name")
    common = set(pos_binary) & set(pos_weighted) & all_evs
    diff = set(pos_binary) - set(pos_weighted)
    for c in sorted(common, key=lambda x: -pos_weighted.get(x,0)):
        name = G.nodes[c].get("name", c) if c in G else c
        v_w = pos_weighted.get(c, 0)
        print(f"  1.0   |  {v_w:.2f}        | {c} | {name[:50]}")

    print(f"\n--- v71 binary ranking (top-5) ---")
    for i, d in enumerate(rank_v71[:5]):
        mark = " ⭐ TRUTH" if d == truth_cui else ""
        print(f"  [{i+1}] {cui2en.get(d, d)} ({d}): {s_v71[d]:.4f}{mark}")
    rk_v71 = rank_v71.index(truth_cui)+1 if truth_cui in rank_v71 else 'N/A'

    print(f"\n--- v100 value-weighted ranking (top-5) ---")
    for i, d in enumerate(rank_v100[:5]):
        mark = " ⭐ TRUTH" if d == truth_cui else ""
        print(f"  [{i+1}] {cui2en.get(d, d)} ({d}): {s_v100[d]:.4f}{mark}")
    rk_v100 = rank_v100.index(truth_cui)+1 if truth_cui in rank_v100 else 'N/A'

    print(f"\n--- Summary ---")
    print(f"  True disease rank — v71: {rk_v71}, v100: {rk_v100}")
    print(f"  v71 truth score: {s_v71.get(truth_cui, '?'):.4f}")
    print(f"  v100 truth score: {s_v100.get(truth_cui, '?'):.4f}")


if __name__ == "__main__":
    main()
