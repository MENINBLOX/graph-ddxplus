#!/usr/bin/env python3
"""v90 — Query-time UMLS parent expansion.

학술적 정당:
- KG (v85): LLM IE에서 도출된 disease-phenotype edges (benchmark-blind).
- 추론 시점: patient evidence CUI + UMLS parents (depth=1) — UMLS는 표준 ontology.
- v89과 차이: KG 무결성 유지, broad hierarchy로 IDF 망치지 않음.

Inference flow:
  pat_cui_set = {C1, C2, ...} from benchmark
  expanded = {Ci} ∪ {parent(Ci) for each Ci, depth=1}
  score(D) = cosine(expanded, profile[D])

Use UMLS RB (broader than) + PAR (parent) relations.
"""
from __future__ import annotations
import sys, json, math, pickle, argparse, csv, ast, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
EV_META = "data/ddxplus/release_evidences.json"


def load_umls_parents(mrrel_path, restrict_cuis=None):
    """Build child → parents map. RB (broader) + PAR relations.
    If restrict_cuis given, only return entries for CUIs in that set."""
    parents = defaultdict(set)
    print(f"  Scanning MRREL for child → parents (RB, PAR)...", flush=True)
    n = 0
    with open(mrrel_path) as f:
        for line in f:
            n += 1
            parts = line.split("|")
            if len(parts) < 5: continue
            cui1, _, _, rel, cui2 = parts[0], parts[1], parts[2], parts[3], parts[4]
            if rel not in {"RB", "PAR"}: continue
            if restrict_cuis is not None and cui1 not in restrict_cuis: continue
            if cui1 == cui2: continue
            parents[cui1].add(cui2)
            if n % 10_000_000 == 0:
                print(f"    {n//1_000_000}M lines, {len(parents):,} CUIs",
                      flush=True)
    print(f"  Done: {len(parents):,} CUIs have parents", flush=True)
    return parents


def build_profile(G, dcs, kappa, pr):
    profile = {}; all_evs = set()
    allowed = {"patient_reportable", "history", "demographic"}
    for d in dcs:
        if d not in G: profile[d] = {}; continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None:
                if p not in pr: continue
            elif cat not in allowed: continue
            ed_w[p] += ed.get("weight", 0.0)
        prof = {p: w/(w+kappa) for p, w in ed_w.items() if w > 0}
        profile[d] = prof
        all_evs.update(prof.keys())
    return profile, all_evs


def compute_idf(profile, df_threshold):
    N = len(profile)
    df = defaultdict(int)
    for prof in profile.values():
        for e, p in prof.items():
            if p >= df_threshold: df[e] += 1
    return {e: math.log((N+1)/(df_e+1))+1.0 for e, df_e in df.items()}


def reweight(profile, idf, alpha, beta):
    return {d: {e: (p**alpha)*(idf.get(e,1.0)**beta) for e,p in prof.items()}
            for d, prof in profile.items()}


def expand_query(pcuis, parents, decay):
    """For each X in pcuis, add parents(X) with decayed weight 1.0 * decay.
    Returns dict[cui] -> weight (1.0 for direct, decay for parent)."""
    weights = {c: 1.0 for c in pcuis}
    for c in pcuis:
        for p in parents.get(c, ()):
            if p not in weights:
                weights[p] = decay
            # if already in (direct or other parent), keep max
    return weights


def score_cosine_weighted(weighted_query, profile, idf, beta):
    """Cosine where query has variable weights (1.0 direct, decay parent)."""
    scores = {}
    # patient vector with IDF
    pat_vec = {e: w * (idf.get(e, 1.0) ** beta) for e, w in weighted_query.items()}
    p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
    for d, prof in profile.items():
        if not prof: scores[d] = -1e9; continue
        dot = sum(pat_vec[e] * prof[e] for e in pat_vec if e in prof)
        d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        scores[d] = dot / (p_norm * d_norm)
    return scores


def precompute_signal_v71(profile, value_cuis, binary_evs, idf, tau, sharpness):
    """Same as v71."""
    signal = defaultdict(dict)
    for ev_id in binary_evs:
        m = value_cuis.get(ev_id, {})
        cuis = set(m.get("_question", []))
        for d, prof in profile.items():
            best = 0.0
            for c in cuis:
                if c in prof:
                    idf_c = idf.get(c, 1.0)
                    factor = 1.0 / (1.0 + math.exp((idf_c - tau) / sharpness))
                    val = prof[c] * factor
                    if val > best: best = val
            if best > 0:
                signal[d][ev_id] = best
    return signal


def score_v71_with_expansion(weighted_query, neg_binary, profile, idf, beta, signal, lam):
    scores = {}
    pat_vec = {e: w * (idf.get(e, 1.0) ** beta) for e, w in weighted_query.items()}
    p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
    for d, prof in profile.items():
        if not prof: scores[d] = -1e9; continue
        dot = sum(pat_vec[e] * prof[e] for e in pat_vec if e in prof)
        d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        pos_score = dot / (p_norm * d_norm)
        sig = signal.get(d, {})
        neg_pen = sum(sig.get(ev, 0.0) for ev in neg_binary)
        neg_norm = math.sqrt(len(neg_binary)) or 1e-9
        neg_score = neg_pen / (neg_norm * d_norm)
        scores[d] = pos_score - lam * neg_score
    return scores


def load_ddxplus_full(n_max):
    value_cuis = json.load(open(VALUE_CUIS))
    ev_meta = json.load(open(EV_META))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    binary_evs = {ev_id for ev_id, m in ev_meta.items()
                  if m.get("data_type") == "B" and m.get("default_value") == 0}
    patients = []; n = 0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= n_max: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pos_pcuis = set()
            answered_binary = set()
            for ev in evs:
                if "_@_" in ev:
                    base, val = ev.split("_@_", 1)
                    m = value_cuis.get(base, {})
                    for k in ("_question", val):
                        v = m.get(k, [])
                        if isinstance(v, list): pos_pcuis.update(v)
                else:
                    if ev in binary_evs: answered_binary.add(ev)
                    m = value_cuis.get(ev, {})
                    pos_pcuis.update(m.get("_question", []))
            neg_binary = binary_evs - answered_binary
            patients.append((true_cui, pos_pcuis, neg_binary)); n += 1
    return dcs_list, patients, binary_evs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="pilot/data/onlykg_graph_v85_s3.pkl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--decay_sweep", type=str, default="0.0,0.2,0.4,0.6")
    ap.add_argument("--mrrel", default="/windows/data/umls_subset/MRREL.RRF")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))

    print(f"=== v90 query-time UMLS expansion — N={args.n}, KG={args.graph} ===", flush=True)
    dcs_list, patients, _ = load_ddxplus_full(args.n)
    print(f"  Patients: {len(patients)}", flush=True)
    base, all_evs = build_profile(G, dcs_list, 20.0, pr)
    idf = compute_idf(base, 0.12)
    profile = reweight(base, idf, 1.0, args.beta)
    print(f"  KG profile CUIs (lay mode): {len(all_evs)}", flush=True)

    # Restrict MRREL load to CUIs that appear as patient evidence (small set)
    # — saves time vs loading full MRREL
    print(f"\n--- Building parent map (restricted) ---", flush=True)
    pat_cui_set = set()
    for _, pos, _ in patients:
        pat_cui_set.update(pos)
    print(f"  Unique patient CUIs: {len(pat_cui_set)}", flush=True)
    parents = load_umls_parents(args.mrrel, restrict_cuis=pat_cui_set)

    n_with_parents = sum(1 for c in pat_cui_set if c in parents)
    print(f"  Patient CUIs with at least one parent: {n_with_parents}/{len(pat_cui_set)} "
          f"({100*n_with_parents/len(pat_cui_set):.1f}%)", flush=True)

    # v71 signal for DDXPlus negative penalty
    value_cuis = json.load(open(VALUE_CUIS))
    ev_meta = json.load(open(EV_META))
    binary_evs = {ev_id for ev_id, m in ev_meta.items()
                  if m.get("data_type") == "B" and m.get("default_value") == 0}
    print(f"\n--- Computing v71 signal (tau=2.0, sharp=0.5) ---", flush=True)
    signal = precompute_signal_v71(profile, value_cuis, binary_evs, idf, 2.0, 0.5)
    lam = 0.4

    # Evaluate each decay
    for decay_s in args.decay_sweep.split(","):
        decay = float(decay_s)
        n = c1 = c3 = c5 = c10 = 0; rr = 0.0
        for true_cui, pos_raw, neg_binary in patients:
            if decay > 0:
                weighted = expand_query(pos_raw, parents, decay)
                weighted = {c: w for c, w in weighted.items() if c in all_evs}
            else:
                weighted = {c: 1.0 for c in pos_raw if c in all_evs}
            if not weighted: continue
            s = score_v71_with_expansion(weighted, neg_binary, profile, idf,
                                          args.beta, signal, lam)
            ranked = sorted(profile.keys(), key=lambda d: -s[d])
            n += 1
            try: rank = ranked.index(true_cui)+1
            except: rank = len(dcs_list)
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr += 1.0/rank
        print(f"  decay={decay:.2f}: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% "
              f"@5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr/n:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
