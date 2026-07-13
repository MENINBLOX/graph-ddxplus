#!/usr/bin/env python3
"""v100 — Value-weighted patient evidence vector for numeric scale evidences.

System-level design (not DDXPlus-specific):
- For any evidence with numeric scale (0-N integer values), weight CUI
  contribution by value/max_value.
- Universal pattern: VAS pain, PHQ-9, GAD-7, BPI, Apgar — all medical
  ordinal severity scales.
- Aggregate multiple tokens hitting same CUI via max().
- Binary YES → weight 1.0 (default).
- Numeric value=0 → weight 0 (effectively no signal).

Algorithm: identical to v71 (cosine + IDF + self-aware negative penalty),
except patient vector weights:
  pat_vec[CUI] = max_over_tokens(token_weight) * (idf[CUI] ** beta)
where token_weight = 1.0 for binary, val/max_val for numeric.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
EV_META = "data/ddxplus/release_evidences.json"


def build_profile(G, dcs, kappa, pr, top_k=None):
    profile = {}; all_evs = set()
    allowed = {"patient_reportable","history","demographic"}
    for d in dcs:
        if d not in G: profile[d]={}; continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None and p not in pr: continue
            if cat is not None and cat not in allowed: continue
            ed_w[p] += ed.get("weight", 0.0)
        prof = {p: w/(w+kappa) for p, w in ed_w.items() if w > 0}
        if top_k and len(prof) > top_k:
            prof = dict(sorted(prof.items(), key=lambda x: -x[1])[:top_k])
        profile[d] = prof
        all_evs.update(prof.keys())
    return profile, all_evs


def compute_idf(profile, df_threshold):
    N = len(profile); df = defaultdict(int)
    for prof in profile.values():
        for e, p in prof.items():
            if p >= df_threshold: df[e] += 1
    return {e: math.log((N+1)/(df_e+1))+1.0 for e, df_e in df.items()}


def reweight(profile, idf, alpha, beta):
    return {d:{e:(p**alpha)*(idf.get(e,1.0)**beta) for e,p in prof.items()}
            for d, prof in profile.items()}


def precompute_signal_v71(profile, value_cuis, binary_evs, idf, tau, sharpness):
    signal = defaultdict(dict)
    for ev_id in binary_evs:
        m = value_cuis.get(ev_id, {})
        cuis = set(m.get("_question", []))
        for d, prof in profile.items():
            best = 0.0
            for c in cuis:
                if c in prof:
                    idf_c = idf.get(c, 1.0)
                    factor = 1.0 / (1.0 + math.exp((idf_c - tau)/sharpness))
                    val = prof[c] * factor
                    if val > best: best = val
            if best > 0:
                signal[d][ev_id] = best
    return signal


def detect_numeric_evidences(ev_meta):
    """Find evidences where possible-values is a list of 0-N consecutive integers."""
    numeric = {}  # ev_id -> max_value
    for ev_id, m in ev_meta.items():
        pv = m.get("possible-values", [])
        if not pv: continue
        try:
            nums = sorted(int(v) for v in pv)
            if len(nums) >= 3 and nums[0] == 0 and nums == list(range(nums[0], nums[-1]+1)):
                numeric[ev_id] = nums[-1]  # max value (e.g., 10)
        except: continue
    return numeric


def load_ddxplus_value_weighted(n_max, numeric_evs):
    """Load patients with value-weighted CUI extraction.
    Returns: dcs_list, patients (each = (true_cui, pos_weighted_dict, neg_binary)),
    binary_evs. pos_weighted_dict: CUI -> weight in [0,1]."""
    value_cuis = json.load(open(VALUE_CUIS))
    ev_meta = json.load(open(EV_META))
    icd = json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    cond = json.load(open("data/ddxplus/release_conditions_en.json"))
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"]
              for dn,info in cond.items() if dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    binary_evs = {e for e, m in ev_meta.items()
                  if m.get("data_type") == "B" and m.get("default_value") == 0}

    patients = []; n = 0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= n_max: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pos_weighted = {}  # cui -> weight (max over tokens)
            answered = set()
            for ev in evs:
                if "_@_" in ev:
                    base, val = ev.split("_@_", 1)
                    m = value_cuis.get(base, {})
                    # Determine weight
                    if base in numeric_evs:
                        try:
                            v_int = int(val)
                            weight = v_int / numeric_evs[base]
                        except:
                            weight = 1.0  # fallback if not parseable
                    else:
                        weight = 1.0
                    # Add CUIs
                    cui_set = set()
                    for k in ("_question", val):
                        v = m.get(k, [])
                        if isinstance(v, list): cui_set.update(v)
                    for c in cui_set:
                        if weight > pos_weighted.get(c, 0):
                            pos_weighted[c] = weight
                else:
                    # Pure binary token
                    if ev in binary_evs: answered.add(ev)
                    m = value_cuis.get(ev, {})
                    for c in m.get("_question", []):
                        pos_weighted[c] = 1.0  # binary YES = weight 1
            neg_binary = binary_evs - answered
            patients.append((true_cui, pos_weighted, neg_binary)); n += 1
    return dcs_list, patients, binary_evs


def score_v100(pos_weighted, neg_binary, profile_rw, idf, beta, signal, lam):
    """Cosine with value-weighted patient vector + v71 negative penalty."""
    # Patient vector: weight * idf**beta
    pat_vec = {e: w * (idf.get(e, 1.0)**beta) for e, w in pos_weighted.items()}
    p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
    scores = {}
    for d, prof in profile_rw.items():
        if not prof: scores[d] = -1e9; continue
        d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        dot = sum(pat_vec[e] * prof[e] for e in pat_vec if e in prof)
        pos_s = dot / (p_norm * d_norm)
        sig = signal.get(d, {})
        neg_pen = sum(sig.get(ev, 0.0) for ev in neg_binary)
        neg_norm = math.sqrt(len(neg_binary)) or 1e-9
        neg_s = neg_pen / (neg_norm * d_norm)
        scores[d] = pos_s - lam * neg_s
    return scores


def evaluate(profile_rw, idf, beta, signal, patients, all_evs, dcs_list, lam):
    n = c1 = c3 = c5 = c10 = 0; rr = 0.0
    for true_cui, pos_w, neg_b in patients:
        pos = {c: w for c, w in pos_w.items() if c in all_evs and w > 0}
        if not pos: continue
        s = score_v100(pos, neg_b, profile_rw, idf, beta, signal, lam)
        ranked = sorted(profile_rw.keys(), key=lambda d: -s[d])
        n += 1
        try: rk = ranked.index(true_cui) + 1
        except: rk = len(dcs_list)
        if rk == 1: c1 += 1
        if rk <= 3: c3 += 1
        if rk <= 5: c5 += 1
        if rk <= 10: c10 += 1
        rr += 1.0/rk
    return {"n": n, "at1": 100*c1/n, "at3": 100*c3/n,
            "at5": 100*c5/n, "at10": 100*c10/n, "mrr": rr/n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--top_k", type=int, default=999)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--tau", type=float, default=1.5)
    ap.add_argument("--sharp", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.4)
    args = ap.parse_args()

    print(f"=== v100 value-weighted (N={args.n}) ===", flush=True)
    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))
    ev_meta = json.load(open(EV_META))
    value_cuis = json.load(open(VALUE_CUIS))

    numeric_evs = detect_numeric_evidences(ev_meta)
    print(f"  Detected {len(numeric_evs)} numeric-scale evidences:", flush=True)
    for ev, mx in numeric_evs.items():
        print(f"    {ev}: 0..{mx}", flush=True)

    dcs_list, patients, binary_evs = load_ddxplus_value_weighted(args.n, numeric_evs)
    base, all_evs = build_profile(G, dcs_list, 20.0, pr, top_k=args.top_k)
    idf = compute_idf(base, 0.12)
    profile = reweight(base, idf, 1.0, args.beta)
    signal = precompute_signal_v71(profile, value_cuis, binary_evs, idf, args.tau, args.sharp)

    r = evaluate(profile, idf, args.beta, signal, patients, all_evs, dcs_list, args.lam)
    print(f"\n  tau={args.tau} sharp={args.sharp} lam={args.lam} beta={args.beta}:")
    print(f"  @1={r['at1']:.2f}% @3={r['at3']:.2f}% @5={r['at5']:.2f}% "
          f"@10={r['at10']:.2f}% MRR={r['mrr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
