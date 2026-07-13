#!/usr/bin/env python3
"""v73 — Multi-choice value-level negative evidence on top of v71.

v71 only handled binary evidences (208). Multi-choice (5) and categorical
(10) evidences contain their own negative information: a patient who
selects 'lancinante' as pain character implicitly denies 15 other
character options.

We focus on `douleurxx_carac` (16 pain-character options) and
`lesions_peau_couleur` (6 rash-color options) — these are clinically
disease-discriminating. Position evidences (165 values) are too noisy
to penalize comprehensively.

For each non-selected value v of a multi-choice evidence ev:
  if value_cuis[ev][v] has CUIs in profile_D → contribute to neg_pen
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

# Discriminative multi-choice evidences (small value set, clinically meaningful)
DISCRIMINATIVE_MULTI = {"douleurxx_carac", "lesions_peau_couleur"}


def build_profile(G, dcs, kappa, pr, top_k=None):
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
        if top_k and len(prof) > top_k:
            prof = dict(sorted(prof.items(), key=lambda x: -x[1])[:top_k])
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


def precompute_binary_signal(profile, value_cuis, binary_evs, idf, tau, sharp):
    signal = defaultdict(dict)
    for ev_id in binary_evs:
        m = value_cuis.get(ev_id, {})
        cuis = set(m.get("_question", []))
        for d, prof in profile.items():
            best = 0.0
            for c in cuis:
                if c in prof:
                    idf_c = idf.get(c, 1.0)
                    factor = 1.0 / (1.0 + math.exp((idf_c - tau) / sharp))
                    val = prof[c] * factor
                    if val > best: best = val
            if best > 0: signal[d][ev_id] = best
    return signal


def precompute_multi_signal(profile, value_cuis, ev_meta, idf, tau, sharp):
    """For each multi-choice evidence in DISCRIMINATIVE_MULTI, precompute
    per-disease per-value signal."""
    signal = defaultdict(lambda: defaultdict(dict))  # signal[d][ev_id][val]
    for ev_id in DISCRIMINATIVE_MULTI:
        if ev_id not in ev_meta: continue
        m = value_cuis.get(ev_id, {})
        values = ev_meta[ev_id].get("possible-values", [])
        default = ev_meta[ev_id].get("default_value")
        for val in values:
            if val == default: continue
            val_cuis = set(m.get(str(val), []))
            if not val_cuis: continue
            for d, prof in profile.items():
                best = 0.0
                for c in val_cuis:
                    if c in prof:
                        idf_c = idf.get(c, 1.0)
                        factor = 1.0 / (1.0 + math.exp((idf_c - tau) / sharp))
                        v = prof[c] * factor
                        if v > best: best = v
                if best > 0:
                    signal[d][ev_id][val] = best
    return signal


def load_ddxplus(n_max):
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
            answered_multi = defaultdict(set)  # ev_id -> set of selected values
            for ev in evs:
                if "_@_" in ev:
                    base, val = ev.split("_@_", 1)
                    if base in DISCRIMINATIVE_MULTI:
                        answered_multi[base].add(val)
                    m = value_cuis.get(base, {})
                    for k in ("_question", val):
                        v = m.get(k, [])
                        if isinstance(v, list): pos_pcuis.update(v)
                else:
                    if ev in binary_evs: answered_binary.add(ev)
                    m = value_cuis.get(ev, {})
                    pos_pcuis.update(m.get("_question", []))
            neg_binary = binary_evs - answered_binary
            patients.append((true_cui, pos_pcuis, neg_binary, dict(answered_multi))); n += 1
    return dcs_list, patients, binary_evs


def score(pos_pcuis, neg_binary, answered_multi, profile, idf, beta,
          bin_signal, multi_signal, lam_bin, lam_multi, ev_meta):
    scores = {}
    pat_vec = {e: idf.get(e, 1.0) ** beta for e in pos_pcuis}
    p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
    # Pre-collect non-selected values per multi ev (constant per patient)
    multi_neg = {}
    for ev_id in DISCRIMINATIVE_MULTI:
        if ev_id in ev_meta:
            all_vals = set(ev_meta[ev_id].get("possible-values", []))
            default = ev_meta[ev_id].get("default_value")
            if default in all_vals: all_vals.discard(default)
            selected = set(answered_multi.get(ev_id, set()))
            multi_neg[ev_id] = all_vals - selected
    for d, prof in profile.items():
        if not prof: scores[d] = -1e9; continue
        dot = sum(pat_vec[e] * prof[e] for e in pos_pcuis if e in prof)
        d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        pos_score = dot / (p_norm * d_norm)
        # Binary negative
        sig = bin_signal.get(d, {})
        neg_b = sum(sig.get(ev, 0.0) for ev in neg_binary)
        neg_b_norm = math.sqrt(len(neg_binary)) or 1e-9
        neg_b_score = neg_b / (neg_b_norm * d_norm)
        # Multi-choice negative
        neg_m = 0.0; neg_m_count = 0
        for ev_id, unsel in multi_neg.items():
            ev_sig = multi_signal.get(d, {}).get(ev_id, {})
            for val in unsel:
                if val in ev_sig:
                    neg_m += ev_sig[val]
                    neg_m_count += 1
        neg_m_norm = math.sqrt(max(neg_m_count, 1))
        neg_m_score = neg_m / (neg_m_norm * d_norm)
        scores[d] = pos_score - lam_bin * neg_b_score - lam_multi * neg_m_score
    return scores


def evaluate(profile, idf, beta, bin_sig, multi_sig, patients, all_evs, dcs_list,
             lam_bin, lam_multi, ev_meta):
    n = c1 = c3 = c5 = c10 = 0; rr = 0.0
    for true_cui, pos_raw, neg_binary, answered_multi in patients:
        pos = pos_raw & all_evs
        if not pos: continue
        s = score(pos, neg_binary, answered_multi, profile, idf, beta,
                  bin_sig, multi_sig, lam_bin, lam_multi, ev_meta)
        ranked = sorted(profile.keys(), key=lambda d: -s[d])
        n += 1
        try: rank = ranked.index(true_cui)+1
        except: rank = len(dcs_list)
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr += 1.0/rank
    return {"n": n, "at1": 100*c1/n, "at3": 100*c3/n, "at5": 100*c5/n,
            "at10": 100*c10/n, "mrr": rr/n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--top_k", type=int, default=80)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--sharp", type=float, default=0.5)
    ap.add_argument("--lam_bin", type=float, default=0.30)
    ap.add_argument("--lam_multi_sweep", type=str, default="0.0,0.05,0.1,0.2,0.3,0.5,1.0")
    args = ap.parse_args()
    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))
    dcs_list, patients, binary_evs = load_ddxplus(args.n)
    base, all_evs = build_profile(G, dcs_list, 20.0, pr, top_k=args.top_k)
    idf = compute_idf(base, 0.12)
    profile = reweight(base, idf, 1.0, args.beta)
    value_cuis = json.load(open(VALUE_CUIS))
    ev_meta = json.load(open(EV_META))
    bin_sig = precompute_binary_signal(profile, value_cuis, binary_evs, idf, args.tau, args.sharp)
    multi_sig = precompute_multi_signal(profile, value_cuis, ev_meta, idf, args.tau, args.sharp)
    print(f"=== v73 multi-choice negative — N={args.n} ===")
    for lam_m in args.lam_multi_sweep.split(","):
        lam_m = float(lam_m)
        r = evaluate(profile, idf, args.beta, bin_sig, multi_sig, patients, all_evs,
                     dcs_list, args.lam_bin, lam_m, ev_meta)
        print(f"  lam_multi={lam_m:.2f}: @1={r['at1']:.2f}% @3={r['at3']:.2f}% "
              f"@5={r['at5']:.2f}% MRR={r['mrr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
