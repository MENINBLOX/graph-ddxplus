#!/usr/bin/env python3
"""v71 — Self-aware negative penalty with disease-specificity attenuation.

v69 issue: when the true disease's profile contains a CUI that the patient
denied, that disease is penalized heavily, hurting accuracy on patients
who don't report all expected symptoms.

v71 fix: attenuate penalty when the denied CUI is DISEASE-SPECIFIC to D.
A CUI that's specific to D (high IDF, narrow disease coverage) should not
penalize D — the patient simply isn't reporting it for typical reasons
(question routing, sub-clinical, age-inappropriate).

penalty(D, ev) = signal[D][ev] * attenuation_factor(ev)
attenuation_factor(ev) = sigmoid((avg_idf(ev_cuis) - tau) / sharpness)
  where tau is the IDF threshold above which penalty fades

Lower IDF (common CUI like Pain, Cough) → high attenuation factor (1.0,
penalty applies). Higher IDF (rare CUI like Sputum, Stridor) →
attenuation drops, lessening self-penalty.
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


def precompute_signal_v71(profile, value_cuis, binary_evs, idf, tau, sharpness):
    """signal[d][ev] = max prof[E] * attenuation_factor over E in ev_cuis.
    Attenuation: low IDF → 1.0 (full penalty), high IDF → 0 (no self-penalty)."""
    signal = defaultdict(dict)
    for ev_id in binary_evs:
        m = value_cuis.get(ev_id, {})
        cuis = set(m.get("_question", []))
        for d, prof in profile.items():
            best = 0.0
            for c in cuis:
                if c in prof:
                    idf_c = idf.get(c, 1.0)
                    # attenuation: sigmoid((tau - idf_c) / sharpness)
                    # idf_c < tau → factor ≈ 1 (apply penalty)
                    # idf_c > tau → factor → 0 (skip penalty)
                    factor = 1.0 / (1.0 + math.exp((idf_c - tau) / sharpness))
                    val = prof[c] * factor
                    if val > best: best = val
            if best > 0:
                signal[d][ev_id] = best
    return signal


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


def score(pos_pcuis, neg_binary, profile, idf, beta, signal, lam):
    scores = {}
    pat_vec = {e: idf.get(e, 1.0) ** beta for e in pos_pcuis}
    p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
    for d, prof in profile.items():
        if not prof: scores[d] = -1e9; continue
        dot = sum(pat_vec[e] * prof[e] for e in pos_pcuis if e in prof)
        d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        pos_score = dot / (p_norm * d_norm)
        sig = signal.get(d, {})
        neg_pen = sum(sig.get(ev, 0.0) for ev in neg_binary)
        neg_norm = math.sqrt(len(neg_binary)) or 1e-9
        neg_score = neg_pen / (neg_norm * d_norm)
        scores[d] = pos_score - lam * neg_score
    return scores


def evaluate(profile, idf, beta, signal, patients, all_evs, dcs_list, lam):
    n = c1 = c3 = c5 = c10 = 0; rr = 0.0
    for true_cui, pos_raw, neg_binary in patients:
        pos = pos_raw & all_evs
        if not pos: continue
        s = score(pos, neg_binary, profile, idf, beta, signal, lam)
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
    ap.add_argument("--tau_sweep", type=str, default="2.0,2.5,3.0,3.5,4.0")
    ap.add_argument("--sharp_sweep", type=str, default="0.5")
    ap.add_argument("--lam_sweep", type=str, default="0.1,0.15,0.2,0.3,0.5")
    args = ap.parse_args()
    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))
    dcs_list, patients, binary_evs = load_ddxplus_full(args.n)
    base, all_evs = build_profile(G, dcs_list, 20.0, pr, top_k=args.top_k)
    idf = compute_idf(base, 0.12)
    profile = reweight(base, idf, 1.0, args.beta)
    value_cuis = json.load(open(VALUE_CUIS))
    print(f"=== v71 self-aware negative — N={args.n} ===")
    print(f"  IDF stats: min={min(idf.values()):.2f} max={max(idf.values()):.2f}")
    for tau in args.tau_sweep.split(","):
        tau = float(tau)
        for sharp in args.sharp_sweep.split(","):
            sharp = float(sharp)
            sig = precompute_signal_v71(profile, value_cuis, binary_evs, idf, tau, sharp)
            for lam in args.lam_sweep.split(","):
                lam = float(lam)
                r = evaluate(profile, idf, args.beta, sig, patients, all_evs, dcs_list, lam)
                print(f"  tau={tau:.1f} sharp={sharp:.1f} lam={lam:.3f}: "
                      f"@1={r['at1']:.2f}% @3={r['at3']:.2f}% MRR={r['mrr']:.4f}",
                      flush=True)


if __name__ == "__main__":
    main()
