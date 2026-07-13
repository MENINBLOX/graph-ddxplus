#!/usr/bin/env python3
"""v77 full metrics — GTPA@k + DDR/DDP/DDF1 on 30K patients.

Standard DDXPlus metrics (Tchango 2022):
  GTPA@k: top-k contains true PATHOLOGY
  DDR (Differential Recall): |pred ∩ GT_diff| / |GT_diff|
  DDP (Differential Precision): |pred ∩ GT_diff| / |pred top-k|
  DDF1: 2·DDR·DDP / (DDR+DDP)

We cut predicted top-k at K = |GT_diff| per patient.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse, statistics
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
EV_META = "data/ddxplus/release_evidences.json"


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


def precompute_signal(profile, value_cuis, binary_evs, idf, tau, sharp):
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


def v71_score(pos, neg_b, profile, idf, beta, signal, lam):
    scores = {}
    pat_vec = {e: idf.get(e, 1.0) ** beta for e in pos}
    p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
    for d, prof in profile.items():
        if not prof: scores[d] = -1e9; continue
        dot = sum(pat_vec[e] * prof[e] for e in pos if e in prof)
        d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        ps = dot / (p_norm * d_norm)
        sig = signal.get(d, {})
        np_ = sum(sig.get(ev, 0.0) for ev in neg_b)
        nn = math.sqrt(len(neg_b)) or 1e-9
        ns = np_ / (nn * d_norm)
        scores[d] = ps - lam * ns
    return scores


def v74_nb_score(yes, no, llm_profs, smooth=1e-3):
    scores = {}
    for d, p in llm_profs.items():
        s = 0.0
        for ev in yes:
            v = p.get(ev, 0.01); v = max(smooth, min(1-smooth, v))
            s += math.log(v)
        for ev in no:
            v = p.get(ev, 0.01); v = max(smooth, min(1-smooth, v))
            s += math.log(1-v)
        scores[d] = s
    return scores


def zscore(scores):
    vals = list(scores.values())
    m = statistics.mean(vals); sd = statistics.stdev(vals) if len(vals)>1 else 1.0
    if sd == 0: sd = 1.0
    return {d: (s-m)/sd for d, s in scores.items()}


def load_llm_profiles(path):
    profs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            p = {}
            for ev, val in r["profile"].items():
                if isinstance(val, (list, tuple)): p[ev] = val[1]
                else: p[ev] = float(val)
            profs[r["dcui"]] = p
    return profs


def load_ddxplus(n_max):
    value_cuis = json.load(open(VALUE_CUIS))
    ev_meta = json.load(open(EV_META))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    binary_evs = {ev_id for ev_id, m in ev_meta.items()
                  if m.get("data_type") == "B" and m.get("default_value") == 0}
    dcs_list = sorted(set(fr2cui.values()))
    patients = []; n = 0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= n_max: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            dd = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
            gt_diff_cuis = []
            for name, prob in dd:
                c = fr2cui.get(name)
                if c and c in dcs_list:
                    gt_diff_cuis.append(c)
            pos_pcuis = set(); yes_b = set()
            for ev in evs:
                if "_@_" in ev:
                    base, val = ev.split("_@_", 1)
                    m = value_cuis.get(base, {})
                    for k in ("_question", val):
                        v = m.get(k, [])
                        if isinstance(v, list): pos_pcuis.update(v)
                else:
                    if ev in binary_evs: yes_b.add(ev)
                    m = value_cuis.get(ev, {})
                    pos_pcuis.update(m.get("_question", []))
            no_b = binary_evs - yes_b
            patients.append((true_cui, pos_pcuis, yes_b, no_b, gt_diff_cuis)); n += 1
    return dcs_list, patients, binary_evs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--llm_path", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--alpha", type=float, default=0.65)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=0.4)
    args = ap.parse_args()
    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))
    dcs_list, patients, binary_evs = load_ddxplus(args.n)
    base, all_evs = build_profile(G, dcs_list, 20.0, pr)
    idf = compute_idf(base, 0.12)
    profile = reweight(base, idf, 1.0, args.beta)
    value_cuis = json.load(open(VALUE_CUIS))
    sig = precompute_signal(profile, value_cuis, binary_evs, idf, args.tau, 0.5)
    llm_profs = load_llm_profiles(args.llm_path)

    n = c1 = c3 = c5 = c10 = 0; rr = 0.0
    ddr_sum = ddp_sum = ddf1_sum = 0.0
    n_dd = 0
    for true_cui, pos_raw, yes_b, no_b, gt_diff in patients:
        pos = pos_raw & all_evs
        if not pos: continue
        s71 = zscore(v71_score(pos, no_b, profile, idf, args.beta, sig, args.lam))
        s74 = zscore(v74_nb_score(yes_b, no_b, llm_profs))
        final = {d: args.alpha * s71.get(d,0) + (1-args.alpha) * s74.get(d,0) for d in dcs_list}
        ranked = sorted(final.keys(), key=lambda d: -final[d])
        n += 1
        try: rank = ranked.index(true_cui)+1
        except: rank = len(dcs_list)
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr += 1.0/rank
        # DDR/DDP/DDF1
        if gt_diff:
            k = len(gt_diff)
            pred_topk = set(ranked[:k])
            gt_set = set(gt_diff)
            overlap = pred_topk & gt_set
            recall = len(overlap) / len(gt_set)
            precision = len(overlap) / len(pred_topk) if pred_topk else 0
            f1 = 2*recall*precision/(recall+precision) if (recall+precision)>0 else 0
            ddr_sum += recall; ddp_sum += precision; ddf1_sum += f1
            n_dd += 1
    print(f"=== v77 N=5 ensemble — N={n} ===")
    print(f"  GTPA@1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}%")
    print(f"  MRR={rr/n:.4f}")
    print(f"  DDR={100*ddr_sum/n_dd:.2f}%  DDP={100*ddp_sum/n_dd:.2f}%  DDF1={100*ddf1_sum/n_dd:.2f}%")


if __name__ == "__main__":
    main()
