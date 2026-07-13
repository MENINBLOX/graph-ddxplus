#!/usr/bin/env python3
"""Full DDXPlus evaluation: GTPA@k + DDR/DDP/DDF1 + dump top-K for Stage 2.

Standard DDXPlus metrics:
- GTPA@k: % patients whose true PATHOLOGY appears in predicted top-k
- DDR (Differential Diagnosis Recall): how much of GT differential is covered
- DDP (Differential Diagnosis Precision): how precisely predicted differential matches GT
- DDF1: harmonic mean

We also dump top-K candidates per patient → input for Stage 2 LLM.
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


def precompute_signal(profile, value_cuis, binary_evs, idf, tau, sharpness):
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
            if best > 0: signal[d][ev_id] = best
    return signal


def score_v71(pos_pcuis, neg_binary, profile, idf, beta, signal, lam):
    scores = {}
    pat_vec = {e: idf.get(e, 1.0) ** beta for e in pos_pcuis}
    p_norm = math.sqrt(sum(v*v for v in pat_vec.values())) or 1e-9
    for d, prof in profile.items():
        if not prof: scores[d] = -1e9; continue
        dot = sum(pat_vec[e] * prof[e] for e in pos_pcuis if e in prof)
        d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        pos = dot / (p_norm * d_norm)
        sig = signal.get(d, {})
        neg_pen = sum(sig.get(ev, 0.0) for ev in neg_binary)
        neg_norm = math.sqrt(len(neg_binary)) or 1e-9
        neg = neg_pen / (neg_norm * d_norm)
        scores[d] = pos - lam * neg
    return scores


def load_ddxplus_with_dd(n_max):
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
            gt_diff = []  # list of (cui, prob)
            for name, prob in dd:
                c = fr2cui.get(name)
                if c and c in dcs_list:
                    gt_diff.append((c, prob))
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
            patients.append((true_cui, pos_pcuis, neg_binary, gt_diff, n)); n += 1
    return dcs_list, patients, binary_evs


def ddr_ddp_ddf1(predicted_ranked, gt_diff, k=None):
    """DDXPlus standard metrics.
    predicted_ranked: list of disease CUIs (top-K)
    gt_diff: list of (cui, prob) ranked by prob
    Returns DDR (recall), DDP (precision), DDF1.
    """
    if not gt_diff:
        return 0.0, 0.0, 0.0
    gt_set = {c for c, _ in gt_diff}
    pred = predicted_ranked[:k] if k else predicted_ranked
    pred_set = set(pred)
    if not pred_set:
        return 0.0, 0.0, 0.0
    overlap = gt_set & pred_set
    recall = len(overlap) / len(gt_set)
    precision = len(overlap) / len(pred_set)
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    return recall, precision, f1


def evaluate(profile, idf, beta, signal, patients, all_evs, dcs_list, lam,
             dump_topk_path=None, top_dump=10):
    n = c1 = c3 = c5 = c10 = 0; rr = 0.0
    ddr_sum = ddp_sum = ddf1_sum = 0.0
    dumps = []
    for true_cui, pos_raw, neg_binary, gt_diff, pid in patients:
        pos = pos_raw & all_evs
        if not pos: continue
        s = score_v71(pos, neg_binary, profile, idf, beta, signal, lam)
        ranked = sorted(profile.keys(), key=lambda d: -s[d])
        n += 1
        try: rank = ranked.index(true_cui)+1
        except: rank = len(dcs_list)
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr += 1.0/rank
        # DDR/DDP/DDF1 at top-len(gt_diff)
        r, p, f = ddr_ddp_ddf1(ranked, gt_diff, k=len(gt_diff))
        ddr_sum += r; ddp_sum += p; ddf1_sum += f
        if dump_topk_path:
            dumps.append({
                "pid": pid,
                "true_cui": true_cui,
                "top": [(d, s[d]) for d in ranked[:top_dump]],
                "gt_diff": gt_diff,
            })
    if dump_topk_path:
        with open(dump_topk_path, "w") as f:
            for d in dumps:
                f.write(json.dumps(d) + "\n")
        print(f"  dumped top-{top_dump} for {len(dumps)} patients → {dump_topk_path}")
    return {"n": n, "at1": 100*c1/n, "at3": 100*c3/n, "at5": 100*c5/n,
            "at10": 100*c10/n, "mrr": rr/n,
            "ddr": 100*ddr_sum/n, "ddp": 100*ddp_sum/n, "ddf1": 100*ddf1_sum/n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--top_k", type=int, default=999)  # full profile by default
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=0.4)
    ap.add_argument("--dump_path", default=None)
    ap.add_argument("--top_dump", type=int, default=10)
    args = ap.parse_args()
    G = pickle.load(open(args.graph, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))
    dcs_list, patients, binary_evs = load_ddxplus_with_dd(args.n)
    base, all_evs = build_profile(G, dcs_list, 20.0, pr, top_k=args.top_k)
    idf = compute_idf(base, 0.12)
    profile = reweight(base, idf, 1.0, args.beta)
    value_cuis = json.load(open(VALUE_CUIS))
    sig = precompute_signal(profile, value_cuis, binary_evs, idf, args.tau, 0.5)
    r = evaluate(profile, idf, args.beta, sig, patients, all_evs, dcs_list,
                 args.lam, args.dump_path, args.top_dump)
    print(f"=== Full eval N={r['n']} ===")
    print(f"  GTPA: @1={r['at1']:.2f}% @3={r['at3']:.2f}% @5={r['at5']:.2f}% @10={r['at10']:.2f}%")
    print(f"  MRR: {r['mrr']:.4f}")
    print(f"  DDR={r['ddr']:.2f}% DDP={r['ddp']:.2f}% DDF1={r['ddf1']:.2f}%")


if __name__ == "__main__":
    main()
