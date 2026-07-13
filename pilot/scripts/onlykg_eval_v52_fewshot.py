#!/usr/bin/env python3
"""v52: Few-shot profile-based scoring.

Same architecture as v51 (text-profile) but profile from few-shot train sampling
instead of PubMed keywords.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse, random
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
COMPOUND_PATH = "pilot/data/compound_pain_lookup_lt5.json"


def normalize_scores(d):
    vals = list(d.values())
    if not vals: return d
    lo, hi = min(vals), max(vals)
    if hi == lo: return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def build_few_shot_profile(N_per_disease, fr2cui, dcs_list, seed=42):
    """Sample N patients per disease from train, compute mean intensity/sudden/precision."""
    random.seed(seed)
    disease_evs = defaultdict(list)
    with open('data/ddxplus/release_train_patients.csv') as f:
        for row in csv.DictReader(f):
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in fr2cui.values(): continue
            evs_str = row["EVIDENCES"]
            disease_evs[true_cui].append(evs_str)

    profile = {}
    for d in dcs_list:
        evs_list = disease_evs.get(d, [])
        if not evs_list:
            profile[d] = {'intens': 0, 'soudain': 0, 'precis': 0, 'count': 0}
            continue
        sample = random.sample(evs_list, min(N_per_disease, len(evs_list)))
        intens, soudain, precis = [], [], []
        char_counter = Counter()
        for evs_str in sample:
            evs = ast.literal_eval(evs_str)
            for ev in evs:
                if '_@_' in ev:
                    base, val = ev.split('_@_', 1)
                    if base == 'douleurxx_intens':
                        try: intens.append(int(val))
                        except: pass
                    elif base == 'douleurxx_soudain':
                        try: soudain.append(int(val))
                        except: pass
                    elif base == 'douleurxx_precis':
                        try: precis.append(int(val))
                        except: pass
                    elif base == 'douleurxx_carac':
                        char_counter[val] += 1
        profile[d] = {
            'intens': sum(intens)/max(len(intens),1) if intens else 0,
            'soudain': sum(soudain)/max(len(soudain),1) if soudain else 0,
            'precis': sum(precis)/max(len(precis),1) if precis else 0,
            'char_top': set([v for v, _ in char_counter.most_common(3)]),
            'count': len(sample),
        }
    return profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--hop2_decay", type=float, default=0.7)
    ap.add_argument("--idf_pow", type=float, default=0.5)
    ap.add_argument("--core_k", type=int, default=35)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--identity_boost", type=float, default=1.5)
    ap.add_argument("--sig_k", type=int, default=10)
    ap.add_argument("--sig_w", type=float, default=9.0)
    ap.add_argument("--w_s1", type=float, default=0.7)
    ap.add_argument("--w_cov", type=float, default=0.1)
    ap.add_argument("--w_prcov", type=float, default=0.1)
    ap.add_argument("--w_compound", type=float, default=0.1)
    ap.add_argument("--w_fs", type=float, default=0.1)
    ap.add_argument("--n_shot", type=int, default=0,
                    help="number of train patients per disease for profile (0 = no fs)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    dcs_set = set(dcs_list)

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)
    PR = set(json.load(open(PR_UNIVERSE))) if Path(PR_UNIVERSE).exists() else set()

    compound = defaultdict(set)
    raw = json.load(open(COMPOUND_PATH))
    for k, v_list in raw.items():
        q, v = k.split('|')
        compound[(q, v)].update(v_list)

    # Build few-shot profile
    fs_profile = None
    if args.n_shot > 0:
        fs_profile = build_few_shot_profile(args.n_shot, fr2cui, dcs_list, seed=args.seed)

    d_q = {}
    for d in dcs_list:
        if d not in G: d_q[d] = {}; continue
        phen_w = {}
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + ed.get("weight", 0)
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, ed2 in G.out_edges(p_direct, data=True):
                if ed2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + args.hop2_decay * dw * ed2.get("weight", 0)
        d_q[d] = {p: w for p, w in phen_w.items() if p in Q}

    phen_freq = Counter()
    for d, qp in d_q.items():
        for p in qp: phen_freq[p] += 1
    N_D = len(dcs_list)
    idf = {p: math.log(N_D / max(c, 1)) ** args.idf_pow for p, c in phen_freq.items()}
    d_q_idf = {d: {p: w * idf.get(p, 1.0) for p, w in qp.items()} for d, qp in d_q.items()}
    d_core = {d: set(sorted(qp.keys(), key=lambda p: -qp[p])[:args.core_k]) for d, qp in d_q_idf.items()}
    d_sig = {d: set(sorted(qp.keys(), key=lambda p: -qp[p])[:args.sig_k]) for d, qp in d_q_idf.items()}

    disease_full_phens = {d: {p for _, p, ed in G.out_edges(d, data=True) if ed.get("etype")=="HAS_PHENOTYPE"} if d in G else set() for d in dcs_list}
    compound_cuis_all = set()
    for cuis in compound.values(): compound_cuis_all.update(cuis)
    compound_doc_freq = {c: sum(1 for p in disease_full_phens.values() if c in p) for c in compound_cuis_all}
    compound_idf = {c: math.log(49 / max(compound_doc_freq.get(c, 1), 1)) for c in compound_cuis_all}

    def parse_features(evs):
        intensity, suddenness, precision = None, None, None
        pcuis = set(); compound_targets = set(); pain_chars = set()
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                q_cuis = m.get("_question", [])
                v_cuis = m.get(val, [])
                for q in q_cuis:
                    for v in v_cuis:
                        if (q, v) in compound: compound_targets.update(compound[(q, v)])
                pcuis.update(q_cuis); pcuis.update(v_cuis)
                if base == 'douleurxx_intens':
                    try: intensity = int(val)
                    except: pass
                elif base == 'douleurxx_soudain':
                    try: suddenness = int(val)
                    except: pass
                elif base == 'douleurxx_precis':
                    try: precision = int(val)
                    except: pass
                elif base == 'douleurxx_carac':
                    pain_chars.add(val)
            else:
                m = value_cuis.get(ev, {})
                pcuis.update(m.get("_question", []))
        return pcuis, compound_targets, intensity, suddenness, precision, pain_chars

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis, compound_targets, intensity, suddenness, precision, pain_chars = parse_features(evs)
            identity_diseases = pcuis & dcs_set

            s1_scores = {}; cov_scores = {}; prcov_scores = {}; comp_scores = {}; fs_scores = {}
            for d in dcs_list:
                qp = d_q_idf.get(d, {})
                pos = sum(w for q, w in qp.items() if q in pcuis)
                core = d_core.get(d, set())
                neg = sum(qp.get(c, 0) for c in core if c not in pcuis)
                s1 = pos - args.alpha * neg
                total = sum(qp.values()) if qp else 1
                s1 = s1 / (math.sqrt(total) or 1)
                sig = d_sig.get(d, set())
                if sig:
                    s1 += args.sig_w * (sum(1 for p in sig if p in pcuis) / len(sig))
                if d in identity_diseases:
                    s1 += args.identity_boost
                s1_scores[d] = s1

                cov_scores[d] = sum(1 for p in pcuis if p in qp) / max(len(pcuis), 1) if pcuis and qp else 0
                if PR and pcuis and qp:
                    pr_pcuis = pcuis & PR
                    pr_qp = {p: w for p, w in qp.items() if p in PR}
                    prcov_scores[d] = sum(1 for p in pr_pcuis if p in pr_qp) / max(len(pr_pcuis), 1) if (pr_pcuis and pr_qp) else 0
                else:
                    prcov_scores[d] = 0

                comp = 0
                if compound_targets and disease_full_phens[d]:
                    comp = sum(compound_idf.get(c, 0) for c in (compound_targets & disease_full_phens[d]))
                comp_scores[d] = comp

                # Few-shot profile matching: distance-based
                fs_score = 0
                if fs_profile is not None:
                    p = fs_profile.get(d, {})
                    # Numeric: -|patient - disease_mean| (closer = higher score)
                    if intensity is not None and p.get('intens', 0) > 0:
                        fs_score += -abs(intensity - p['intens']) / 10.0
                    if suddenness is not None and p.get('soudain', 0) > 0:
                        fs_score += -abs(suddenness - p['soudain']) / 10.0
                    if precision is not None and p.get('precis', 0) > 0:
                        fs_score += -abs(precision - p['precis']) / 10.0
                    # Character: matching top chars
                    if pain_chars and p.get('char_top'):
                        match = len(pain_chars & p['char_top'])
                        fs_score += match * 0.5
                fs_scores[d] = fs_score

            s1_n = normalize_scores(s1_scores)
            cov_n = normalize_scores(cov_scores)
            prcov_n = normalize_scores(prcov_scores)
            comp_n = normalize_scores(comp_scores)
            fs_n = normalize_scores(fs_scores) if args.n_shot > 0 else {d: 0 for d in dcs_list}

            final = {d: args.w_s1*s1_n[d] + args.w_cov*cov_n[d] + args.w_prcov*prcov_n[d] + args.w_compound*comp_n[d] + args.w_fs*fs_n[d] for d in dcs_list}
            ranked = sorted(dcs_list, key=lambda d: -final.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"v52 n_shot={args.n_shot:>5d} w_fs={args.w_fs}: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
