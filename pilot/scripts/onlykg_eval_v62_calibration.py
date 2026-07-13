#!/usr/bin/env python3
"""v62 — Test multiple KG profile calibrations + scoring on DDXPlus.

Variants:
  raw_hill        : Hill function P = w/(w+kappa) — current baseline
  multinomial     : Normalize Σ_E P(E|D) = 1 (probability simplex)
  sigmoid         : 1/(1 + exp(-(w-w0)/sigma))
  rank_based      : P(E|D) = (rank/N) for top-N edges
  freq_norm       : P(E|D) = freq / max_freq_in_disease
"""
import json, math, pickle, csv, ast, sys, argparse
from collections import defaultdict
sys.path.insert(0, 'pilot/scripts')
from medkg_paths import MEDKG_ROOT

PR_UNIVERSE = "pilot/data/pr_universe.json"


def build_profile_calibrated(G, dcs, mode, method, kappa=20.0, top_k=None):
    """Build profile with various calibration methods."""
    if mode == 'lay':
        allowed = {'patient_reportable', 'history', 'demographic'}
        pr = set(json.load(open(PR_UNIVERSE)))
    else:
        allowed = {'clinical_sign', 'lab_finding', 'imaging_finding', 'history', 'demographic'}
        pr = None

    profile = {}
    for d in dcs:
        if d not in G:
            profile[d] = {}; continue
        ed_w = defaultdict(float)
        ed_freq = defaultdict(int)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get('etype') != 'HAS_PHENOTYPE': continue
            cat = ed.get('category')
            if cat is None:
                if mode == 'lay' and pr and p not in pr: continue
            else:
                if cat not in allowed: continue
            ed_w[p] += ed.get('weight', 0.0)
            ed_freq[p] += ed.get('freq', 1)

        if not ed_w:
            profile[d] = {}; continue

        # Optionally restrict to top-K
        if top_k:
            sorted_e = sorted(ed_w.items(), key=lambda x: -x[1])[:top_k]
            ed_w = dict(sorted_e)
            ed_freq = {e: ed_freq.get(e, 1) for e in ed_w}

        if method == 'raw_hill':
            profile[d] = {p: w/(w+kappa) for p, w in ed_w.items() if w > 0}
        elif method == 'multinomial':
            total = sum(ed_w.values())
            profile[d] = {p: w/total for p, w in ed_w.items() if w > 0}
        elif method == 'sigmoid':
            # Per-disease median as midpoint
            vals = sorted(ed_w.values())
            w0 = vals[len(vals)//2]
            sigma = max(1.0, w0)
            profile[d] = {p: 1.0/(1.0 + math.exp(-(w-w0)/sigma)) for p, w in ed_w.items()}
        elif method == 'rank_based':
            sorted_e = sorted(ed_w.items(), key=lambda x: -x[1])
            n = len(sorted_e)
            profile[d] = {p: 1.0 - (i/n) for i, (p, _) in enumerate(sorted_e)}
        elif method == 'freq_norm':
            max_freq = max(ed_freq.values())
            profile[d] = {p: ed_freq[p]/max_freq for p in ed_w}
        else:
            raise ValueError(method)
    return profile


def cosine(pcuis, profile):
    scores = {}
    p_norm = math.sqrt(len(pcuis))
    for d, prof in profile.items():
        if not prof: scores[d] = -1e9; continue
        dot = sum(prof[e] for e in pcuis if e in prof)
        d_norm = math.sqrt(sum(v*v for v in prof.values()))
        scores[d] = dot / (p_norm * d_norm + 1e-9)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()

    G = pickle.load(open(args.graph, 'rb'))
    value_cuis = json.load(open(MEDKG_ROOT/'kg'/'ddxplus_evidence_value_cuis.json'))
    icd = json.load(open('data/ddxplus/disease_icd10_cui_mapping.json'))
    cond = json.load(open('data/ddxplus/release_conditions_en.json'))
    fr2cui = {info.get('cond-name-fr',''): icd[dn]['cui'] for dn,info in cond.items() if dn in icd}
    dcs = sorted(set(fr2cui.values()))

    pats = []
    with open('data/ddxplus/release_test_patients.csv') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= args.n: break
            true_cui = fr2cui.get(row['PATHOLOGY'])
            if true_cui not in dcs: continue
            evs = ast.literal_eval(row['EVIDENCES'])
            pcuis = set()
            for ev in evs:
                if '_@_' in ev:
                    base, val = ev.split('_@_', 1)
                    m = value_cuis.get(base, {})
                    for k in ('_question', val):
                        v = m.get(k, [])
                        if isinstance(v, list): pcuis.update(v)
                else:
                    m = value_cuis.get(ev, {})
                    pcuis.update(m.get('_question', []))
            pats.append((true_cui, pcuis))

    print(f"=== v62 KG calibration on DDXPlus {args.n} ===")
    for method in ['raw_hill', 'multinomial', 'sigmoid', 'rank_based', 'freq_norm']:
        for top_k in [None, 50, 100]:
            prof = build_profile_calibrated(G, dcs, 'lay', method, top_k=top_k)
            all_evs = set()
            for p in prof.values(): all_evs.update(p.keys())
            n=c1=c3=c10=0; rr=0
            for true_cui, raw in pats:
                pcuis = set(raw) & all_evs
                if not pcuis: continue
                scores = cosine(pcuis, prof)
                ranked = sorted(prof.keys(), key=lambda d: -scores[d])
                n += 1
                try: rank = ranked.index(true_cui)+1
                except: rank = len(dcs)
                if rank == 1: c1 += 1
                if rank <= 3: c3 += 1
                if rank <= 10: c10 += 1
                rr += 1/rank
            k_str = f"top{top_k}" if top_k else "all"
            print(f"  {method:<12} {k_str:<5}: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @10={100*c10/n:.2f}% MRR={rr/n:.4f}")


if __name__ == "__main__":
    main()
