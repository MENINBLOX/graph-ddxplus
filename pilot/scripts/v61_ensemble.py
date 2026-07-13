#!/usr/bin/env python3
"""v61: NB + cosine ensemble. Both scores normalized, weighted sum."""
import json, math, pickle, csv, ast, sys
from collections import defaultdict
sys.path.insert(0, 'pilot/scripts')
from medkg_paths import MEDKG_ROOT

G = pickle.load(open('/mnt/medkg/kg/onlykg_graph_v43_with_wiki.pkl', 'rb'))
pr = set(json.load(open('pilot/data/pr_universe.json')))
KAPPA, P_BASELINE, SMOOTH = 20.0, 0.01, 1e-3

def build_profile(dcs, mode):
    allowed = {'patient_reportable','history','demographic'} if mode=='lay' else \
              {'clinical_sign','lab_finding','imaging_finding','history','demographic'}
    profile = {}
    for d in dcs:
        if d not in G: profile[d] = {}; continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get('etype') != 'HAS_PHENOTYPE': continue
            cat = ed.get('category')
            if cat is None:
                if mode == 'lay' and p not in pr: continue
            else:
                if cat not in allowed: continue
            ed_w[p] += ed.get('weight', 0.0)
        profile[d] = {p: w/(w+KAPPA) for p, w in ed_w.items() if w > 0}
    return profile

def normalize(scores):
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo: return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

def nb(pcuis, profile, all_evs, log_prior):
    scores = {}
    for d, prof in profile.items():
        s = log_prior
        for e in all_evs:
            p = prof.get(e, P_BASELINE)
            p = max(SMOOTH, min(1-SMOOTH, p))
            s += math.log(p) if e in pcuis else math.log(1-p)
        scores[d] = s
    return scores

def cosine(pcuis, profile):
    scores = {}
    p_norm = math.sqrt(len(pcuis))
    for d, prof in profile.items():
        if not prof: scores[d] = 0; continue
        dot = sum(prof[e] for e in pcuis if e in prof)
        d_norm = math.sqrt(sum(v*v for v in prof.values()))
        scores[d] = dot / (p_norm * d_norm + 1e-9)
    return scores

value_cuis = json.load(open(MEDKG_ROOT/'kg'/'ddxplus_evidence_value_cuis.json'))
icd = json.load(open('data/ddxplus/disease_icd10_cui_mapping.json'))
cond = json.load(open('data/ddxplus/release_conditions_en.json'))
fr2cui = {info.get('cond-name-fr',''): icd[dn]['cui'] for dn,info in cond.items() if dn in icd}
ddx_dcs = sorted(set(fr2cui.values()))

profile = build_profile(ddx_dcs, 'lay')
all_evs = set()
for p in profile.values(): all_evs.update(p.keys())
log_prior = math.log(1.0/len(ddx_dcs))

pats = []
with open('data/ddxplus/release_test_patients.csv') as f:
    for i, row in enumerate(csv.DictReader(f)):
        if i >= 5000: break
        true_cui = fr2cui.get(row['PATHOLOGY'])
        if true_cui not in ddx_dcs: continue
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

print("=== DDXPlus 5K NB + cosine ensemble sweep ===")
for w_nb in [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]:
    n=c1=c3=c10=0; rr=0
    for true_cui, raw in pats:
        pcuis = set(raw) & all_evs
        if not pcuis: continue
        sc_nb = normalize(nb(pcuis, profile, all_evs, log_prior))
        sc_cos = normalize(cosine(pcuis, profile))
        final = {d: w_nb*sc_nb[d] + (1-w_nb)*sc_cos[d] for d in profile}
        ranked = sorted(profile.keys(), key=lambda d: -final[d])
        n += 1
        try: rank = ranked.index(true_cui)+1
        except: rank = len(ddx_dcs)
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 10: c10 += 1
        rr += 1/rank
    print(f"  w_nb={w_nb}: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @10={100*c10/n:.2f}% MRR={rr/n:.4f}")
