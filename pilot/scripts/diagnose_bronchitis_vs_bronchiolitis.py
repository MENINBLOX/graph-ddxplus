#!/usr/bin/env python3
"""Direct comparison: Bronchitis vs Bronchiolitis profile under v64 weighting."""
import json, math, pickle, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "pilot/scripts")
from medkg_paths import MEDKG_ROOT

GRAPH = "pilot/data/onlykg_graph_v49_v5_full.pkl"
PR_UNIVERSE = "pilot/data/pr_universe.json"
MRCONSO = "/windows/data/umls_subset/MRCONSO.RRF"
KAPPA = 20.0
DF_THRESHOLD = 0.12
TOP_K = 80
BETA = 0.75

# Target pair
BRONCHITIS = "C0006277"
BRONCHIOLITIS = "C0001311"

G = pickle.load(open(GRAPH, "rb"))
with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
dcs_list = sorted(set(fr2cui.values()))
pr_set = set(json.load(open(PR_UNIVERSE)))

def build(d):
    if d not in G: return {}
    ed_w = defaultdict(float)
    for _, p, ed in G.out_edges(d, data=True):
        if ed.get("etype") != "HAS_PHENOTYPE": continue
        cat = ed.get("category")
        if cat is None:
            if p not in pr_set: continue
        else:
            if cat not in {"patient_reportable","history","demographic"}: continue
        ed_w[p] += ed.get("weight", 0.0)
    prof = {p: w/(w+KAPPA) for p, w in ed_w.items() if w > 0}
    if len(prof) > TOP_K:
        prof = dict(sorted(prof.items(), key=lambda x: -x[1])[:TOP_K])
    return prof

profiles = {d: build(d) for d in dcs_list}
N = len(profiles)
df = defaultdict(int)
for p in profiles.values():
    for e, w in p.items():
        if w >= DF_THRESHOLD: df[e] += 1
idf = {e: math.log((N+1)/(df_e+1))+1.0 for e, df_e in df.items()}

# Load names for CUIs involved
all_target = set(profiles[BRONCHITIS].keys()) | set(profiles[BRONCHIOLITIS].keys())
names = {c: '' for c in all_target}
with open(MRCONSO) as f:
    for line in f:
        parts = line.split('|')
        if len(parts) < 15: continue
        c, lang = parts[0], parts[1]
        if lang != 'ENG': continue
        if c in names and not names[c]: names[c] = parts[14]

b1 = profiles[BRONCHITIS]
b2 = profiles[BRONCHIOLITIS]
shared = set(b1.keys()) & set(b2.keys())
only1 = set(b1.keys()) - set(b2.keys())
only2 = set(b2.keys()) - set(b1.keys())

print(f"=== Bronchitis ({BRONCHITIS}) vs Bronchiolitis ({BRONCHIOLITIS}) ===")
print(f"Bronchitis profile size: {len(b1)}")
print(f"Bronchiolitis profile size: {len(b2)}")
print(f"Shared: {len(shared)}, Bronchitis-only: {len(only1)}, Bronchiolitis-only: {len(only2)}\n")

print(f"### Bronchitis-only CUIs (top 15 by weight)")
print(f"{'CUI':<12} {'name':<40} {'w':>6} {'idf':>6}")
for c in sorted(only1, key=lambda x: -b1[x])[:15]:
    print(f"{c:<12} {names.get(c,'?')[:40]:<40} {b1[c]:>6.3f} {idf.get(c,1.0):>6.2f}")

print(f"\n### Bronchiolitis-only CUIs (top 15 by weight)")
for c in sorted(only2, key=lambda x: -b2[x])[:15]:
    print(f"{c:<12} {names.get(c,'?')[:40]:<40} {b2[c]:>6.3f} {idf.get(c,1.0):>6.2f}")

print(f"\n### Shared CUIs - where Bronchiolitis weight > Bronchitis (potential 'theft')")
diff = [(c, b1[c], b2[c], idf.get(c,1.0)) for c in shared if b2[c] > b1[c]]
print(f"{'CUI':<12} {'name':<40} {'w_b1':>6} {'w_b2':>6} {'diff':>7} {'idf':>6}")
for c, w1, w2, i in sorted(diff, key=lambda x: -(x[2]-x[1]))[:15]:
    print(f"{c:<12} {names.get(c,'?')[:40]:<40} {w1:>6.3f} {w2:>6.3f} {w2-w1:>+7.3f} {i:>6.2f}")

print(f"\n### Shared CUIs - where Bronchitis weight > Bronchiolitis")
diff = [(c, b1[c], b2[c], idf.get(c,1.0)) for c in shared if b1[c] > b2[c]]
print(f"{'CUI':<12} {'name':<40} {'w_b1':>6} {'w_b2':>6} {'diff':>7} {'idf':>6}")
for c, w1, w2, i in sorted(diff, key=lambda x: -(x[1]-x[2]))[:15]:
    print(f"{c:<12} {names.get(c,'?')[:40]:<40} {w1:>6.3f} {w2:>6.3f} {w1-w2:>+7.3f} {i:>6.2f}")
