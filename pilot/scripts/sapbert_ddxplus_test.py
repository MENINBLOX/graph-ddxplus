"""Does SapBERT soft CUI-alignment close the DDXPlus vocabulary gap? (KG-only, §3 safe)"""
import sys, json, csv, ast, math
from collections import defaultdict
sys.path.insert(0, "pilot/scripts")
import onlykg_eval_unified as U
import onlykg_eval_unified2 as U2
import sapbert_align as SB

kg, prof, idf, dz_names = U.load_kg("pilot/cache/prod/kg.json")
children = json.load(open("pilot/cache/umls_maps/disease_children.json"))
child_adj, parent_adj = U2.build_disease_hierarchy(children)
resolver = U.Resolver(prof, dz_names, children)
PH = kg["phenotypes"]
N_DZ = len(kg["diseases"]); KGDZ = set(kg["diseases"])
icc = {}
def ic_dz(c):
    if c not in icc: icc[c] = math.log(N_DZ / (1 + len(U2.closure(c, child_adj, 6) & KGDZ)))
    return icc[c]

cand, pats = U.load_ddxplus(3000)
# candidate profiles + disease signature
prof_of = {}; dsig_of = {}
for k, (name, cu) in cand.items():
    cl = resolver.cluster(name, cu)
    prof_of[k] = U.merged_profile(cl, prof)
    ds = {}
    for b in ({cu} if cu else set()) | cl:
        for r in U2.closure(b, parent_adj, 4) | U2.closure(b, child_adj, 4) | {b}:
            ds[r] = max(ds.get(r, 0), ic_dz(r))
    dsig_of[k] = ds

# CUIs to embed: union of all candidate-profile phenotypes
pcuis = sorted({c for pf in prof_of.values() for c in pf})
pnames = [PH.get(c, {}).get("name", c) for c in pcuis]

umap = json.load(open("data/ddxplus/umls_mapping.json"))["mapping"]
ev_cui2name = {v["cui"]: v["name"] for v in umap.values()}
ev_cuis = sorted(ev_cui2name)
ev_names = [ev_cui2name[c] for c in ev_cuis]
print(f"embedding {len(pcuis)} KG-profile CUIs + {len(ev_cuis)} DDXPlus evidence CUIs via SapBERT...", flush=True)
align = SB.build_alignment(ev_cuis, ev_names, pcuis, pnames, topk=10, tau=0.75)

def ev_c(e):
    for c in (e, e.split("_@_")[0], e.split("_")[0]):
        if c in umap: return umap[c]["cui"]
    return None

def evaluate(soft):
    keys = list(cand)
    n = c1 = c3 = c5 = c10 = 0
    for tk, ev, ds in pats:
        t = tk[0]
        sc = {}
        for k in keys:
            pf = prof_of[k]; dsig = dsig_of[k]; s = 0.0
            for e in ev:
                best = idf.get(e, 0.0) if e in pf else 0.0     # exact phenotype
                if soft and e in align:                        # SapBERT-aligned phenotype
                    for kc, sim in align[e]:
                        if kc in pf:
                            v = sim * idf.get(kc, 0.0)
                            if v > best: best = v
                d = dsig.get(e, 0.0)
                if d > best: best = d                          # unified disease match
                s += best
            sc[k] = s / max(len(pf) + len(dsig), 1) ** 0.5
        ranked = sorted(keys, key=lambda k: -sc[k])
        r = ranked.index(t) + 1 if t in ranked else 99
        n += 1; c1 += r == 1; c3 += r <= 3; c5 += r <= 5; c10 += r <= 10
    return 100*c1/n, 100*c3/n, 100*c5/n, 100*c10/n

for soft in (False, True):
    a1, a3, a5, a10 = evaluate(soft)
    tag = "SapBERT-soft" if soft else "exact-only  "
    print(f"  {tag}  @1={a1:.1f} @3={a3:.1f} @5={a5:.1f} @10={a10:.1f}")
