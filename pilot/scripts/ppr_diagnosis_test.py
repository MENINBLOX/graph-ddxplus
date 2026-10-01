"""Personalized PageRank (random-walk-with-restart) diagnosis vs LSA fusion.

Classic graph method (Tong 2006; used in gene/disease prioritization -- e.g. Phen2Disease,
RWR). We build the bipartite disease<->phenotype graph (IDF-weighted edges), restart from the
patient's evidence phenotypes, and rank diseases by stationary visitation. Captures MULTI-HOP
association (patient symptom -> disease -> its other phenotypes -> related diseases) that LSA's
global low-rank and sparse overlap miss. KG-only, no benchmark labels (§3 safe).
"""
import sys, json, argparse
import numpy as np
from scipy.sparse import csr_matrix
sys.path.insert(0, "pilot/scripts")
import onlykg_eval_unified as U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg.json")
    ap.add_argument("--children", default="pilot/cache/umls_maps/disease_children.json")
    ap.add_argument("--alpha", type=float, default=0.7, help="walk continuation prob")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--ppd", type=int, default=30)
    a = ap.parse_args()

    kg, prof, idf, dz_names = U.load_kg(a.kg)
    children = json.load(open(a.children))
    resolver = U.Resolver(prof, dz_names, children)

    dz = sorted(prof); ph = sorted(kg["phenotypes"])
    Nd, Np = len(dz), len(ph)
    di = {d: i for i, d in enumerate(dz)}
    pi = {p: i + Nd for i, p in enumerate(ph)}     # phenotype node ids offset by Nd
    # bipartite adjacency (symmetric), edge weight = idf(phenotype)
    r, c, v = [], [], []
    for d in dz:
        for p, w in prof[d].items():
            if p in kg["phenotypes"]:
                a_, b_ = di[d], pi[p]
                r += [a_, b_]; c += [b_, a_]; v += [w, w]
    N = Nd + Np
    A = csr_matrix((v, (r, c)), shape=(N, N), dtype=np.float32)
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1.0
    Dinv = csr_matrix((1.0 / deg, (range(N), range(N))), shape=(N, N), dtype=np.float32)
    W = (A @ Dinv).tocsr()                          # column-stochastic transition
    print(f"graph: {Nd} dz + {Np} ph, {A.nnz//2} edges", flush=True)

    def ppr(seed_ids, seed_w):
        e = np.zeros(N, dtype=np.float32)
        for i, w in zip(seed_ids, seed_w):
            e[i] = w
        s = e.sum()
        if s == 0:
            return None
        e /= s
        rvec = e.copy()
        for _ in range(a.iters):
            rvec = (1 - a.alpha) * e + a.alpha * (W @ rvec)
        return rvec

    def run(name, cand, pats, coref=None):
        keys = sorted(cand)
        cl_of = {k: (resolver.cluster(*cand[k]) | ({cand[k][1]} if cand[k][1] else set())) for k in keys}
        par = {k: k for k in keys}
        def find(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        for grp in (coref or []):
            g = [k for k in grp if k in par]
            for k in g[1:]: par[find(k)] = find(g[0])
        rankable = [k for k in keys if any(cc in di for cc in cl_of[k])]
        n = c1 = c3 = c5 = c10 = 0
        for tkeys, ev, ds in pats:
            seeds = [(pi[p], idf.get(p, 1.0)) for p in ev if p in pi]
            seeds += [(di[t], 5.0) for t in ev if t in di]   # disease-CUI evidence (§11) as seeds
            if not seeds:
                n += 1; continue
            rvec = ppr([s for s, _ in seeds], [w for _, w in seeds])
            if rvec is None:
                n += 1; continue
            gsc = {}
            for k in rankable:
                sc = max((rvec[di[cc]] for cc in cl_of[k] if cc in di), default=0.0)
                g = find(k)
                if sc > gsc.get(g, -1): gsc[g] = sc
            ranked = sorted(gsc, key=lambda g: -gsc[g])
            tg = {find(t) for t in tkeys if t in par}
            rr = [ranked.index(g) + 1 for g in tg if g in gsc]
            rk = min(rr) if rr else len(ranked) + 1
            n += 1; c1 += rk == 1; c3 += rk <= 3; c5 += rk <= 5; c10 += rk <= 10
        print(f"  {name:18} @1={100*c1/n:.2f} @3={100*c3/n:.2f} @5={100*c5/n:.2f} @10={100*c10/n:.2f}")

    def coref_of(pats):
        par = {}
        def find(x):
            par.setdefault(x, x)
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        for tk, _, _ in pats:
            for t in tk[1:]:
                par.setdefault(tk[0], tk[0]); par[find(t)] = find(tk[0])
        from collections import defaultdict
        g = defaultdict(list)
        for k in par: g[find(k)].append(k)
        return list(g.values())

    print(f"=== PPR (alpha={a.alpha}, iters={a.iters}) ===")
    cand, pats = U.load_ddxplus(3000); run("DDXPlus", cand, pats)
    cand, pats = U.load_symcat(a.ppd); run("SymCat", cand, pats)
    for dsn in ["RAMEDIS", "LIRICAL", "MME", "HMS"]:
        cand, pats = U.load_rarebench([dsn]); run(f"RareBench:{dsn}", cand, pats, coref_of(pats))


if __name__ == "__main__":
    main()
