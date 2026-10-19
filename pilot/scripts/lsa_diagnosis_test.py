"""Latent-semantic (SVD/LSA) diagnosis vs sparse overlap.

Motivation: our KG disease->phenotype profiles are sparse, so exact/hierarchy overlap misses
associations the KG never wrote as an edge. LSA factorizes the disease x phenotype matrix into
a low-rank latent space: co-occurring phenotypes get similar latent vectors, so a patient's
symptoms project to a point whose nearby diseases include ones whose *latent* profile matches
even without literal shared edges. This is classic LSA/collaborative-filtering (Deerwester
1990) -- academically standard, KG-only, no benchmark labels (§3 safe).
"""
import sys, json, math, argparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
sys.path.insert(0, "pilot/scripts")
import onlykg_eval_unified as U
import onlykg_eval_unified2 as U2


def build_lsa(kg, prof, idf, k=256):
    dz = sorted(prof)
    ph = sorted(kg["phenotypes"])
    di = {d: i for i, d in enumerate(dz)}
    pi = {p: i for i, p in enumerate(ph)}
    rows, cols, vals = [], [], []
    for d in dz:
        for p, w in prof[d].items():
            if p in pi:
                rows.append(di[d]); cols.append(pi[p]); vals.append(w)
    M = csr_matrix((vals, (rows, cols)), shape=(len(dz), len(ph)), dtype=np.float32)
    # L2-normalize disease rows (cosine geometry)
    U_, S, Vt = svds(M, k=k)
    S = S[::-1]; U_ = U_[:, ::-1]; Vt = Vt[::-1]
    d_emb = U_ * S                    # (Ndz, k)
    p_emb = Vt.T                      # (Nph, k)
    d_emb /= (np.linalg.norm(d_emb, axis=1, keepdims=True) + 1e-9)
    return di, pi, d_emb, p_emb, idf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg.json")
    ap.add_argument("--children", default="pilot/cache/umls_maps/disease_children.json")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--ppd", type=int, default=30)
    a = ap.parse_args()

    kg, prof, idf, dz_names = U.load_kg(a.kg)
    children = json.load(open(a.children))
    resolver = U.Resolver(prof, dz_names, children)
    print(f"building LSA (k={a.k}) on {len(prof)} dz x {len(kg['phenotypes'])} ph ...", flush=True)
    di, pi, d_emb, p_emb, idf = build_lsa(kg, prof, idf, a.k)

    def cluster_emb(cl):
        idxs = [di[c] for c in cl if c in di]
        if not idxs:
            return None
        v = d_emb[idxs].mean(0)
        return v / (np.linalg.norm(v) + 1e-9)

    def query_vec(ev):
        vs = [idf.get(p, 1.0) * p_emb[pi[p]] for p in ev if p in pi]
        if not vs:
            return None
        v = np.sum(vs, 0)
        return v / (np.linalg.norm(v) + 1e-9)

    def run(name, cand, pats, coref=None):
        keys = sorted(cand)
        cemb = {}
        for k in keys:
            nm, cu = cand[k]
            cemb[k] = cluster_emb(resolver.cluster(nm, cu))
        # co-reference groups
        par = {k: k for k in keys}
        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        for grp in (coref or []):
            g = [k for k in grp if k in par]
            for k in g[1:]:
                par[find(k)] = find(g[0])
        rankable = [k for k in keys if cemb[k] is not None]
        E = np.stack([cemb[k] for k in rankable])           # (R, k)
        n = c1 = c3 = c5 = c10 = 0
        for tkeys, ev, ds in pats:
            q = query_vec(ev)
            tg = {find(t) for t in tkeys if t in par}
            if q is None:
                n += 1; continue
            sims = E @ q
            gsc = {}
            for j, k in enumerate(rankable):
                g = find(k)
                if sims[j] > gsc.get(g, -1e9):
                    gsc[g] = sims[j]
            ranked = sorted(gsc, key=lambda g: -gsc[g])
            rr = [ranked.index(g) + 1 for g in tg if g in gsc]
            r = min(rr) if rr else len(ranked) + 1
            n += 1; c1 += r == 1; c3 += r <= 3; c5 += r <= 5; c10 += r <= 10
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

    print("=== LSA latent-semantic ranking ===")
    cand, pats = U.load_ddxplus(3000); run("DDXPlus", cand, pats)
    cand, pats = U.load_symcat(a.ppd); run("SymCat", cand, pats)
    for dsn in ["RAMEDIS", "LIRICAL", "MME", "HMS"]:
        cand, pats = U.load_rarebench([dsn]); run(f"RareBench:{dsn}", cand, pats, coref_of(pats))


if __name__ == "__main__":
    main()
