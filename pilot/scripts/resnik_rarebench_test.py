"""Phenomizer/Resnik phenotype-semantic-similarity scoring for RareBench (HPO benchmarks).

The achievable ceiling (true disease rankable via hierarchy) is 84% (RAMEDIS) / 90% (LIRICAL),
but the current overlap scorer reaches only ~23% Recall@1 -> the bottleneck is DISCRIMINATION
among rankable candidates, not content. The domain-standard remedy for HPO-based rare-disease
diagnosis is information-content semantic similarity (Resnik 1995; Phenomizer, Köhler 2009):
a SPECIFIC shared ancestor (high IC) counts far more than a generic one. We score a candidate
by the query->disease best-match Resnik similarity, weighting each patient phenotype by the IC
of its most-informative common ancestor with the disease's profile.
"""
import sys, json, math, argparse
from collections import defaultdict
sys.path.insert(0, "pilot/scripts")
import onlykg_eval_unified as U
import onlykg_eval_unified2 as U2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg.json")
    ap.add_argument("--children", default="pilot/cache/umls_maps/disease_children.json")
    ap.add_argument("--parents", default="pilot/cache/umls_maps/phenotype_parents.json")
    ap.add_argument("--cap", type=int, default=3, help="phenotype ancestor hops")
    ap.add_argument("--sym", type=float, default=0.0, help="weight of disease->query direction")
    a = ap.parse_args()

    kg, prof, idf, dz_names = U.load_kg(a.kg)
    children = json.load(open(a.children))
    child_adj, parent_adj = U2.build_disease_hierarchy(children)
    parents = json.load(open(a.parents))
    resolver = U.Resolver(prof, dz_names, children)

    # ancestor-IC closure for a phenotype: {ancestor cui -> idf}, self included, cap hops
    _aic = {}
    def anc_ic(c):
        if c in _aic:
            return _aic[c]
        out = {c: idf.get(c, 0.0)}
        seen, fr = {c}, {c}
        for _ in range(a.cap):
            nx = set()
            for x in fr:
                for p in parents.get(x, ()):
                    if p not in seen:
                        seen.add(p); nx.add(p)
                        if p in idf:
                            out[p] = idf[p]
            fr = nx
        _aic[c] = out
        return out

    def run(ds):
        cand, pats = U.load_rarebench([ds])
        keys = sorted(cand)
        # profile ancestor-IC union per candidate; also raw profile for the reverse direction
        panc, prawset = {}, {}
        for k in keys:
            nm, cu = cand[k]
            raw = U.merged_profile(resolver.cluster(nm, cu), prof)
            prawset[k] = set(raw)
            u = {}
            for p in raw:
                for anc, ic in anc_ic(p).items():
                    if ic > u.get(anc, 0):
                        u[anc] = ic
            panc[k] = u
        rankable = [k for k in keys if panc[k]]
        # co-reference collapse (benchmark-declared)
        par = {k: k for k in keys}
        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        for tk, _, _ in pats:
            for t in tk[1:]:
                if t in par and tk[0] in par:
                    par[find(t)] = find(tk[0])
        grp = {k: find(k) for k in keys}

        n = c1 = c3 = c5 = c10 = 0
        for tk, ev, _ in pats:
            qa = {e: anc_ic(e) for e in ev}
            sc = {}
            for k in rankable:
                pu = panc[k]
                # query -> disease best-match Resnik: for each q, max shared-ancestor IC
                s = 0.0
                for e in ev:
                    best = 0.0
                    for anc, ic in qa[e].items():
                        if anc in pu:
                            v = min(ic, pu[anc])
                            if v > best:
                                best = v
                    s += best
                sc[k] = s
            gsc = {}
            for k in rankable:
                if sc[k] > gsc.get(grp[k], -1):
                    gsc[grp[k]] = sc[k]
            ranked = sorted(gsc, key=lambda g: -gsc[g])
            tg = {grp[t] for t in tk if t in grp}
            rr = [ranked.index(g) + 1 for g in tg if g in gsc]
            r = min(rr) if rr else len(ranked) + 1
            n += 1; c1 += r == 1; c3 += r <= 3; c5 += r <= 5; c10 += r <= 10
        print(f"  {ds:8} @1={100*c1/n:.1f} @3={100*c3/n:.1f} @5={100*c5/n:.1f} @10={100*c10/n:.1f}")

    print(f"=== Resnik query->disease best-match (cap={a.cap}) ===")
    for ds in ["RAMEDIS", "LIRICAL", "MME", "HMS"]:
        run(ds)


if __name__ == "__main__":
    main()
