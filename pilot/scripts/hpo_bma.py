"""HPO-space bidirectional IC-weighted Best-Match-Average (BMA) scorer — the recall@3 method.

Prior-work synthesis: every high-recall phenotype ranker (Phenomizer, PubCaseFinder, Phen2Disease,
Xrare) replaces exact set-overlap with SOFT, IC-weighted, BIDIRECTIONAL best-match matching over the
HPO is-a DAG. A patient term one level off the disease's annotated term gets graded partial credit
(IC of their most-informative common ancestor, MICA) instead of 0. §3-safe: IC = our KG-derived IDF
over HPO terms; uses only the HPO DAG (data/medkg/hp.json via hpo_link); no frequency tables.

Disease profile = KG findings mapped to HPO (via the cui2hpo bridge). Patient evidence = the raw HPO
codes RareBench provides. Score = S_patient-centric + S_disease-centric BMA (Phen2Disease form).

Usage: python pilot/scripts/hpo_bma.py --kg pilot/cache/prod/kg_merged3c.json --sim lin --datasets RAMEDIS,LIRICAL,MME,HMS
"""
from __future__ import annotations
import json, math, argparse, sys, functools
from collections import defaultdict, Counter
sys.path.insert(0, "pilot/scripts")
import onlykg_eval_unified as U
import hpo_link as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg_merged3c.json")
    ap.add_argument("--bridge", default="pilot/cache/prod/hpo_bridge.json")
    ap.add_argument("--children", default="pilot/cache/umls_maps/disease_children.json")
    ap.add_argument("--sim", choices=["resnik", "lin"], default="lin")
    ap.add_argument("--datasets", default="RAMEDIS,LIRICAL,MME,HMS")
    a = ap.parse_args()

    kg = json.load(open(a.kg))
    c2h = json.load(open(a.bridge))["cui2hpo"]
    # disease CUI -> HPO profile (map each phenotype-CUI edge through the bridge)
    prof = defaultdict(set)
    for e in kg["edges"]:
        h = c2h.get(e["p"])
        if h:
            prof[e["d"]].add(h)
    prof = {d: hs for d, hs in prof.items() if hs}
    print(f"diseases with HPO profile: {len(prof)}", flush=True)

    dz_names = {c: v.get("name", "") for c, v in kg["diseases"].items()}
    resolver = U.Resolver({c: {h: 1 for h in hs} for c, hs in prof.items()}, dz_names,
                          json.load(open(a.children)))

    # IC over HPO terms = -log(df/Ndz); df = # diseases whose profile contains t (self only; the
    # BMA already generalizes via MICA so we keep df on annotated terms, not their ancestors)
    Ndz = len(prof)
    df = Counter()
    for hs in prof.values():
        for h in hs:
            df[h] += 1
    IC = {h: math.log(Ndz / c) for h, c in df.items()}
    IC_MAX = math.log(Ndz)

    @functools.lru_cache(maxsize=400000)
    def anc(h):
        return frozenset({h} | H.ancestors(h, 8))

    lin = (a.sim == "lin")

    @functools.lru_cache(maxsize=2000000)
    def sim(t1, t2):
        common = anc(t1) & anc(t2)
        if not common:
            return 0.0
        mica = max((IC.get(x, 0.0) for x in common), default=0.0)   # Resnik = IC(MICA)
        if lin:
            denom = IC.get(t1, IC_MAX) + IC.get(t2, IC_MAX)
            return (2 * mica / denom) if denom > 0 else 0.0
        return mica

    def bma(P, D):
        if not P or not D:
            return 0.0
        # patient-centric: each patient term's best match into D, IC-weighted
        num_p = sum(IC.get(t, IC_MAX) * max(sim(t, d) for d in D) for t in P)
        den_p = sum(IC.get(t, IC_MAX) for t in P) or 1.0
        num_d = sum(IC.get(d, IC_MAX) * max(sim(d, t) for t in P) for d in D)
        den_d = sum(IC.get(d, IC_MAX) for d in D) or 1.0
        return num_p / den_p + num_d / den_d

    dm = json.load(open("data/rarebench/disease_umls_mapping.json"))["mapping"]

    for dsn in a.datasets.split(","):
        cand, pats = U.load_rarebench([dsn])
        # candidate HPO signatures (resolved cluster profile)
        sig = {}
        for k, (nm, cu) in cand.items():
            hs = set()
            for c in resolver.cluster(nm, cu):
                hs |= prof.get(c, set())
            sig[k] = hs
        pat = []
        for line in open(f"data/rarebench/data/{dsn}.jsonl"):
            r = json.loads(line)
            gold = {t for t in r["RareDisease"] if t in cand}
            ev = set(r["Phenotype"])
            if gold and ev:
                pat.append((gold, ev))
        n = len(pat); c1 = c3 = c10 = 0; rr = 0.0
        for tset, ev in pat:
            scores = {k: bma(ev, hs) for k, hs in sig.items()}
            ranked = sorted(scores, key=lambda k: -scores[k])
            rk = min((ranked.index(t) + 1 for t in tset if t in ranked), default=n + 1)
            c1 += rk == 1; c3 += rk <= 3; c10 += rk <= 10; rr += 1.0 / rk
        print(f"=== BMA({a.sim}) {dsn}  (n={n})  @1={100*c1/n:.2f} @3={100*c3/n:.2f} "
              f"@10={100*c10/n:.2f} MRR={rr/n:.4f}", flush=True)


if __name__ == "__main__":
    main()
