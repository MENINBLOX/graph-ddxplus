"""HPO-native RareBench evaluation (redesign validation).

Disease profiles = IE findings re-normalized to HPO terms (hpo_link). Patient evidence = the raw
HPO codes RareBench already provides (NO CUI round-trip). Matching is in HPO space with is-a
ancestor expansion + IC weighting (Phenomizer-style), so near-synonyms bridge by shared ancestor
and off-topic non-phenotype findings are filtered out by failing to link.

Compares against the old CUI-based numbers to decide whether the redesign is worth a full build.
Usage: python pilot/scripts/hpo_eval.py --ie pilot/cache/prod/ie_pred_merged3c.json --hier 3
"""
from __future__ import annotations
import json, math, argparse, sys
from collections import defaultdict, Counter
sys.path.insert(0, "pilot/scripts")
import onlykg_eval_unified as U
import hpo_link as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie", default="pilot/cache/prod/ie_pred_merged3c.json")
    ap.add_argument("--children", default="pilot/cache/umls_maps/disease_children.json")
    ap.add_argument("--hier", type=int, default=3, help="HPO ancestor expansion depth for query")
    ap.add_argument("--datasets", default="RAMEDIS,LIRICAL,MME,HMS")
    a = ap.parse_args()

    ie = json.load(open(a.ie))
    children = json.load(open(a.children))
    # disease CUI -> HPO profile (link every finding name to an HPO term; non-HPO drops out)
    prof = {}
    linked = total = 0
    for cui, rec in ie.items():
        hset = set()
        for f in rec.get("findings", []):
            total += 1
            h = H.link(f.get("name", ""))
            if h:
                hset.add(h); linked += 1
        if hset:
            prof[cui] = hset
    print(f"IE findings linked to HPO: {linked}/{total} ({100*linked/max(1,total):.1f}%); "
          f"diseases with HPO profile: {len(prof)}", flush=True)

    dz_names = {c: r.get("disease", "") for c, r in ie.items()}
    resolver = U.Resolver({c: {h: 1 for h in hs} for c, hs in prof.items()}, dz_names, children)

    # IDF over HPO terms
    df = Counter()
    for hs in prof.values():
        for h in hs:
            df[h] += 1
    N = len(prof)
    idf = {h: math.log(N / c) for h, c in df.items()}

    dm = json.load(open("data/rarebench/disease_umls_mapping.json"))["mapping"]

    def cluster_prof(cui, name):
        cl = resolver.cluster(name, cui)
        pf = set()
        for c in cl:
            pf |= prof.get(c, set())
        return pf

    for dsn in a.datasets.split(","):
        cand, pats = U.load_rarebench([dsn])
        # candidate universe = all gold CUIs (same as the CUI eval)
        cset = {}
        for k, (nm, cu) in cand.items():
            cset[k] = (nm, cu)
        # precompute candidate HPO signatures (profile ∪ ancestors, IC-weighted)
        sig = {}
        for k, (nm, cu) in cset.items():
            pf = cluster_prof(cu, nm)
            exp = {}
            for h in pf:
                for x in {h} | H.ancestors(h, a.hier):
                    exp[x] = max(exp.get(x, 0), idf.get(x, math.log(N)) if x == h else idf.get(x, 0) * 0.5)
            sig[k] = (pf, exp)
        # raw HPO patient evidence
        raw = json.load(open(f"data/rarebench/data/{dsn}.jsonl").read().splitlines()[0]) if False else None
        pat_hpo = []
        for line in open(f"data/rarebench/data/{dsn}.jsonl"):
            r = json.loads(line)
            gold = [t for t in r["RareDisease"] if t in cand]
            if gold and r.get("Phenotype"):
                pat_hpo.append((set(gold), set(r["Phenotype"])))

        n = len(pat_hpo); c1 = c3 = c10 = 0; rrsum = 0.0
        for tset, ev in pat_hpo:
            q = {}
            for h in ev:
                for x in {h} | H.ancestors(h, a.hier):
                    q[x] = max(q.get(x, 0), 1.0 if x in ev else 0.4)
            scores = {}
            for k, (pf, exp) in sig.items():
                s = 0.0
                for x, qw in q.items():
                    if x in exp:
                        s += qw * exp[x]
                scores[k] = s / (math.sqrt(len(pf)) or 1.0)
            ranked = sorted(scores, key=lambda k: -scores[k])
            ranks = [ranked.index(t) + 1 for t in tset if t in ranked]
            rank = min(ranks) if ranks else n + 1
            c1 += rank == 1; c3 += rank <= 3; c10 += rank <= 10; rrsum += 1.0 / rank
        print(f"=== HPO-native {dsn}  (n={n})  @1={100*c1/n:.2f} @3={100*c3/n:.2f} "
              f"@10={100*c10/n:.2f} MRR={rrsum/n:.4f}", flush=True)


if __name__ == "__main__":
    main()
