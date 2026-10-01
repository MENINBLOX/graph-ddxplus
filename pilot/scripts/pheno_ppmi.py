"""Phenotype-phenotype association derived from the KG itself (positive PMI).

The graph has been strictly bipartite and single-hop: a patient's finding either sits in the
candidate's profile or contributes nothing. That discards what the KG knows implicitly -- two
phenotypes that co-occur across disease profiles far more often than chance are clinically linked,
so a patient finding the candidate lacks can still be partly explained by a finding it has.

Restricted to POSITIVE pointwise mutual information, which is the standard way to keep
co-occurrence from degenerating into "everything is related to the hubs": pairs occurring at or
below chance get no edge, and hub phenotypes (Pain, Fever) score low PMI with everything precisely
because they are ubiquitous. Every count comes from the KG's own disease profiles -- no benchmark
file is read, so it stays §5 benchmark-blind and no inference-time LLM is involved (§4).

Output: {cui: [[neighbour_cui, ppmi_weight], ...]} -> pheno_ppmi.json
"""
from __future__ import annotations
import argparse, collections, json, math


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--out", default="pilot/cache/prod/pheno_ppmi.json")
    ap.add_argument("--min-df", type=int, default=3,
                    help="ignore phenotypes on fewer diseases than this; their PMI is dominated by "
                         "sampling noise")
    ap.add_argument("--max-df-frac", type=float, default=0.25,
                    help="ignore phenotypes on more than this fraction of diseases; a phenotype "
                         "shared by a quarter of the corpus separates nothing")
    ap.add_argument("--topk", type=int, default=12, help="neighbours kept per phenotype")
    ap.add_argument("--min-ppmi", type=float, default=1.0)
    a = ap.parse_args()

    kg = json.load(open(a.kg, encoding="utf-8"))
    prof = collections.defaultdict(set)
    for e in kg["edges"]:
        prof[e["d"]].add(e["p"])
    N = len(prof)
    df = collections.Counter()
    for ps in prof.values():
        df.update(ps)
    lo, hi = a.min_df, int(a.max_df_frac * N)
    keep = {p for p, c in df.items() if lo <= c <= hi}
    print(f"diseases {N}; phenotypes {len(df)}; kept after df filter [{lo},{hi}]: {len(keep)}")

    co = collections.Counter()
    for ps in prof.values():
        ks = sorted(ps & keep)
        if len(ks) > 400:            # a handful of huge profiles would dominate every count
            continue
        for i, x in enumerate(ks):
            for y in ks[i + 1:]:
                co[(x, y)] += 1
    print(f"co-occurring pairs: {len(co)}")

    nbr = collections.defaultdict(list)
    for (x, y), c in co.items():
        if c < 2:
            continue
        # PMI over the disease-profile "corpus"
        pmi = math.log((c * N) / (df[x] * df[y]))
        if pmi < a.min_ppmi:
            continue
        nbr[x].append((y, round(pmi, 3)))
        nbr[y].append((x, round(pmi, 3)))
    out = {}
    for p, lst in nbr.items():
        lst.sort(key=lambda t: -t[1])
        out[p] = [[c, w] for c, w in lst[:a.topk]]
    json.dump(out, open(a.out, "w"))
    sizes = [len(v) for v in out.values()]
    print(f"phenotypes with neighbours: {len(out)}; mean neighbours "
          f"{sum(sizes)/max(1,len(sizes)):.1f}")
    ph = kg["phenotypes"]
    for probe in list(out)[:3]:
        nm = ph.get(probe, {}).get("name", probe)[:26]
        ns = [ph.get(c, {}).get("name", c)[:20] for c, _ in out[probe][:4]]
        print(f"  {nm:28} -> {ns}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
