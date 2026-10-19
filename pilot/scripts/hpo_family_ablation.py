"""Leave-one-out ablation over HPO Clinical-modifier FAMILIES, not over hand-picked slots.

The old harness ablated six attributes we had chosen ourselves, which invites the question "why
those six" and cannot test anything outside them. Here the unit of ablation is a family of the HPO
Clinical modifier hierarchy (Clinical course, Position, Severity, Triggered by, ...): the inventory
comes from the ontology, and which families exist in our KG is a measurement rather than a choice.

Both sides -- the KG edge and the benchmark's patient finding -- are normalized into HPO terms
before comparison, so a match no longer depends on a class table we wrote. A patient value matches
either the same HPO term (full credit) or a sibling in the same family (half credit).

Significance is judged exactly as before: the observed leave-one-in delta is compared against a
permutation null that shuffles that family's values across the edges carrying them (marginal
preserved, disease-attribute association destroyed), plus a patient-level bootstrap CI.

Usage: python hpo_family_ablation.py --kg KG.json --queries nlice_queries_v2.json
"""
from __future__ import annotations
import argparse, json, math, random, sys, collections
sys.path.insert(0, "pilot/scripts")
import hpo_modifier

SEED = 42
LAMBDA = 1.0
SLOTS = ("location", "character", "severity", "onset", "timing", "aggravating", "relieving")


def famname(h, voc):
    """Axis a normalized value belongs to: an HPO modifier family, or the UMLS anatomy axis."""
    return "Location (UMLS anatomy)" if h.startswith("LOC:") else voc.label.get(voc.family(h), "?")


def hpo_terms(obj: dict, voc) -> set[str]:
    """Every HPO clinical modifier stated by an object, whether it arrived as a fixed slot or as an
    open-schema qualifier."""
    out = set()
    for q in obj.get("qualifiers", ()) or ():
        out |= voc.classes(q.get("quote", ""), q.get("relation", ""))
    for k in SLOTS:
        v = obj.get(k) or obj.get(k + "_cui")
        if isinstance(v, str) and v:
            out |= voc.classes(v, k)
    return out


def load(kg_path: str, q_path: str, voc):
    kg = json.load(open(kg_path, encoding="utf-8"))
    df = collections.Counter()
    prof: dict[str, dict] = collections.defaultdict(dict)
    for e in kg["edges"]:
        d, p = e["d"], e["p"]
        if p not in prof[d]:
            df[p] += 1
        at = e.get("attrs", {}) or {}
        terms = hpo_terms(at, voc)
        # Location is NOT an HPO clinical modifier -- it lives on the UMLS anatomy axis -- but it
        # is the one channel the slot ablation credited, so it is carried as its own axis rather
        # than dropped just because the ontology puts it elsewhere.
        if at.get("location_cui"):
            terms = terms | {"LOC:" + at["location_cui"]}
        prof[d][p] = terms
    N = max(1, len(kg["diseases"]))
    idf = {p: math.log(N / c) for p, c in df.items()}
    Q = json.load(open(q_path, encoding="utf-8"))
    pats = []
    for pt in Q["patients"]:
        fs = []
        for f in pt["findings"]:
            t = hpo_terms(f, voc)
            if f.get("location_cui"):
                t = t | {"LOC:" + f["location_cui"]}
            fs.append((f["cui"], t))
        pats.append((pt["true_cui"], fs))
    cand = sorted(set(Q["meta"]["diseases"].values()) & set(prof))
    return prof, idf, pats, cand


def rank(pats, cand, prof, idf, voc, fams):
    """GTPA@1 and MRR when only `fams` (None = none) contribute attribute credit."""
    hit = rr = 0.0
    for truth, fs in pats:
        best = {}
        for d in cand:
            pf = prof.get(d)
            if not pf:
                best[d] = 0.0
                continue
            s = 0.0
            for cui, qt in fs:
                cell = pf.get(cui)
                if cell is None:
                    continue
                w = idf.get(cui, 0.0)
                s += w
                if fams and qt and cell:
                    a = {h for h in qt if famname(h, voc) in fams}
                    b = {h for h in cell if famname(h, voc) in fams}
                    if a & b:
                        s += LAMBDA * w
                    elif {famname(h, voc) for h in a} & {famname(h, voc) for h in b}:
                        s += 0.5 * LAMBDA * w
            best[d] = s / max(math.sqrt(len(pf)), 1.0)
        order = sorted(cand, key=lambda d: -best.get(d, 0.0))
        r = order.index(truth) + 1 if truth in order else len(order)
        hit += r == 1
        rr += 1.0 / r
    n = len(pats)
    return 100 * hit / n, rr / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--queries", default="pilot/cache/prod/nlice_queries_v2.json")
    ap.add_argument("--perm", type=int, default=30)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    rng = random.Random(SEED)
    voc = hpo_modifier.load()
    prof, idf, pats, cand = load(a.kg, a.queries, voc)

    present = collections.Counter()
    distinct = collections.defaultdict(set)
    for pf in prof.values():
        for terms in pf.values():
            for h in terms:
                f = famname(h, voc)
                present[f] += 1
                distinct[f].add(h)
    fams = [f for f, n in present.most_common() if n >= 20]
    print(f"candidates={len(cand)} patients={len(pats)}")
    print(f"HPO modifier families present in this KG (>=20 edge values): {fams}")
    print(f"  counts: {dict(present.most_common(10))}")
    # a family whose values are all identical has nothing to permute; say so instead of printing
    # a p-value that looks meaningful
    print(f"  distinct values: {{{', '.join(f'{f}: {len(distinct[f])}' for f in fams)}}}\n")

    base1, baseM = rank(pats, cand, prof, idf, voc, None)
    print(f"  {'model':22}{'GTPA@1':>9}{'MRR':>9}")
    print(f"  {'base (no attributes)':22}{base1:9.2f}{baseM:9.4f}")
    per = {}
    for f in fams:
        per[f] = rank(pats, cand, prof, idf, voc, {f})
        print(f"  {('+' + f)[:22]:22}{per[f][0]:9.2f}{per[f][1]:9.4f}")
    allf = rank(pats, cand, prof, idf, voc, set(fams))
    print(f"  {'+all families':22}{allf[0]:9.2f}{allf[1]:9.4f}\n")

    print(f"  {'family':22}{'d@1':>8}{'null_mean':>11}{'null_sd':>9}{'p':>7}")
    for f in fams:
        obs = per[f][0] - base1
        null = []
        for _ in range(a.perm):
            vals, slots = [], []
            for d, pf in prof.items():
                for p, terms in pf.items():
                    keep = {h for h in terms if famname(h, voc) == f}
                    if keep:
                        vals.append(keep); slots.append((d, p))
            rng.shuffle(vals)
            shuf = {d: dict(pf) for d, pf in prof.items()}
            for (d, p), v in zip(slots, vals):
                shuf[d][p] = (prof[d][p] - {h for h in prof[d][p]
                                            if famname(h, voc) == f}) | v
            null.append(rank(pats, cand, shuf, idf, voc, {f})[0] - base1)
        mu = sum(null) / len(null)
        sd = (sum((x - mu) ** 2 for x in null) / max(1, len(null) - 1)) ** 0.5
        p = sum(1 for x in null if x >= obs) / len(null)
        note = "  <- only 1 distinct value: null is degenerate" if len(distinct[f]) < 2 else ""
        print(f"  {f[:22]:22}{obs:8.3f}{mu:11.3f}{sd:9.3f}{p:7.3f}{note}")
    print("\n  (a family is credited when its observed delta beats the permutation null, p<0.05)")


if __name__ == "__main__":
    main()
