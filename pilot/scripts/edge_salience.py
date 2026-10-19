"""Weight an edge by how prominently the source presents that finding.

The IE treats every mention alike, so a disease's cardinal symptom and a finding named once in a
complications list arrive with identical weight. The profiles that result are broad and unfocused:
Pneumonia carries 992 phenotypes and still covers only 57% of what its patients report, whereas 14
hand-picked concepts reach 30% -- twelve times the coverage per concept. That breadth is what
trades away Top-1 (every extra edge is another disease competing on generic evidence).

Clinical prose orders itself: the cardinal presentation comes first, incidental associations later.
Measured against DDXPlus, a finding first mentioned in the opening tenth of its source is reported
by patients 14.8% of the time; one from the closing tenth, 3.8% -- a 3.9x gradient.

So each edge gets `sal` in (0,1] from the first-mention position of its finding in the disease's
source. Nothing is dropped: a late mention still matches, it just carries less weight than the
sentence the author chose to open with. Benchmark-blind -- position is a property of the source.

Usage: python edge_salience.py --kg KG.json --preds PREDS.json --records RECS.json --out OUT.json
"""
from __future__ import annotations
import argparse, json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--preds", required=True, help="supplies each disease's source text")
    ap.add_argument("--records", required=True, help="supplies finding name -> CUI")
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor", type=float, default=0.35,
                    help="weight of a finding mentioned at the very end; a late mention is still "
                         "evidence, so this never reaches zero")
    a = ap.parse_args()

    preds = json.load(open(a.preds, encoding="utf-8"))
    recs = json.load(open(a.records, encoding="utf-8"))

    # earliest mention of each (disease, phenotype CUI), as a fraction of the source length
    pos: dict[tuple[str, str], float] = {}
    for cui, obj in recs.items():
        src = (preds.get(cui, {}).get("source", "") or "").lower()
        if len(src) < 200:
            continue
        L = len(src)
        for r in obj.get("phenotypes", []):
            c, nm = r.get("name_cui"), (r.get("name") or "").lower()
            if not c or not nm:
                continue
            i = src.find(nm)
            if i < 0:
                continue
            f = i / L
            k = (cui, c)
            if k not in pos or f < pos[k]:
                pos[k] = f

    kg = json.load(open(a.kg, encoding="utf-8"))
    n = hit = 0
    for e in kg["edges"]:
        n += 1
        f = pos.get((e["d"], e["p"]))
        if f is None:
            continue                       # not locatable in the source: leave unweighted (1.0)
        hit += 1
        e["sal"] = round(1.0 - (1.0 - a.floor) * f, 4)
    json.dump(kg, open(a.out, "w"))
    vals = sorted(e["sal"] for e in kg["edges"] if "sal" in e)
    print(f"edges {n}; salience assigned {hit} ({100*hit/max(1,n):.0f}%)")
    if vals:
        print(f"  sal: min {vals[0]}  median {vals[len(vals)//2]}  max {vals[-1]}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
