"""Merge two IE prediction files at the finding level: union the findings of each disease CUI.
Used to add GeneReviews-derived phenotypes (esp. pathognomonic biochemistry) to the base
PubMed KG without losing the base content. Dedup findings by normalized name (case/space).
"""
from __future__ import annotations
import json, re, argparse


def norm(nm):
    return re.sub(r"\s+", " ", str(nm).lower().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="pilot/cache/prod/ie_pred_full.json")
    ap.add_argument("--add", default="pilot/cache/prod/ie_pred_gr.json")
    ap.add_argument("--out", default="pilot/cache/prod/ie_pred_merged.json")
    a = ap.parse_args()
    base = json.load(open(a.base))
    add = json.load(open(a.add))
    merged = 0; added_findings = 0
    for cui, rec in add.items():
        if cui in base:
            seen = {norm(f.get("name", "")) for f in base[cui].get("findings", [])}
            for f in rec.get("findings", []):
                if norm(f.get("name", "")) not in seen:
                    base[cui].setdefault("findings", []).append(f)
                    seen.add(norm(f.get("name", ""))); added_findings += 1
            # keep the richer source text for provenance
            if len(rec.get("source", "")) > len(base[cui].get("source", "")):
                base[cui]["source"] = rec["source"]
            merged += 1
        else:
            base[cui] = rec
    json.dump(base, open(a.out, "w"))
    print(f"merged {merged} diseases; +{added_findings} new findings; total diseases {len(base)}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
