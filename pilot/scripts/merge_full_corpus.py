"""Merge the full-corpus extraction into the production prediction corpus.

The full-corpus artifact is 1.1 GB and the base corpus 519 MB; decoding both into Python dicts at
once needs more memory than this host has, and an OOM kill part-way through would leave a corrupt
output. So the large file is streamed one disease at a time with ijson and applied to the base
incrementally: peak memory is the base plus a single disease record.

Additive on all three axes, so nothing already extracted is lost:
  findings   -- a finding absent from the base is appended (matched on normalized name)
  attributes -- only EMPTY slots are filled, so pass-1/pass-2 values always win
  associations -- unioned by item, after the same negation and case-description cleaning used
                  elsewhere (a negated statement stored as positive evidence is worse than none)

A disease the base does not have is added outright. An earlier merge skipped those with
`if cui not in base: continue`, which silently dropped diseases and cost one of them every patient.
"""
from __future__ import annotations
import argparse, json, re

import ijson

ATTRS = ("location", "character", "severity", "onset", "timing", "aggravating", "relieving")
NEG = re.compile(r"^(no|not|non|without|absence of|absent|denies|negative|lack of)\b", re.I)
CASE = re.compile(r"\b\d{1,3}[- ]?(year|yr|month)s?[- ]?old\b|\bmean age\b", re.I)


def nz(s: object) -> str:
    return " ".join(str(s).lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="production predictions to merge INTO")
    ap.add_argument("--incoming", required=True, help="large full-corpus extraction (streamed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", default="", help="existing records; changed diseases are dropped "
                                                 "so normalization only redoes those")
    ap.add_argument("--records-out", default="")
    a = ap.parse_args()

    base = json.load(open(a.base, encoding="utf-8"))
    print(f"base corpus: {len(base)} diseases", flush=True)

    n_new = n_find = n_cell = n_assoc = 0
    changed: set[str] = set()
    seen_in = 0

    with open(a.incoming, "rb") as fh:
        for cui, v in ijson.kvitems(fh, ""):
            seen_in += 1
            if seen_in % 2000 == 0:
                print(f"  streamed {seen_in} diseases; findings +{n_find} cells +{n_cell} "
                      f"assoc +{n_assoc}", flush=True)
            if cui not in base:
                base[cui] = v
                changed.add(cui)
                n_new += 1
                continue
            tgt_list = base[cui].setdefault("findings", [])
            index = {nz(f.get("name")): f for f in tgt_list}
            for f in v.get("findings", []) or []:
                k = nz(f.get("name"))
                if not k:
                    continue
                cur = index.get(k)
                if cur is None:
                    tgt_list.append(f)
                    index[k] = f
                    n_find += 1
                    changed.add(cui)
                    continue
                for at in ATTRS:
                    if str(f.get(at, "") or "").strip() and not str(cur.get(at, "") or "").strip():
                        cur[at] = f[at]
                        n_cell += 1
                        changed.add(cui)
                if f.get("qualifiers") and not cur.get("qualifiers"):
                    cur["qualifiers"] = f["qualifiers"]
                    changed.add(cui)
            bag = base[cui].setdefault("associations", [])
            have = {x.get("item") for x in bag}
            for x in v.get("associations", []) or []:
                it = x.get("item")
                if not it or it in have or x.get("kind") == "demographic":
                    continue
                if NEG.search(it) or CASE.search(it):
                    continue
                bag.append(x)
                have.add(it)
                n_assoc += 1
                changed.add(cui)
            src = v.get("source", "") or ""
            if len(src) > len(base[cui].get("source", "") or ""):
                base[cui]["source"] = src
                changed.add(cui)

    with open(a.out, "w") as fh:
        json.dump(base, fh)
    print(f"\nincoming diseases streamed {seen_in}")
    print(f"new diseases {n_new}  findings +{n_find}  attribute cells +{n_cell}  "
          f"associations +{n_assoc}")
    print(f"diseases changed {len(changed)}   corpus now {len(base)}")

    if a.records and a.records_out:
        rec = json.load(open(a.records, encoding="utf-8"))
        kept = {k: v for k, v in rec.items() if k not in changed}
        json.dump(kept, open(a.records_out, "w"))
        print(f"records seeded {len(kept)} (of {len(rec)}); to re-normalize {len(changed)}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
