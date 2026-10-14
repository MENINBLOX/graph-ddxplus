"""Corpus v3 raw downloads -> IE source files holding only what the v2 corpus lacks.

Per disease: PMC clinical-section text of articles not in v2 (v2 stored the same sections with the
same cap), the tables of every PMC article (v2 dropped tables), PubMed abstracts whose PMID v2 does
not hold, and the gene-found abstracts and PMC articles. Each abstract and each article's tables
stay a separate document, because provenance is recorded per document.

    python v3_sources.py --raw pilot/cache/v3/raw --out pilot/cache/v3/sources_new
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os

V2 = "pilot/cache/v2"


def linearize(ts: list[dict]) -> str:
    return "\n\n".join(f"Table: {t['caption']}\n" + "\n".join(" | ".join(r) for r in t["rows"]) for t in ts)


def convert(rec: dict, v2: dict | None) -> dict:
    have = set()
    for d in (v2 or {}).get("docs", []):
        have |= {d["doc_id"], d.get("pmcid") or "", "PMID" + d["pmid"] if d.get("pmid") else ""}
    docs, seen = [], set()

    def add(doc_id, pmid, kind, title, year, text):
        if not text.strip() or doc_id in seen:
            return
        seen.add(doc_id)
        docs.append({"doc_id": doc_id, "pmid": pmid, "kind": kind, "title": title, "year": year, "text": text})

    for d in rec["pmc"] + rec["gene_pmc"]:
        if d["id"] not in have and "PMID" + d["pmid"] not in have:
            add(d["id"], d["pmid"], "pmc", d["title"], d["year"], d["clinical"])
        add("TBL" + d["id"], d["pmid"], "table", d["title"], d["year"], linearize(d["tables"]))
    for d in rec["abstracts"] + rec["gene_abstracts"]:
        if d["id"] not in have:
            add(d["id"], d["pmid"], "abstract", d["title"], d["year"], d["full"])
    return {"id": rec["key"], "name": rec["name"], "docs": docs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(f"{a.raw}/*.json.gz")):
        dst = f"{a.out}/{os.path.basename(f)[:-3]}"
        if os.path.exists(dst):
            continue
        rec = json.load(gzip.open(f))
        p = f"{V2}/sources/{rec['key'].replace(':', '_')}.json"
        out = convert(rec, json.load(open(p)) if os.path.exists(p) else None)
        tmp = dst + ".tmp"
        json.dump(out, open(tmp, "w"))
        os.replace(tmp, dst)
        n += 1
    print(f"converted {n}")


if __name__ == "__main__":
    main()
