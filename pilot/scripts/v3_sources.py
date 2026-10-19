"""Corpus v3 raw downloads -> IE source files holding only what the v2 corpus lacks.

Per disease: PMC clinical-section text of articles not in v2 (v2 stored the same sections with the
same cap), the tables of every PMC article (v2 dropped tables), PubMed abstracts whose PMID v2 does
not hold, and the gene-found abstracts and PMC articles. Each abstract and each article's tables
stay a separate document, because provenance is recorded per document. Gene-found documents carry
via="gene", so a time-split build can drop them (build_graph_v2.py --gene-cut).

    python v3_sources.py --raw pilot/cache/v3/raw --out pilot/cache/v3/sources_new

With --full, every PMC article (v2 ones included, same doc_id) carries its whole body instead of the
12k clinical sections; list that directory before the v2 sources so the whole body wins the merge.
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


def convert(rec: dict, v2: dict | None, full: bool = False) -> dict:
    """Name-found documents come first, so an article found by both routes is not marked gene-found.
    An abstract is dropped when the same article's body is already a document."""
    have = set()
    for d in (v2 or {}).get("docs", []):
        have |= {d["doc_id"], d.get("pmcid") or "", "PMID" + d["pmid"] if d.get("pmid") else ""}
    docs, seen = [], set()

    def add(doc_id, d, kind, text, by_gene):
        if not text.strip() or doc_id in seen:
            return
        seen.add(doc_id)
        docs.append({"doc_id": doc_id, "pmid": d["pmid"], "kind": kind, "title": d["title"], "year": d["year"],
                     "text": text, **({"via": "gene"} if by_gene else {})})

    for d, by_gene in [(d, False) for d in rec["pmc"]] + [(d, True) for d in rec["gene_pmc"]]:
        if full:
            add(d["id"], d, "pmc", d["full"], by_gene)
        elif d["id"] not in have and "PMID" + d["pmid"] not in have:
            add(d["id"], d, "pmc", d["clinical"], by_gene)
        add("TBL" + d["id"], d, "table", linearize(d["tables"]), by_gene)
    bodies = {"PMID" + x["pmid"] for x in docs if x["kind"] == "pmc" and x["pmid"]}
    for d, by_gene in [(d, False) for d in rec["abstracts"]] + [(d, True) for d in rec["gene_abstracts"]]:
        if d["id"] not in have and d["id"] not in bodies:
            add(d["id"], d, "abstract", d["full"], by_gene)
    return {"id": rec["key"], "name": rec["name"], "docs": docs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="convert again files already converted")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(f"{a.raw}/*.json.gz")):
        dst = f"{a.out}/{os.path.basename(f)[:-3]}"
        if os.path.exists(dst) and not a.overwrite:
            continue
        rec = json.load(gzip.open(f))
        p = f"{V2}/sources/{rec['key'].replace(':', '_')}.json"
        out = convert(rec, json.load(open(p)) if os.path.exists(p) else None, a.full)
        tmp = f"{dst}.{os.getpid()}.tmp"     # the IE and dictionary loops may convert concurrently
        json.dump(out, open(tmp, "w"))
        os.replace(tmp, dst)
        n += 1
    print(f"converted {n}")


if __name__ == "__main__":
    main()
