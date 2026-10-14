"""Turn collection_pretest.py downloads into IE sources for a pilot subset.

Writes (1) <out>/new/<key>.json with only the articles the current corpus lacks, for ie_v2.py, and
(2) <out>/merged/ with every current source file, where the pilot diseases' files carry both the
current and the new articles, for build_graph_v2.py (titles, years and the generic-article rule).

    python collect_pilot_sources.py --omims pilot/cache/v2/collect_pretest/pilot_omims.json \
        --out pilot/cache/v2/collect_pretest
"""
from __future__ import annotations

import argparse
import glob
import json
import os

V2 = "pilot/cache/v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omims", required=True, help="JSON list of OMIM ids with downloads")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/new", exist_ok=True)
    os.makedirs(f"{a.out}/merged", exist_ok=True)
    pilot = {}
    for o in json.load(open(a.omims)):
        rec = json.load(open(f"{a.out}/docs/{o.replace(':', '_')}.json"))
        base_p = f"{V2}/sources/{rec['key'].replace(':', '_')}.json"
        base = json.load(open(base_p)) if os.path.exists(base_p) else {"id": rec["key"], "name": rec["name"], "docs": []}
        have = {d["doc_id"] for d in base["docs"]} | {d.get("pmcid") for d in base["docs"]} | \
            {"PMID" + d["pmid"] for d in base["docs"] if d.get("pmid")}
        new, seen = [], set()
        for d in rec["pmc"] + rec["gene_pmc"] + rec["abstracts"] + rec["gene_abstracts"]:
            pm = "PMID" + d["pmid"] if d.get("pmid") else ""
            if d["id"] in have or pm in have or d["id"] in seen or (pm and pm in seen) or not d["full"]:
                continue
            seen |= {d["id"], pm}
            new.append({"doc_id": d["id"], "pmid": d.get("pmid", ""), "kind": "pmc" if d["id"].startswith("PMC") else "abstract",
                        "title": d["title"], "year": d["year"], "text": d["full"]})
        json.dump({"id": rec["key"], "name": rec["name"], "docs": new}, open(f"{a.out}/new/{rec['key'].replace(':', '_')}.json", "w"))
        pilot[rec["key"]] = {**base, "docs": base["docs"] + new}
    for f in glob.glob(f"{V2}/sources/*.json"):
        dst = f"{a.out}/merged/{os.path.basename(f)}"
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(f), dst)
    for k, rec in pilot.items():
        dst = f"{a.out}/merged/{k.replace(':', '_')}.json"
        if os.path.lexists(dst):
            os.remove(dst)
        json.dump(rec, open(dst, "w"))
    n = sum(len(json.load(open(f))["docs"]) for f in glob.glob(f"{a.out}/new/*.json"))
    print(f"pilot diseases {len(pilot)}, new articles {n}")


if __name__ == "__main__":
    main()
