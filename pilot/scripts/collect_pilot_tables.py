"""Table documents for the collection pilot: each article's tables as one IE source document.

Rows are written one per line with cells joined by " | " so the IE sees the row structure (a
feature name followed by per-patient +/- values). The table document keeps the article's title
and year, so the topic and time-cut rules treat it like its article.

    python collect_pilot_tables.py --pretest pilot/cache/v2/collect_pretest
"""
from __future__ import annotations

import argparse
import json
import os

V2 = "pilot/cache/v2"


def linearize(ts: list[dict]) -> str:
    return "\n\n".join(f"Table: {t['caption']}\n" + "\n".join(" | ".join(r) for r in t["rows"]) for t in ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretest", required=True)
    a = ap.parse_args()
    os.makedirs(f"{a.pretest}/new_tables", exist_ok=True)
    os.makedirs(f"{a.pretest}/merged_tables", exist_ok=True)
    pilot = json.load(open(f"{a.pretest}/pilot_omims.json"))
    keys = set()
    for o in pilot:
        rec = json.load(open(f"{a.pretest}/docs/{o.replace(':', '_')}.json"))
        tb = json.load(open(f"{a.pretest}/tables/{o.replace(':', '_')}.json"))
        merged = json.load(open(f"{a.pretest}/merged/{rec['key'].replace(':', '_')}.json"))
        meta = {d.get("pmcid") or d["doc_id"]: d for d in merged["docs"]}
        docs = []
        for pid, ts in tb["tables"].items():
            text = linearize(ts)
            if text.strip():
                src = meta.get(pid, {})
                docs.append({"doc_id": "TBL" + pid, "pmcid": pid, "kind": "table", "title": src.get("title", ""),
                             "year": src.get("year", ""), "text": text})
        json.dump({"id": rec["key"], "name": rec["name"], "docs": docs},
                  open(f"{a.pretest}/new_tables/{rec['key'].replace(':', '_')}.json", "w"))
        json.dump({**merged, "docs": merged["docs"] + docs},
                  open(f"{a.pretest}/merged_tables/{rec['key'].replace(':', '_')}.json", "w"))
        keys.add(rec["key"].replace(":", "_") + ".json")
    for f in os.listdir(f"{a.pretest}/merged"):
        dst = f"{a.pretest}/merged_tables/{f}"
        if f not in keys and not os.path.lexists(dst):
            os.symlink(os.path.realpath(f"{a.pretest}/merged/{f}"), dst)
    print("pilot table documents written for", len(keys), "diseases")


if __name__ == "__main__":
    main()
