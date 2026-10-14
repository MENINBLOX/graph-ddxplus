"""LLM-free extraction baseline: HPO label/exact-synonym matches in each sentence of a disease's articles.

Writes the ie_v2.py JSONL format, so build_graph_v2.py --hpo-names turns the matches into edges the
same way it treats LLM items. A match preceded by a negation cue within the five tokens before it
is dropped. Comparing graphs built from the same articles isolates what the LLM extraction adds
over dictionary matching.

    python dict_ie.py --sources DIR --out OUT.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_graph_v2 import hpo_name_index  # noqa: E402

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n+")
TOK = re.compile(r"[a-z0-9]+")
NEG = {"no", "not", "without", "absent", "absence", "negative", "denied", "denies", "normal", "lack"}
MAX_N = 8


def matches(sent: str, names: set[str]) -> list[str]:
    toks = TOK.findall(sent.lower())
    out, i = [], 0
    while i < len(toks):
        for n in range(min(MAX_N, len(toks) - i), 0, -1):
            cand = " ".join(toks[i:i + n])
            if cand in names:
                if not NEG & set(toks[max(0, i - 5):i]):
                    out.append(cand)
                i += n - 1
                break
        i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = set(hpo_name_index())
    with open(a.out, "w") as fo:
        for f in sorted(glob.glob(f"{a.sources}/*.json")):
            rec = json.load(open(f))
            items = []
            for d in rec["docs"]:
                for s in SENT.split(d["text"]):
                    for m in matches(s, names):
                        items.append({"t": m, "k": "dict", "doc": d["doc_id"], "sent": s.strip()[:600]})
            fo.write(json.dumps({"id": rec["id"], "name": rec["name"], "items": items}) + "\n")
    print("done")


if __name__ == "__main__":
    main()
