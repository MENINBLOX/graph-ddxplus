"""LLM spans, dictionary concepts: re-link the LLM items the phrase table leaves without a concept.

The phrase table rejects most LLM phrases as not grounded ("right ptosis", "serum creatine kinase was elevated": the
linked concept's names cover under half of the phrase). Here such an item is replaced by the UMLS finding strings
dict_ie.py --vocab umls finds inside the phrase itself (longest match, negation cue dropped), so the LLM still decides
which text is a finding of the disease and the dictionary only names it. Items the table links stay as they are.
--keep-original also keeps the unlinked item itself, so what the builder links in it (coordinated parts, HPO names)
is not lost. Output keeps the ie_v2.py format; a re-linked item also carries the LLM phrase it was found in as
"src_t", which build_graph_v2.py stores on the Mention as its span.

    python llm_dict_relink.py --ie IN.jsonl --table phrase2cui_v3.json --out OUT.jsonl.gz [--keep-original] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dict_ie  # noqa: E402
from jsonl_io import open_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-original", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="diseases to process (0 = all), for a size estimate")
    a = ap.parse_args()
    table = json.load(open(a.table))
    names = dict_ie.umls_index()
    stats: Counter = Counter()
    with open_text(a.out, "w") as fo, open_text(a.ie) as fi:
        for i, ln in enumerate(fi):
            if a.limit and i >= a.limit:
                break
            r = json.loads(ln)
            items = []
            for it in r.get("items", []):
                v = table.get(it["t"].lower().strip())
                if v is None or v.get("cui") or v.get("why") in ("negated", "too_long"):
                    items.append(it)
                    stats["kept" if v and v.get("cui") else "kept_unlinked"] += 1
                    continue
                found = dict_ie.matches(it["t"], names)
                stats["relinked" if found else "no_match"] += 1
                if a.keep_original:
                    items.append(it)
                for m in found:
                    new = {**it, "t": m, "k": "llm_dict", "src_t": it["t"]}
                    if names[m]:
                        new["cui"], new["tui"] = names[m]
                    items.append(new)
                    stats["new_items"] += 1
            fo.write(json.dumps({**r, "items": items}) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
