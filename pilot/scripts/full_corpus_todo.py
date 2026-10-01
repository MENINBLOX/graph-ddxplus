"""List the corpus diseases whose literature has not yet been collected at depth.

The final KG adds a full-corpus pass on top of the development KG, whose benchmark gold diseases
were already collected at depth. This writes the remaining diseases as {cui: name} -- the input to
`deep_fetch.py` in `reproduce.sh sources`. Only disease names leave this script (§C3).

    python full_corpus_todo.py --out pilot/cache/prod/full_todo.json
"""
from __future__ import annotations

import argparse
import json
import os

P = "pilot/cache/prod/"
DEEP_DIRS = ("sources_ddx_deep2", "sources_nlice_deep", "sources_bench_deep", "sources_thin_deep")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=P + "ie_pred_merged27.json", help="development-KG predictions")
    ap.add_argument("--out", default=P + "full_todo.json")
    a = ap.parse_args()

    preds = json.load(open(a.preds, encoding="utf-8"))
    done: set[str] = set()
    for d in DEEP_DIRS:
        p = P + d
        if os.path.isdir(p):
            done |= {f[:-5] for f in os.listdir(p) if f.endswith(".json") and "manifest" not in f}
    todo = {cui: v["disease"] for cui, v in sorted(preds.items())
            if cui not in done and v.get("disease")}
    print(f"corpus {len(preds)}   deep-treated {len(done & set(preds))}   remaining {len(todo)}")
    json.dump(todo, open(a.out, "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
