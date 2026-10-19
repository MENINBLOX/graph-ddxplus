"""LLM-free extraction baselines: dictionary matches in each sentence of a disease's articles.

--vocab hpo: HPO labels and exact synonyms; build_graph_v2.py --hpo-names links them.
--vocab umls: every English UMLS string of a concept with a finding semantic type (QuickUMLS-style);
  the item carries its CUI. A string naming several such concepts takes the one it is the preferred
  name of; when there is none the item carries no CUI and the builder links it like any phrase
  (HPO name, phrase table).
Longest match first; a match preceded by a negation cue within the five tokens before it is dropped.
Output is the ie_v2.py JSONL format, so comparing graphs built from the same articles isolates what
the LLM extraction adds over dictionary matching.

    python dict_ie.py --sources DIR --out OUT.jsonl.gz [--vocab hpo|umls] [--workers 8] [--resume]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_graph_v2 import _norm, hpo_name_index  # noqa: E402
from jsonl_io import complete_lines, open_text, repair  # noqa: E402
from v152_normalize_pipeline import FINDING_TUI  # noqa: E402

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n+")
TOK = re.compile(r"[a-z0-9]+")
ID = re.compile(r'\{"id": "([^"]+)"')
NEG = {"no", "not", "without", "absent", "absence", "negative", "denied", "denies", "normal", "lack"}
MAX_N = 8


def matches(sent: str, names) -> list[str]:
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


def umls_index() -> dict[str, tuple[str, str] | None]:
    """Normalized English string -> (CUI, finding TUI) over UMLS concepts with a finding semantic type."""
    tui: dict[str, str] = {}
    for ln in open("data/umls_extracted/MRSTY.RRF"):
        c, t = ln.split("|", 2)[:2]
        if t in FINDING_TUI:
            tui.setdefault(c, t)
    cuis, pref = {}, {}
    for ln in open("data/umls_extracted/MRCONSO.RRF", encoding="utf-8"):
        f = ln.split("|", 15)
        if f[1] != "ENG" or f[0] not in tui:
            continue
        k = _norm(f[14]).strip()
        if len(k) < 4 or k.count(" ") >= MAX_N:
            continue
        cuis.setdefault(k, set()).add(f[0])
        if f[2] == "P" and f[4] == "PF":
            pref.setdefault(k, set()).add(f[0])
    out: dict[str, tuple[str, str] | None] = {}
    for k, cs in cuis.items():
        c = next(iter(cs)) if len(cs) == 1 else next(iter(pref[k])) if len(pref.get(k, ())) == 1 else None
        out[k] = (c, tui[c]) if c else None
    return out


NAMES: dict[str, tuple[str, str] | None] = {}


def extract(path: str) -> str:
    rec = json.load(open(path))
    items = []
    for d in rec["docs"]:
        for s in SENT.split(d["text"]):
            for m in matches(s, NAMES):
                it = {"t": m, "k": "dict", "doc": d["doc_id"], "sent": s.strip()[:600]}
                if NAMES[m]:
                    it["cui"], it["tui"] = NAMES[m]
                items.append(it)
    return json.dumps({"id": rec["id"], "name": rec["name"], "items": items}) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="append, skipping diseases already in --out")
    ap.add_argument("--vocab", choices=["hpo", "umls"], default="hpo")
    a = ap.parse_args()
    NAMES.update(umls_index() if a.vocab == "umls" else dict.fromkeys(hpo_name_index()))
    done = set()
    if a.resume and os.path.exists(a.out):
        repair(a.out)
        done = {m.group(1).replace(":", "_") for ln in complete_lines(a.out) if (m := ID.match(ln))}
    files = [f for f in sorted(glob.glob(f"{a.sources}/*.json")) if os.path.basename(f)[:-5] not in done]
    with open_text(a.out, "a" if a.resume else "w") as fo, Pool(a.workers) as pool:
        fo.writelines(pool.imap(extract, files, chunksize=8))
    print(f"done {len(files)}")


if __name__ == "__main__":
    main()
