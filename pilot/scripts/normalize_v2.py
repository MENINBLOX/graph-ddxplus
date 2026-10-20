"""Corpus-v2 normalization: link every distinct extracted phrase to a UMLS concept once.

IE v2 output repeats the same phrases across diseases and articles ("fever" appears tens of
thousands of times), so phrases are linked once each and the table is reused by graph assembly.
Linking is v152's link_top over the scispaCy UMLS candidate generator: character n-gram similarity
>= COV_THRESH (0.70), and a candidate below HIGH_CONF (0.88) must share a content word with the
concept name; grounded() then requires a concept name or alias inside the phrase covering at least
half of its content words. Incremental: phrases already in the table are skipped, so the step can
follow the pipelined IE.

Run in .venv-sci:
    .venv-sci/bin/python pilot/scripts/normalize_v2.py --ie OUT.jsonl --table phrase2cui.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re  # noqa: E402

import v152_normalize_pipeline as V  # noqa: E402
from v152_normalize_pipeline import build_linker, link_top, load_maps  # noqa: E402
from v154_prod_normalize import EVIDENCE_TUI, clean_name_for_link  # noqa: E402
from v152_normalize_pipeline import FINDING_TUI  # noqa: E402

VALID_TUI = set(FINDING_TUI) | set(EVIDENCE_TUI)
MAX_WORDS = 8


MIN_COVER = 0.5
# NegEx-style cues: an item the text states is absent must not become a positive edge.
NEG = re.compile(r"\b(no|not|without|absence of|absent|negative for|ruled out|rule out|denie[sd]|"
                 r"free of|lack of|unremarkable|normal)\b")
STOP = {"of", "the", "a", "an", "and", "or", "with", "in", "on", "to", "for", "at", "by", "as",
        "is", "was", "were", "be", "from", "that", "this", "these", "its", "their"}


def _lem(w: str) -> str:
    for suf, rep in (("ies", "y"), ("ses", "sis"), ("es", "e"), ("s", "")):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)] + rep
    return w


def _toks(s: str) -> list[str]:
    return [_lem(w) for w in re.findall(r"[a-z0-9]+", s.lower())]


def grounded(phrase: str, cui: str) -> bool:
    """Accept a link only if one of the concept's names/aliases occurs as a contiguous word sequence
    in the phrase AND covers at least half of the phrase's content words. The linker falls back to
    a phrase's last word(s), so without the coverage rule a generic one-word alias ("close",
    "history", "syndrome", "population") captures a whole clause."""
    ent = V._KB.cui_to_entity.get(cui) if V._KB is not None else None
    if ent is None:
        return False
    pt = _toks(phrase)
    content = [w for w in pt if w not in STOP]
    ps = " " + " ".join(pt) + " "
    for alias in [ent.canonical_name, *ent.aliases]:
        at = _toks(alias)
        if not at or len(" ".join(at)) <= 2:
            continue
        if " " + " ".join(at) + " " in ps:
            cover = len([w for w in at if w not in STOP]) / max(1, len(content))
            if cover >= MIN_COVER:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--checkpoint", type=int, default=20000)
    a = ap.parse_args()
    table = json.load(open(a.table)) if os.path.exists(a.table) else {}
    phrases = set()
    for line in (ln for path in a.ie.split(",") for ln in open(path)):
        try:
            r = json.loads(line)
        except Exception:
            continue
        phrases.update(i["t"].lower().strip() for i in r.get("items", []))
    # re-link phrases rejected only because their concept had no type in the old map
    for p in [p for p, v in table.items() if v.get("why") == "type" and not v.get("tui")]:
        del table[p]
    todo = sorted(p for p in phrases if p not in table)
    print(f"phrases {len(phrases)}, already linked {len(phrases) - len(todo)}, to link {len(todo)}", flush=True)
    # semantic types straight from MRSTY: the v152 STY map lacks some concepts (e.g. Pallor), and a
    # missing type looked exactly like an invalid one -- 206k mentions were rejected that way.
    sty_all: dict[str, set[str]] = {}
    with open("data/umls_extracted/MRSTY.RRF", encoding="utf-8") as fh:
        for line in fh:
            x = line.split("|")
            sty_all.setdefault(x[0], set()).add(x[1])
    nlp = build_linker()

    def save():
        tmp = a.table + ".tmp"
        json.dump(table, open(tmp, "w"))
        os.replace(tmp, a.table)

    for i, p in enumerate(todo, 1):
        if len(p.split()) > MAX_WORDS:
            table[p] = {"cui": None, "tui": "", "why": "too_long"}
            continue
        if NEG.search(p):
            table[p] = {"cui": None, "tui": "", "why": "negated"}
            continue
        cid, _sc = link_top(nlp, clean_name_for_link(p), valid_fn=lambda tui: tui in VALID_TUI)
        tuis = sty_all.get(cid, set()) if cid else set()
        tui = sorted(tuis & VALID_TUI)[0] if tuis & VALID_TUI else (sorted(tuis)[0] if tuis else "")
        if cid and not tuis & VALID_TUI:
            table[p] = {"cui": None, "tui": tui, "why": "type"}
        elif cid and not grounded(p, cid):
            table[p] = {"cui": None, "tui": tui, "why": "not_grounded"}
        else:
            table[p] = {"cui": cid, "tui": tui}
        if i % a.checkpoint == 0:
            save()
            print(f"  {i}/{len(todo)}", flush=True)
    save()
    n = sum(1 for p in phrases if table.get(p, {}).get("cui"))
    print(f"linked {n}/{len(phrases)} phrases ({100 * n / max(1, len(phrases)):.1f}%)")


if __name__ == "__main__":
    main()
