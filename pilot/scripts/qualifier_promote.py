"""Promote open-schema qualifiers into the fixed attribute slots they already correspond to.

The IE writes both: seven fixed slots and a free `qualifiers` list whose relation label the model
chooses itself. The open half was never scored -- it normalized to HPO at 20% and every HPO family
measured a delta of 0.000 -- so 127,621 KG edges carry qualifier text that the diagnosis ignores.

Measured, that text is not on the wrong findings: 47,623 qualifier cells sit on findings DDXPlus
patients actually report, eight times the opportunity of all seven fixed slots combined (~5,900).
The labels the model chose map onto our channels almost directly -- `quality` is character,
`intensity` is severity, `time course` is timing -- so the signal needs promotion, not another
extraction pass.

Only EMPTY slots are filled, so a value pass 1 or pass 2 extracted always wins; the qualifier list
is left in place for provenance and RAG. Re-running this is minutes, against 20+ hours for re-IE.

Usage: python qualifier_promote.py --preds IN.json --out OUT.json
"""
from __future__ import annotations
import argparse, json, re, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attr_score as A

# Relation label (as the model wrote it) -> the slot it belongs in. Substring match, lowercased.
_REL_SLOT = (
    ("quality", "character"), ("character", "character"), ("nature", "character"),
    ("type", "character"), ("description", "character"), ("consistency", "character"),
    ("intensity", "severity"), ("severity", "severity"), ("level", "severity"),
    ("degree", "severity"), ("grade", "severity"),
    ("time course", "timing"), ("course", "timing"), ("frequency", "timing"),
    ("duration", "timing"), ("timing", "timing"), ("chronicity", "timing"),
    ("pattern", "timing"), ("progression", "timing"),
    ("onset", "onset"), ("speed", "onset"),
    ("location", "location"), ("site", "location"), ("distribution", "location"),
    ("region", "location"), ("radiation", "location"),
    ("trigger", "aggravating"), ("aggravat", "aggravating"), ("exacerbat", "aggravating"),
    ("worsen", "aggravating"), ("precipitat", "aggravating"),
    ("relie", "relieving"), ("ameliorat", "relieving"), ("improv", "relieving"),
    ("alleviat", "relieving"),
)
# A slot only accepts a quote that its own controlled vocabulary recognizes, so promotion cannot
# inject free text the scorer would never match.
_VALIDATE = {"character": A.char_classes, "timing": A.timing_classes,
             "aggravating": A.trigger_classes, "relieving": A.trigger_classes}
_SEV = re.compile(r"(?i)\b(mild|moderate|severe|intense|slight|marked|profound|extreme)\b")
_ONSET = re.compile(r"(?i)\b(sudden|abrupt|acute|gradual|insidious|progressive|rapid)\b")


def slot_of(relation: str) -> str | None:
    r = (relation or "").lower()
    for key, slot in _REL_SLOT:
        if key in r:
            return slot
    return None


def value_for(slot: str, quote: str) -> str | None:
    """The quote, if this slot's controlled vocabulary recognizes it."""
    q = (quote or "").strip()
    if not q or len(q) > 60:
        return None
    if slot in _VALIDATE:
        return q if _VALIDATE[slot](q) else None
    if slot == "severity":
        m = _SEV.search(q)
        return m.group(1).lower() if m else None
    if slot == "onset":
        m = _ONSET.search(q)
        if not m:
            return None
        w = m.group(1).lower()
        return "gradual" if w in ("gradual", "insidious", "progressive") else "sudden"
    if slot == "location":
        return q if len(q.split()) <= 4 else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    preds = json.load(open(a.preds, encoding="utf-8"))

    seen = filled = 0
    per = {}
    for v in preds.values():
        for f in v.get("findings", []):
            for q in f.get("qualifiers", []) or []:
                seen += 1
                slot = slot_of(q.get("relation", ""))
                if not slot or str(f.get(slot, "") or "").strip():
                    continue
                val = value_for(slot, q.get("quote", ""))
                if not val:
                    continue
                f[slot] = val
                filled += 1
                per[slot] = per.get(slot, 0) + 1
    json.dump(preds, open(a.out, "w"))
    tot = sum(len(v.get("findings", [])) for v in preds.values())
    print(f"qualifier cells seen {seen}; promoted into empty slots {filled} "
          f"({100*filled/max(1,seen):.1f}%)")
    print(f"  by slot: {dict(sorted(per.items(), key=lambda kv: -kv[1]))}")
    fill = {s: sum(1 for v in preds.values() for f in v.get("findings", [])
                   if str(f.get(s, "") or "").strip())
            for s in ("location", "character", "severity", "onset", "timing",
                      "aggravating", "relieving")}
    print(f"  corpus fill now: {{{', '.join(f'{k}: {100*c/max(1,tot):.1f}%' for k, c in fill.items())}}}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
