"""Recover attributes the IE left inside the finding NAME instead of in an attribute field.

The IE is told to strip the anatomical SITE out of a finding name into `location`, and it does
(location is the best-filled attribute at 33%). No such rule exists for the temporal, intensity or
quality modifiers, so they stay lexicalized in the name: measured over 478,603 production findings,
12,275 names contain a temporal word ("progressive", "recurrent", "chronic", "acute", "nocturnal")
while their `timing` field is empty -- more than the number of findings that HAVE a timing value.

This step copies such a modifier into the empty attribute field. It deliberately does NOT rewrite
the name: the name determines which UMLS concept the finding links to, and rewriting it would
silently change the graph's vocabulary. So the operation is purely additive and cannot regress
anything that already works.

Only modifiers drawn from the controlled vocabularies the scorer actually matches on are
recovered (attr_score's timing/character classes, the severity ordinal, the onset binary); a word
that no scoring channel can consume is left alone rather than stored as decoration.

Usage: python attr_lexicalized.py --preds IN.json --out OUT.json
"""
from __future__ import annotations
import argparse, json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attr_score as A

# The surface word is kept verbatim (not the class label): downstream matching re-derives the class
# from the text on both sides, and the raw word stays auditable against the source.
_TIMING = re.compile(r"\b(nocturnal|postprandial|episodic|intermittent|paroxysmal|periodic|exertional|"
                     r"morning|recurrent|recurring|constant|persistent|chronic|transient|progressive|"
                     r"relapsing|cyclic|cyclical|daily|nightly|seasonal)\b", re.I)
_CHARACTER = re.compile(r"\b(sharp|dull|burning|crushing|throbbing|colicky|cramping|stabbing|"
                        r"squeezing|tearing|aching|shooting|gnawing|pulsating|lancinating)\b", re.I)
_SEVERITY = {"mild": "mild", "slight": "mild", "minimal": "mild",
             "moderate": "moderate",
             "severe": "severe", "profound": "severe", "marked": "severe", "massive": "severe"}
_ONSET = {"sudden": "sudden", "abrupt": "sudden", "gradual": "gradual", "insidious": "gradual"}


def recover(finding: dict) -> dict:
    """Fill empty attribute fields from modifiers present in the finding's own name."""
    name = str(finding.get("name", "") or "")
    if not name:
        return {}
    low = name.lower()
    got = {}

    def empty(k):
        return not str(finding.get(k, "") or "").strip()

    if empty("timing"):
        m = _TIMING.search(low)
        # only credit it if the scorer can actually turn it into a class
        if m and A.timing_classes(m.group(1)):
            got["timing"] = m.group(1)
    if empty("character"):
        m = _CHARACTER.search(low)
        if m and A.char_classes(m.group(1)):
            got["character"] = m.group(1)
    if empty("severity"):
        for w, v in _SEVERITY.items():
            if re.search(rf"\b{w}\b", low):
                got["severity"] = v
                break
    if empty("onset"):
        for w, v in _ONSET.items():
            if re.search(rf"\b{w}\b", low):
                got["onset"] = v
                break
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    preds = json.load(open(a.preds, encoding="utf-8"))
    n = 0
    added = {k: 0 for k in ("timing", "character", "severity", "onset")}
    for v in preds.values():
        for f in v.get("findings", []):
            n += 1
            for k, val in recover(f).items():
                f[k] = val
                added[k] += 1
    json.dump(preds, open(a.out, "w"))
    have = {k: sum(1 for v in preds.values() for f in v["findings"]
                   if str(f.get(k, "") or "").strip())
            for k in ("timing", "character", "severity", "onset")}
    print(f"findings {n} (names unchanged)")
    print(f"{'attr':12}{'recovered':>11}{'now filled':>13}")
    for k in ("timing", "character", "severity", "onset"):
        print(f"  {k:10}{added[k]:11}{have[k]:9} ({100*have[k]/max(1,n):.2f}%)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
