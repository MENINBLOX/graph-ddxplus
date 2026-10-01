"""Split a qualifier out of the finding NAME so the finding can join the patient's vocabulary.

The IE is told to strip the anatomical SITE out of a finding name, and it does. Nothing tells it to
strip the quality, intensity or course, so those stay in the name and the finding becomes a nearly
unique string. Measured on 478,603 production findings, 7.1% carry a qualifier word in the name,
and for the ones where both forms exist in the KG the difference is decisive:

    document frequency   median   p90
      name as extracted      13   102      "severe pain"   -> df 0
      qualifier stripped    204  4903      "pain"          -> df 4903

A df-0 node is unreachable: no patient ever writes "severe pain", they write "pain" plus a severity.
This is the mechanism behind the 26x gap between the findings patients report (median df 2,249) and
the findings our IE qualified (median df 87), and therefore behind attribute families that hold
tens of thousands of KG values yet overlap the patient side zero times.

The decomposition is ADDITIVE: the original finding is kept and a second one is emitted with the
core name and the qualifier moved into the attribute where it belongs. Nothing that already worked
can regress, and the hub node gains the qualifier it needs to be discriminative.

Usage: python attr_decompose.py --preds IN.json --out OUT.json
"""
from __future__ import annotations
import argparse, json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# qualifier word -> the attribute it belongs in. Only words that a downstream vocabulary can
# actually consume are split out; anything else stays in the name where it is at least readable.
_QUAL = {
    "character": ("vague", "colicky", "diffuse", "dull", "sharp", "burning", "crushing",
                  "throbbing", "cramping", "stabbing", "squeezing", "aching", "shooting",
                  "tearing", "painful", "pruritic", "watery", "bloody", "purulent", "productive"),
    "severity": ("severe", "mild", "moderate", "profound", "marked", "slight", "minimal",
                 "massive", "extensive", "low-grade", "high-grade"),
    "timing": ("acute", "chronic", "recurrent", "persistent", "progressive", "intermittent",
               "episodic", "paroxysmal", "transient", "nocturnal", "postprandial", "exertional",
               "periodic", "relapsing", "fluctuating", "daily"),
    "onset": ("sudden", "abrupt", "gradual", "insidious", "congenital", "neonatal"),
}
# a bare qualifier is not a finding; and stripping must leave a real symptom behind
_MIN_CORE_CHARS = 3
_PATTERN = {a: re.compile(r"\b(" + "|".join(map(re.escape, ws)) + r")\b", re.I)
            for a, ws in _QUAL.items()}


def decompose(name: str) -> tuple[str, dict] | None:
    """"severe colicky pain" -> ("pain", {"severity": "severe", "character": "colicky"})."""
    core = name.strip().lower()
    found: dict[str, str] = {}
    for attr, pat in _PATTERN.items():
        m = pat.search(core)
        if m:
            found[attr] = m.group(1).lower()
            core = pat.sub(" ", core)
    core = " ".join(core.split())
    if not found or len(core) < _MIN_CORE_CHARS:
        return None
    if not re.search(r"[a-z]", core):
        return None
    return core, found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    preds = json.load(open(a.preds, encoding="utf-8"))

    n_f = n_split = n_new = 0
    for v in preds.values():
        fs = v.get("findings", [])
        have = {f["name"].strip().lower() for f in fs}
        add = []
        for f in fs:
            n_f += 1
            d = decompose(f["name"])
            if not d:
                continue
            core, quals = d
            n_split += 1
            if core in have:
                # the hub finding already exists -- either from the IE or created earlier in this
                # same pass -- so give it the qualifier it was missing rather than duplicating it
                tgt = next((x for x in fs if x["name"].strip().lower() == core),
                           None) or next((x for x in add if x["name"] == core), None)
                if tgt is None:
                    continue
                for k, val in quals.items():
                    if not str(tgt.get(k, "") or "").strip():
                        tgt[k] = val
                continue
            g = {k: v2 for k, v2 in f.items() if k not in ("name", "qualifiers")}
            g["name"] = core
            for k, val in quals.items():
                g[k] = val
            if f.get("qualifiers"):
                g["qualifiers"] = list(f["qualifiers"])
            add.append(g)
            have.add(core)
            n_new += 1
        fs.extend(add)
    json.dump(preds, open(a.out, "w"))
    total = sum(len(v.get("findings", [])) for v in preds.values())
    print(f"findings scanned {n_f}; names carrying a qualifier {n_split} "
          f"({100*n_split/max(1,n_f):.1f}%)")
    print(f"  hub findings newly created: {n_new}   (the rest already existed and were enriched)")
    print(f"  findings after decomposition: {total}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
