"""Emit the generic form of a finding whose name is a qualified phrase.

A benchmark asks its questions in coarse vocabulary -- DDXPlus records plain `Pain` (C0030193) --
while the literature writes the qualified form, so the KG holds `chest pain`, `epigastric pain`,
`chest wall or substernal pain` and never the bare concept. Measured on DDXPlus, 31% of all missed
evidence is this: the profile already contains a concept sharing the patient's words, just not the
concept the patient's vocabulary uses.

`site_decompose` handled this for a fixed list of anatomical sites. That list is the limitation:
"chest wall or substernal pain" and "increased sweating" have no site token to strip. Here the rule
is positional instead -- an English symptom phrase puts the concept last, so the trailing word IS
the generic concept -- which covers the qualified forms a fixed vocabulary misses.

Additive, like every other decomposition step: the qualified finding keeps its edges and its
discriminative weight; the generic node is what lets coarse patient vocabulary reach the disease.

Usage: python generalize_findings.py --preds IN.json --out OUT.json
"""
from __future__ import annotations
import argparse, json, re

# concepts a benchmark is likely to ask about in bare form. Restricted on purpose: generalizing an
# arbitrary head noun would collapse specific findings into uninformative hubs.
_GENERIC_HEAD = {
    "pain", "ache", "discomfort", "tenderness", "swelling", "edema", "rash", "lesion", "lesions",
    "numbness", "weakness", "cramp", "cramps", "stiffness", "itching", "pruritus", "bleeding",
    "hemorrhage", "paralysis", "spasm", "tingling", "cough", "sweating", "diaphoresis", "fever",
    "nausea", "vomiting", "diarrhea", "diarrhoea", "constipation", "dyspnea", "dyspnoea",
    "wheezing", "stridor", "palpitations", "dizziness", "vertigo", "syncope", "fatigue",
    "malaise", "anxiety", "confusion", "seizure", "seizures", "tremor", "jaundice", "pallor",
    "cyanosis", "erythema", "ulcer", "ulcers", "nodule", "nodules", "mass", "discharge",
    "congestion", "obstruction", "incontinence", "retention", "anemia", "anaemia",
}
_WORD = re.compile(r"[a-z]+")


def generic_of(name: str) -> str | None:
    """"chest wall or substernal pain" -> "pain"; None when the name is already generic."""
    ws = _WORD.findall(name.lower())
    if len(ws) < 2:
        return None
    head = ws[-1]
    if head not in _GENERIC_HEAD:
        return None
    return head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    preds = json.load(open(a.preds, encoding="utf-8"))
    n_f = n_hit = n_new = 0
    for v in preds.values():
        fs = v.get("findings", [])
        have = {f["name"].strip().lower() for f in fs}
        add = []
        for f in fs:
            n_f += 1
            g = generic_of(f["name"])
            if not g:
                continue
            n_hit += 1
            if g in have:
                continue
            rec = {k: val for k, val in f.items() if k not in ("name", "qualifiers")}
            rec["name"] = g
            add.append(rec)
            have.add(g)
            n_new += 1
        fs.extend(add)
    json.dump(preds, open(a.out, "w"))
    total = sum(len(v.get("findings", [])) for v in preds.values())
    print(f"findings {n_f}; qualified phrases with a generic head {n_hit} "
          f"({100*n_hit/max(1,n_f):.2f}%)")
    print(f"  generic findings created: {n_new}")
    print(f"  findings after generalization: {total}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
