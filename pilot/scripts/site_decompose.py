"""Split an anatomical site out of a finding NAME so the generic symptom node exists.

The IE is instructed to strip the site into `location` ("substernal chest pain" -> name "pain",
location "substernal chest"), and it does so inconsistently: 9,660 findings still carry the site in
the name. The cost is measured and large. `Pain` (C0030193) is the single most-missed piece of
DDXPlus evidence -- 750 patient instances whose gold disease has no bare pain node -- and those
diseases are not missing pain at all:

    Pulmonary embolism   has "pleuritic chest pain", "chest pain"   but not "pain"
    GERD                 has "epigastric pain", "chest pain"        but not "pain"
    Myocarditis          has "radiating chest pain", "joint pain"   but not "pain"

The hierarchy cannot rescue it either: the phenotype parent map does not lift chest pain, abdominal
pain, headache or myalgia to Pain (it does lift arthralgia, and it wrongly makes chest pain a parent
of back pain). So a patient reporting generic pain never meets a disease documented with a specific
pain.

This emits the generic form ALONGSIDE the specific one, with the site moved into `location`. Purely
additive: the specific node keeps its edges and its own discriminative value, and the generic node
gives the coarse patient vocabulary something to match.

Usage: python site_decompose.py --preds IN.json --out OUT.json
"""
from __future__ import annotations
import argparse, json, re

_CORE = ("pain", "ache", "discomfort", "tenderness", "swelling", "edema", "rash", "numbness",
         "weakness", "cramp", "cramps", "stiffness", "itching", "burning", "bleeding",
         "paralysis", "spasm", "tingling", "pressure", "heaviness")
_SITE = ("chest", "abdomen", "abdominal", "epigastric", "substernal", "retrosternal", "precordial",
         "head", "neck", "back", "lumbar", "flank", "pelvic", "joint", "joints", "muscle",
         "muscles", "limb", "limbs", "leg", "legs", "arm", "arms", "shoulder", "shoulders", "knee",
         "hip", "throat", "ear", "eye", "eyes", "facial", "face", "tooth", "dental", "jaw", "calf",
         "thigh", "foot", "hand", "hands", "wrist", "ankle", "scrotal", "testicular", "breast",
         "rectal", "anal", "pleuritic", "maxillary", "frontal", "temporal", "occipital",
         "periumbilical", "hypochondrial", "iliac", "umbilical", "substernal", "abdomen")
_CORE_RE = "|".join(map(re.escape, sorted(_CORE, key=len, reverse=True)))
_SITE_RE = "|".join(map(re.escape, sorted(_SITE, key=len, reverse=True)))
# "<site> <core>"  or  "<core> in/of the <site>"
_A = re.compile(rf"\b(?P<site>{_SITE_RE})\w*\b[\s-]+(?P<core>{_CORE_RE})\b", re.I)
_B = re.compile(rf"\b(?P<core>{_CORE_RE})\b\s+(?:in|of|at)\s+(?:the\s+)?(?P<site>{_SITE_RE})\w*\b",
                re.I)


def split_site(name: str) -> tuple[str, str] | None:
    """"epigastric pain" -> ("pain", "epigastric"); None when there is no site to move."""
    for pat in (_A, _B):
        m = pat.search(name.lower())
        if m:
            core, site = m.group("core"), m.group("site")
            if core and site and core != site:
                return core, site
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    preds = json.load(open(a.preds, encoding="utf-8"))

    n_f = n_split = n_new = n_enrich = 0
    for v in preds.values():
        fs = v.get("findings", [])
        have = {f["name"].strip().lower() for f in fs}
        add = []
        for f in fs:
            n_f += 1
            sp = split_site(f["name"])
            if not sp:
                continue
            core, site = sp
            n_split += 1
            if core in have:
                # the generic node already exists; make sure it records this site
                tgt = next((x for x in fs if x["name"].strip().lower() == core), None) \
                    or next((x for x in add if x["name"] == core), None)
                if tgt is not None and not str(tgt.get("location", "") or "").strip():
                    tgt["location"] = site
                    n_enrich += 1
                continue
            g = {k: val for k, val in f.items() if k not in ("name", "qualifiers")}
            g["name"] = core
            g["location"] = f.get("location") or site
            if f.get("qualifiers"):
                g["qualifiers"] = list(f["qualifiers"])
            add.append(g)
            have.add(core)
            n_new += 1
        fs.extend(add)
    json.dump(preds, open(a.out, "w"))
    total = sum(len(v.get("findings", [])) for v in preds.values())
    print(f"findings scanned {n_f}; site baked into the name {n_split} "
          f"({100*n_split/max(1,n_f):.2f}%)")
    print(f"  generic findings created : {n_new}")
    print(f"  existing generics given a site: {n_enrich}")
    print(f"  findings after split     : {total}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
