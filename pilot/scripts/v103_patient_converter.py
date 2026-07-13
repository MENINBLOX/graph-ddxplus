#!/usr/bin/env python3
"""v103 patient evidence → CUI + attribute JSON converter.

DDXPlus token을 attribute-rich evidence로 변환:
- Binary token (douleurxx YES) → {cui: C0030193, attrs: {}}
- Multi-choice (douleurxx_carac_@_cramp) → {cui: C0030193, attrs: {character: [cramp]}}
- Location (douleurxx_endroitducorps_@_abdomen) → {cui: C0030193, attrs: {location: [abdomen]}}
- Numeric (douleurxx_intens_@_4) → {cui: C0030193, attrs: {severity: moderate}}  (VAS mapping)

학술적: DDXPlus value를 controlled attribute로 정규화. Numeric→VAS severity는
clinical convention (0=none, 1-3 mild, 4-6 moderate, 7-10 severe).
"""
from __future__ import annotations
import sys, json, csv, ast
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
EV_META = "data/ddxplus/release_evidences.json"

# DDXPlus French body-location value → controlled location enum
FR_LOC_MAP = {
    "tete": "head", "front": "head", "joue": "face", "joue_D_": "face", "joue_G_": "face",
    "oeil": "eye", "nez": "nose", "bouche": "mouth", "levre": "lip", "langue": "tongue",
    "gorge": "throat", "cou": "neck", "arriere_du_cou": "neck", "thorax": "chest",
    "haut_du_thorax": "chest", "bas_du_thorax": "chest", "poitrine": "chest",
    "ventre": "abdomen", "epigastre": "epigastric", "abdomen": "abdomen",
    "flanc": "abdomen", "flanc_D_": "abdomen", "flanc_G_": "abdomen",
    "fosse_iliaque": "abdomen", "fosse_iliaque_D_": "abdomen", "fosse_iliaque_G_": "abdomen",
    "hypochondre_D_": "abdomen", "hypochondre_G_": "abdomen", "dos": "back",
    "bassin": "pelvis", "aine": "groin", "bras": "arm", "bras_D_": "arm", "bras_G_": "arm",
    "epaule": "shoulder", "coude": "elbow", "poignet": "wrist", "main": "hand",
    "doigt": "finger", "jambe": "leg", "cuisse": "thigh", "genou": "knee",
    "cheville": "ankle", "pied": "foot", "biceps_D_": "arm", "biceps_G_": "arm",
    "cartilage_thyroidien": "throat",
}

# Pain character (douleurxx_carac) FR → controlled
FR_CHAR_MAP = {
    "une_crampe": "cramping", "vive": "sharp", "lancinante_/_choc_électrique": "stabbing",
    "sensible": "tender", "un_tiraillement": "tugging", "une_brûlure_ou_chaleur": "burning",
    "un_serrement": "pressing", "pénible": "dull", "pulsatile": "throbbing",
    "une_pulsation": "throbbing",
}


def numeric_to_severity(val):
    """VAS clinical convention."""
    try:
        v = int(val)
        if v == 0: return None
        if v <= 3: return "mild"
        if v <= 6: return "moderate"
        return "severe"
    except: return None


def convert_patient(evs, value_cuis, ev_meta, numeric_evs):
    """DDXPlus EVIDENCES list → list of {cui, attrs}."""
    # Group by base evidence
    by_base = defaultdict(lambda: {"cuis": set(), "location": set(),
                                    "character": set(), "severity": None})
    for ev in evs:
        if "_@_" in ev:
            base, val = ev.split("_@_", 1)
        else:
            base, val = ev, None
        m = value_cuis.get(base, {})
        # Base CUIs
        for c in m.get("_question", []):
            by_base[base]["cuis"].add(c)
        if val is not None:
            for c in m.get(val, []):
                by_base[base]["cuis"].add(c)
            # Attribute extraction
            if "endroitducorps" in base or "irrad" in base:
                loc = FR_LOC_MAP.get(val)
                if loc: by_base[base]["location"].add(loc)
            elif "carac" in base:
                ch = FR_CHAR_MAP.get(val)
                if ch: by_base[base]["character"].add(ch)
            elif base in numeric_evs:
                sev = numeric_to_severity(val)
                if sev: by_base[base]["severity"] = sev

    # Emit per-CUI evidence (merge attrs across tokens of same base)
    cui_evidence = {}
    for base, data in by_base.items():
        attrs = {}
        if data["location"]: attrs["location"] = sorted(data["location"])
        if data["character"]: attrs["character"] = sorted(data["character"])
        if data["severity"]: attrs["severity"] = data["severity"]
        for cui in data["cuis"]:
            if cui not in cui_evidence:
                cui_evidence[cui] = {"cui": cui, "attributes": dict(attrs)}
            else:
                # Merge: union location/character, keep severity
                ex = cui_evidence[cui]["attributes"]
                for k in ("location","character"):
                    if k in attrs:
                        ex[k] = sorted(set(ex.get(k,[])) | set(attrs[k]))
                if "severity" in attrs and "severity" not in ex:
                    ex["severity"] = attrs["severity"]
    return list(cui_evidence.values())


def detect_numeric(ev_meta):
    numeric = set()
    for ev_id, m in ev_meta.items():
        pv = m.get("possible-values", [])
        try:
            nums = sorted(int(v) for v in pv)
            if len(nums) >= 3 and nums[0] == 0: numeric.add(ev_id)
        except: pass
    return numeric


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="pilot/data/cache/v103_patients.jsonl")
    args = ap.parse_args()

    value_cuis = json.load(open(VALUE_CUIS))
    ev_meta = json.load(open(EV_META))
    numeric_evs = detect_numeric(ev_meta)
    icd = json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    cond = json.load(open("data/ddxplus/release_conditions_en.json"))
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}

    n = 0
    with open(args.out, "w") as fout:
        with open("data/ddxplus/release_test_patients.csv") as f:
            for row in csv.DictReader(f):
                if n >= args.n: break
                true_cui = fr2cui.get(row["PATHOLOGY"])
                if not true_cui: continue
                evs = ast.literal_eval(row["EVIDENCES"])
                evidence = convert_patient(evs, value_cuis, ev_meta, numeric_evs)
                fout.write(json.dumps({"true_cui": true_cui, "evidence": evidence}) + "\n")
                n += 1
    print(f"Converted {n} patients → {args.out}", flush=True)
    # Sample
    import subprocess
    print("\nSample patient 0:")
    line = open(args.out).readline()
    p = json.loads(line)
    print(f"  true_cui: {p['true_cui']}")
    for e in p["evidence"][:8]:
        print(f"  {e['cui']}: {e['attributes']}")


if __name__ == "__main__":
    main()
