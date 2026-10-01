"""Attribute-aware DDXPlus loader — keeps Bates clinical attributes instead of collapsing.

`onlykg_eval_unified.load_ddxplus` throws every attribute-valued evidence away, mapping
`douleurxx_carac_@_...`, `douleurxx_endroitducorps_@_...`, `douleurxx_intens_@_6` all to the
single base CUI "Pain". This module keeps the attributes DDXPlus encodes per finding:

    DDXPlus sub-question suffix -> Bates attribute
      carac            -> character   (M, multi-categorical; only on `douleurxx`)
      endroitducorps   -> location    (M; on douleurxx / lesions_peau / oedeme)
      irrad            -> location    (M; radiation absorbed into location, CLAUDE.md)
      intens           -> severity    (C, NRS 0-10; on douleurxx / lesions_peau)
      soudain          -> onset       (C, 0-10, higher = more sudden; on douleurxx)
      precis           -> (location precision; meta, ignored)

Values are decoded through `release_evidences.json` `value_meaning` (keyed by the exact
patient value code -> {fr,en}), then:
  * location: english value -> UMLS CUI via the project linker (LOC_TUI), tokens kept too;
  * character: english tokens kept (no controlled vocabulary — CLAUDE.md);
  * severity (intens NRS): 0-3 mild, 4-6 moderate, 7-10 severe;
  * onset (soudain):       0-3 gradual, 7-10 sudden, 4-6 uncertain -> None (tertile split,
                           mirrors the NRS bins; avoids forcing the ambiguous middle).

`load(n=3000)` -> `(cand, patients)` where cand = {en_key: (name, cui)} (same as the unified
loader) and each patient = `(target_keys, findings, "DDXPlus")`, findings a list of:
    {"cui": str, "location": set[str], "location_txt": set[str],
     "character": set[str], "severity": str|None, "onset": str|None}
"""
from __future__ import annotations

import ast
import csv
import json
import re
import sys
from collections import defaultdict
from typing import Optional

sys.path.insert(0, "pilot/scripts")
import os
from umls_link import link as umls_link  # noqa: E402

# compose located pain into specific-symptom concepts (join to semiology-enriched KG edges)
COMPOSE = os.environ.get("DDX_COMPOSE_PAIN", "1") != "0"

# UMLS semantic types accepted as anatomical location (T017 anatomy .. T031 body substance)
LOC_TUI = {"T023", "T029", "T030", "T024", "T025", "T018", "T031", "T022", "T017"}

# DDXPlus sub-question suffix -> Bates attribute channel.
ATTR_SUFFIX = {
    "carac": "character",
    "endroitducorps": "location",
    "irrad": "location",
    "intens": "severity",
    "soudain": "onset",
}
_NON_LOCATION_VALUES = {"nowhere", "nulle_part", "na", ""}
_LOC_STOP = {"the", "of", "a", "an", "side", "aspect", "region", "area", "part", "face"}

_DATA = "data/ddxplus"

# DDXPlus represents "chest pain" compositionally as pain(C0030193) + location(chest); the
# semiology KG stores it as the specific concept chest-pain(C0008031). To let a patient's
# located pain JOIN the semiology-enriched disease edges, a pain finding whose location matches
# a body region is DUPLICATED into the corresponding specific-symptom concept (same attributes).
# Region cue tokens -> specific pain concept CUI.
_PAIN_CUIS = {"C0030193", "C0234238", "C0234254"}   # pain / pain family base concepts
_REGION_SYMPTOM = [
    ({"chest", "thorax", "thoracic", "sternum", "sternal", "precordial", "substernal",
      "retrosternal"}, "C0008031"),                                   # chest pain
    ({"abdomen", "abdominal", "epigastric", "epigastrium", "hypochondrium", "hypochondriac",
      "flank", "iliac", "umbilical", "periumbilical", "pelvic", "belly"}, "C0000737"),  # abdominal pain
    ({"back", "lumbar", "loin"}, "C0004604"),                          # back pain
    ({"head", "temporal", "frontal", "occipital", "forehead"}, "C0018681"),  # headache
    ({"throat", "pharynx", "pharyngeal"}, "C0242429"),                # throat pain
]


def _composed_symptom(location_txt: set[str]) -> Optional[str]:
    for cues, cui in _REGION_SYMPTOM:
        if location_txt & cues:
            return cui
    return None


def _int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _severity(nrs: Optional[str]) -> Optional[str]:
    """NRS 0-10 -> tertile controlled severity (CLAUDE.md: 1-3/4-6/7-10)."""
    v = _int(nrs)
    if v is None:
        return None
    return "mild" if v <= 3 else "moderate" if v <= 6 else "severe"


def _onset(soudain: Optional[str]) -> Optional[str]:
    """`soudain` (0-10, higher = faster appearance) -> {sudden, gradual}. Ambiguous
    middle (4-6) -> None so the uncertain band adds no spurious signal."""
    v = _int(soudain)
    if v is None:
        return None
    if v <= 3:
        return "gradual"
    if v >= 7:
        return "sudden"
    return None


def _loc_tokens(text: str) -> set[str]:
    """English location phrase -> content tokens (drop laterality/filler)."""
    return {
        w
        for w in re.findall(r"[a-z]+", text.lower())
        if w not in _LOC_STOP and len(w) > 2
    }


class _Parser:
    """Decodes a patient's french evidence list into attribute-aware findings."""

    def __init__(self) -> None:
        self._umap: dict = json.load(open(f"{_DATA}/umls_mapping.json"))["mapping"]
        self._ev: dict = json.load(open(f"{_DATA}/release_evidences.json"))
        self._loc_cui_cache: dict[str, Optional[str]] = {}

    def _split_code(self, left: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """`left` (evidence code without value) -> (base_code, attribute, subq_code).

        Base finding => (base, None, None). Sub-question => (base, attribute, left)."""
        if left in self._umap:
            return left, None, None
        for suffix, attr in ATTR_SUFFIX.items():
            if left.endswith("_" + suffix):
                base = left[: -(len(suffix) + 1)]
                if base in self._umap:
                    return base, attr, left
        return None, None, None

    def _english(self, subq_code: str, value: str) -> Optional[str]:
        info = self._ev.get(subq_code, {}).get("value_meaning", {}).get(value)
        if not info:
            return None
        en = info.get("en")
        return en if en and en != "NA" else None

    def _loc_cui(self, english: str) -> Optional[str]:
        if english not in self._loc_cui_cache:
            cui, _ = umls_link(english, LOC_TUI)
            self._loc_cui_cache[english] = cui
        return self._loc_cui_cache[english]

    def parse(self, evidences: list[str]) -> list[dict]:
        findings: dict[str, dict] = {}

        def finding(cui: str) -> dict:
            return findings.setdefault(
                cui,
                {
                    "cui": cui,
                    "location": set(),
                    "location_txt": set(),
                    "character": set(),
                    "severity": None,
                    "onset": None,
                },
            )

        for raw in evidences:
            left, _, value = raw.partition("_@_")
            value = value or None
            base, attr, subq_code = self._split_code(left)
            if base is None:
                continue
            cui = self._umap[base].get("cui")
            if not cui:
                continue
            f = finding(cui)
            if attr is None:
                continue
            if attr == "severity":
                sev = _severity(value)
                if sev:
                    f["severity"] = sev
            elif attr == "onset":
                ons = _onset(value)
                if ons:
                    f["onset"] = ons
            elif value:
                english = self._english(subq_code, value)  # type: ignore[arg-type]
                if not english or english.lower() in _NON_LOCATION_VALUES:
                    continue
                if attr == "character":
                    f["character"] |= _loc_tokens(english)
                elif attr == "location":
                    f["location_txt"] |= _loc_tokens(english)
                    loc_cui = self._loc_cui(english)
                    if loc_cui:
                        f["location"].add(loc_cui)

        # compose located pain -> specific-symptom concept so it joins the semiology KG edges
        if COMPOSE:
            for f in list(findings.values()):
                if f["cui"] in _PAIN_CUIS:
                    sc = _composed_symptom(f["location_txt"])
                    if sc and sc not in findings:
                        findings[sc] = {
                            "cui": sc,
                            "location": set(f["location"]),
                            "location_txt": set(f["location_txt"]),
                            "character": set(f["character"]),
                            "severity": f["severity"],
                            "onset": f["onset"],
                        }
        return list(findings.values())


def load(n: int = 3000) -> tuple[dict, list]:
    """DDXPlus with per-finding Bates attributes preserved.

    Returns (cand, patients): cand = {en_key: (name_en, umls_cui)};
    patients = list of (target_keys, findings, "DDXPlus")."""
    dmap = json.load(open(f"{_DATA}/disease_umls_mapping.json"))["mapping"]
    cond = json.load(open(f"{_DATA}/release_conditions_en.json"))
    fr2en_dis = {
        info.get("cond-name-fr") or info.get("condition_name"): en
        for en, info in cond.items()
    }
    cand = {
        en: (info.get("name_en", en), info["umls_cui"])
        for en, info in dmap.items()
        if info.get("umls_cui")
    }

    parser = _Parser()
    patients: list = []
    with open(f"{_DATA}/release_test_patients.csv") as fh:
        for row in csv.DictReader(fh):
            if len(patients) >= n:
                break
            en = fr2en_dis.get(row["PATHOLOGY"])
            if en not in cand:
                continue
            findings = parser.parse(ast.literal_eval(row["EVIDENCES"]))
            if findings:
                patients.append(([en], findings, "DDXPlus"))
    return cand, patients


if __name__ == "__main__":  # smoke test
    cand, patients = load(200)
    print(f"cand={len(cand)} patients={len(patients)}")
    ex = patients[0]
    print("target:", ex[0])
    for f in ex[1][:5]:
        print("  ", {k: (sorted(v) if isinstance(v, set) else v) for k, v in f.items()})
