"""Attribute-aware RareBench loader.

RareBench encodes clinical attributes *inside* the HPO term NAME
(e.g. "Severe intellectual disability", "Neonatal onset", "Generalized
hypotonia", "Recurrent infections"). This loader parses those embedded
modifiers into the same attribute schema the production KG edges use
(severity / onset / timing / location / character), so that patient
findings can be compared against KG disease->phenotype edge attributes.

Design decisions
----------------
- The phenotype CUI is kept EXACTLY as mapped (the specific compound HPO
  CUI, e.g. the CUI for "Severe intellectual disability"). We do NOT
  re-link to a base concept. Modifier stripping is only used to *detect*
  the attribute from the name; the emitted ``cui`` is unchanged.
- Attribute keyword lists are grounded in the vocabulary actually present
  across the 4 sub-datasets (RAMEDIS/LIRICAL/MME/HMS), probed empirically.
- ``onset`` is normalised to the KG's controlled {sudden, gradual} axis;
  ``timing`` keeps the raw temporal token (free-text, richer than onset);
  ``location`` collects laterality / distribution modifiers as a set.

Public API
----------
    load(datasets) -> (cand, patients)
        cand     : {disease_key -> (name, cui)}
        patients : list of (target_keys, findings, dataset)
                   findings = list of {"cui", "severity", "onset",
                                        "timing", "location", "character"}

Run as a script for the full attribute-encoding / coverage /
discrimination report:
    python pilot/scripts/attr_loader_rarebench.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DATA_DIR = Path("data/rarebench")
HPO_MAP_PATH = DATA_DIR / "hpo_umls_mapping.json"
DIS_MAP_PATH = DATA_DIR / "disease_umls_mapping.json"

# --------------------------------------------------------------------------
# Attribute keyword lexicons (grounded in the RareBench HPO name vocabulary)
# --------------------------------------------------------------------------

# severity -> controlled {mild, moderate, severe}
_SEVERITY: Dict[str, str] = {
    "mild": "mild",
    "borderline": "mild",
    "moderate": "moderate",
    "severe": "severe",
    "profound": "severe",
}

# temporal / onset tokens kept verbatim in `timing`
_TIMING_WORDS: Set[str] = {
    "neonatal", "congenital", "prenatal", "antenatal", "perinatal", "fetal",
    "embryonal", "infantile", "childhood", "juvenile", "adult", "adulthood",
    "acute", "subacute", "chronic", "progressive", "nonprogressive",
    "recurrent", "episodic", "intermittent", "paroxysmal", "transient",
    "persistent", "insidious",
}

# onset normalisation onto the KG's {sudden, gradual} axis
_ONSET_SUDDEN: Set[str] = {
    "neonatal", "congenital", "prenatal", "antenatal", "perinatal", "fetal",
    "embryonal", "acute", "paroxysmal", "sudden",
}
_ONSET_GRADUAL: Set[str] = {
    "progressive", "chronic", "insidious", "adult", "adulthood",
    "juvenile", "childhood", "late",
}

# laterality / distribution modifiers -> `location` token set
_LOCATION_WORDS: Set[str] = {
    "unilateral", "bilateral", "generalized", "generalised", "focal",
    "diffuse", "localized", "localised", "proximal", "distal", "central",
    "peripheral", "left", "right", "upper", "lower", "global", "segmental",
    "multifocal", "patchy",
}

# character / quality tokens (sparse in HPO names, but present)
_CHARACTER_WORDS: Set[str] = {
    "painful", "painless", "burning", "aching", "stabbing", "throbbing",
    "cramping", "purulent", "productive", "dry", "spastic", "flaccid",
    "rigid", "pulsatile", "colicky",
}

_TOKEN_RE = re.compile(r"[a-z]+")


def _tokens(name: str) -> List[str]:
    return _TOKEN_RE.findall(name.lower())


def parse_attributes(name: str) -> Dict[str, object]:
    """Extract embedded attributes from an HPO term name.

    Returns a dict with keys severity/onset/timing/location/character.
    location is a set (possibly empty); the rest are str or None.
    """
    toks = _tokens(name)
    tok_set = set(toks)

    severity: Optional[str] = None
    for t in toks:
        if t in _SEVERITY:
            severity = _SEVERITY[t]
            break

    timing: Optional[str] = None
    for t in toks:
        if t in _TIMING_WORDS:
            timing = t
            break

    onset: Optional[str] = None
    if tok_set & _ONSET_SUDDEN:
        onset = "sudden"
    elif tok_set & _ONSET_GRADUAL:
        onset = "gradual"

    location: Set[str] = {t for t in toks if t in _LOCATION_WORDS}

    character: Optional[str] = None
    for t in toks:
        if t in _CHARACTER_WORDS:
            character = t
            break

    return {
        "severity": severity,
        "onset": onset,
        "timing": timing,
        "location": location,
        "character": character,
    }


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------


def load(datasets: List[str]) -> Tuple[Dict[str, Tuple[str, str]], List[tuple]]:
    """Attribute-aware RareBench loader.

    Mirrors ``onlykg_eval_unified.load_rarebench`` but each patient's
    evidence is a list of attribute-annotated finding dicts instead of a
    bare CUI set.
    """
    dis_map = json.load(open(DIS_MAP_PATH))["mapping"]
    hpo_map = json.load(open(HPO_MAP_PATH))["mapping"]
    # Compound -> base concept. Keeping only the compound CUI (the original design decision above)
    # means the base concept our KG actually stores is never reached: on LIRICAL the gold profile
    # explains a median 28.6% of a patient's phenotypes and nothing at all for 13.7% of them,
    # while `short stature` sits on 1,594 KG diseases against `very short stature` on 94. Emitting
    # the base ALONGSIDE the compound, carrying the modifier as an ATTRIBUTE, is what makes the
    # match reachable without making it uninformative -- the base match is common, and the
    # attribute agreement is what discriminates.
    try:
        cui2base = json.load(open("pilot/cache/prod/hpo_base_bridge.json"))["cui2base"]
    except Exception:
        cui2base = {}
    cand: Dict[str, Tuple[str, str]] = {}
    patients: List[tuple] = []

    for ds in datasets:
        path = DATA_DIR / "data" / f"{ds}.jsonl"
        if not path.exists():
            continue
        for line in open(path):
            r = json.loads(line)

            tkeys: List[str] = []
            for t in r["RareDisease"]:
                info = dis_map.get(t)
                if info and info.get("umls_cui"):
                    cand[t] = (info.get("disease_name", t), info["umls_cui"])
                    tkeys.append(t)
            if not tkeys:
                continue

            findings: List[dict] = []
            seen: Set[str] = set()
            for hp in r["Phenotype"]:
                info = hpo_map.get(hp)
                if not isinstance(info, dict):
                    continue
                cui = info.get("umls_cui")
                if not cui or cui in seen:
                    continue
                seen.add(cui)
                attrs = parse_attributes(info.get("hpo_name") or "")
                findings.append({"cui": cui, **attrs})
                base = cui2base.get(cui)
                if base and base not in seen:
                    seen.add(base)
                    findings.append({"cui": base, **attrs})

            if findings:
                patients.append((tkeys, findings, ds))

    return cand, patients


# --------------------------------------------------------------------------
# Report (run as script)
# --------------------------------------------------------------------------


def _build_kg_index(kg_path: str = "pilot/cache/prod/kg.json"):
    """disease_cui -> {phenotype_cui -> list[attrs]} ; plus global pheno set."""
    kg = json.load(open(kg_path))
    idx: Dict[str, Dict[str, List[dict]]] = {}
    pheno_all: Set[str] = set()
    for e in kg["edges"]:
        d, p, a = e["d"], e["p"], e.get("attrs", {})
        idx.setdefault(d, {}).setdefault(p, []).append(a)
        pheno_all.add(p)
    return idx, pheno_all


def _report() -> None:
    datasets = ["RAMEDIS", "LIRICAL", "MME", "HMS"]
    cand, patients = load(datasets)
    dis_map = json.load(open(DIS_MAP_PATH))["mapping"]

    print("=" * 72)
    print("RAREBENCH ATTRIBUTE-AWARE LOADER — REPORT")
    print("=" * 72)

    # ---- (a) per-dataset attribute coverage --------------------------------
    print("\n[a] PER-DATASET ATTRIBUTE COVERAGE "
          "(% of phenotype instances carrying each attribute)")
    attr_axes = ["severity", "onset", "timing", "location", "character"]
    hdr = f"{'dataset':9s} {'patients':>8s} {'pheno':>7s} " + " ".join(
        f"{a[:8]:>8s}" for a in attr_axes)
    print(hdr)
    per_ds = {ds: {"pat": 0, "ph": 0, **{a: 0 for a in attr_axes}}
              for ds in datasets}
    for tkeys, findings, ds in patients:
        per_ds[ds]["pat"] += 1
        for f in findings:
            per_ds[ds]["ph"] += 1
            for a in attr_axes:
                v = f[a]
                if (v if a != "location" else len(v)):
                    per_ds[ds][a] += 1
    tot = {"pat": 0, "ph": 0, **{a: 0 for a in attr_axes}}
    for ds in datasets:
        s = per_ds[ds]
        ph = s["ph"] or 1
        row = f"{ds:9s} {s['pat']:>8d} {s['ph']:>7d} " + " ".join(
            f"{100*s[a]/ph:>7.1f}%" for a in attr_axes)
        print(row)
        for k in tot:
            tot[k] += s[k]
    ph = tot["ph"] or 1
    print(f"{'ALL':9s} {tot['pat']:>8d} {tot['ph']:>7d} " + " ".join(
        f"{100*tot[a]/ph:>7.1f}%" for a in attr_axes))
    print(f"\n(any-attribute phenotype instances: "
          f"{_any_attr_count(patients)} / {tot['ph']} = "
          f"{100*_any_attr_count(patients)/ph:.1f}%)")

    # ---- (b) do target-disease KG edges carry attributes? ------------------
    print("\n[b] TARGET-DISEASE KG EDGE ATTRIBUTE AVAILABILITY")
    kg_idx, pheno_all = _build_kg_index()
    # patient phenotype CUI coverage in KG
    ph_in_kg = ph_tot = 0
    for _, findings, _ in patients:
        for f in findings:
            ph_tot += 1
            if f["cui"] in pheno_all:
                ph_in_kg += 1
    print(f"  patient phenotype CUIs present as a KG phenotype: "
          f"{ph_in_kg}/{ph_tot} = {100*ph_in_kg/max(ph_tot,1):.1f}%")

    # for each patient, does the true disease have any attributed edge?
    tgt_with_attr = tgt_seen = 0
    edge_attr_axes = {"severity": 0, "onset": 0, "location": 0, "character": 0}
    for tkeys, _, _ in patients:
        cuis = {dis_map[t]["umls_cui"] for t in tkeys
                if dis_map.get(t, {}).get("umls_cui")}
        cuis &= set(kg_idx)
        if not cuis:
            continue
        tgt_seen += 1
        has = False
        for dc in cuis:
            for attr_list in kg_idx[dc].values():
                for a in attr_list:
                    if a:
                        has = True
                    for ax in edge_attr_axes:
                        if a.get(ax):
                            edge_attr_axes[ax] += 1
        if has:
            tgt_with_attr += 1
    print(f"  patients whose true disease is IN the KG: {tgt_seen}/{len(patients)}")
    print(f"  ...of those, true disease has >=1 attributed edge: "
          f"{tgt_with_attr} ({100*tgt_with_attr/max(tgt_seen,1):.1f}%)")

    # ---- (c) discrimination potential --------------------------------------
    print("\n[c] DISCRIMINATION ANALYSIS")
    print("  Mechanism A (exact compound-CUI match): patient phenotype CUI"
          " already encodes the attribute; can it be matched on a KG edge?")
    _discrimination(patients, dis_map, kg_idx)


def _any_attr_count(patients) -> int:
    n = 0
    for _, findings, _ in patients:
        for f in findings:
            if (f["severity"] or f["onset"] or f["timing"]
                    or f["location"] or f["character"]):
                n += 1
    return n


def _discrimination(patients, dis_map, kg_idx) -> None:
    """Two complementary views of whether embedded attributes can
    distinguish the true disease from distractors."""
    # Mechanism A: the specific (compound) CUI is shared between true disease
    # and distractors — how often does an attributed patient phenotype's CUI
    # even appear on the true disease edge, and is it shared with others?
    all_dis = list(kg_idx)
    a_shared = a_true_hit = a_attributed = 0
    for tkeys, findings, _ in patients:
        tcuis = {dis_map[t]["umls_cui"] for t in tkeys
                 if dis_map.get(t, {}).get("umls_cui")} & set(kg_idx)
        if not tcuis:
            continue
        for f in findings:
            if not (f["severity"] or f["onset"] or f["location"]
                    or f["character"]):
                continue
            a_attributed += 1
            pc = f["cui"]
            in_true = any(pc in kg_idx[t] for t in tcuis)
            if in_true:
                a_true_hit += 1
    print(f"    attributed patient phenotypes: {a_attributed}")
    print(f"    ...whose compound CUI is on the true disease's edges: "
          f"{a_true_hit} ({100*a_true_hit/max(a_attributed,1):.1f}%)")

    # Mechanism B: attribute-vs-attribute comparison on a SHARED base
    # phenotype. For patient phenotypes carrying severity/onset, find KG
    # edges (any disease) to the SAME CUI that also carry that axis; check
    # whether the true disease agrees and distractors disagree.
    match_true = mismatch_true = 0
    discr_cases = 0            # true agrees AND >=1 distractor disagrees
    axes = ["severity", "onset"]
    for tkeys, findings, _ in patients:
        tcuis = {dis_map[t]["umls_cui"] for t in tkeys
                 if dis_map.get(t, {}).get("umls_cui")} & set(kg_idx)
        if not tcuis:
            continue
        for f in findings:
            for ax in axes:
                val = f[ax]
                if not val:
                    continue
                pc = f["cui"]
                # KG edges to this exact CUI that carry this axis
                true_vals, distractor_vals = set(), set()
                for dc, phmap in kg_idx.items():
                    if pc not in phmap:
                        continue
                    for a in phmap[pc]:
                        if a.get(ax):
                            (true_vals if dc in tcuis
                             else distractor_vals).add(a[ax])
                if val in true_vals:
                    match_true += 1
                    if distractor_vals and val not in distractor_vals:
                        discr_cases += 1
                elif true_vals:
                    mismatch_true += 1
    print("\n  Mechanism B (attribute agreement on shared CUI, severity+onset):")
    print(f"    patient attr == true-disease edge attr : {match_true}")
    print(f"    patient attr != true-disease edge attr : {mismatch_true}")
    print(f"    discriminative cases (true agrees, a distractor disagrees): "
          f"{discr_cases}")


if __name__ == "__main__":
    _report()
