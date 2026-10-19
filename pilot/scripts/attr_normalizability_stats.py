"""Per-benchmark, per-attribute normalizability statistics.

For each of the 5 evaluation benchmarks, report the % of PATIENTS that carry >=1 finding whose
Bates attribute value can be NORMALIZED to our standard target:
  location  -> UMLS CUI / SNOMED Body structure   (anatomical site present)
  severity  -> controlled {mild,moderate,severe}   (ordinal / NRS / HPO Severity modifier)
  onset     -> controlled {sudden,gradual} / HPO temporal
  character -> FREE TEXT, no controlled standard    (present but not normalizable -> reported separately)
  aggravating / relieving -> UMLS CUI               (mostly absent)

Method differs only in how each benchmark exposes the value (that is the point of the table):
  DDXPlus, NLICE  -> attr-aware loaders already decode+normalize the value -> read directly.
  SymCat          -> attributes are lexicalized in the symptom string -> lexical parse of names.
  RareBench       -> attributes are fused into the HPO term label -> lexical parse of hpo_name,
                     mappable to HPO Clinical Modifier subontology (HP:0012823).
"""
from __future__ import annotations
import json, re, sys, random
from collections import defaultdict
sys.path.insert(0, "pilot/scripts")

# ---- lexicons (conservative) ------------------------------------------------
CHAR = re.compile(r"\b(sharp|dull|burning|throbbing|stabbing|aching|cramp\w*|colicky|pressure|"
                  r"tight\w*|heav\w*|gnawing|pulsating|shooting|tearing|squeezing|tingling|"
                  r"pricking|knife|pulse-like|exhausting|sensitive|haunting|tedious|violent)\b", re.I)
SEV = re.compile(r"\b(mild|moderate|severe|profound|marked|slight|minimal|extreme|"
                 r"heavy|scanty|excessive|diminished)\b", re.I)
TIMING = re.compile(r"\b(sudden|gradual|acute|chronic|progressive|episodic|recurrent|intermittent|"
                    r"insidious|paroxysmal|relapsing)\b", re.I)   # course/pace (excludes age-of-onset
# and prognosis terms e.g. 'neonatal death' that inflate the count)
# RareBench separable-location = laterality/distribution qualifiers ONLY (anatomical site is
# baked into the HPO concept itself, e.g. 'Hepatomegaly', so it is not a separable attribute).
LOC_LAT = re.compile(r"\b(bilateral|unilateral|left|right|generalized|generalised|focal|multifocal|"
                     r"diffuse|asymmetric|symmetric|proximal|distal)\b", re.I)
AGGREL = re.compile(r"\b(worse|aggravat\w*|exacerbat\w*|reliev\w*|ameliorat\w*|better with)\b", re.I)
# anatomical location: broad body-part lexicon (proxy for UMLS-anatomy normalizability)
LOC = re.compile(r"\b(head|skull|scalp|face|forehead|eye|ear|nose|nasal|mouth|oral|jaw|tongue|throat|"
                 r"neck|cervical|chest|thora\w*|breast|sternum|rib|back|lumbar|spine|spinal|abdom\w*|"
                 r"epigastr\w*|flank|groin|pelvi\w*|hip|buttock|arm|shoulder|elbow|wrist|hand|finger|"
                 r"thumb|leg|thigh|knee|ankle|foot|feet|toe|heel|calf|skin|joint|muscle|bone|"
                 r"kidney|liver|hepatic|renal|cardiac|pulmonary|lung|gastric|intestin\w*|colon|rectal|"
                 r"axilla|tonsil|anus|frontal|occipital|temporal|bilateral|unilateral|left|right|"
                 r"generalized|focal|diffuse|proximal|distal|upper|lower)\b", re.I)


def pat_stats(patient_symptom_texts, loc_rx=LOC):
    """given a list of patients, each a list of symptom-name strings, return per-attribute
    fraction of patients whose ANY symptom string matches that attribute lexicon."""
    n = len(patient_symptom_texts)
    hit = defaultdict(int)
    for texts in patient_symptom_texts:
        joined = " ; ".join(texts)
        for key, rx in [("location", loc_rx), ("character", CHAR), ("severity", SEV),
                        ("timing", TIMING), ("aggr/reliev", AGGREL)]:
            if rx.search(joined):
                hit[key] += 1
    return {k: 100 * hit[k] / n for k in ("location", "character", "severity", "timing", "aggr/reliev")}, n


def from_loader_findings(patients, get_findings):
    """DDXPlus / NLICE: findings already carry normalized fields. % patients with >=1 non-empty."""
    n = len(patients)
    hit = defaultdict(int)
    for p in patients:
        finds = get_findings(p)
        flags = {"location": False, "character": False, "severity": False, "onset": False}
        for f in finds:
            if f.get("location") or f.get("location_txt"): flags["location"] = True
            if f.get("character"): flags["character"] = True
            if f.get("severity"): flags["severity"] = True
            if f.get("onset"): flags["onset"] = True
        for k, v in flags.items():
            if v: hit[k] += 1
    return {k: 100 * hit[k] / n for k in ("location", "character", "severity", "onset")}, n


def main():
    rows = {}

    # DDXPlus (normalized loader) — FULL test set (134,530 patients), not a 3000 cap
    import attr_loader_ddxplus as DL
    _, dpats = DL.load(200000)
    s, n = from_loader_findings(dpats, lambda p: p[1])
    rows["DDXPlus"] = (n, s)

    # NLICE (normalized loader)
    import attr_loader_nlice as NL
    ncand, npats = NL.load(150)
    def nlice_finds(p):
        ev = p[1]
        return ev if (ev and isinstance(ev[0], dict)) else []
    s, n = from_loader_findings(npats, nlice_finds)
    rows["NLICE"] = (n, s)

    # SymCat (lexical parse of symptom names) — simulate patients like load_symcat
    sc = json.load(open("data/symcat/symcat_parsed.json"))
    sym_umls = json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    slug2name = {i["symcat_slug"]: nm for nm, i in sym_umls.items() if i.get("symcat_slug")}
    rng = random.Random(42)
    spats = []
    for dn, sym_list in sc["disease_symptom_pairs"].items():
        pairs = [(s, f) for s, f in sym_list]
        for _ in range(50):
            texts = [slug2name.get(s, s) for s, fr in pairs if rng.random() * 100 < fr]
            if texts:
                spats.append(texts)
    s, n = pat_stats(spats)
    rows["SymCat"] = (n, s)

    # RareBench per dataset (lexical parse of HPO term labels)
    import onlykg_eval_unified as U
    hm = json.load(open("data/rarebench/hpo_umls_mapping.json"))["mapping"]
    cui2name = {}
    for hp, info in hm.items():
        if isinstance(info, dict) and info.get("umls_cui"):
            cui2name.setdefault(info["umls_cui"], info.get("hpo_name", ""))
    for dsn in ["RAMEDIS", "LIRICAL", "MME", "HMS"]:
        _, pats = U.load_rarebench([dsn])
        texts_per = [[cui2name.get(c, "") for c in ev] for _, ev, _ in pats]
        s, n = pat_stats(texts_per, loc_rx=LOC_LAT)   # separable location = laterality only
        rows[f"RareBench:{dsn}"] = (n, s)

    # ---- print table --------------------------------------------------------
    cols = ["location", "character", "severity", "onset", "timing", "aggr/reliev"]
    print(f"\n% of patients carrying >=1 finding with a NORMALIZABLE value for the attribute")
    print(f"(character = free-text, not normalized to a controlled standard; shown for presence)\n")
    print(f"{'benchmark':16} {'n':>6} " + " ".join(f"{c:>11}" for c in cols))
    for bench, (n, s) in rows.items():
        cells = []
        for c in cols:
            v = s.get(c)
            cells.append(f"{v:10.1f}%" if v is not None else f"{'-':>11}")
        print(f"{bench:16} {n:>6} " + " ".join(cells))
    print("\nnote: DDXPlus/NLICE from normalized attr-loaders (onset column); SymCat/RareBench from "
          "lexical parse of symptom/HPO labels (timing column). location=UMLS/SNOMED, "
          "severity->{mild,mod,severe}, onset/timing->controlled, aggr/reliev->UMLS CUI.")


if __name__ == "__main__":
    main()
