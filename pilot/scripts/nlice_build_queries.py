"""Build attribute-bearing patient queries from NLICE sample_data.

NLICE (arXiv:2401.13756) augments SymCat-grounded synthetic patients with per-symptom
Bates attributes. This is the ONLY substrate on which our attribute leave-one-out
ablation can produce a diagnosis signal, because our other benchmarks (SymCat/RareBench/
DDXPlus) give attribute-FREE queries (verified: symptom -> bare CUI list).

NLICE SYMPTOMS field = ';'-separated symptom tokens, each ':'-delimited 10 fields:
  [name : nature : location : sublocation : intensity : duration : onset : excitation : frequency : prob%]
Mapping onto our schema (production IE extracts name/location/severity/onset/character):
  character  <- nature       (field 2, free-text; also linked to CHAR CUI)
  location   <- location+sub (fields 3,4 -> UMLS CUI, matches KG edge location_cui)
  onset      <- onset        (field 7 -> {sudden,gradual}, matches KG edge onset)
  severity   <- intensity    (field 5 -- EMPTY in sample_data; covered by public gold)
  aggravating<- excitation   (field 8; filled for 12% of findings)
  timing     <- frequency    (field 9; filled for 7%)
  duration   <- duration     (field 6; free text, NOT scored -- CLAUDE.md keeps it out of scoring)
NOTE: excitation/frequency were previously dropped here on the stated grounds that "our IE does not
extract aggravating". That was not true -- the IE extracted it and the frozen 4-attribute cleaner
(v145.clean) deleted it before storage. The bug thus became a documented design rationale for not
testing three of the six candidate attributes at all. They are emitted now.
  relieving  <- (absent in NLICE)

Symptom/disease names are linked to CUI with the SAME threshold-free MRXNS linker used to
build the KG (umls_link), so namespaces align. Only patients whose true disease CUI is in
our KG are kept. Deterministic: reads first --cap rows per disease (no sampling RNG).

Output: {meta, patients:[{true_cui, disease, findings:[{cui, character, location_cui, onset,
                          aggravating, timing, duration}]}]}
"""
from __future__ import annotations
import json, os, re, sys, glob, argparse
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from umls_link import link as umls_link
from attr_normalize import normalize_onset

FINDING_TUI = {"T184", "T047", "T048", "T046", "T191", "T019", "T190", "T037",
               "T020", "T049", "T050", "T033", "T034"}
LOC_TUI = {"T023", "T029", "T030", "T024", "T025", "T018", "T031", "T022", "T017"}
CHAR_TUI = FINDING_TUI | {"T080", "T081"}

NLICE_DIR_DEFAULT = ("/tmp/claude-1000/-home-max-thesis/"
                     "edda7dda-7b76-45bb-a79a-ce62d7e31a1b/scratchpad/NLICE/sample_data")


def clean_tok(s: str) -> str:
    """NLICE tokens use '_' for spaces and trailing '_'; drop parenthetical glosses."""
    s = s.replace("_", " ").strip()
    s = re.sub(r"\(.*?\)", "", s).strip()   # 'Decreased(anorexia)' -> 'Decreased'
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlice-dir", default=NLICE_DIR_DEFAULT)
    ap.add_argument("--kg", default="pilot/cache/prod/kg.json")
    ap.add_argument("--out", default="pilot/cache/prod/nlice_queries.json")
    ap.add_argument("--cap", type=int, default=200, help="max patients per disease")
    a = ap.parse_args()

    kg = json.load(open(a.kg))
    kg_dz = set(kg["diseases"])
    kg_ph = set(kg["phenotypes"])

    # memoised linkers -------------------------------------------------------
    dz_cache: dict[str, str | None] = {}
    def link_disease(name: str):
        if name not in dz_cache:
            cid, _ = umls_link(clean_tok(name), FINDING_TUI)
            dz_cache[name] = cid
        return dz_cache[name]

    sym_cache: dict[str, str | None] = {}
    def link_symptom(name: str):
        if name not in sym_cache:
            cid, _ = umls_link(clean_tok(name), FINDING_TUI)
            sym_cache[name] = cid
        return sym_cache[name]

    loc_cache: dict[str, str | None] = {}
    def link_location(tokens: list[str]):
        key = "|".join(tokens)
        if key not in loc_cache:
            cid = None
            for t in tokens:
                t = clean_tok(t)
                if not t:
                    continue
                cid, _ = umls_link(t, LOC_TUI)
                if cid:
                    break
            loc_cache[key] = cid
        return loc_cache[key]

    char_cache: dict[str, str | None] = {}
    def link_character(tok: str):
        if tok not in char_cache:
            cid, _ = umls_link(clean_tok(tok), CHAR_TUI)
            char_cache[tok] = cid
        return char_cache[tok]

    files = sorted(glob.glob(os.path.join(a.nlice_dir, "*.csv")))
    print(f"NLICE CSVs: {len(files)}  cap/disease={a.cap}", flush=True)

    patients = []
    stat = Counter()
    dz_kept = {}
    for fp in files:
        dname = os.path.basename(fp)[:-4]
        tcui = link_disease(dname)
        stat["disease_seen"] += 1
        if not tcui:
            stat["disease_no_cui"] += 1
            continue
        if tcui not in kg_dz:
            stat["disease_not_in_kg"] += 1
            continue
        dz_kept[dname] = tcui
        kept = 0
        with open(fp, encoding="utf-8") as fh:
            header = fh.readline()
            for line in fh:
                if kept >= a.cap:
                    break
                cell = line.rstrip("\n").split(",")[-1]
                if not cell:
                    continue
                findings = []
                for tok in cell.split(";"):
                    parts = tok.split(":")
                    if not parts or not parts[0].strip():
                        continue
                    parts += [""] * (10 - len(parts))
                    sname, nature, loc1, loc2, intens, dur, onset, excit, freq, _p = parts[:10]
                    scui = link_symptom(sname)
                    if not scui:
                        stat["symptom_no_cui"] += 1
                        continue
                    stat["symptom_linked"] += 1
                    if scui in kg_ph:
                        stat["symptom_in_kg"] += 1
                    f = {"cui": scui}
                    nat = clean_tok(nature).lower()
                    if nat:
                        f["character"] = nat
                        cc = link_character(nature)
                        if cc:
                            f["character_cui"] = cc
                    lcui = link_location([loc1, loc2])
                    if lcui:
                        f["location_cui"] = lcui
                    on = normalize_onset(clean_tok(onset)) if onset.strip() else ""
                    if on:
                        f["onset"] = on
                    if excit.strip():
                        f["aggravating"] = clean_tok(excit)
                    if freq.strip():
                        f["timing"] = clean_tok(freq)
                    if dur.strip():
                        f["duration"] = clean_tok(dur)
                    findings.append(f)
                if findings:
                    patients.append({"true_cui": tcui, "disease": dname, "findings": findings})
                    kept += 1
        stat[f"kept_{dname[:20]}"] = kept

    # attribute density on the query side -----------------------------------
    nf = sum(len(p["findings"]) for p in patients)
    dens = Counter()
    for p in patients:
        for f in p["findings"]:
            for k in ("character", "character_cui", "location_cui", "onset",
                      "aggravating", "timing", "duration"):
                if f.get(k):
                    dens[k] += 1

    print(f"\n=== NLICE query build ===")
    print(f"  diseases: seen={stat['disease_seen']} no_cui={stat['disease_no_cui']} "
          f"not_in_kg={stat['disease_not_in_kg']} kept={len(dz_kept)}")
    print(f"  patients: {len(patients)}   findings: {nf}   findings/patient={nf/max(1,len(patients)):.1f}")
    sl, snc = stat["symptom_linked"], stat["symptom_no_cui"]
    print(f"  symptom linking: linked={sl} no_cui={snc} ({100*sl/max(1,sl+snc):.0f}%)  "
          f"in_kg={stat['symptom_in_kg']} ({100*stat['symptom_in_kg']/max(1,sl):.0f}% of linked)")
    print(f"  query-side attribute density (of {nf} findings):")
    for k in ("character", "character_cui", "location_cui", "onset",
              "aggravating", "timing", "duration"):
        print(f"    {k:14} {dens[k]:6} ({100*dens[k]/max(1,nf):.1f}%)")

    meta = {"n_patients": len(patients), "n_diseases": len(dz_kept),
            "n_findings": nf, "cap": a.cap,
            "diseases": dz_kept, "query_attr_density": dict(dens)}
    json.dump({"meta": meta, "patients": patients}, open(a.out, "w"))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
