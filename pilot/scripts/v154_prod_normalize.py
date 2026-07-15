"""Production NORMALIZATION + gold-free VALIDATION stage (context B = disease -> phenotypes).

Input  = v153 IE output {cui: {disease, findings:[{name,location,character,severity,onset,...}]}}
Source = the disease's clinical text (v105_sources/<CUI>.txt) to ground against.
Output = KG-ready records: each phenotype -> CUI + SNOMED + semantic type + NORMALIZED
         categorical attrs (severity/onset via attr_normalize) + grounding flag.

Reuses the scispaCy UMLS linker + CUI->SNOMED/STY maps from v152 and the deterministic
categorical normalizer (attr_normalize). Same gold-free signals: normalization coverage,
clinical-type validity, semantic grounding, hallucination. Run in .venv-sci.

Usage: python v154_prod_normalize.py --pred v153_prod_pred.json --out prod_normalized.json"""
import json, os, re, sys, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v152_normalize_pipeline import (load_maps, build_linker, link_top, doc_cuis,
                                      FINDING_TUI, LOC_TUI, CHAR_TUI, THRESH)
from attr_normalize import normalize_severity, normalize_onset, SEV_LEX, ONSET_LEX

SRC_DIR = "pilot/cache/v105_sources"
CONCEPT = ["name", "location", "character"]

# severity/onset adjectives to strip from a finding NAME before CUI linking (they are captured
# separately as attributes; leaving them in the name mislinks to the qualifier CUI, e.g.
# "severe tachypnea" -> "Severe" C0205082 instead of "tachypnea").
STRIP_WORDS = set(SEV_LEX) | {s for syns in ONSET_LEX.values() for s in syns if " " not in s}
def clean_name_for_link(name):
    toks = re.findall(r"[a-z]+(?:'[a-z]+)?|[a-z]+", name.lower())
    kept = [w for w in re.split(r'\s+', name.strip()) if w.lower().strip("',.") not in STRIP_WORDS]
    return " ".join(kept) if kept else name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="pilot/cache/prod/ie_pred.json")
    ap.add_argument("--out", default="pilot/cache/prod/kg_records.json")
    a = ap.parse_args()
    preds = json.load(open(a.pred, encoding="utf-8"))
    snomed, sty = load_maps()
    nlp = build_linker()
    print(f"maps {len(snomed)} SNOMED / {len(sty)} STY. normalizing {len(preds)} diseases", flush=True)

    tot = defaultdict(int); linked = defaultdict(int); snomed_ok = defaultdict(int)
    clinical = defaultdict(int); ground = defaultdict(int); halluc = defaultdict(int)
    sev_raw = sev_norm = ons_raw = ons_norm = 0
    out = {}

    for di, (cui, obj) in enumerate(preds.items()):
        src_l = obj.get("source", "").lower()  # source embedded by v153 (self-contained)
        scuis = doc_cuis(nlp, src_l) if src_l else set()
        recs = []
        for f in obj.get("findings", []):
            rec = {"name": f.get("name", "")}
            for fld in CONCEPT:
                v = str(f.get(fld, "")).strip()
                if not v: continue
                tot[fld] += 1
                link_term = clean_name_for_link(v) if fld == "name" else v
                cid, sc = link_top(nlp, link_term)
                verbatim = all(w in src_l for w in re.findall(r'[a-z]+', v.lower()) if len(w) > 3)
                if cid:
                    linked[fld] += 1
                    tui = sty.get(cid, ("", ""))[0]; sn = snomed.get(cid)
                    if sn: snomed_ok[fld] += 1
                    okset = LOC_TUI if fld == "location" else CHAR_TUI if fld == "character" else FINDING_TUI
                    if tui in okset: clinical[fld] += 1
                    if cid in scuis or verbatim: ground[fld] += 1
                    elif not verbatim: halluc[fld] += 1
                    key = "name" if fld == "name" else fld
                    rec[f"{key}_cui"] = cid; rec[f"{key}_snomed"] = sn; rec[f"{key}_tui"] = tui
                    if fld != "name": rec[fld] = v
                elif verbatim:
                    ground[fld] += 1
            # categorical: keep raw + normalized
            sr = str(f.get("severity", "")).strip()
            if sr:
                sev_raw += 1; sn_ = normalize_severity(sr)
                if sn_: sev_norm += 1
                rec["severity_raw"] = sr; rec["severity"] = sn_
            orr = str(f.get("onset", "")).strip()
            if orr:
                ons_raw += 1; on_ = normalize_onset(orr)
                if on_: ons_norm += 1
                rec["onset_raw"] = orr; rec["onset"] = on_
            for extra in ("timing", "aggravating", "relieving"):
                if str(f.get(extra, "")).strip(): rec[extra] = f[extra]
            recs.append(rec)
        out[cui] = {"disease": obj.get("disease", cui), "phenotypes": recs}
        if (di + 1) % 5 == 0: print(f"  {di+1}/{len(preds)}", flush=True)

    print(f"\n=== PRODUCTION normalization+validation ({len(preds)} diseases) ===")
    print(f"  {'field':9} {'n':>5} {'CUI':>6} {'SNOMED':>7} {'clin-type':>10} {'grounding':>10} {'halluc':>8}")
    for fld in CONCEPT:
        n = tot[fld]
        if not n: continue
        print(f"  {fld:9} {n:5} {linked[fld]/n:6.2f} {snomed_ok[fld]/n:7.2f} {clinical[fld]/n:10.2f} "
              f"{ground[fld]/n:10.2f} {halluc[fld]/n:8.2f}")
    print(f"  severity  raw={sev_raw} normalized->vocab={sev_norm} ({sev_norm/max(1,sev_raw):.0%})")
    print(f"  onset     raw={ons_raw} normalized->vocab={ons_norm} ({ons_norm/max(1,ons_raw):.0%})")
    json.dump(out, open(a.out, "w"))
    print(f"KG-ready -> {a.out}")
    for cui, o in list(out.items())[:1]:
        print(f"\n[{cui}] {o['disease']} sample phenotypes:")
        for r in o["phenotypes"][:4]:
            print(" ", json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
