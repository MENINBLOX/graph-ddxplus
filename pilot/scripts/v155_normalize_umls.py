"""Production NORMALIZATION — threshold-FREE (replaces v154's scispaCy candidate-generation).

Every step is deterministic, no similarity threshold, citable:
  concept (finding/location/character) -> UMLS CUI via umls_link (MRXNS exact-match +
      morphological reduction; disambiguated to a clinically-valid semantic type) -> SNOMED CT.
  categorical (severity/onset) -> controlled vocab via attr_normalize (adjectives, NRS 0-10).
  faithfulness: finding is source-grounded by construction (IE clean() keeps only source tokens);
      reported here as the fraction whose content tokens appear in the embedded source.

Input  = v153 IE output {cui:{disease, source, findings:[...]}}.
Output = KG-ready records (CUI + SNOMED + semantic type + normalized attrs). Plain python3."""
import json, re, os, sys, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from umls_link import link as umls_link, norm as umls_norm
from attr_normalize import normalize_severity, normalize_onset, SEV_LEX, ONSET_LEX

UMLS = "pilot/cache/umls_maps"
CONCEPT = ["name", "location", "character"]
FINDING_TUI = {"T184", "T047", "T048", "T046", "T191", "T019", "T190", "T037", "T020", "T049", "T050", "T033", "T034"}
LOC_TUI = {"T023", "T029", "T030", "T024", "T025", "T018", "T031", "T022", "T017"}
CHAR_TUI = FINDING_TUI | {"T080", "T081"}
STRIP_WORDS = set(SEV_LEX) | {s for syns in ONSET_LEX.values() for s in syns if " " not in s}
STOP = set("a an and or of the to in on with at for is are".split())

def clean_name(name):
    kept = [w for w in re.split(r'\s+', name.strip()) if w.lower().strip("',.") not in STRIP_WORDS]
    return " ".join(kept) if kept else name

def grounded(term, src_l):
    ws = [w for w in re.findall(r'[a-z]+', term.lower()) if len(w) > 3 and w not in STOP]
    return bool(ws) and all(w in src_l for w in ws)

def load_maps():
    snomed = {}
    for line in open(f"{UMLS}/cui_snomed.tsv", encoding="utf-8"):
        c, s = line.rstrip("\n").split("\t"); snomed[c] = s
    sty = {}
    for line in open(f"{UMLS}/cui_sty.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2: sty.setdefault(p[0], p[1])
    return snomed, sty

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="pilot/cache/prod/ie_pred_scaleup.json")
    ap.add_argument("--out", default="pilot/cache/prod/kg_records.json")
    a = ap.parse_args()
    preds = json.load(open(a.pred, encoding="utf-8"))
    snomed, sty = load_maps()
    print(f"maps {len(snomed)} SNOMED / {len(sty)} STY. normalizing {len(preds)} diseases", flush=True)

    tot = defaultdict(int); linked = defaultdict(int); snomed_ok = defaultdict(int)
    clinical = defaultdict(int); ground = defaultdict(int)
    sev_raw = sev_norm = ons_raw = ons_norm = 0
    out = {}
    for di, (cui, obj) in enumerate(preds.items()):
        src_l = obj.get("source", "").lower()
        recs = []
        for f in obj.get("findings", []):
            rec = {"name": f.get("name", "")}
            for fld in CONCEPT:
                v = str(f.get(fld, "")).strip()
                if not v: continue
                tot[fld] += 1
                term = clean_name(v) if fld == "name" else v
                okset = LOC_TUI if fld == "location" else CHAR_TUI if fld == "character" else FINDING_TUI
                cid, mode = umls_link(term, okset)
                if grounded(v, src_l): ground[fld] += 1
                if cid:
                    linked[fld] += 1
                    tui = sty.get(cid, ""); sn = snomed.get(cid)
                    if sn: snomed_ok[fld] += 1
                    if tui in okset: clinical[fld] += 1
                    key = "name" if fld == "name" else fld
                    rec[f"{key}_cui"] = cid; rec[f"{key}_snomed"] = sn; rec[f"{key}_tui"] = tui; rec[f"{key}_mode"] = mode
                    if fld != "name": rec[fld] = v
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

    print(f"\n=== THRESHOLD-FREE normalization ({len(preds)} diseases) ===")
    print(f"  {'field':9} {'n':>5} {'CUI':>6} {'SNOMED':>7} {'clin-type':>10} {'grounded':>9}")
    for fld in CONCEPT:
        n = tot[fld]
        if not n: continue
        print(f"  {fld:9} {n:5} {linked[fld]/n:6.2f} {snomed_ok[fld]/n:7.2f} {clinical[fld]/n:10.2f} {ground[fld]/n:9.2f}")
    print(f"  severity  raw={sev_raw} -> vocab={sev_norm} ({sev_norm/max(1,sev_raw):.0%})")
    print(f"  onset     raw={ons_raw} -> vocab={ons_norm} ({ons_norm/max(1,ons_raw):.0%})")
    json.dump(out, open(a.out, "w"))
    print(f"KG-ready -> {a.out}")

if __name__ == "__main__":
    main()
