"""Production NORMALIZATION + gold-free VALIDATION stage (context B = disease -> phenotypes).

Input  = v153 IE output {cui: {disease, findings:[{name,location,character,severity,onset,...}]}}
Source = the disease's clinical text (v105_sources/<CUI>.txt) to ground against.
Output = KG-ready records: each phenotype -> CUI + SNOMED + semantic type + NORMALIZED
         categorical attrs (severity/onset via attr_normalize) + grounding flag.

Reuses the scispaCy UMLS linker + CUI->SNOMED/STY maps from v152 and the deterministic
categorical normalizer (attr_normalize). Same gold-free signals: normalization coverage,
clinical-type validity, semantic grounding, hallucination. Run in .venv-sci.

Usage: python v154_prod_normalize.py --pred v153_prod_pred.json --out prod_normalized.json"""
import json, zlib, os, re, sys, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v152_normalize_pipeline import (load_maps, build_linker, link_top, doc_cuis,
                                      FINDING_TUI, LOC_TUI, CHAR_TUI, THRESH)
from attr_normalize import normalize_severity, normalize_onset, SEV_LEX, ONSET_LEX

# Semantic types an ASSOCIATION may take. Wider than FINDING_TUI because §11's evidence pool
# includes past disease (T047), procedures (T061), drugs (T121), social behaviour (T054),
# occupation/population (T097/T098) -- exactly the types whose recall was 0.0% on DDXPlus.
EVIDENCE_TUI = {"T047", "T048", "T046", "T191", "T019", "T020", "T037", "T190",
                "T033", "T184", "T061", "T060", "T121", "T109", "T195", "T200",
                "T054", "T055", "T056", "T097", "T098", "T031", "T204", "T007", "T005"}

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
    ap.add_argument("--resume", action="store_true",
                    help="skip diseases already present in --out and append to it (a full 35k run "
                         "takes ~1h, so it must survive an interrupted shell)")
    ap.add_argument("--checkpoint", type=int, default=2000,
                    help="persist --out every N diseases (atomic temp+rename)")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker's index; with --nshards it normalizes only the diseases whose "
                         "CUI hashes to this shard. scispaCy linking is single-process and leaves "
                         "most of a 24-core host idle, so a full-corpus pass is run as N "
                         "independent workers writing N record files that are merged afterwards.")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N diseases this invocation (0 = all); use with --resume to "
                         "run a long normalization as a series of bounded slices")
    a = ap.parse_args()
    preds = json.load(open(a.pred, encoding="utf-8"))
    snomed, sty = load_maps()
    nlp = build_linker()
    print(f"maps {len(snomed)} SNOMED / {len(sty)} STY. normalizing {len(preds)} diseases", flush=True)

    tot = defaultdict(int); linked = defaultdict(int); snomed_ok = defaultdict(int)
    clinical = defaultdict(int); ground = defaultdict(int); halluc = defaultdict(int)
    sev_raw = sev_norm = ons_raw = ons_norm = 0
    out = {}
    if a.resume and os.path.exists(a.out):
        out = json.load(open(a.out, encoding="utf-8"))
        print(f"resume: {len(out)} already normalized", flush=True)

    def save():
        tmp = a.out + ".tmp"
        json.dump(out, open(tmp, "w"))
        os.replace(tmp, a.out)

    todo = [(c, o) for c, o in preds.items() if c not in out]
    if a.nshards > 1:
        # stable partition: zlib.crc32 is deterministic across processes and runs, unlike hash()
        todo = [(c, o) for c, o in todo if zlib.crc32(c.encode()) % a.nshards == a.shard]
        print(f"shard {a.shard}/{a.nshards}: {len(todo)} diseases", flush=True)
    if a.limit:
        todo = todo[: a.limit]
    print(f"normalizing {len(todo)} diseases this run", flush=True)

    for di, (cui, obj) in enumerate(todo):
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
                okset = LOC_TUI if fld == "location" else CHAR_TUI if fld == "character" else FINDING_TUI
                cid, sc = link_top(nlp, link_term, valid_fn=lambda tui: tui in okset)
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
            # open-schema qualifiers ride through as-is: their controlled vocabulary is HPO
            # Clinical modifier, applied at scoring time (hpo_modifier), not a UMLS CUI link.
            # Dropping them here would repeat the failure that cost three attributes for months.
            if f.get("qualifiers"):
                rec["qualifiers"] = f["qualifiers"]
            for extra in ("timing", "aggravating", "relieving"):
                if str(f.get(extra, "")).strip(): rec[extra] = f[extra]
            recs.append(rec)
        # §11 treats patient evidence as ONE pool -- a history of myocardial infarction counts like
        # a report of pain -- so the association IE's history/exposure/risk items are linked and
        # emitted as evidence records too. They are marked so the two layers stay separable.
        for asc in obj.get("associations", []) or []:
            term = str(asc.get("item", "")).strip()
            if not term:
                continue
            cid, _ = link_top(nlp, term, valid_fn=lambda tui: tui in EVIDENCE_TUI)
            if not cid:
                continue
            recs.append({"name": term, "name_cui": cid, "name_tui": sty.get(cid, ("", ""))[0],
                         "name_snomed": snomed.get(cid), "assoc_kind": asc.get("kind", "")})
        out[cui] = {"disease": obj.get("disease", cui), "phenotypes": recs}
        if (di + 1) % 200 == 0: print(f"  {di+1}/{len(todo)} (total {len(out)})", flush=True)
        if (di + 1) % a.checkpoint == 0: save()
    save()

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
