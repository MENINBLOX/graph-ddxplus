"""Gold-free IE normalization + validation PIPELINE (replaces benchmark-gold IE validation).

For each IE conceptual term (finding/location/character) it produces a normalized record
and the validation signals in one pass:
  term -> scispaCy UMLS link -> CUI + canonical + semantic-type(TUI) + score
       -> CUI->SNOMED CT code (MRCONSO)  [KG target vocabulary]
       -> semantic-type validity (is the CUI a clinical type, not demographic/qualifier noise)
       -> grounding: CUI present in the SOURCE's CUI set (concept-level, string-independent)
          else verbatim-in-source; else = semantic HALLUCINATION (concept not in source)

Outputs:
  (1) normalized records per doc/finding (KG-ready): name/location/character -> CUI/SNOMED/type
  (2) aggregate validation: norm-coverage, SNOMED-mappable, clinical-type rate, grounding, halluc

Categorical fields (severity/onset) -> controlled-vocab conformance + synonym grounding (v151 logic).
Deterministic (set membership + terminology lookup); no LLM/NLI judge.

Usage: python v152_normalize_pipeline.py --pred <pred.json> --n 200 [--out normalized.json] [--sample K]
Run in .venv-sci."""
import json, glob, os, re, sys, argparse
from collections import defaultdict

ANN_DIR = "pilot/cache/maccrobat/brat"
UMLS = "pilot/cache/umls_maps"
CONCEPT_FIELDS = ["name", "location", "character"]
CAT_VOCAB = {"severity": {"mild", "moderate", "severe"}, "onset": {"sudden", "gradual"}}
THRESH = 0.85

# valid semantic types for a FINDING (disease/symptom/finding/pathologic) — NOT anatomy.
# a finding linking to a Body-Part CUI (e.g. "Ebstein's anomaly" -> "tricuspid valve" C0040960/T023)
# is a scispaCy mislink and must FAIL the finding clin-type check.
FINDING_TUI = {
    "T184",  # Sign or Symptom
    "T047", "T048", "T046", "T191", "T019", "T190", "T037", "T020", "T049", "T050",  # disease/pathologic/...
    "T033", "T034",  # Finding / Lab or Test Result
}
LOC_TUI = {"T023", "T029", "T030", "T024", "T025", "T018", "T031", "T022", "T017"}  # body site types (for location)
CHAR_TUI = FINDING_TUI | {"T080", "T081"}  # character rarely links cleanly; qualitative concepts

CAT_SYN = {
    "sudden": ["sudden", "suddenly", "acute", "acutely", "abrupt", "abruptly", "rapid", "rapidly",
               "immediate", "immediately", "explosive", "thunderclap", "within minutes", "within hours"],
    "gradual": ["gradual", "gradually", "insidious", "insidiously", "slow", "slowly", "progressive",
                "progressively", "chronic", "chronically", "over days", "over weeks", "over months",
                "over years", "long-standing", "longstanding"],
    "mild": ["mild", "mildly", "slight", "slightly", "minimal", "minimally", "faint", "low-grade"],
    "moderate": ["moderate", "moderately"],
    "severe": ["severe", "severely", "marked", "markedly", "florid", "massive", "profound", "extensive",
               "serious", "significant", "high-grade", "strong", "intense"],
}

def load_maps():
    snomed = {}
    for line in open(f"{UMLS}/cui_snomed.tsv", encoding="utf-8"):
        c, s = line.rstrip("\n").split("\t"); snomed[c] = s
    sty = {}
    for line in open(f"{UMLS}/cui_sty.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3: sty.setdefault(p[0], (p[1], p[2]))  # cui -> (TUI, name)
    return snomed, sty

def build_linker():
    import spacy
    from scispacy.linking import EntityLinker  # noqa
    nlp = spacy.load("en_core_sci_lg", disable=["parser", "tagger"])
    nlp.add_pipe("scispacy_linker",
                 config={"resolve_abbreviations": True, "linker_name": "umls",
                         "threshold": THRESH, "max_entities_per_mention": 1})
    return nlp

def link_top(nlp, term):
    """top (cui, score) for a term, or (None, 0)."""
    best = (None, 0.0)
    for ent in nlp(term).ents:
        for cui, score in ent._.kb_ents:
            if score > best[1]: best = (cui, score)
    return best

def doc_cuis(nlp, text):
    cuis = set()
    for ent in nlp(text).ents:
        for cui, score in ent._.kb_ents:
            if score >= THRESH: cuis.add(cui)
    return cuis

def cat_grounded(v, src_l):
    return any(syn in src_l for syn in CAT_SYN.get(v, [v]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    preds = json.load(open(a.pred, encoding="utf-8"))
    txts = sorted(glob.glob(f"{ANN_DIR}/*.txt"))[:a.n]
    if a.sample: txts = txts[:a.sample]
    snomed, sty = load_maps()
    nlp = build_linker()
    print(f"maps: {len(snomed)} CUI->SNOMED, {len(sty)} CUI->STY. scoring {len(txts)} docs", flush=True)

    tot = defaultdict(int); linked = defaultdict(int); snomed_ok = defaultdict(int)
    clinical = defaultdict(int); ground = defaultdict(int); halluc = defaultdict(int)
    cat_tot = defaultdict(int); cat_vocab = defaultdict(int); cat_ground = defaultdict(int)
    normalized = {}

    for i, tp in enumerate(txts):
        pmid = os.path.basename(tp)[:-4]
        src = open(tp, encoding="utf-8").read(); src_l = src.lower()
        scuis = doc_cuis(nlp, src)
        recs = []
        for f in preds.get(pmid, []):
            rec = {"name": f.get("name", "")}
            for fld in CONCEPT_FIELDS:
                v = str(f.get(fld, "")).strip()
                if not v: continue
                tot[fld] += 1
                cui, sc = link_top(nlp, v)
                verbatim = all(w in src_l for w in re.findall(r'[a-z]+', v.lower()) if len(w) > 3)
                if cui:
                    linked[fld] += 1
                    tui = sty.get(cui, ("", ""))[0]
                    sn = snomed.get(cui)
                    if sn: snomed_ok[fld] += 1
                    ok_type = tui in (LOC_TUI if fld == "location" else CHAR_TUI if fld == "character" else FINDING_TUI)
                    if ok_type: clinical[fld] += 1
                    if cui in scuis or verbatim: ground[fld] += 1
                    elif not verbatim: halluc[fld] += 1
                    if fld != "name":
                        rec[f"{fld}"] = v; rec[f"{fld}_cui"] = cui; rec[f"{fld}_snomed"] = sn; rec[f"{fld}_tui"] = tui
                    else:
                        rec["name_cui"] = cui; rec["name_snomed"] = sn; rec["name_tui"] = tui
                else:
                    if verbatim: ground[fld] += 1  # in source but not a UMLS concept (free-text)
            for fld, vocab in CAT_VOCAB.items():
                v = str(f.get(fld, "")).strip().lower()
                if not v: continue
                cat_tot[fld] += 1
                if v in vocab: cat_vocab[fld] += 1
                if cat_grounded(v, src_l): cat_ground[fld] += 1
                rec[fld] = v
            recs.append(rec)
        normalized[pmid] = recs
        if (i + 1) % 25 == 0: print(f"  {i+1}/{len(txts)}", flush=True)

    print(f"\n=== NORMALIZATION + VALIDATION ({os.path.basename(a.pred)}, {len(txts)} docs) ===")
    print("CONCEPTUAL (of extracted terms):")
    print(f"  {'field':9} {'n':>5} {'CUI-link':>9} {'SNOMED':>8} {'clin-type':>10} {'grounding':>10} {'halluc':>8}")
    for fld in CONCEPT_FIELDS:
        n = tot[fld]
        if not n: continue
        print(f"  {fld:9} {n:5} {linked[fld]/n:9.2f} {snomed_ok[fld]/n:8.2f} {clinical[fld]/n:10.2f} "
              f"{ground[fld]/n:10.2f} {halluc[fld]/n:8.2f}")
    print("CATEGORICAL (of extracted values):")
    print(f"  {'field':9} {'n':>5} {'vocab':>9} {'grounded':>10}")
    for fld in CAT_VOCAB:
        n = cat_tot[fld]
        if not n: continue
        print(f"  {fld:9} {n:5} {cat_vocab[fld]/n:9.2f} {cat_ground[fld]/n:10.2f}")
    if a.out:
        json.dump(normalized, open(a.out, "w"))
        print(f"\nnormalized KG-ready records -> {a.out}")

if __name__ == "__main__":
    main()
