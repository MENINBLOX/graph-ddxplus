"""Gold-free IE validation via UMLS-CUI grounding (user-proposed). No external gold.

Three unified metrics, all computed by normalizing BOTH the IE output and the SOURCE text
to UMLS CUIs with scispaCy, then comparing at the CONCEPT (not string) level:

  (1) normalization coverage : % of IE conceptual terms (finding/location/character) that
      link to a UMLS CUID above threshold  -> "are these real medical concepts / KG-usable"
  (2) semantic grounding     : % of IE CUIs that also appear in the SOURCE's CUI set
      (string-independent: SOB / dyspnea / "short of breath" share CUI C0013404)
  (3) semantic hallucination : IE CUI NOT in source AND term not verbatim in source
      -> concept the model introduced that the source never states

Categorical fields (severity/onset) are not UMLS concepts -> validated by controlled-vocab
conformance ({mild,moderate,severe}/{sudden,gradual}) + source presence of the value word.

Deterministic (set membership), NOT an NLI/LLM judge -> compatible with the professor's
"no AI-judges-AI" constraint and with the pipeline's own scispaCy UMLS normalization.

Usage: python v151_cui_grounding.py --pred pilot/cache/maccrobat/v147_V4_pred.json --n 200 [--sample 20]
Run in .venv (scispacy + en_core_sci_lg + UMLS linker)."""
import json, glob, os, re, sys, argparse
from collections import defaultdict

ANN_DIR = "pilot/cache/maccrobat/brat"
CONCEPT_FIELDS = ["name", "location", "character"]
CAT_VOCAB = {"severity": {"mild", "moderate", "severe"}, "onset": {"sudden", "gradual"}}
# categorical values are NORMALIZED -> ground via source SYNONYMS, not the literal vocab word
CAT_SYN = {
    "sudden": ["sudden", "suddenly", "acute", "acutely", "abrupt", "abruptly", "rapid", "rapidly",
               "immediate", "immediately", "explosive", "thunderclap", "instant", "within minutes", "within hours"],
    "gradual": ["gradual", "gradually", "insidious", "insidiously", "slow", "slowly", "progressive",
                "progressively", "chronic", "chronically", "over days", "over weeks", "over months", "over years",
                "long-standing", "longstanding", "developed over"],
    # severity values also often appear via synonyms of the grade in source
    "mild": ["mild", "mildly", "slight", "slightly", "minimal", "minimally", "faint", "low-grade", "trivial"],
    "moderate": ["moderate", "moderately"],
    "severe": ["severe", "severely", "marked", "markedly", "florid", "massive", "profound", "extensive",
               "serious", "significant", "high-grade", "strong", "intense"],
}
THRESH = 0.85
KKB = 5

def cat_grounded(value, src_l):
    """categorical value grounded if any of its source-synonyms appears in source."""
    for syn in CAT_SYN.get(value, [value]):
        if syn in src_l:
            return True
    return False

def build_linker():
    import spacy
    from scispacy.linking import EntityLinker  # noqa: registers pipe
    nlp = spacy.load("en_core_sci_lg", disable=["parser", "tagger"])
    nlp.add_pipe("scispacy_linker",
                 config={"resolve_abbreviations": True, "linker_name": "umls",
                         "threshold": THRESH, "max_entities_per_mention": 3})
    return nlp

def doc_cuis(nlp, text):
    """all CUIs scispaCy links anywhere in `text` (source-side concept set)."""
    cuis = set()
    for ent in nlp(text).ents:
        for cui, score in ent._.kb_ents:
            if score >= THRESH:
                cuis.add(cui)
    return cuis

def term_cuis(nlp, term):
    """top CUIs for a short IE term (returns set, possibly empty = unlinkable)."""
    cuis = set()
    for ent in nlp(term).ents:
        for cui, score in ent._.kb_ents:
            if score >= THRESH:
                cuis.add(cui)
    return cuis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sample", type=int, default=0, help="limit docs (0=all n)")
    a = ap.parse_args()
    preds = json.load(open(a.pred, encoding="utf-8"))
    txts = sorted(glob.glob(f"{ANN_DIR}/*.txt"))[:a.n]
    if a.sample: txts = txts[:a.sample]

    nlp = build_linker()
    print(f"linker ready. scoring {len(txts)} docs from {os.path.basename(a.pred)}", flush=True)

    # counters
    concept_tot = defaultdict(int); concept_link = defaultdict(int)
    concept_ground = defaultdict(int); concept_halluc = defaultdict(int)
    cat_tot = defaultdict(int); cat_vocab_ok = defaultdict(int); cat_src = defaultdict(int)
    halluc_ex = []

    for i, tp in enumerate(txts):
        pmid = os.path.basename(tp)[:-4]
        src = open(tp, encoding="utf-8").read()
        src_l = src.lower()
        scuis = doc_cuis(nlp, src)
        for f in preds.get(pmid, []):
            for fld in CONCEPT_FIELDS:
                v = str(f.get(fld, "")).strip()
                if not v: continue
                concept_tot[fld] += 1
                tc = term_cuis(nlp, v)
                if tc: concept_link[fld] += 1
                verbatim = all(w in src_l for w in re.findall(r'[a-z]+', v.lower()) if len(w) > 3)
                if (tc & scuis) or verbatim:
                    concept_ground[fld] += 1
                elif tc and not (tc & scuis) and not verbatim:
                    concept_halluc[fld] += 1
                    if len(halluc_ex) < 25: halluc_ex.append((pmid, fld, v))
            for fld, vocab in CAT_VOCAB.items():
                v = str(f.get(fld, "")).strip().lower()
                if not v: continue
                cat_tot[fld] += 1
                if v in vocab: cat_vocab_ok[fld] += 1
                if cat_grounded(v, src_l): cat_src[fld] += 1  # synonym-aware grounding
        if (i + 1) % 25 == 0: print(f"  {i+1}/{len(txts)} docs", flush=True)

    print(f"\n=== CUI-grounding IE validation ({os.path.basename(a.pred)}, {len(txts)} docs) ===")
    print("CONCEPTUAL fields (UMLS CUI):")
    print(f"  {'field':10} {'n':>5} {'norm-cov':>9} {'grounding':>10} {'halluc':>8}")
    for fld in CONCEPT_FIELDS:
        n = concept_tot[fld]
        if not n: continue
        print(f"  {fld:10} {n:5} {concept_link[fld]/n:9.2f} {concept_ground[fld]/n:10.2f} {concept_halluc[fld]/n:8.2f}")
    print("CATEGORICAL fields (controlled vocab; grounding = source synonym present):")
    print(f"  {'field':10} {'n':>5} {'vocab-ok':>9} {'grounded':>10}")
    for fld in CAT_VOCAB:
        n = cat_tot[fld]
        if not n: continue
        print(f"  {fld:10} {n:5} {cat_vocab_ok[fld]/n:9.2f} {cat_src[fld]/n:12.2f}")
    print("\nsemantic-hallucination examples (IE CUI not in source, term not verbatim):")
    for pmid, fld, v in halluc_ex[:15]:
        print(f"  [{pmid}] {fld}={v!r}")

if __name__ == "__main__":
    main()
