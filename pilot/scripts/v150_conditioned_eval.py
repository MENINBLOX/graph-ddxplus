"""Corrected, field-standard evaluation of ATTRIBUTE extraction (SemEval-2015 Task14 style):
attributes are scored CONDITIONED on the finding, not as standalone relation pairs. This
removes the confounds that broke pair-level binding F1:
  - multi-site duplication: a finding linked to N sites in gold is ONE finding with a SET of
    acceptable sites; the model is correct if it gives ANY of them (not forced to enumerate N).
  - finding-extraction vs attribute-assignment are REPORTED SEPARATELY (our claim = the latter).
  - noise findings ('normal','changes') are flagged and excluded from the conditioned metric.

Metrics per attribute, over unique (doc, finding) items that gold gives an attribute value:
  coverage      = of gold findings, how many the model also extracted (finding recall; corpus-dependent)
  fill          = of co-identified findings, how many the model gave the attribute for
  ACCURACY      = of those it filled, how many match ANY gold value  <- attribute precision (our claim)
  cond-recall   = of co-identified findings, how many correctly attributed (fill*accuracy)
Stdlib only; runs on cached preds."""
import json, re, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_maccrobat_attrs import parse_gold, sym_match, val_match, toks

ANN_DIR = "pilot/cache/maccrobat/brat"
NOISE = set("normal changes change remarkable unremarkable finding findings appearance".split())
def is_noise(fn):
    w = set(re.findall(r'[a-z]+', fn.lower()))
    return bool(w & NOISE) and len(fn.split()) <= 2

# --- anatomical normalization: clinical adjective-form -> organ noun, + singularize ---
# Interim stand-in for the pipeline's scispaCy UMLS CUI normalization; fixes string variants
# (abdominal<->abdomen, pulmonary<->lung, lungs->lung) that string-match spuriously misses.
ADJ2ORGAN = {
    "abdominal": "abdomen", "pulmonary": "lung", "hepatic": "liver", "cardiac": "heart",
    "renal": "kidney", "gastric": "stomach", "cerebral": "brain", "pancreatic": "pancreas",
    "splenic": "spleen", "pleural": "pleura", "cutaneous": "skin", "dermal": "skin",
    "ocular": "eye", "nasal": "nose", "oral": "mouth", "thoracic": "chest", "pericardial": "pericardium",
    "colonic": "colon", "duodenal": "duodenum", "ileal": "ileum", "vaginal": "vagina",
    "uterine": "uterus", "ovarian": "ovary", "prostatic": "prostate", "vertebral": "vertebra",
    "articular": "joint", "muscular": "muscle", "osseous": "bone", "vascular": "vessel",
    "arterial": "artery", "venous": "vein", "lymphatic": "lymph node", "cervical": "neck",
}
def anat_norm(s):
    out = []
    for w in re.findall(r'[a-z]+', s.lower()):
        w = ADJ2ORGAN.get(w, w)
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"): w = w[:-1]  # singularize
        out.append(w)
    return set(out)
def anat_match(pv, gv):
    a, b = anat_norm(pv), anat_norm(gv)
    return bool(a and b and (a & b))

def gold_by_finding(gold):
    """pmid -> {finding: set(attrvals)} deduping multi-site; drop noise findings."""
    out = {}
    for pmid, pairs in gold.items():
        d = defaultdict(set)
        for fn, av in pairs:
            if is_noise(fn) or not av: continue
            d[fn].add(av)
        out[pmid] = d
    return out

def evaluate(preds, gold, attr, anat=False):
    G = cov = fill = correct = 0
    gbf = gold_by_finding(gold)
    misfill = []
    def vmatch(pv, gv):
        if val_match(pv, gv): return True
        return anat and attr == "location" and anat_match(pv, gv)
    for pmid, fmap in gbf.items():
        doc = preds.get(pmid, [])
        for gfind, gvals in fmap.items():
            G += 1
            # did the model extract this finding?
            cand = [x for x in doc if sym_match(x["name"], gfind)]
            if not cand: continue
            cov += 1
            # among matched preds, did any give the attribute?
            pv = next((x[attr] for x in cand if x.get(attr)), "")
            if not pv: continue
            fill += 1
            if any(vmatch(pv, gv) for gv in gvals):
                correct += 1
            else:
                misfill.append((pmid, gfind, pv, sorted(gvals)))
    coverage = cov / G if G else 0
    fillrate = fill / cov if cov else 0
    accuracy = correct / fill if fill else 0
    cond_recall = correct / cov if cov else 0
    return dict(G=G, cov=cov, fill=fill, correct=correct, coverage=coverage,
                fillrate=fillrate, accuracy=accuracy, cond_recall=cond_recall, misfill=misfill)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", default=[
        "gemma4-E4B=pilot/cache/maccrobat/v147_V4_pred.json"])
    ap.add_argument("--show-misfill", type=int, default=0)
    ap.add_argument("--anat", action="store_true", help="anatomical normalization for location matching")
    a = ap.parse_args()
    loaded = {}
    for spec in a.preds:
        lab, p = spec.split("=", 1)
        if os.path.exists(p): loaded[lab] = json.load(open(p))
    for attr in ("location", "severity"):
        gold = parse_gold(ANN_DIR, "severity" if attr == "severity" else "biological_structure")
        gbf = gold_by_finding(gold)
        uniq = sum(len(v) for v in gbf.values())
        tag = " [anat-normalized]" if (a.anat and attr == "location") else ""
        print(f"\n### {attr.upper()}  conditioned eval{tag}  (unique gold findings w/ attr, noise-dropped = {uniq})")
        print(f"{'model':14} {'coverage':>9} {'fill':>6} {'ACCURACY':>9} {'cond-rec':>9}")
        for lab, preds in loaded.items():
            r = evaluate(preds, gold, attr, anat=a.anat)
            print(f"{lab:14} {r['coverage']:9.2f} {r['fillrate']:6.2f} {r['accuracy']:9.2f} {r['cond_recall']:9.2f}"
                  f"   (cov {r['cov']}/{r['G']}, filled {r['fill']}, correct {r['correct']})")
            for pmid, gf, pv, gv in r["misfill"][:a.show_misfill]:
                print(f"      MISS [{pmid}] {gf!r}: pred={pv!r} gold={gv}")

if __name__ == "__main__":
    main()
