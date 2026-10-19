"""Bucket-A forensic: gold LOCATION pairs whose finding the model extracted NOWHERE.
Decides whether the recall ceiling is extraction quality or CORPUS MISMATCH — MACCROBAT
is radiology/pathology case reports, so many 'findings' are imaging/path observations
(mass, lesion, opacity) whose site is annotated, not patient symptoms. Categorize each
missed finding as imaging/path vs symptom-like. Stdlib only."""
import json, glob, re, os, argparse
from collections import Counter
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_maccrobat_attrs import parse_gold, sym_match

ANN_DIR = "pilot/cache/maccrobat/brat"
IMAGING = set("mass lesion lesions opacity opacities nodule nodules enhancement thickening "
              "dilation dilatation effusion effusions calcification calcifications hyperintensity "
              "hypodensity hyperdensity infiltrate infiltrates consolidation stenosis atrophy "
              "hypertrophy edema tumor tumour cyst cysts fracture aneurysm thrombus thrombosis "
              "collection fluid septum membrane structure hematoma hemorrhage haemorrhage plaque "
              "polyp scarring fibrosis necrosis metastasis metastases distension distention".split())
SYMPTOM = set("pain fever cough nausea vomiting fatigue weakness dyspnea dyspnoea headache "
              "bleeding rash swelling numbness dizziness diarrhea diarrhoea constipation malaise "
              "weight loss anorexia sweats chills palpitations syncope seizure jaundice pruritus "
              "cyanosis rales murmur wheeze wheezing tremor".split())

def cat(name):
    ws = set(re.findall(r'[a-z]+', name.lower()))
    if ws & IMAGING: return "imaging/path"
    if ws & SYMPTOM: return "symptom"
    return "other"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--show", type=int, default=20)
    a = ap.parse_args()
    preds = json.load(open(a.pred, encoding="utf-8"))
    gold = parse_gold(ANN_DIR, "biological_structure")
    cats = Counter(); samples = {"imaging/path": [], "symptom": [], "other": []}
    total_miss = 0
    for pmid, gp in gold.items():
        names = [x["name"] for x in preds.get(pmid, [])]
        for gn, gv in gp:
            if any(sym_match(an, gn) for an in names):
                continue  # extracted -> not bucket A
            total_miss += 1
            c = cat(gn); cats[c] += 1
            if len(samples[c]) < a.show: samples[c].append((pmid, gn, gv))
    print(f"\n=== bucket-A (finding NOT extracted) forensic  n={total_miss}  pred={os.path.basename(a.pred)} ===")
    for c, n in cats.most_common():
        print(f"  {c:14} {n:4}  ({100*n/total_miss:.0f}%)")
    for c in ("imaging/path", "symptom", "other"):
        print(f"\n--- {c} missed findings (up to {a.show}) ---")
        for pmid, gn, gv in samples[c]:
            print(f"  [{pmid}] {gn!r}  (gold_loc={gv!r})")

if __name__ == "__main__":
    main()
