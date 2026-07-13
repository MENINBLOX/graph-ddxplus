"""D-bucket (precision-FP) forensic for a strategy's LOCATION predictions.

A prediction lands in bucket D when its finding name matches NO gold
MODIFY(Biological_structure->symptom) pair in that doc. But MACCROBAT relation gold is
SPARSE — annotators do not link every finding to its site. So a 'FP' may be a CORRECT
(finding, location) extraction that gold simply never annotated. This decides whether
the D jump under decomposition is real error or gold-incompleteness.

For each D prediction (finding, location), test against the SOURCE .txt:
  STATED-IN-SRC   location tokens appear in the same sentence as the finding  -> correct extraction, gold-incomplete
  SITE-ELSEWHERE  location in source but not in finding's sentence             -> possibly wrong binding
  NOT-IN-SRC      location tokens absent from source                          -> genuine hallucination/error

Usage: python3 v146_D_forensic.py --pred pilot/cache/maccrobat/v146_V3_pred.json [--show 20]
Stdlib only."""
import json, glob, re, os, argparse
from collections import Counter
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_maccrobat_attrs import parse_gold, toks, sym_match

ANN_DIR = "pilot/cache/maccrobat/brat"

def sentences(txt): return [s for s in re.split(r'(?<=[.\n])\s+', txt) if s.strip()]
def loc_in(loc, sl):
    lt = toks(loc)
    return bool(lt) and (loc.lower() in sl or sum(1 for w in lt if w in sl) >= max(1, len(lt)//2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--show", type=int, default=20)
    a = ap.parse_args()
    preds = json.load(open(a.pred, encoding="utf-8"))
    gold = parse_gold(ANN_DIR, "biological_structure")

    cats = Counter(); samples = {"STATED-IN-SRC": [], "SITE-ELSEWHERE": [], "NOT-IN-SRC": []}
    for pmid, gp in gold.items():
        doc = preds.get(pmid, [])
        pp = [(x["name"], x["location"]) for x in doc if x.get("location")]
        txt = ""
        tp = f"{ANN_DIR}/{pmid}.txt"
        if os.path.exists(tp): txt = open(tp, encoding="utf-8").read()
        sl_full = txt.lower(); sents = sentences(txt)
        for pn, pv in pp:
            if any(sym_match(pn, gn) for gn, gv in gp):
                continue  # not a D FP (matches a gold location pair)
            # find finding's sentence
            fs = next((s for s in sents if sym_match(pn, s)), None)
            if fs is not None and loc_in(pv, fs.lower()):
                cat = "STATED-IN-SRC"
            elif loc_in(pv, sl_full):
                cat = "SITE-ELSEWHERE"
            else:
                cat = "NOT-IN-SRC"
            cats[cat] += 1
            if len(samples[cat]) < a.show:
                snip = (fs.strip()[:160] if fs else "(finding sentence not located)")
                samples[cat].append((pmid, pn, pv, snip))

    total = sum(cats.values())
    print(f"\n=== LOCATION D-bucket (precision-FP) forensic  n={total}  pred={os.path.basename(a.pred)} ===")
    for c, n in cats.most_common():
        print(f"  {c:15} {n:4}  ({100*n/total:.0f}%)")
    for c in ("STATED-IN-SRC", "SITE-ELSEWHERE", "NOT-IN-SRC"):
        print(f"\n--- {c} samples (up to {a.show}) ---")
        for pmid, pn, pv, snip in samples[c]:
            print(f"[{pmid}] finding={pn!r} loc={pv!r}\n    src: {snip}")

if __name__ == "__main__":
    main()
