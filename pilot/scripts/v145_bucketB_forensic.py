"""Forensic on the dominant LOCATION sink (bucket B = finding extracted, location EMPTY)
for the winning strategy. Decides whether B is a RECOVERABLE local miss or a corpus ceiling.

For each bucket-B gold pair (symptom the model extracted but left location=""):
  - locate the symptom mention in the source .txt
  - test whether the gold location string appears in the SAME sentence as the symptom
    SAME-SENTENCE  -> location was locally stated -> genuine model miss (lever exists)
    OTHER-SENTENCE -> gold links across sentences  -> corpus/annotation ceiling (not prompt-fixable)
    NOT-IN-SOURCE  -> gold value not verbatim in text -> annotation normalization gap

Usage: python3 v145_bucketB_forensic.py --pred pilot/cache/maccrobat/v145_C_pred.json [--show 25]
Stdlib only."""
import json, glob, re, os, argparse
from collections import Counter
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_maccrobat_attrs import parse_gold, toks, sym_match

ANN_DIR = "pilot/cache/maccrobat/brat"

def sentences(txt):
    # crude sentence split good enough for locality test
    return [s for s in re.split(r'(?<=[.\n])\s+', txt) if s.strip()]

def loc_in_sent(loc, sent):
    lt = toks(loc)
    if not lt: return False
    sl = sent.lower()
    if loc.lower() in sl: return True
    return sum(1 for w in lt if w in sl) >= max(1, len(lt) // 2)

def find_sent(name, sents):
    for s in sents:
        if sym_match(name, s): return s
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--show", type=int, default=25)
    a = ap.parse_args()
    preds = json.load(open(a.pred, encoding="utf-8"))
    gold = parse_gold(ANN_DIR, "biological_structure")

    cats = Counter(); samples = []
    for pmid, gp in gold.items():
        doc = preds.get(pmid, [])
        names = [x["name"] for x in doc]
        filled = {x["name"] for x in doc if x.get("location")}
        txt = ""
        tp = f"{ANN_DIR}/{pmid}.txt"
        if os.path.exists(tp): txt = open(tp, encoding="utf-8").read()
        sents = sentences(txt)
        for gn, gv in gp:
            extracted = any(sym_match(an, gn) for an in names)
            bound = any(sym_match(fn, gn) for fn in filled)
            if not extracted or bound:  # only bucket B (extracted, location empty)
                continue
            fs = find_sent(gn, sents)
            if fs is None:
                cat = "NAME-NOT-IN-SRC"
            elif loc_in_sent(gv, fs):
                cat = "SAME-SENTENCE"      # recoverable local miss
            elif gv.lower() in txt.lower() or sum(1 for w in toks(gv) if w in txt.lower()) >= max(1, len(toks(gv))//2):
                cat = "OTHER-SENTENCE"     # cross-sentence gold link
            else:
                cat = "GOLD-NOT-IN-SRC"    # annotation/normalization gap
            cats[cat] += 1
            if cat == "SAME-SENTENCE" and len(samples) < a.show:
                samples.append((pmid, gn, gv, fs.strip()[:200]))

    total = sum(cats.values())
    print(f"\n=== bucket-B (location extracted-but-empty) forensic  n={total}  pred={os.path.basename(a.pred)} ===")
    for c, n in cats.most_common():
        print(f"  {c:16} {n:4}  ({100*n/total:.0f}%)")
    print(f"\n--- SAME-SENTENCE samples (recoverable local misses; up to {a.show}) ---")
    for pmid, gn, gv, sent in samples:
        print(f"[{pmid}] finding={gn!r}  gold_loc={gv!r}\n    src: {sent}")

if __name__ == "__main__":
    main()
