"""Source-adjudicated LOCATION scoring — corrects MACCROBAT relation-gold incompleteness.

Forensic finding (v146_D_forensic, all strategies): 0% of location predictions are
hallucinated (NOT-IN-SRC); 83-87% of nominal FPs are STATED-IN-SRC (correct but never
annotated in the sparse MODIFY gold). Raw binding-F1 therefore measures gold coverage,
not extraction quality. Here we recompute precision by counting a prediction as correct
if it EITHER matches a gold pair OR its location is stated in the finding's own sentence
(STATED-IN-SRC). Recall stays against gold (unchanged). Applied uniformly to every
strategy — this is bias correction by source verification, not per-strategy tuning.

Reports raw vs adjudicated P / R / F1 side by side. Stdlib only."""
import json, re, os, argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_maccrobat_attrs import parse_gold, score_one, toks, sym_match

ANN_DIR = "pilot/cache/maccrobat/brat"

def sentences(txt): return [s for s in re.split(r'(?<=[.\n])\s+', txt) if s.strip()]
def loc_in(loc, sl):
    lt = toks(loc)
    return bool(lt) and (loc.lower() in sl or sum(1 for w in lt if w in sl) >= max(1, len(lt)//2))

def adjudicate(preds, gold, lenient=False):
    """Return adjudicated (P, R, F1) for location. TP_adj = preds matching gold OR stated-in-src
    (lenient also credits site-elsewhere-in-doc). NP, and gold recall from score_one."""
    r = score_one(preds, gold, "location")
    NP, NG, TPb = r["NP"], r["NG"], r["bound"]
    correct_extra = 0
    for pmid, gp in gold.items():
        doc = preds.get(pmid, [])
        pp = [(x["name"], x["location"]) for x in doc if x.get("location")]
        txt = ""
        tp = f"{ANN_DIR}/{pmid}.txt"
        if os.path.exists(tp): txt = open(tp, encoding="utf-8").read()
        sl = txt.lower(); sents = sentences(txt)
        for pn, pv in pp:
            if any(sym_match(pn, gn) for gn, gv in gp):
                continue  # already counted toward gold precision
            fs = next((s for s in sents if sym_match(pn, s)), None)
            if fs is not None and loc_in(pv, fs.lower()):
                correct_extra += 1
            elif lenient and loc_in(pv, sl):
                correct_extra += 1
    P_adj = (TPb + correct_extra) / NP if NP else 0
    R = TPb / NG if NG else 0
    F = 2 * P_adj * R / (P_adj + R) if P_adj + R else 0
    return P_adj, R, F, NP

def main():
    files = {"S": "v145_S", "C": "v145_C", "V1": "v145_V1", "V2": "v145_V2",
             "V3": "v146_V3", "C3": "v146_C3"}
    gold = parse_gold(ANN_DIR, "biological_structure")
    ng = sum(len(v) for v in gold.values())
    print(f"### LOCATION adjudicated  (gold pairs={ng})")
    print(f"{'strat':5} {'rawP':>5} {'rawR':>5} {'rawF1':>6} | {'adjP':>5} {'adjP+':>5} {'adjF1':>6} {'adjF1+':>7} | {'bucketB':>7}")
    for k, stem in files.items():
        p = f"pilot/cache/maccrobat/{stem}_pred.json"
        if not os.path.exists(p): continue
        preds = json.load(open(p, encoding="utf-8"))
        r = score_one(preds, gold, "location")
        Pb, Rb, Fb = r["bind"]
        Pa, Ra, Fa, NP = adjudicate(preds, gold, lenient=False)
        Pa2, _, Fa2, _ = adjudicate(preds, gold, lenient=True)
        print(f"{k:5} {Pb:5.2f} {Rb:5.2f} {Fb:6.2f} | {Pa:5.2f} {Pa2:5.2f} {Fa:6.2f} {Fa2:7.2f} | {r['B']:7}")
    print("\nrawP/rawR/rawF1 = vs sparse gold; adjP = +stated-in-src correct; adjP+ = +site-elsewhere;")
    print("adjF1 = harmonic(adjP, gold recall). Recall stays gold-bound (sparse gold caps it).")

if __name__ == "__main__":
    main()
