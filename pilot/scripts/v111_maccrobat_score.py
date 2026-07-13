"""Standard mention-level NER P/R/F1 of our IE output vs MACCROBAT gold.
Recognized metric: per-entity-type precision/recall/F1, micro-averaged over docs,
with greedy one-to-one mention matching. Reports STRICT (exact normalized string)
and RELAXED (token-overlap) — both are standard in clinical IE evaluation.

Our extractor is ZERO-SHOT (no MACCROBAT training); supervised fine-tuned token
classifiers are the usual baselines. Comparison is reported with that caveat.
"""
import json, re, argparse
from collections import defaultdict

STOP=set("the a an of to in on with and or for is are be may can at as by from "
         "his her their patient left right both bilateral".split())
def norm(s): return re.sub(r'\s+',' ',str(s).strip().lower())
def toks(s): return {w for w in re.findall(r'[a-z0-9]+',norm(s)) if w not in STOP and len(w)>1}

def match(pred_list, gold_list, relaxed):
    """greedy one-to-one; returns (tp, n_pred, n_gold)."""
    gold=[norm(g) for g in gold_list]; pred=[norm(p) for p in pred_list]
    used=[False]*len(gold); tp=0
    for p in pred:
        pt=toks(p)
        for j,g in enumerate(gold):
            if used[j]: continue
            if (p==g) if not relaxed else (p==g or (pt and toks(g) and (pt&toks(g)) and (p in g or g in p or len(pt&toks(g))>=max(1,min(len(pt),len(toks(g))))))):
                used[j]=True; tp+=1; break
    return tp, len(pred), len(gold)

def prf(tp,np_,ng):
    P=tp/np_ if np_ else 0.0; R=tp/ng if ng else 0.0
    F=2*P*R/(P+R) if (P+R) else 0.0; return P,R,F

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="pilot/data/cache/maccrobat/MACCROBAT2020-V2.json")
    ap.add_argument("--pred",default="pilot/data/cache/maccrobat/v111_pred.json")
    a=ap.parse_args()
    docs=json.load(open(a.data))["data"]
    preds=json.load(open(a.pred))
    assert len(docs)==len(preds), f"{len(docs)} vs {len(preds)}"

    TYPES={"SIGN_SYMPTOM":"name","SEVERITY":"severity","BIOLOGICAL_STRUCTURE":"location"}
    for mode,relaxed in [("STRICT",False),("RELAXED",True)]:
        print(f"\n=== {mode} mention-level micro P/R/F1 (zero-shot IE vs MACCROBAT gold, 200 docs) ===")
        print(f"{'type':<24}{'gold':>7}{'pred':>7}{'TP':>6}{'P':>8}{'R':>8}{'F1':>8}")
        for gtype,pkey in TYPES.items():
            TP=NP=NG=0
            for d,p in zip(docs,preds):
                gold=[e["text"] for e in d["ner_info"] if e["label"]==gtype]
                if pkey=="name": pr=[x["name"] for x in p]
                else: pr=[x[pkey] for x in p if x.get(pkey)]
                tp,np_,ng=match(pr,gold,relaxed); TP+=tp; NP+=np_; NG+=ng
            P,R,F=prf(TP,NP,NG)
            print(f"{gtype:<24}{NG:>7}{NP:>7}{TP:>6}{P:>8.3f}{R:>8.3f}{F:>8.3f}")

if __name__=="__main__":
    main()
