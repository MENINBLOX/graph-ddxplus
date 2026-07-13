"""Standard mention-level P/R/F1 of our IE output vs CADEC gold.
Two reference targets:
  (a) verbatim ADR spans (standard NER span eval) — lay colloquial phrasing.
  (b) MedDRA Preferred Term normalization — concept-level match (fairer to a
      normalizing extractor that outputs medical terms).
Strict (exact) + relaxed (token-overlap). Zero-shot extractor; supervised
CADEC NER baselines (e.g. CRF/BERT) report F1 ~0.6-0.7 on ADR spans."""
import json, re

STOP=set("the a an of to in on with and or for is are be may can at as by from "
         "his her their patient i you my me could not no had have was were been".split())
def norm(s): return re.sub(r'\s+',' ',str(s).strip().lower())
def toks(s): return {w for w in re.findall(r'[a-z0-9]+',norm(s)) if w not in STOP and len(w)>1}

def match(pred_list, gold_list, relaxed):
    gold=[norm(g) for g in gold_list]; pred=[norm(p) for p in pred_list]
    used=[False]*len(gold); tp=0
    for p in pred:
        pt=toks(p)
        for j,g in enumerate(gold):
            if used[j]: continue
            ok=(p==g) if not relaxed else (p==g or (pt and toks(g) and (pt&toks(g)) and (p in g or g in p or len(pt&toks(g))>=max(1,min(len(pt),len(toks(g)))))))
            if ok: used[j]=True; tp+=1; break
    return tp, len(pred), len(gold)

def prf(tp,np_,ng):
    P=tp/np_ if np_ else 0.0; R=tp/ng if ng else 0.0
    return P,R,(2*P*R/(P+R) if (P+R) else 0.0)

def main():
    gold=json.load(open("pilot/data/cache/cadec/gold.json"))
    preds=json.load(open("pilot/data/cache/cadec/v111_pred.json"))
    assert len(gold)==len(preds)
    print(f"=== CADEC mention-level micro P/R/F1 (zero-shot IE, {len(gold)} patient posts) ===")
    for tgt,key in [("ADR verbatim span","ade"),("MedDRA PT (normalized)","pt")]:
        for mode,relaxed in [("STRICT",False),("RELAXED",True)]:
            TP=NP=NG=0
            for g,p in zip(gold,preds):
                tp,np_,ng=match(p,g[key],relaxed); TP+=tp; NP+=np_; NG+=ng
            P,R,F=prf(TP,NP,NG)
            print(f"  {tgt:<24} {mode:<8} gold={NG:>5} pred={NP:>5} TP={TP:>5}  P={P:.3f} R={R:.3f} F1={F:.3f}")

if __name__=="__main__": main()
