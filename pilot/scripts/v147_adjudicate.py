"""Source-adjudicated scoring for BOTH location and severity, across S/C/V3/V4 — corrects
MACCROBAT relation-gold incompleteness (verified 0% hallucination, ~85% of nominal FPs
correct-but-unannotated for location). Same uniform bias-correction rule as v146:
a prediction counts correct if it matches a gold pair OR its value is stated in the
finding's own sentence. Recall stays gold-bound. Stdlib only."""
import json, re, os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_maccrobat_attrs import parse_gold, score_one, toks, sym_match

ANN_DIR = "pilot/cache/maccrobat/brat"
FILES = {"S": "v145_S", "C": "v145_C", "V3": "v146_V3", "V4": "v147_V4"}

INTENS = {  # source intensity terms -> normalized severity, for severity adjudication
    "severe": "severe", "severely": "severe", "marked": "severe", "markedly": "severe",
    "florid": "severe", "massive": "severe", "profound": "severe", "extensive": "severe",
    "moderate": "moderate", "moderately": "moderate",
    "mild": "mild", "mildly": "mild", "slight": "mild", "slightly": "mild",
    "minimal": "mild", "faint": "mild",
}

def sentences(txt): return [s for s in re.split(r'(?<=[.\n])\s+', txt) if s.strip()]
def loc_stated(pv, sent):
    lt = toks(pv)
    return bool(lt) and (pv.lower() in sent or sum(1 for w in lt if w in sent) >= max(1, len(lt)//2))
def sev_stated(pv, sent):
    words = set(re.findall(r'[a-z]+', sent))
    return any(term in words and norm == pv for term, norm in INTENS.items())

def adjudicate(preds, gold, attr):
    r = score_one(preds, gold, attr)
    NP, NG, TPb = r["NP"], r["NG"], r["bound"]
    extra = 0
    for pmid, gp in gold.items():
        doc = preds.get(pmid, [])
        pp = [(x["name"], x[attr]) for x in doc if x.get(attr)]
        txt = ""
        tp = f"{ANN_DIR}/{pmid}.txt"
        if os.path.exists(tp): txt = open(tp, encoding="utf-8").read()
        sents = sentences(txt)
        for pn, pv in pp:
            if any(sym_match(pn, gn) for gn, gv in gp):
                continue
            fs = next((s for s in sents if sym_match(pn, s)), None)
            if fs is None: continue
            sl = fs.lower()
            if (attr == "location" and loc_stated(pv, sl)) or (attr == "severity" and sev_stated(pv, sl)):
                extra += 1
    P_adj = (TPb + extra) / NP if NP else 0
    R = TPb / NG if NG else 0
    F = 2 * P_adj * R / (P_adj + R) if P_adj + R else 0
    return r["bind"], (P_adj, R, F), r["B"]

def main():
    preds = {k: json.load(open(f"pilot/cache/maccrobat/{v}_pred.json")) for k, v in FILES.items()
             if os.path.exists(f"pilot/cache/maccrobat/{v}_pred.json")}
    for attr in ("location", "severity"):
        src_type = "severity" if attr == "severity" else "biological_structure"
        gold = parse_gold(ANN_DIR, src_type)
        ng = sum(len(v) for v in gold.values())
        print(f"\n### {attr.upper()} adjudicated  (gold pairs={ng})")
        print(f"{'strat':5} {'rawP':>5} {'rawR':>5} {'rawF1':>6} | {'adjP':>5} {'adjR':>5} {'adjF1':>6} | {'bktB':>5}")
        for k in preds:
            (Pb, Rb, Fb), (Pa, Ra, Fa), B = adjudicate(preds[k], gold, attr)
            tag = "  <-SINGLE" if k == "V4" else ""
            print(f"{k:5} {Pb:5.2f} {Rb:5.2f} {Fb:6.2f} | {Pa:5.2f} {Ra:5.2f} {Fa:6.2f} | {B:5}{tag}")

if __name__ == "__main__":
    main()
