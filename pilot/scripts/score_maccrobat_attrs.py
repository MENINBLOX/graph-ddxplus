"""Score location/severity extraction on MACCROBAT relation gold, with error-bucket
decomposition. Stdlib only — runs under system python3, no model/torch needed.

Gold = MACCROBAT MODIFY(Severity->Sign_symptom) and MODIFY(Biological_structure->Sign_symptom).
Metric = binding P/R/F1 (right symptom got an attr) + value-aware P/R/F1 (value also right),
identical scoring to v144, PLUS a decomposition of where recall/precision is lost:
  A  symptom not extracted at all           (finding-recall loss)
  B  symptom extracted but attr left EMPTY   (attr-assignment recall loss)  <- usual dominant sink
  C  binding matched but VALUE wrong         (normalization loss)
  D  predicted attr with no gold symptom     (precision FP)

Usage:
  python3 score_maccrobat_attrs.py --pred single=pilot/cache/maccrobat/v144_single_pred.json \
                                   --pred cot=pilot/cache/maccrobat/v144_cot_pred.json
"""
import json, glob, re, os, sys, argparse
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_GOLD = "pilot/cache/maccrobat/brat"
STOP = set("the a an of to in on with and or for is are be may can at as by from his her their patient left right".split())

def toks(s): return {w for w in re.findall(r'[a-z]+', s.lower()) if w not in STOP and len(w) > 2}
def sym_match(a, b):
    ta, tb = toks(a), toks(b)
    return bool(ta and tb and (a.lower() in b.lower() or b.lower() in a.lower() or ta & tb))
def val_match(pv, gv):
    return bool(pv) and bool(gv) and (pv in gv or gv in pv or bool(toks(pv) & toks(gv)))

def parse_gold(gold_dir, src_type):
    """MODIFY(src_type -> sign_symptom) gold pairs per pmid: (symptom_text, attr_text)."""
    g = defaultdict(list)
    for ann in glob.glob(f"{gold_dir}/*.ann"):
        pmid = os.path.basename(ann)[:-4]
        L = open(ann, encoding="utf-8").read().splitlines(); ents = {}; evs = {}
        for line in L:
            p = line.split("\t")
            if p[0].startswith("T") and len(p) >= 3: ents[p[0]] = (p[1].split()[0].lower(), p[2])
            elif p[0].startswith("E") and len(p) >= 2:
                t = p[1].split()[0].split(":"); evs[p[0]] = (t[0].lower(), t[1])
        def tt(arg):
            if arg in ents: return ents[arg]
            if arg in evs: return (evs[arg][0], ents.get(evs[arg][1], ("", ""))[1])
            return ("", "")
        for line in L:
            p = line.split("\t")
            if p[0].startswith("R") and len(p) >= 2 and p[1].split()[0] == "MODIFY":
                m = re.findall(r'Arg\d:(\w+)', p[1])
                if len(m) != 2: continue
                if ents.get(m[0], ("", ""))[0] == src_type:
                    typ, symtext = tt(m[1])
                    if typ == "sign_symptom" and symtext:
                        g[pmid].append((symtext.lower(), ents[m[0]][1].lower()))
    return g

def score_one(preds, gold, attr):
    NP = NG = TPb = TPv = A = B = C = D = 0
    val_ex = []
    for pmid, gp in gold.items():
        doc = preds.get(pmid, [])
        names = [x["name"] for x in doc]
        pp = [(x["name"], x[attr]) for x in doc if x.get(attr)]
        NP += len(pp); NG += len(gp)
        ub = [False]*len(gp); uv = [False]*len(gp)
        for pn, pv in pp:
            for j, (gn, gv) in enumerate(gp):
                if not ub[j] and sym_match(pn, gn): ub[j] = True; TPb += 1; break
            for j, (gn, gv) in enumerate(gp):
                if not uv[j] and sym_match(pn, gn) and val_match(pv, gv): uv[j] = True; TPv += 1; break
        for j, (gn, gv) in enumerate(gp):
            if ub[j]:
                if not uv[j]:
                    C += 1
                    for pn, pv in pp:
                        if sym_match(pn, gn): val_ex.append((pv, gv)); break
                continue
            if any(sym_match(an, gn) for an in names): B += 1
            else: A += 1
        for pn, pv in pp:
            if not any(sym_match(pn, gn) for gn, gv in gp): D += 1
    def prf(tp): P = tp/NP if NP else 0; R = tp/NG if NG else 0; return P, R, (2*P*R/(P+R) if P+R else 0)
    return {"NP": NP, "NG": NG, "bind": prf(TPb), "val": prf(TPv),
            "A": A, "B": B, "C": C, "D": D, "bound": TPb, "val_ex": val_ex}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True, help="label=path.json (repeatable)")
    ap.add_argument("--gold-dir", default=DEFAULT_GOLD)
    ap.add_argument("--attrs", nargs="+", default=["location", "severity"])
    a = ap.parse_args()
    preds = {}
    for spec in a.pred:
        label, path = spec.split("=", 1)
        preds[label] = json.load(open(path, encoding="utf-8"))
    for attr in a.attrs:
        src_type = "severity" if attr == "severity" else "biological_structure"
        gold = parse_gold(a.gold_dir, src_type)
        print(f"\n{'='*80}\n### {attr.upper()}  (gold pairs={sum(len(v) for v in gold.values())})")
        for label, pd in preds.items():
            r = score_one(pd, gold, attr)
            Pb, Rb, Fb = r["bind"]; Pv, Rv, Fv = r["val"]
            print(f"[{label:16}] pred={r['NP']:4} gold={r['NG']:4} | "
                  f"binding P/R/F1={Pb:.2f}/{Rb:.2f}/{Fb:.2f}  value={Pv:.2f}/{Rv:.2f}/{Fv:.2f}")
            print(f"   buckets: A(sym-miss)={r['A']}  B(attr-empty)={r['B']}  "
                  f"C(val-wrong)={r['C']}  D(precision-FP)={r['D']}/{r['NP']}  bound={r['bound']}")

if __name__ == "__main__":
    main()
