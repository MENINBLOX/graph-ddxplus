"""Test the PROMPT-SCOPE hypothesis (user): the location recall ceiling on MACCROBAT is not
model/corpus-bound but caused by the V4 prompt NARROWING extraction to patient symptoms and
EXCLUDING test/imaging/pathology findings that the gold DOES annotate. A-forensic showed 98%
of recall misses are imaging/pathology observations (necrosis, metastatic lesions, cholestasis).

V5 = V4 with a BROADENED STEP1: extract every abnormal finding INCLUDING imaging, pathology/
histology, and lab/exam abnormalities (a finding counts even if discovered by a test); exclude
only the test/procedure/med NAMES, normal findings, headings. Same atomization + attribute
rules. Same model (gemma-4-E4B). If recall/A-bucket recovers -> it was a prompt problem.

Scores V4 vs V5 side by side (location + severity, raw + adjudicated + buckets)."""
import json, glob, re, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v143_unified_4attr import SEV_R2, CHAR, ONSET
from v146_decomp import LOC_DECOMP, build_backend
from v147_single6 import HEAD_V4, TIMING, AGGR, RELV
from v145_strategy_scope import parse_json, clean, ANN_DIR, MODEL
from v147_adjudicate import adjudicate
from score_maccrobat_attrs import parse_gold, score_one

ONSET_RULE = ONSET["O2"]

STEP1_BROAD = ('STEP 1 — List EVERY abnormal finding described ANYWHERE in the case, ATOMICALLY. '
               'Include not only patient-reported symptoms and signs, but ALSO abnormalities seen on '
               'IMAGING (e.g. mass, lesion, opacity, effusion, nodule, dilation, thickening), '
               'PATHOLOGY/HISTOLOGY (e.g. necrosis, fibrosis, steatosis, atrophy, tumor cells, '
               'infiltration), and EXAMINATION/LAB abnormalities. A finding counts EVEN IF it was '
               'discovered by a test, scan, or biopsy — extract the abnormal finding itself, not the '
               'test. When a finding phrase contains an anatomical SITE as a modifier, strip the site '
               'OFF the name and keep only the core finding ("abdominal pain" -> "pain", "hepatic '
               'necrosis" -> "necrosis"); the site goes to location in STEP 2. Do NOT strip intensity/'
               'quality words. EXCLUDE ONLY: the test/procedure/medication NAMES themselves, findings '
               'explicitly stated as normal, and section headings.')

def build_prompt(step1):
    return (HEAD_V4.replace("{step1}", step1).replace("{onset}", ONSET_RULE)
            .replace("{severity}", SEV_R2).replace("{location}", LOC_DECOMP).replace("{character}", CHAR)
            .replace("{timing}", TIMING).replace("{aggravating}", AGGR).replace("{relieving}", RELV))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=2048)
    a = ap.parse_args()
    txts = sorted(glob.glob(f"{ANN_DIR}/*.txt"))[:a.n]
    docs = [(os.path.basename(t)[:-4], open(t, encoding="utf-8").read()) for t in txts]
    head_v5 = build_prompt(STEP1_BROAD)

    chat = build_backend("hf", a.bs, a.max_new)
    texts = chat([head_v5.format(src=s) for _, s in docs])
    preds = {"V5": {pmid: clean(parse_json(t), s.lower()) for (pmid, s), t in zip(docs, texts)}}
    json.dump(preds["V5"], open("pilot/cache/maccrobat/v149_V5_pred.json", "w"))
    # V4 baseline (same model, narrow scope) for side-by-side
    if os.path.exists("pilot/cache/maccrobat/v147_V4_pred.json"):
        preds["V4"] = json.load(open("pilot/cache/maccrobat/v147_V4_pred.json"))

    for attr in ("location", "severity"):
        gold = parse_gold(ANN_DIR, "severity" if attr == "severity" else "biological_structure")
        print(f"\n### {attr.upper()}  (gold pairs={sum(len(v) for v in gold.values())})")
        for k in ("V4", "V5"):
            if k not in preds: continue
            r = score_one(preds[k], gold, attr)
            (Pb, Rb, Fb), (Pa, Ra, Fa), B = adjudicate(preds[k], gold, attr)
            tag = " <-BROAD" if k == "V5" else ""
            print(f"[{k}] raw F1={Fb:.2f} ({Pb:.2f}/{Rb:.2f}) | adj F1={Fa:.2f} ({Pa:.2f}/{Ra:.2f}) "
                  f"| A={r['A']} B={r['B']} C={r['C']} D={r['D']}/{r['NP']}{tag}", flush=True)

if __name__ == "__main__":
    main()
