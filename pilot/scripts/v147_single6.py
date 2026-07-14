"""V4 — the production SINGLE-PROMPT architecture: one LLM call per document extracts all
6 candidate attributes at once via internal STEP1(list)->STEP2(assign) reasoning (NOT
multiple calls, NOT per-attribute sequential).

Design locked from the v145/v146 forensics:
  * location: ATOMIZE — strip an anatomical SITE pre-modifier off the finding name into
    location ("abdominal pain"->name "pain"). This was V3's win (adjF1 0.64).
  * severity: CONSERVATIVE — do NOT decompose intensity (V3's aggressive intensity-stripping
    over-triggered severity to F1 0.46). Use the validated SEV_R2 rule (S/C got 0.61-0.68;
    the 0.61 vs 0.68 gap on 39 gold pairs is ~2 pairs = noise).
  * onset/character + timing/aggravating/relieving: extracted in the SAME prompt; only
    location/severity have public gold, the other 4 are extrinsic-only (not in paper).

Validates on MACCROBAT case text (gold exists), scored + source-adjudicated against cached
v145 S/C and v146 V3. Goal: single prompt reaches location ~0.64 AND severity ~0.61+."""
import json, glob, re, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v143_unified_4attr import SEV_R2, CHAR, ONSET          # validated conservative severity
from v146_decomp import LOC_DECOMP, build_backend           # winning location decomposition
from v145_strategy_scope import parse_json, clean, ANN_DIR, MODEL
from score_maccrobat_attrs import parse_gold, score_one

ONSET_RULE = ONSET["O2"]
# new free-text attributes (no public gold; extrinsic-only)
TIMING = ('- timing: the temporal pattern the text states for THIS finding (e.g. "at night", '
          '"after meals", "intermittent", "for 3 days"), copied briefly; else "".')
AGGR = ('- aggravating: a factor the text says WORSENS THIS finding (e.g. "exertion", "lying '
        'down", "cold air"); else "".')
RELV = ('- relieving: a factor the text says RELIEVES THIS finding (e.g. "rest", "antacids", '
        '"sitting up"); else "".')

STEP1_SITE = ('STEP 1 — List the patient\'s abnormal findings (symptoms, signs, lesions, abnormal '
              'conditions) ATOMICALLY. When a finding phrase contains an anatomical SITE as a modifier, '
              'strip the site OFF the name and keep ONLY the core finding — "abdominal pain" -> name '
              '"pain", "leg edema" -> name "edema", "dilated left ventricle" -> name "dilated"; the site '
              'is recorded in location at STEP 2. Do NOT strip intensity or quality words from the name. '
              'EXCLUDE tests/procedures, medications, normal findings, headings.')

HEAD_V4 = '''You are a clinical information extractor. Read the clinical CASE TEXT and extract, USING ONLY facts explicitly stated in it, every abnormal clinical finding the patient has, atomically, with its attributes.

CASE TEXT:
"""{src}"""

{step1}
STEP 2 — For each ATOMIC finding output its attributes (fill ONLY if the CASE TEXT states it for THAT finding; otherwise ""):
{onset}
{severity}
{location}
{character}
{timing}
{aggravating}
{relieving}

Output exactly:
JSON: {{"findings":[{{"name":"","onset":"","severity":"","location":"","character":"","timing":"","aggravating":"","relieving":""}}]}}'''


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="hf")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=2048)
    a = ap.parse_args()
    txts = sorted(glob.glob(f"{ANN_DIR}/*.txt"))[:a.n]
    docs = [(os.path.basename(t)[:-4], open(t, encoding="utf-8").read()) for t in txts]

    head = (HEAD_V4.replace("{step1}", STEP1_SITE).replace("{onset}", ONSET_RULE)
            .replace("{severity}", SEV_R2).replace("{location}", LOC_DECOMP).replace("{character}", CHAR)
            .replace("{timing}", TIMING).replace("{aggravating}", AGGR).replace("{relieving}", RELV))

    chat = build_backend(a.backend, a.bs, a.max_new)
    texts = chat([head.format(src=s) for _, s in docs])
    preds = {"V4": {pmid: clean(parse_json(t), s.lower()) for (pmid, s), t in zip(docs, texts)}}
    json.dump(preds["V4"], open("pilot/cache/maccrobat/v147_V4_pred.json", "w"))

    for k in ("S", "C", "V3"):
        p = (f"pilot/cache/maccrobat/v145_{k}_pred.json" if k in ("S", "C")
             else f"pilot/cache/maccrobat/v146_{k}_pred.json")
        if os.path.exists(p): preds[k] = json.load(open(p))

    order = [k for k in ("S", "C", "V3", "V4") if k in preds]
    for attr in ("location", "severity"):
        src_type = "severity" if attr == "severity" else "biological_structure"
        gold = parse_gold(ANN_DIR, src_type)
        print(f"\n{'='*80}\n### {attr.upper()}  (gold pairs={sum(len(v) for v in gold.values())})")
        for k in order:
            r = score_one(preds[k], gold, attr)
            Pb, Rb, Fb = r["bind"]; Pv, Rv, Fv = r["val"]
            tag = " <-SINGLE-PROMPT" if k == "V4" else ""
            print(f"[{k:3}] pred={r['NP']:4} | binding F1={Fb:.2f} ({Pb:.2f}/{Rb:.2f})  value F1={Fv:.2f} "
                  f"| A={r['A']} B={r['B']} C={r['C']} D={r['D']}/{r['NP']}{tag}", flush=True)


if __name__ == "__main__":
    main()
