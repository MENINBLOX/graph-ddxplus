"""PRODUCTION IE stage of the large-scale attribute pipeline (context B = disease -> phenotypes).

Input = a disease's clinical description text (v105_sources/<CUI>.txt); output = atomic
findings + 6 candidate attributes. Reuses the VALIDATED recipe from MACCROBAT tuning:
gemma-4-E4B, single prompt, STEP1 site-atomization, conservative severity, 6-attribute schema
(v147 V4) — but re-framed for DISEASE descriptions ("symptoms a patient with this disease can
have") instead of case reports ("the patient has"). Benchmark-blind, no few-shot, temp=0.

Raw severity/onset are extracted VERBATIM here (faithful); the deterministic normalization to
controlled vocab (incl. 0-10 NRS) happens in the normalization layer (v152/v154), not here.

Usage: python v153_production_ie.py --n 15 [--out preds.json]   (run in .venv)"""
import json, glob, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v143_unified_4attr import SEV_R2, CHAR, ONSET
from v146_decomp import LOC_DECOMP, build_backend
from v147_single6 import TIMING, AGGR, RELV
from v145_strategy_scope import parse_json, clean

SRC_V105 = "pilot/cache/v105_sources"          # Wikipedia intros (symptom-poor)
SRC_SCALEUP = "pilot/cache/scaleup_sources"    # symptom-rich "Signs and symptoms" sections
SEED = "data/medkg/seeds/combined_seed.jsonl"
MODEL = "google/gemma-4-E4B-it"
ONSET_RULE = ONSET["O2"]

STEP1_PROD = ('STEP 1 — List the abnormal clinical findings (symptoms, signs, phenotypes) the SOURCE '
              'attributes to patients with this disease, ATOMICALLY. When a finding phrase contains an '
              'anatomical SITE as a modifier, strip the site OFF the name and keep ONLY the core finding '
              '("abdominal pain" -> "pain", "hepatic fibrosis" -> "fibrosis"); the site goes to location '
              'in STEP 2. Do NOT strip intensity/quality words. EXCLUDE the disease name itself, causes/'
              'risk factors, epidemiology, treatments/drugs, diagnostics, normal findings, and headings.')

HEAD_PROD = '''You are a clinical information extractor. Read the SOURCE describing the disease "{disease}" and extract, USING ONLY facts explicitly stated in the SOURCE, the symptoms/signs/phenotypes a patient with this disease can have, atomically, with their attributes.

SOURCE:
"""{src}"""

{step1}
STEP 2 — For each ATOMIC finding output its attributes (fill ONLY if the SOURCE states it for THAT finding; otherwise ""):
{onset}
{severity}
{location}
{character}
{timing}
{aggravating}
{relieving}

Output exactly:
JSON: {{"findings":[{{"name":"","onset":"","severity":"","location":"","character":"","timing":"","aggravating":"","relieving":""}}]}}'''


def load_names():
    m = {}
    for line in open(SEED, encoding="utf-8"):
        try:
            d = json.loads(line); m[d["cui"]] = d["name"]
        except Exception:
            pass
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--source", choices=["v105", "scaleup"], default="scaleup")
    ap.add_argument("--out", default="pilot/cache/prod/ie_pred.json")
    a = ap.parse_args()
    docs = []  # (key, disease_name, source_text)
    if a.source == "v105":
        names = load_names()
        for p in sorted(glob.glob(f"{SRC_V105}/*.txt"))[:a.n]:
            cui = os.path.basename(p)[:-4]
            docs.append((cui, names.get(cui, cui), open(p, encoding="utf-8").read()))
    else:  # scaleup: symptom-rich sections, keyed by index; {disease,title,text}
        for p in sorted(glob.glob(f"{SRC_SCALEUP}/*.json"))[:a.n]:
            d = json.load(open(p, encoding="utf-8"))
            key = os.path.basename(p)[:-5]
            docs.append((key, d.get("disease", key), d.get("text", "")))

    prompt = (HEAD_PROD.replace("{step1}", STEP1_PROD).replace("{onset}", ONSET_RULE)
              .replace("{severity}", SEV_R2).replace("{location}", LOC_DECOMP).replace("{character}", CHAR)
              .replace("{timing}", TIMING).replace("{aggravating}", AGGR).replace("{relieving}", RELV))

    chat = build_backend("hf", a.bs, a.max_new)
    texts = chat([prompt.replace("{disease}", nm).replace("{src}", s) for _, nm, s in docs])
    preds = {}
    for (cui, nm, s), t in zip(docs, texts):
        preds[cui] = {"disease": nm, "source": s, "findings": clean(parse_json(t), s.lower())}
    json.dump(preds, open(a.out, "w"))
    # quick stats
    nf = sum(len(v["findings"]) for v in preds.values())
    filled = {k: 0 for k in ("onset", "severity", "location", "character")}
    for v in preds.values():
        for f in v["findings"]:
            for k in filled:
                if str(f.get(k, "")).strip(): filled[k] += 1
    print(f"diseases={len(preds)} findings={nf} ({nf/max(1,len(preds)):.1f}/disease)", flush=True)
    print("attr fill:", {k: f"{filled[k]}/{nf} ({filled[k]/max(1,nf):.0%})" for k in filled}, flush=True)
    print(f"saved -> {a.out}", flush=True)
    for cui, v in list(preds.items())[:2]:
        print(f"\n[{cui}] {v['disease']}: {[f['name'] for f in v['findings']][:12]}")


if __name__ == "__main__":
    main()
