"""Decomposition variant — targets the confirmed dominant LOCATION bucket-B sink:
anatomical site fused into the finding NAME as a pre-modifier ("abdominal pain",
"leg edema", "pulmonary edema") so `location` was left EMPTY.  Forensic
(v145_bucketB_forensic): 77% of location bucket-B misses are SAME-SENTENCE, and the
site is inside the finding phrase.  V1/V2 (sentence-scope / quote) FAILED because they
searched WIDER and grabbed distant sites (precision collapse); the real misses are the
fused pre-modifier.  Fix = the project's atomization principle: split the site OUT of
the name INTO location (and fused intensity OUT into severity).

Strategies generated here:
  V3  single-pass, atomic decomposition of fused site/intensity pre-modifiers
  C3  2-stage cot (the v145 winner) with the same decomposition rule in both stages

Scored alongside CACHED v145 preds (S/C/V1/V2) so numbers are directly comparable
(same gold, same scorer).  Run in .venv (hf backend)."""
import json, glob, re, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v143_unified_4attr import CHAR, ONSET
from v145_strategy_scope import parse_json, clean, ANN_DIR, MODEL
from score_maccrobat_attrs import parse_gold, score_one

ONSET_RULE = ONSET["O2"]

# --- decomposition attribute rules: a fused pre-modifier of the finding phrase counts ---
LOC_DECOMP = ('- location: the anatomical body site of THIS finding, copied verbatim. INCLUDE a site '
              'even when it appeared as a modifier INSIDE the finding phrase — e.g. "abdominal" from '
              '"abdominal pain", "leg" from "leg edema", "pulmonary" from "pulmonary edema", "left '
              'ventricle" from "dilated left ventricle". EXCLUDE non-anatomical words (severity, '
              'quality, size, laterality-only). If the text names no body site for this finding, "".')
SEV_DECOMP = ('- severity: one word "mild", "moderate", or "severe" for THIS symptom\'s intensity, '
              'INCLUDING when it appeared as a modifier inside the finding phrase (e.g. "severe" from '
              '"severe stenosis"). EXCLUDE disease/case-wide severity ("in severe cases"). Else "".')

STEP1_DECOMP = ('STEP 1 — List the patient\'s abnormal findings (symptoms, signs, lesions, abnormal '
                'conditions). Extract each finding ATOMICALLY: strip modifiers OFF the finding name and '
                'record them as attributes instead — an anatomical site modifier goes to location '
                '("abdominal pain" -> name "pain"; "leg edema" -> name "edema"; "dilated left ventricle" '
                '-> name "dilated"), an intensity word goes to severity ("severe stenosis" -> name '
                '"stenosis"). Keep ONLY the core finding word(s) in name. EXCLUDE tests/procedures, '
                'medications, normal findings, headings.')

HEAD_DECOMP = '''You are a clinical information extractor. Read the clinical CASE TEXT and extract, USING ONLY facts explicitly stated in it, every abnormal clinical finding the patient has, atomically, with its attributes.

CASE TEXT:
"""{src}"""

{step1}
STEP 2 — For each ATOMIC finding output its attributes (fill ONLY if the CASE TEXT states it; otherwise ""):
{onset}
{severity}
{location}
{character}

Output exactly:
JSON: {{"findings":[{{"name":"","onset":"","severity":"","location":"","character":""}}]}}'''

COT1_DECOMP = '''You are a clinical information extractor. Read the clinical CASE TEXT and list every abnormal clinical finding the patient has, ATOMICALLY. Strip modifiers off the finding name: an anatomical site ("abdominal pain" -> "pain", "leg edema" -> "edema") and an intensity word ("severe stenosis" -> "stenosis") do NOT belong in the name. Keep only the core finding word(s). EXCLUDE tests/procedures, medications, normal findings, headings.

CASE TEXT:
"""{src}"""

Output ONLY JSON: {{"findings":["...","..."]}}'''

COT2_DECOMP = '''You are a clinical information extractor. For EACH listed atomic finding, decide its attributes USING ONLY facts explicitly stated in the CASE TEXT. The finding names were atomized, so a body site or intensity that appeared next to the finding in the text IS its attribute.

CASE TEXT:
"""{src}"""

FINDINGS: {finds}

For each finding fill attributes ONLY if the CASE TEXT states it for THAT finding; otherwise "":
{onset}
{severity}
{location}
{character}

Output exactly:
JSON: {{"findings":[{{"name":"","onset":"","severity":"","location":"","character":""}}]}}'''


def build_backend(backend, bs, max_new):
    if backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85,
                  enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
        SP = SamplingParams(temperature=0.0, max_tokens=max_new)
        def chat(prompts):
            outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], SP, use_tqdm=True)
            return [o.outputs[0].text for o in outs]
        return chat
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as _AutoModel
    except Exception:
        from transformers import AutoModelForCausalLM as _AutoModel
    proc = AutoProcessor.from_pretrained(MODEL)
    tok = getattr(proc, "tokenizer", proc); tok.padding_side = "left"
    model = _AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    def chat(prompts):
        res = []
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            msgs = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in batch]
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                gen = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
            L = inp["input_ids"].shape[1]
            res += [proc.decode(g[L:], skip_special_tokens=True) for g in gen]
            print(f"  hf gen {min(i + bs, len(prompts))}/{len(prompts)}", flush=True)
        return res
    return chat


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

    head_v3 = (HEAD_DECOMP.replace("{step1}", STEP1_DECOMP).replace("{onset}", ONSET_RULE)
               .replace("{severity}", SEV_DECOMP).replace("{location}", LOC_DECOMP).replace("{character}", CHAR))
    cot2_d = (COT2_DECOMP.replace("{onset}", ONSET_RULE).replace("{severity}", SEV_DECOMP)
              .replace("{location}", LOC_DECOMP).replace("{character}", CHAR))

    chat = build_backend(a.backend, a.bs, a.max_new)

    # V3 single-pass
    txt_v3 = chat([head_v3.format(src=s) for _, s in docs])
    preds = {"V3": {pmid: clean(parse_json(t), s.lower()) for (pmid, s), t in zip(docs, txt_v3)}}

    # C3 2-stage
    t1 = chat([COT1_DECOMP.format(src=s) for _, s in docs])
    finds1 = []
    for t in t1:
        m = re.search(r'(\{.*\})', t, re.DOTALL); fl = []
        if m:
            try: fl = [str(x) for x in json.loads(m.group(1)).get("findings", []) if isinstance(x, str)]
            except: pass
        finds1.append(fl[:40])
    t2 = chat([cot2_d.format(src=s, finds=json.dumps(fl)) for (_, s), fl in zip(docs, finds1)])
    preds["C3"] = {pmid: clean(parse_json(t), s.lower()) for (pmid, s), t in zip(docs, t2)}

    for k, pd in preds.items():
        json.dump(pd, open(f"pilot/cache/maccrobat/v146_{k}_pred.json", "w"))

    # load cached v145 for side-by-side
    for k in ("S", "C", "V1", "V2"):
        p = f"pilot/cache/maccrobat/v145_{k}_pred.json"
        if os.path.exists(p): preds[k] = json.load(open(p))

    order = [k for k in ("S", "C", "V1", "V2", "V3", "C3") if k in preds]
    for attr in ("location", "severity"):
        src_type = "severity" if attr == "severity" else "biological_structure"
        gold = parse_gold(ANN_DIR, src_type)
        print(f"\n{'='*80}\n### {attr.upper()}  (gold pairs={sum(len(v) for v in gold.values())})")
        for k in order:
            r = score_one(preds[k], gold, attr)
            Pb, Rb, Fb = r["bind"]; Pv, Rv, Fv = r["val"]
            tag = " <-NEW" if k in ("V3", "C3") else ""
            print(f"[{k:3}] pred={r['NP']:4} | binding F1={Fb:.2f} ({Pb:.2f}/{Rb:.2f})  value F1={Fv:.2f} "
                  f"| A={r['A']} B={r['B']} C={r['C']} D={r['D']}/{r['NP']}{tag}", flush=True)


if __name__ == "__main__":
    main()
