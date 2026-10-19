"""4-way IE extraction-STRATEGY comparison on MACCROBAT (model FIXED = gemma-4-E4B-it).
Targets the dominant F1 sink found by forensic decomposition: bucket B = "symptom
extracted but attribute left EMPTY", caused by over-strict adjacency in LOC_L1/SEV_R2.

Strategies (all temp=0, source-grounded, NO few-shot):
  S   single           v144 HEAD_SINGLE + LOC_L1/SEV_R2 (baseline)
  C   cot              v144 2-stage list-then-assign      (baseline)
  V1  sentence-scope   single, but location/severity may bind within the SAME
                       sentence/clause (not just immediately adjacent) -> recover bucket B
  V2  quote-grounded   single, per finding QUOTE the source span (or NONE) THEN normalize
                       -> forces sentence search (recall) while grounding (precision guard)

Run in .venv (vllm). Writes preds to pilot/cache/maccrobat/v145_<S>_pred.json and prints
binding/value F1 + A/B/C/D buckets per strategy (via score_maccrobat_attrs)."""
import json, glob, re, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v143_unified_4attr import SEV_R2, LOC_L1, CHAR, ONSET
from score_maccrobat_attrs import parse_gold, score_one

ANN_DIR = "pilot/cache/maccrobat/brat"
MODEL = "google/gemma-4-E4B-it"
ONSET_RULE = ONSET["O2"]

# ---- relaxed rules (V1): sentence/clause scope instead of strict adjacency ----
LOC_V1 = ('- location: the anatomical body site the SOURCE associates with THIS finding within the '
          'SAME sentence or clause (it need NOT sit immediately next to the finding word), copied '
          'verbatim (e.g. "left adrenal gland","leg","cheeks"). Include it when the sentence makes '
          'clear this is where the finding is. EXCLUDE non-anatomical locations. If the finding\'s '
          'sentence names no body site for it, leave EMPTY.')
SEV_V1 = ('- severity: one word "mild", "moderate", or "severe" when the SOURCE states the intensity '
          'of THIS symptom anywhere in its own sentence/clause (need NOT be adjacent). EXCLUDE '
          'severity describing the disease/case/episode as a whole (e.g. "in severe cases", "severe '
          'disease"). If this symptom\'s own intensity is not stated, leave EMPTY.')

HEAD_SINGLE = '''You are a clinical information extractor. Read the clinical CASE TEXT and extract, USING ONLY facts explicitly stated in it, every abnormal clinical finding the patient has, with its attributes.

CASE TEXT:
"""{src}"""

STEP 1 — Reasoning: list the patient's abnormal findings (symptoms, signs, lesions, abnormal conditions). EXCLUDE tests/procedures, medications, normal findings, headings.
STEP 2 — For each finding output its attributes (fill ONLY if the CASE TEXT states it; otherwise ""):
{onset}
{severity}
{location}
{character}

Output exactly:
JSON: {{"findings":[{{"name":"","onset":"","severity":"","location":"","character":""}}]}}'''

COT1 = '''You are a clinical information extractor. Read the clinical CASE TEXT and list every abnormal clinical finding the patient has (symptom, sign, lesion, abnormal condition). EXCLUDE tests/procedures, medications, normal findings, headings.

CASE TEXT:
"""{src}"""

Output ONLY JSON: {{"findings":["...","..."]}}'''

COT2 = '''You are a clinical information extractor. For EACH listed finding, decide its attributes USING ONLY facts explicitly stated in the CASE TEXT. Think step by step about which words in the text directly modify each finding before answering.

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

# V2: quote-then-normalize (grounded single pass). Model must quote the source span first.
HEAD_V2 = '''You are a clinical information extractor. Read the clinical CASE TEXT and extract every abnormal clinical finding the patient has, USING ONLY facts explicitly stated in it.

CASE TEXT:
"""{src}"""

STEP 1 — List the patient's abnormal findings (symptoms, signs, lesions, abnormal conditions). EXCLUDE tests/procedures, medications, normal findings, headings.
STEP 2 — For EACH finding, first search the CASE TEXT for the exact span that states its body site and its intensity, quote it verbatim (or "NONE" if the text gives none for THIS finding), THEN write the normalized value consistent with the quote:
- loc_quote: verbatim source span naming THIS finding's body site, else "NONE".
- location: the anatomical body site copied from loc_quote (EXCLUDE non-anatomical); "" if loc_quote is NONE.
- sev_quote: verbatim source span stating THIS symptom's own intensity, else "NONE" (NOT disease-wide severity like "in severe cases").
- severity: "mild"/"moderate"/"severe" from sev_quote; "" if NONE.
- onset: one word "sudden" or "gradual" if the CASE TEXT states how THIS finding begins, else "".
- character: the QUALITY of the symptom (e.g. "sharp","burning","productive") if stated, else "". Do NOT restate the symptom.

Output exactly:
JSON: {{"findings":[{{"name":"","loc_quote":"","location":"","sev_quote":"","severity":"","onset":"","character":""}}]}}'''

STOP = set("the a an of to in on with and or for is are be may can at as by from his her their patient left right".split())
def kt(s): return [w for w in re.findall(r'[a-z]+', s.lower()) if len(w) > 3 and w not in STOP]
def in_src(v, srcl): ws = kt(v); return bool(ws) and sum(1 for w in ws if w in srcl) >= max(1, len(ws)//2)

def parse_json(txt):
    m = re.search(r'(\{.*\})', txt, re.DOTALL)
    if not m: return []
    try: return json.loads(m.group(1)).get("findings", [])
    except: return []

def clean(finds, srcl):
    """Identical filtering to v144 for apples-to-apples: drop non-source names/values."""
    out = []
    for f in finds:
        if not isinstance(f, dict): continue
        nm = str(f.get("name", "")).strip().lower()
        if len(nm) < 2 or not in_src(nm, srcl): continue
        g = {"name": nm}
        for at in ("onset", "severity", "location", "character"):
            v = str(f.get(at, "")).strip().lower()
            if at == "onset": v = v if v in ("sudden", "gradual") else ""
            elif v and not in_src(v, srcl): v = ""
            g[at] = v
        out.append(g)
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="hf")
    ap.add_argument("--bs", type=int, default=8, help="hf batch size")
    ap.add_argument("--max-new", type=int, default=2048, help="max new tokens per generation")
    a = ap.parse_args()
    txts = sorted(glob.glob(f"{ANN_DIR}/*.txt"))[:a.n]
    docs = [(os.path.basename(t)[:-4], open(t, encoding="utf-8").read()) for t in txts]

    single_S = HEAD_SINGLE.replace("{onset}", ONSET_RULE).replace("{severity}", SEV_R2).replace("{location}", LOC_L1).replace("{character}", CHAR)
    single_V1 = HEAD_SINGLE.replace("{onset}", ONSET_RULE).replace("{severity}", SEV_V1).replace("{location}", LOC_V1).replace("{character}", CHAR)
    cot2_tmpl = COT2.replace("{onset}", ONSET_RULE).replace("{severity}", SEV_R2).replace("{location}", LOC_L1).replace("{character}", CHAR)

    # chat(prompts:list[str]) -> list[str] output text, backend-selectable.
    if a.backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85,
                  enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
        SP = SamplingParams(temperature=0.0, max_tokens=a.max_new)
        def chat(prompts):
            outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], SP, use_tqdm=True)
            return [o.outputs[0].text for o in outs]
    else:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _AutoModel
        except Exception:
            from transformers import AutoModelForCausalLM as _AutoModel
        proc = AutoProcessor.from_pretrained(MODEL)
        tok = getattr(proc, "tokenizer", proc)
        tok.padding_side = "left"
        model = _AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
        def chat(prompts):
            res = []
            for i in range(0, len(prompts), a.bs):
                batch = prompts[i:i + a.bs]
                msgs = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in batch]
                inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt", padding=True).to("cuda")
                with torch.inference_mode():
                    gen = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False)
                L = inp["input_ids"].shape[1]
                res += [proc.decode(g[L:], skip_special_tokens=True) for g in gen]
                print(f"  hf gen {min(i + a.bs, len(prompts))}/{len(prompts)}", flush=True)
            return res

    def run_single(tmpl):
        texts = chat([tmpl.format(src=s) for _, s in docs])
        return {pmid: clean(parse_json(t), s.lower()) for (pmid, s), t in zip(docs, texts)}

    preds = {}
    preds["S"] = run_single(single_S)
    preds["V1"] = run_single(single_V1)
    preds["V2"] = run_single(HEAD_V2)  # V2 JSON has extra quote keys; clean() ignores them

    # C: 2-stage cot
    t1 = chat([COT1.format(src=s) for _, s in docs])
    finds1 = []
    for t in t1:
        m = re.search(r'(\{.*\})', t, re.DOTALL); fl = []
        if m:
            try: fl = [str(x) for x in json.loads(m.group(1)).get("findings", []) if isinstance(x, str)]
            except: pass
        finds1.append(fl[:40])
    t2 = chat([cot2_tmpl.format(src=s, finds=json.dumps(fl)) for (_, s), fl in zip(docs, finds1)])
    preds["C"] = {pmid: clean(parse_json(t), s.lower()) for (pmid, s), t in zip(docs, t2)}

    for k, pd in preds.items():
        json.dump(pd, open(f"pilot/cache/maccrobat/v145_{k}_pred.json", "w"))

    # score inline
    order = ["S", "C", "V1", "V2"]
    for attr in ("location", "severity"):
        src_type = "severity" if attr == "severity" else "biological_structure"
        gold = parse_gold(ANN_DIR, src_type)
        print(f"\n{'='*80}\n### {attr.upper()}  (gold pairs={sum(len(v) for v in gold.values())})")
        for k in order:
            r = score_one(preds[k], gold, attr)
            Pb, Rb, Fb = r["bind"]; Pv, Rv, Fv = r["val"]
            print(f"[{k:3}] pred={r['NP']:4} | binding F1={Fb:.2f} ({Pb:.2f}/{Rb:.2f})  value F1={Fv:.2f} "
                  f"| A={r['A']} B={r['B']} C={r['C']} D={r['D']}/{r['NP']}", flush=True)

if __name__ == "__main__":
    main()
