"""IE MODEL PANEL — hold the winning single-prompt architecture (V4) FIXED and vary the
MODEL, to select the IE model by open-panel comparison (project mandate: gemma/Qwen/Llama).
gemma-4-E4B is the smallest model; bigger/stronger models are the largest untried lever
for raising IE quality. One model per process (clean VRAM on the single 4090).

Loads any HF chat model (gemma multimodal via ImageTextToText, Qwen/Llama via CausalLM),
bf16 or 4-bit (bitsandbytes) to fit 24GB. Runs the V4 6-attribute single prompt on n
MACCROBAT docs, saves preds, prints raw + source-adjudicated location/severity F1.

Usage: python v148_model_panel.py --model Qwen/Qwen2.5-14B-Instruct --tag qwen25-14b --load-4bit --n 200
"""
import json, glob, re, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v143_unified_4attr import SEV_R2, CHAR, ONSET
from v146_decomp import LOC_DECOMP
from v147_single6 import HEAD_V4, STEP1_SITE, TIMING, AGGR, RELV
from v145_strategy_scope import parse_json, clean, ANN_DIR
from v147_adjudicate import adjudicate
from score_maccrobat_attrs import parse_gold, score_one

ONSET_RULE = ONSET["O2"]

def build_v4_prompt():
    return (HEAD_V4.replace("{step1}", STEP1_SITE).replace("{onset}", ONSET_RULE)
            .replace("{severity}", SEV_R2).replace("{location}", LOC_DECOMP).replace("{character}", CHAR)
            .replace("{timing}", TIMING).replace("{aggravating}", AGGR).replace("{relieving}", RELV))

def build_chat(model_id, load_4bit, bs, max_new):
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    kw = {"device_map": "cuda"}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_use_double_quant=True)
    else:
        kw["dtype"] = torch.bfloat16
    multimodal = True
    try:
        from transformers import AutoModelForImageTextToText as AM
        model = AM.from_pretrained(model_id, **kw).eval()
        proc = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"  [load] not image-text-to-text ({type(e).__name__}); using CausalLM", flush=True)
        from transformers import AutoModelForCausalLM as AM
        model = AM.from_pretrained(model_id, **kw).eval()
        try:
            proc = AutoProcessor.from_pretrained(model_id)
        except Exception:
            proc = AutoTokenizer.from_pretrained(model_id)
        multimodal = False
    tok = getattr(proc, "tokenizer", proc)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    def content(p):
        return [{"type": "text", "text": p}] if multimodal else p
    def chat(prompts):
        res = []
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            msgs = [[{"role": "user", "content": content(p)}] for p in batch]
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                gen = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
            L = inp["input_ids"].shape[1]
            res += [tok.decode(g[L:], skip_special_tokens=True) for g in gen]
            print(f"  gen {min(i + bs, len(prompts))}/{len(prompts)}", flush=True)
        return res
    return chat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=2048)
    a = ap.parse_args()
    txts = sorted(glob.glob(f"{ANN_DIR}/*.txt"))[:a.n]
    docs = [(os.path.basename(t)[:-4], open(t, encoding="utf-8").read()) for t in txts]
    head = build_v4_prompt()
    print(f"### MODEL={a.model} tag={a.tag} 4bit={a.load_4bit} n={a.n}", flush=True)
    chat = build_chat(a.model, a.load_4bit, a.bs, a.max_new)
    texts = chat([head.format(src=s) for _, s in docs])
    preds = {pmid: clean(parse_json(t), s.lower()) for (pmid, s), t in zip(docs, texts)}
    out = f"pilot/cache/maccrobat/v148_{a.tag}_pred.json"
    json.dump(preds, open(out, "w"))
    print(f"saved {out}", flush=True)
    for attr in ("location", "severity"):
        gold = parse_gold(ANN_DIR, "severity" if attr == "severity" else "biological_structure")
        (Pb, Rb, Fb), (Pa, Ra, Fa), B = adjudicate(preds, gold, attr)
        print(f"[{a.tag}] {attr:9} raw F1={Fb:.2f} ({Pb:.2f}/{Rb:.2f}) | adj F1={Fa:.2f} ({Pa:.2f}/{Ra:.2f}) | bktB={B}", flush=True)

if __name__ == "__main__":
    main()
