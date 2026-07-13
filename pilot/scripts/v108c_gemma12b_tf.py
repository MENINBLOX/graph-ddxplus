"""gemma-4-12b-it IE via transformers (vLLM 0.21 can't run gemma4_unified yet).
v106 prompt (CoT + heuristic + controlled vocab). Batched generation across GPUs."""
import json, glob, re, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
ATTRS=["location","onset","duration","character","severity","radiation","timing","aggravating","relieving","associated","course","context","prior_episodes"]
PROMPT=open("pilot/scripts/v106_grounded_ie.py").read()
PROMPT=re.search(r"PROMPT='''(.*?)'''",PROMPT,re.DOTALL).group(1)
OUT="pilot/data/cache/v108_gemma12b_ie"; Path(OUT).mkdir(parents=True,exist_ok=True)
icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json")); cui2name={i["cui"]:d for d,i in icd.items() if "cui" in i}
items=[(fp.split("/")[-1][:-4],cui2name[fp.split("/")[-1][:-4]],open(fp).read()) for fp in sorted(glob.glob("pilot/data/cache/v105_sources/*.txt")) if fp.split("/")[-1][:-4] in cui2name]
mid="google/gemma-4-12b-it"
tok=AutoTokenizer.from_pretrained(mid); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.bfloat16,device_map="auto").eval()
print("model loaded",flush=True)
def gen_batch(prompts):
    texts=[tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True) for p in prompts]
    enc=tok(texts,return_tensors="pt",padding=True,truncation=True,max_length=4096,add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out=model.generate(**enc,max_new_tokens=2048,do_sample=False,pad_token_id=tok.pad_token_id)
    return [tok.decode(out[i][enc["input_ids"].shape[1]:],skip_special_tokens=True) for i in range(len(prompts))]
STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
def in_src(v,srcl):
    ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
B=4; kept=0
for i in range(0,len(items),B):
    batch=items[i:i+B]
    prompts=[PROMPT.format(disease=dn,src=src[:2200]) for c,dn,src in batch]
    outs=gen_batch(prompts)
    for (c,dn,src),txt in zip(batch,outs):
        srcl=src.lower()
        m=re.search(r'JSON:\s*(\{.*\})',txt,re.DOTALL) or re.search(r'(\{.*\})',txt,re.DOTALL)
        finds=[]
        if m:
            try: finds=json.loads(m.group(1)).get("findings",[])
            except: pass
        clean=[]
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower()
            if len(nm)<2 or nm==dn.lower() or not in_src(nm,srcl): continue
            rec={"name":nm}
            for at in ATTRS:
                v=str(f.get(at,"")).strip().lower(); rec[at]=v if (v and in_src(v,srcl)) else ""
                if rec[at]: kept+=1
            clean.append(rec)
        json.dump({"disease":dn,"cui":c,"findings":clean},open(f"{OUT}/{c}.json","w"))
        print(f"  {dn}: {len(clean)} findings",flush=True)
print(f"\nattr kept={kept}")
