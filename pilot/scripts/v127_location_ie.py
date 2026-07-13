"""Location (body site) binding validation on MACCROBAT BIOLOGICAL_STRUCTURE->finding
gold (dev 946 / test 856 pairs). Same recognized relation-level method as severity.
baseline vs L1 (anatomical site, verbatim, bound to finding). gemma-4-E4B, DEV-select."""
import json, glob, re
HEAD='''You are a clinical information extractor. Read the clinical CASE TEXT and extract, USING ONLY facts explicitly stated in it, every abnormal clinical finding the patient has, with its body location.

CASE TEXT:
"""{src}"""

STEP 1 — Reasoning: list the patient's abnormal clinical findings — symptoms, exam signs, and abnormal conditions/lesions (e.g. mass, nodule, embolism, hemorrhage, effusion).
STEP 2 — Output one JSON object.

RULES — what to INCLUDE: any abnormal finding, sign, symptom, lesion, or abnormal condition. EXCLUDE only tests/procedures, medications, normal findings, headings.
{locrule}

Output exactly:
JSON: {{"findings":[{{"name":"...","location":""}}]}}'''
BASE='- location: the body site if the text states it.'
L1='- location: the specific anatomical body site the text attaches DIRECTLY to THIS finding (e.g. "left lung","liver","left ear","both lungs","left lower lobe"), copied verbatim. Fill ONLY when the site directly modifies THIS finding (e.g. "mass in the left lower lobe" -> "left lower lobe"). EXCLUDE non-anatomical locations (hospital, bed). If no body site is attached to this finding, leave EMPTY.'
VARIANTS={"v127_loc_base":BASE,"v127_loc_L1":L1}
def main():
    docs=json.load(open("pilot/data/cache/maccrobat/MACCROBAT2020-V2.json"))["data"]
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    sp=SamplingParams(temperature=0.0,max_tokens=2048)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    for vname,locrule in VARIANTS.items():
        prompts=[[{"role":"user","content":HEAD.format(src=d["full_text"][:3500],locrule=locrule)}] for d in docs]
        outs=llm.chat(prompts,sp,use_tqdm=True)
        res=[]
        for d,o in zip(docs,outs):
            srcl=d["full_text"].lower(); txt=o.outputs[0].text
            m=re.search(r'JSON:\s*(\{.*\})',txt,re.DOTALL) or re.search(r'(\{.*\})',txt,re.DOTALL)
            finds=[]
            if m:
                try: finds=json.loads(m.group(1)).get("findings",[])
                except Exception: pass
            out=[]
            for f in finds:
                if not isinstance(f,dict): continue
                nm=str(f.get("name","")).strip().lower()
                if len(nm)<2 or not in_src(nm,srcl): continue
                loc=str(f.get("location","")).strip().lower()
                out.append({"name":nm,"location":loc if loc else ""})
            res.append(out)
        json.dump(res,open(f"pilot/data/cache/maccrobat/{vname}_pred.json","w"))
        print(f"{vname}: loc-bearing={sum(1 for d in res for x in d if x['location'])}",flush=True)
main()
