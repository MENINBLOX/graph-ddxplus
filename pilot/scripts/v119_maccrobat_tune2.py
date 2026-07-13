"""Round 2 MACCROBAT dev-tuning: M2 (verbatim severity + sign inclusion) had best
dev value-F1 but low precision (over-predict). M4/M5 add R2's adjacency discipline
to lift precision while keeping verbatim+sign recall. DEV-only selection; test held out."""
import json, re
from pathlib import Path

HEAD='''You are a clinical information extractor. Read the clinical CASE TEXT and extract, USING ONLY facts explicitly stated in it, every sign or symptom the patient shows, with its severity.

CASE TEXT:
"""{src}"""

STEP 1 — Reasoning: list the concrete patient signs/symptoms the text mentions.
STEP 2 — Output one JSON object.

RULES — sign/symptom: INCLUDE a specific symptom or clinical sign the patient feels or shows. INCLUDE physical examination signs too (e.g. murmur, hepatomegaly, edema, rash, pallor, swelling), not only patient-reported symptoms. EXCLUDE tests/procedures, medications, final diagnoses, lab values, headings.
{sevrule}

Output exactly:
JSON: {{"findings":[{{"name":"...","severity":""}}]}}'''

VERBATIM='the short intensity word the text attaches DIRECTLY to THIS symptom, copied verbatim (e.g. "severe","mild","massive","profuse","marked")'
# M4: verbatim + adjacency requirement
M4=f'- severity: {VERBATIM}. Fill ONLY when the word stands directly next to this symptom in the text (as in "<word> <symptom>", e.g. "severe headache"). If the text only describes the disease/case severity as a whole, or no intensity word is directly attached to this symptom, leave EMPTY.'
# M5: verbatim + adjacency + explicit case-level exclusion list
M5=f'- severity: {VERBATIM}. Fill ONLY if the word directly modifies THIS symptom. Leave EMPTY when: the word describes the overall disease/case ("severe case","critically ill","in severe disease"), or the word is separated from the symptom by other words, or no intensity word is attached. When unsure, leave EMPTY.'

VARIANTS={"v119_M4":M4,"v119_M5":M5}

def main():
    docs=json.load(open("pilot/data/cache/maccrobat/MACCROBAT2020-V2.json"))["data"]
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    sp=SamplingParams(temperature=0.0,max_tokens=2048)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    for vname,sevrule in VARIANTS.items():
        prompts=[[{"role":"user","content":HEAD.format(src=d["full_text"][:3500],sevrule=sevrule)}] for d in docs]
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
                sev=str(f.get("severity","")).strip().lower()
                out.append({"name":nm,"severity":sev if sev else ""})
            res.append(out)
        json.dump(res,open(f"pilot/data/cache/maccrobat/{vname}_pred.json","w"))
        print(f"{vname}: severity-bearing={sum(1 for doc in res for x in doc if x['severity'])}",flush=True)

if __name__=="__main__": main()
