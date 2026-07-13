"""Precision fix: gold Severity vocab = intensity-GRADING words (severe/mild/moderate/
marked/slight/extensive...); course/temporal words (sustained/recurring/progressive)
are NOT severity. M2 over-captured course words -> FP. M6 restricts to grading words +
excludes timing/course; M7 = M6 + adjacency. DEV selection, test held out."""
import json, glob, re
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

GRADE='a single intensity-GRADING word the text attaches directly to THIS finding, describing HOW SEVERE or MILD it is, copied verbatim (e.g. "mild","moderate","severe","marked","slight","extensive","significant","minimal","minor","small","large","high","massive","profuse","advanced","serious"). Do NOT output timing/course/onset/frequency words (e.g. "sudden","gradual","acute","chronic","recurrent","persistent","progressive","sustained","intermittent","frequent","aggravated") — those describe COURSE, not severity.'
M6=f'- severity: {GRADE} Fill ONLY when the grading word directly modifies THIS finding; if it describes the disease/case as a whole, or no grading word is attached, leave EMPTY.'
M7=f'- severity: {GRADE} Confirm the grading word stands directly next to this finding ("<word> <finding>"); fill ONLY then. If it describes the disease/case overall, or is not directly attached, leave EMPTY.'

VARIANTS={"v123_M6":M6,"v123_M7":M7}

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
