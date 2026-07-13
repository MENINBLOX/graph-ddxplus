"""Recognized severity-binding validation on MACCROBAT2020 relation gold.
Runs our extractor on 200 case reports with BASELINE vs R2 severity rule (only the
severity rule differs). Captures (symptom, severity) pairs. Scored against
MACCROBAT MODIFY(Severity->Sign_symptom) gold relations (276 pairs) -> recognized
relation-level P/R/F1, replacing the self-defined NLI proxy."""
import json, re
from pathlib import Path

HEAD='''You are a clinical information extractor. Read the clinical CASE TEXT and extract, USING ONLY facts explicitly stated in it, every sign or symptom the patient shows, with its severity.

CASE TEXT:
"""{src}"""

STEP 1 — Reasoning: list the concrete patient signs/symptoms the text mentions.
STEP 2 — Output one JSON object.

RULES — sign/symptom: INCLUDE a specific symptom or clinical sign the patient feels or shows (e.g. palpitations, dyspnea, fever, chest pain, murmur, rash). EXCLUDE tests/procedures, medications, final diagnoses, lab values, headings.
{sevrule}

Output exactly:
JSON: {{"findings":[{{"name":"...","severity":""}}]}}'''

BASELINE='- severity: one word "mild", "moderate", or "severe" if the text states it.'
R2='- severity: one word "mild", "moderate", or "severe" — BUT fill ONLY when the text states the severity of THIS SPECIFIC symptom (e.g. "severe headache", "mild cough"). If the text only describes the severity of the disease/case/episode as a whole (e.g. "severe case", "critically ill"), leave severity EMPTY. The word must describe the symptom\'s own intensity and stand directly next to it; otherwise leave EMPTY.'

VARIANTS={"v117_base":BASELINE,"v117_R2":R2}

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
        ns=sum(1 for doc in res for x in doc if x["severity"])
        print(f"{vname}: docs={len(res)} severity-bearing={ns}",flush=True)

if __name__=="__main__": main()
