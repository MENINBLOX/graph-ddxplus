"""Reconcile combined-vs-separate prompt. Production v106 uses ONE prompt for all
attributes; validation used per-attribute prompts. Test: does a COMBINED prompt
(same validated severity+location rules, extracted together) match the focused
per-attribute F1? gemma-4-E4B, same 200 docs, score both attrs vs MACCROBAT gold."""
import json, glob, re
HEAD='''You are a clinical information extractor. Read the clinical CASE TEXT and extract, USING ONLY facts explicitly stated in it, every abnormal clinical finding the patient has, with its severity and body location.

CASE TEXT:
"""{src}"""

STEP 1 — Reasoning: list the patient's abnormal clinical findings (symptoms, exam signs, abnormal conditions/lesions).
STEP 2 — Output one JSON object.

RULES — INCLUDE any abnormal finding/sign/symptom/lesion/condition. EXCLUDE tests/procedures, medications, normal findings, headings.
- severity: a single intensity-GRADING word the text attaches directly to THIS finding (e.g. mild, moderate, severe, marked, slight, extensive, significant, massive). Do NOT output timing/course words (sudden, acute, chronic, recurrent, persistent, progressive) — not severity. Fill ONLY when it directly modifies this finding; else "".
- location: the specific anatomical body site the text attaches DIRECTLY to THIS finding (e.g. left lung, liver, left ear), copied verbatim. EXCLUDE non-anatomical sites. Fill ONLY when the site modifies this finding; else "".

Output exactly:
JSON: {{"findings":[{{"name":"...","severity":"","location":""}}]}}'''
BAD=re.compile(r'\d|cm|mm|°|%|kg|mg|ml|grade|fatal|benign')
def main():
    docs=json.load(open("pilot/data/cache/maccrobat/MACCROBAT2020-V2.json"))["data"]
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    outs=llm.chat([[{"role":"user","content":HEAD.format(src=d["full_text"][:3500])}] for d in docs],SamplingParams(temperature=0.0,max_tokens=2048),use_tqdm=True)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
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
            loc=str(f.get("location","")).strip().lower()
            out.append({"name":nm,"severity":sev if (sev and not BAD.search(sev)) else "","location":loc if loc else ""})
        res.append(out)
    json.dump(res,open("pilot/data/cache/maccrobat/v128_combined_pred.json","w"))
    print(f"v128 combined: sev={sum(1 for d in res for x in d if x['severity'])} loc={sum(1 for d in res for x in d if x['location'])}")
main()
