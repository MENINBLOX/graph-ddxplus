"""M8: fix recall. Prior finding rule excluded 'final diagnoses' -> dropped disease
findings (mass/regurgitation/nodule/dysfunction) that MACCROBAT tags with severity
(45 missed findings = MY prompt defect, NOT model capacity). M8 includes abnormal
clinical findings/conditions the patient presents with. Severity rule = M7. DEV select."""
import json, glob, re
HEAD='''You are a clinical information extractor. Read the clinical CASE TEXT and extract, USING ONLY facts explicitly stated in it, every abnormal clinical finding the patient has, with its severity.

CASE TEXT:
"""{src}"""

STEP 1 — Reasoning: list EVERY abnormal clinical finding the text reports about THIS patient — symptoms, physical/exam signs (murmur, edema, pallor), and abnormal conditions/lesions found (e.g. mass, nodule, regurgitation, stenosis, dysfunction, hemorrhage, enlargement, effusion). Be exhaustive.
STEP 2 — Output one JSON object.

RULES — what to INCLUDE: any abnormal finding, sign, symptom, lesion, or abnormal condition the patient is reported to have. EXCLUDE only: tests/procedures themselves (e.g. echocardiography, biopsy), medications, normal findings, and section headings.
- severity: a single intensity-GRADING word the text attaches directly to THIS finding, describing HOW SEVERE or MILD it is, copied verbatim (e.g. "mild","moderate","severe","marked","slight","extensive","significant","minimal","minor","small","large","high","massive","profuse","advanced","serious"). Do NOT output timing/course/onset/frequency words (e.g. "sudden","gradual","acute","chronic","recurrent","persistent","progressive","sustained","intermittent","frequent") — those describe COURSE, not severity. Confirm the grading word stands directly next to this finding ("<word> <finding>"); fill ONLY then. If it describes the disease/case overall, or is not directly attached, leave EMPTY.

Output exactly:
JSON: {{"findings":[{{"name":"...","severity":""}}]}}'''
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
            out.append({"name":nm,"severity":sev if sev else ""})
        res.append(out)
    json.dump(res,open("pilot/data/cache/maccrobat/v125_M8_pred.json","w"))
    print(f"v125_M8: sev-bearing={sum(1 for doc in res for x in doc if x['severity'])}, total findings={sum(len(d) for d in res)}")
main()
