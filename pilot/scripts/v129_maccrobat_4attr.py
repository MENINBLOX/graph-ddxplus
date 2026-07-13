"""First-pass 4-attribute IE on MACCROBAT (for professor validation set).
Combined prompt (validated rules): finding + onset + character + severity + location.
Run on the held-out TEST split docs (consistent with severity/location validation).
Output feeds the Label Studio converter as predictions (1차 라벨)."""
import json, glob, re
HEAD='''You are a clinical information extractor. Read the clinical CASE TEXT and extract, USING ONLY facts explicitly stated in it, every abnormal clinical finding the patient has, with four attributes.

CASE TEXT:
"""{src}"""

STEP 1 — Reasoning: list the patient's abnormal clinical findings (symptoms, exam signs, abnormal conditions/lesions).
STEP 2 — Output one JSON object.

RULES — INCLUDE any abnormal finding/sign/symptom/lesion/condition. EXCLUDE tests/procedures, medications, normal findings, headings.
For each finding, fill an attribute ONLY when the text states it directly for THAT finding; otherwise "":
- onset: exactly "sudden" or "gradual" — how the finding began. Not "acute/chronic/recurrent" (those are course).
- character: the QUALITY of the finding (e.g. "sharp","dull","burning","throbbing","productive","itchy","bluish"). Do NOT restate/define the finding itself.
- severity: a single intensity-GRADING word directly on this finding (e.g. mild, moderate, severe, marked, slight, extensive, massive). NOT timing/course words.
- location: the specific anatomical body site directly on this finding (e.g. left lung, liver), verbatim. EXCLUDE non-anatomical sites.

Output exactly:
JSON: {{"findings":[{{"name":"...","onset":"","character":"","severity":"","location":""}}]}}'''
BADSEV=re.compile(r'\d|cm|mm|°|%|kg|mg|ml|grade|fatal|benign')
def main():
    docs=json.load(open("pilot/data/cache/maccrobat/MACCROBAT2020-V2.json"))["data"]
    split=json.load(open("pilot/data/cache/maccrobat/split.json")); test=set(split["test"])
    txt2pmid={open(t).read().strip()[:120]:t.split("/")[-1][:-4] for t in glob.glob("pilot/data/cache/maccrobat/brat/*.txt")}
    idx=[i for i,d in enumerate(docs) if txt2pmid.get(d["full_text"].strip()[:120]) in test]
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    outs=llm.chat([[{"role":"user","content":HEAD.format(src=docs[i]["full_text"][:3500])}] for i in idx],SamplingParams(temperature=0.0,max_tokens=2048),use_tqdm=True)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    out={}
    for i,o in zip(idx,outs):
        d=docs[i]; srcl=d["full_text"].lower(); txt=o.outputs[0].text
        m=re.search(r'JSON:\s*(\{.*\})',txt,re.DOTALL) or re.search(r'(\{.*\})',txt,re.DOTALL)
        finds=[]
        if m:
            try: finds=json.loads(m.group(1)).get("findings",[])
            except Exception: pass
        clean=[]
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower()
            if len(nm)<2 or not in_src(nm,srcl): continue
            sev=str(f.get("severity","")).strip().lower()
            clean.append({"name":nm,
                          "onset":str(f.get("onset","")).strip().lower() if str(f.get("onset","")).strip().lower() in ("sudden","gradual") else "",
                          "character":str(f.get("character","")).strip().lower(),
                          "severity":sev if (sev and not BADSEV.search(sev)) else "",
                          "location":str(f.get("location","")).strip().lower()})
        pmid=txt2pmid.get(d["full_text"].strip()[:120])
        out[pmid]={"text":d["full_text"],"findings":clean}
    json.dump(out,open("pilot/data/cache/maccrobat/v129_4attr_pred.json","w"))
    nf=sum(len(v["findings"]) for v in out.values())
    print(f"v129 4-attr: docs={len(out)} findings={nf}")
    for a in ["onset","character","severity","location"]:
        print(f"  {a}: {sum(1 for v in out.values() for x in v['findings'] if x[a])}")
main()
