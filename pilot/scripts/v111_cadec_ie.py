"""Cross-genre IE certification on CADEC (patient-forum posts = lay vocabulary).
Same zero-shot extractor as MACCROBAT (v111). Gold = ADR mention spans (verbatim)
+ MedDRA Preferred Term normalization. Tests robustness on lay patient language
(directly relevant: DDXPlus patient evidence is lay-phrased)."""
import json, re
from collections import defaultdict
import pyarrow.parquet as pq

PROMPT='''You are a clinical information extractor. Read the patient's TEXT (an online post describing their experience) and extract, USING ONLY what is stated, every symptom, sign, or adverse reaction the patient reports.

TEXT:
"""{src}"""

STEP 1 — Reasoning: list the concrete symptoms / bodily reactions the patient describes (in their own words is fine).
STEP 2 — Output one JSON object.

RULES:
- INCLUDE any symptom, sign, pain, or adverse bodily reaction the patient reports (e.g. nausea, weakness, joint pain, dizziness, could not sleep).
- EXCLUDE the drug/medication names, dosages, and the disease they were treating.
- For each, give severity ("mild"/"moderate"/"severe") and body location ONLY if the text states it.

Output exactly:
JSON: {{"findings":[{{"name":"...","severity":"","location":""}}]}}'''

def main():
    rows=[]
    for sp in ["train","test"]:
        rows+=pq.read_table(f"pilot/data/cache/cadec/{sp}.parquet").to_pylist()
    g=defaultdict(lambda:{"ade":set(),"pt":set()})
    for r in rows:
        t=r["text"].strip()
        if r.get("ade"): g[t]["ade"].add(str(r["ade"]).strip().lower())
        if r.get("term_PT"): g[t]["pt"].add(str(r["term_PT"]).strip().lower())
    texts=list(g.keys())
    gold=[{"text":t,"ade":sorted(g[t]["ade"]),"pt":sorted(g[t]["pt"])} for t in texts]
    json.dump(gold,open("pilot/data/cache/cadec/gold.json","w"))

    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,
            gpu_memory_utilization=0.85,enforce_eager=True,
            limit_mm_per_prompt={"image":0,"audio":0})
    outs=llm.chat([[{"role":"user","content":PROMPT.format(src=t[:2000])}] for t in texts],
                  SamplingParams(temperature=0.0,max_tokens=1024),use_tqdm=True)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    preds=[]
    for t,o in zip(texts,outs):
        srcl=t.lower(); txt=o.outputs[0].text
        m=re.search(r'JSON:\s*(\{.*\})',txt,re.DOTALL) or re.search(r'(\{.*\})',txt,re.DOTALL)
        finds=[]
        if m:
            try: finds=json.loads(m.group(1)).get("findings",[])
            except Exception: pass
        names=[]
        for f in finds:
            if isinstance(f,dict):
                nm=str(f.get("name","")).strip().lower()
                if len(nm)>=2 and in_src(nm,srcl): names.append(nm)
        preds.append(names)
    json.dump(preds,open("pilot/data/cache/cadec/v111_pred.json","w"))
    print(f"wrote {len(preds)} texts; total extracted symptoms={sum(len(p) for p in preds)}")

if __name__=="__main__": main()
