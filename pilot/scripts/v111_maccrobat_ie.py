"""Intrinsic IE certification on a recognized public gold corpus (MACCROBAT2020).
Runs our source-grounded structured extractor (v106 philosophy: CoT + heuristic
rules + controlled vocab) on each clinical case report and outputs, per document,
the signs/symptoms with their severity and body-location attributes.

This is ALGORITHM-INDEPENDENT: it measures the IE output against expert gold
annotations (SIGN_SYMPTOM / SEVERITY / BIOLOGICAL_STRUCTURE), so the downstream
diagnosis algorithm plays no role -> defeats the 'algorithm co-adapted to IE' attack.
Scored separately by v111_maccrobat_score.py with standard mention-level P/R/F1.
"""
import json, re, argparse
from pathlib import Path

PROMPT='''You are a clinical information extractor. Read the clinical CASE TEXT and extract, USING ONLY facts explicitly stated in it, every sign or symptom the patient shows or experiences, with its discriminative attributes.

CASE TEXT:
"""{src}"""

Work in two steps.
STEP 1 — Reasoning: list the concrete patient signs/symptoms the text mentions, and for each note ONLY attributes the text explicitly states.
STEP 2 — Output one JSON object.

RULES — what counts as a "sign/symptom":
- INCLUDE a specific symptom or clinical sign the patient feels or shows (e.g. palpitations, dyspnea, fever, chest pain, murmur, cyanosis, rash, swelling, weakness).
- EXCLUDE diagnostic tests/procedures (e.g. echocardiography, ECG, biopsy, MRI), medications, final disease diagnoses, and lab measurement values.
- EXCLUDE headings/categories (e.g. "symptoms", "physical examination findings").

RULES — attributes (fill ONLY if the text states it for that symptom; else ""):
- severity: one word only: "mild" or "moderate" or "severe".
- location: the specific body site of that symptom only (e.g. "chest","left arm","abdomen").
- Do NOT restate or define the symptom in any attribute.

Output exactly:
JSON: {{"findings":[{{"name":"...","severity":"","location":""}}]}}'''


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="pilot/data/cache/maccrobat/MACCROBAT2020-V2.json")
    ap.add_argument("--out",default="pilot/data/cache/maccrobat/v111_pred.json")
    a=ap.parse_args()
    docs=json.load(open(a.data))["data"]
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,
            gpu_memory_utilization=0.85,enforce_eager=True,
            limit_mm_per_prompt={"image":0,"audio":0})
    prompts=[[{"role":"user","content":PROMPT.format(src=d["full_text"][:3500])}] for d in docs]
    outs=llm.chat(prompts,SamplingParams(temperature=0.0,max_tokens=2048),use_tqdm=True)

    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl):
        ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    preds=[]
    for d,o in zip(docs,outs):
        srcl=d["full_text"].lower(); txt=o.outputs[0].text
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
            loc=str(f.get("location","")).strip().lower()
            clean.append({"name":nm,
                          "severity":sev if (sev and in_src(sev,srcl)) else "",
                          "location":loc if (loc and in_src(loc,srcl)) else ""})
        preds.append(clean)
    json.dump(preds,open(a.out,"w"))
    print(f"wrote {len(preds)} docs -> {a.out}; total symptoms={sum(len(p) for p in preds)}")


if __name__=="__main__":
    main()
