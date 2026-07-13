"""Prompt-iteration to fix severity mis-binding (faithfulness 12%): source gives
case-level severity ("in severe cases: X,Y"), model glues it to each symptom.
Fix via PROMPT (not bigger model — gemini confirmed same defect). gemma-4-E4B.
Generates 3 severity-rule variants in ONE vllm session; v106 cache = baseline.
Everything else IDENTICAL to v106."""
import json, glob, re
from pathlib import Path

ATTRS=["location","onset","duration","character","severity","radiation","timing","aggravating","relieving","associated","course","context","prior_episodes"]

HEAD='''You are a clinical information extractor. Read the SOURCE about "{disease}" and extract, USING ONLY facts explicitly stated in the SOURCE, the symptoms/signs a patient experiences and their clinically discriminative attributes.

SOURCE:
"""{src}"""

Work in two steps.

STEP 1 — Reasoning (think before extracting): list the concrete patient symptoms/signs the SOURCE mentions, and for each, note ONLY the attributes the SOURCE explicitly states.{step1_extra}

STEP 2 — Output one JSON object.

RULES — what counts as a "finding":
- INCLUDE only a specific symptom or sign a patient feels or shows (e.g. fever, productive cough, chest pain, malar rash, groin bulge).
- EXCLUDE headings/categories/disease-names. Never output items such as "general symptoms", "systemic symptoms", "eye symptoms", "<disease> presentation", "symptoms of <disease>", "signs and symptoms", "child presentation". If a phrase is a section heading or a category, it is NOT a finding.

RULES — attribute values (fill ONLY if the SOURCE states it AND it adds NEW information; otherwise ""):
- onset: exactly one word: "sudden" or "gradual".
{sevrule}
- location: a specific body site only (e.g. "cheeks","groin","left chest").
- character: the QUALITY of the symptom (e.g. "sharp","dull","burning","throbbing","productive"). Do NOT define or restate the symptom itself (for "cyanosis" do NOT write "bluish skin"; for "hypoxia" do NOT write "low oxygen").
- radiation/timing/aggravating/relieving/duration: short standard term only if explicitly stated.
- Never put the disease name or a category into any attribute (no context="{disease}", no context="common symptom").
- associated: another specific finding that co-occurs, if stated.

Output exactly:
JSON: {{"findings":[{{"name":"...","location":"","onset":"","duration":"","character":"","severity":"","radiation":"","timing":"","aggravating":"","relieving":"","associated":"","course":"","context":"","prior_episodes":""}}]}}'''

BASE_SEV='- severity: exactly one word: "mild" or "moderate" or "severe". Never "severe cases"/"severe disease".'

R1=('- severity: exactly one word: "mild", "moderate", or "severe" — BUT fill it ONLY when the SOURCE states the severity of THIS SPECIFIC symptom (e.g. "severe headache", "mild cough"). If the SOURCE only describes the severity of the disease/case/episode/infection as a whole (e.g. "in severe cases", "severe disease", "severe pneumonia", "severe infection", "more severe cases may have ..."), leave severity EMPTY. The word must describe the symptom\'s own intensity, not the illness.', "")

R2=(R1[0]+' Confirm the severity word stands directly next to this symptom in the SOURCE (as in "<severity> <symptom>"); if it is not directly attached to the symptom, leave it EMPTY.', "")

R3=(R1[0], '\n- IMPORTANT for severity: for each symptom you plan to give a severity, first quote the exact SOURCE phrase. If that phrase describes the overall disease/case rather than this one symptom, set severity to "".')

VARIANTS={"v115_R1":R1,"v115_R2":R2,"v115_R3":R3}

def main():
    icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    cui2name={info["cui"]:dn for dn,info in icd.items() if "cui" in info}
    items=[]
    for fp in sorted(glob.glob("pilot/data/cache/v105_sources/*.txt")):
        c=fp.split("/")[-1][:-4]
        if c in cui2name: items.append((c,cui2name[c],open(fp).read()))
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    sp=SamplingParams(temperature=0.0,max_tokens=4096)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)

    for vname,(sevrule,step1) in VARIANTS.items():
        Path(f"pilot/data/cache/{vname}").mkdir(parents=True,exist_ok=True)
        prompts=[[{"role":"user","content":HEAD.format(disease=dn,src=src[:2200],sevrule=sevrule,step1_extra=step1)}] for c,dn,src in items]
        outs=llm.chat(prompts,sp,use_tqdm=True)
        for (c,dn,src),o in zip(items,outs):
            srcl=src.lower(); txt=o.outputs[0].text
            m=re.search(r'JSON:\s*(\{.*\})',txt,re.DOTALL) or re.search(r'(\{.*\})',txt,re.DOTALL)
            finds=[]
            if m:
                try: finds=json.loads(m.group(1)).get("findings",[])
                except Exception: pass
            clean=[]
            for f in finds:
                if not isinstance(f,dict): continue
                nm=str(f.get("name","")).strip().lower()
                if len(nm)<2 or nm==dn.lower() or not in_src(nm,srcl): continue
                rec={"name":nm}
                for at in ATTRS:
                    v=str(f.get(at,"")).strip().lower()
                    rec[at]=v if (v and in_src(v,srcl)) else ""
                clean.append(rec)
            json.dump({"disease":dn,"cui":c,"findings":clean},open(f"pilot/data/cache/{vname}/{c}.json","w"))
        print(f"{vname}: done ({len(items)} diseases)",flush=True)

if __name__=="__main__": main()
