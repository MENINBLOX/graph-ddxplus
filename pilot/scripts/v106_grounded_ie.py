"""v106 source-grounded IE with improved prompt (CoT + heuristic rules + controlled
vocabulary). Literature (JMIR 2024): CoT+heuristic best for attribute extraction.
Fixes via PROMPT (not system filters): reject heading/category 'findings', forbid
meta/definitional/disease-restating attribute values, force controlled enums.
Only system safeguard kept = source-grounding (extracted term must appear in source)."""
import sys, json, glob, re, argparse
from pathlib import Path
ATTRS=["location","onset","duration","character","severity","radiation","timing","aggravating","relieving","associated","course","context","prior_episodes"]
PROMPT='''You are a clinical information extractor. Read the SOURCE about "{disease}" and extract, USING ONLY facts explicitly stated in the SOURCE, the symptoms/signs a patient experiences and their clinically discriminative attributes.

SOURCE:
"""{src}"""

Work in two steps.

STEP 1 — Reasoning (think before extracting): list the concrete patient symptoms/signs the SOURCE mentions, and for each, note ONLY the attributes the SOURCE explicitly states.

STEP 2 — Output one JSON object.

RULES — what counts as a "finding":
- INCLUDE only a specific symptom or sign a patient feels or shows (e.g. fever, productive cough, chest pain, malar rash, groin bulge).
- EXCLUDE headings/categories/disease-names. Never output items such as "general symptoms", "systemic symptoms", "eye symptoms", "<disease> presentation", "symptoms of <disease>", "signs and symptoms", "child presentation". If a phrase is a section heading or a category, it is NOT a finding.

RULES — attribute values (fill ONLY if the SOURCE states it AND it adds NEW information; otherwise ""):
- onset: exactly one word: "sudden" or "gradual".
- severity: exactly one word: "mild" or "moderate" or "severe". Never "severe cases"/"severe disease".
- location: a specific body site only (e.g. "cheeks","groin","left chest").
- character: the QUALITY of the symptom (e.g. "sharp","dull","burning","throbbing","productive"). Do NOT define or restate the symptom itself (for "cyanosis" do NOT write "bluish skin"; for "hypoxia" do NOT write "low oxygen").
- radiation/timing/aggravating/relieving/duration: short standard term only if explicitly stated.
- Never put the disease name or a category into any attribute (no context="{disease}", no context="common symptom").
- associated: another specific finding that co-occurs, if stated.

Output exactly:
JSON: {{"findings":[{{"name":"...","location":"","onset":"","duration":"","character":"","severity":"","radiation":"","timing":"","aggravating":"","relieving":"","associated":"","course":"","context":"","prior_episodes":""}}]}}'''
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--src_dir",default="pilot/data/cache/v105_sources"); ap.add_argument("--out_dir",default="pilot/data/cache/v106_grounded_ie")
    a=ap.parse_args(); Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json")); cui2name={info["cui"]:dn for dn,info in icd.items() if "cui" in info}
    items=[]
    for fp in sorted(glob.glob(f"{a.src_dir}/*.txt")):
        c=fp.split("/")[-1][:-4]; 
        if c in cui2name: items.append((c,cui2name[c],open(fp).read()))
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    outs=llm.chat([[{"role":"user","content":PROMPT.format(disease=dn,src=src[:2200])}] for c,dn,src in items],SamplingParams(temperature=0.0,max_tokens=4096),use_tqdm=True)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl):
        ws=kt(v)
        return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    kept=dropped=0
    for (c,dn,src),o in zip(items,outs):
        srcl=src.lower(); txt=o.outputs[0].text
        m=re.search(r'JSON:\s*(\{.*\})',txt,re.DOTALL) or re.search(r'(\{.*\})',txt,re.DOTALL)
        finds=[]
        if m:
            try: finds=json.loads(m.group(1)).get("findings",[])
            except: pass
        clean=[]
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower()
            if len(nm)<2 or nm==dn.lower() or not in_src(nm,srcl): dropped+=1; continue
            rec={"name":nm}
            for at in ATTRS:
                v=str(f.get(at,"")).strip().lower()
                rec[at]=v if (v and in_src(v,srcl)) else ""
                if rec[at]: kept+=1
            clean.append(rec)
        json.dump({"disease":dn,"cui":c,"findings":clean},open(f"{a.out_dir}/{c}.json","w"))
        print(f"  {dn}: {len(clean)} findings",flush=True)
    print(f"\nattr kept={kept} dropped(source-ungrounded)={dropped}")
main()
