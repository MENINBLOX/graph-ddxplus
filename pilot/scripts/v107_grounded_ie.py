"""v107 grounded IE — prompt-internal Extract-Then-Normalize (ABSA/e-commerce
insight). Single CoT prompt: STEP A extract verbatim specific symptoms (reject
vague aggregates/category headings); STEP B normalize each attribute to a
controlled value via explicit maps (rapidly->sudden, 'severe cases'->severe),
drop definitions/disease-names/pathogens. No few-shot. Source-grounding kept."""
import sys, json, glob, re, argparse
from pathlib import Path
ATTRS=["location","onset","duration","character","severity","radiation","timing","aggravating","relieving","associated","course","context","prior_episodes"]
PROMPT='''You are a clinical information extractor. Read the SOURCE about "{disease}" and extract patient symptoms/signs and their attributes, using ONLY facts in the SOURCE.

SOURCE:
"""{src}"""

Reason in this exact order (this order prevents errors), then output JSON.

STEP A — Extract specific findings (verbatim from SOURCE):
A finding must be ONE specific symptom or sign a patient feels or shows (e.g. "fever", "productive cough", "malar rash", "groin bulge", "hemoptysis").
REJECT and do NOT output: vague aggregates or category/heading phrases such as "general malaise", "systemic symptoms", "psychiatric symptoms", "cold or flu-like symptoms", "constitutional symptoms", "{disease} presentation", "symptoms of {disease}", "signs and symptoms". If a phrase names a CATEGORY of symptoms (plural "...symptoms") rather than one concrete symptom, reject it.

STEP B — Normalize attributes (fill ONLY if SOURCE states it; otherwise ""):
- onset: normalize {{sudden, abrupt, acute, rapid, rapidly}} -> "sudden"; {{gradual, insidious, slow, slowly, progressive}} -> "gradual"; otherwise "".
- severity: normalize to exactly "mild", "moderate", or "severe" (strip words like "cases"/"disease"/"form"); otherwise "".
- location: a specific body site (e.g. "cheeks","groin","left chest","calf"). NOT vague words like "body" or "area".
- character: the QUALITY of the symptom (e.g. "sharp","dull","burning","throbbing","cramping","productive","pruritic"). NEVER a definition/restatement of the symptom (for "cyanosis" do NOT write "bluish skin"; for "hypoxia" do NOT write "low oxygen").
- radiation, timing, duration, aggravating, relieving: a short standard term, only if explicitly stated.
- context: a TRIGGERING circumstance or exposure only (e.g. "after exertion","sun exposure","recent travel"). NEVER a disease name, a pathogen (e.g. not "streptococcus pneumoniae"), nor "common symptom".
- associated: another specific co-occurring finding.

Output exactly one line:
JSON: {{"findings":[{{"name":"","location":"","onset":"","duration":"","character":"","severity":"","radiation":"","timing":"","aggravating":"","relieving":"","associated":"","course":"","context":"","prior_episodes":""}}]}}'''
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--src_dir",default="pilot/data/cache/v105_sources"); ap.add_argument("--out_dir",default="pilot/data/cache/v107_grounded_ie")
    a=ap.parse_args(); Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json")); cui2name={info["cui"]:dn for dn,info in icd.items() if "cui" in info}
    items=[(fp.split("/")[-1][:-4],cui2name[fp.split("/")[-1][:-4]],open(fp).read()) for fp in sorted(glob.glob(f"{a.src_dir}/*.txt")) if fp.split("/")[-1][:-4] in cui2name]
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    outs=llm.chat([[{"role":"user","content":PROMPT.format(disease=dn,src=src[:2200])}] for c,dn,src in items],SamplingParams(temperature=0.0,max_tokens=4096),use_tqdm=True)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl):
        ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    kept=0
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
            if len(nm)<2 or nm==dn.lower() or not in_src(nm,srcl): continue
            rec={"name":nm}
            for at in ATTRS:
                v=str(f.get(at,"")).strip().lower()
                rec[at]=v if (v and in_src(v,srcl)) else ""
                if rec[at]: kept+=1
            clean.append(rec)
        json.dump({"disease":dn,"cui":c,"findings":clean},open(f"{a.out_dir}/{c}.json","w"))
        print(f"  {dn}: {len(clean)} findings",flush=True)
    print(f"\nattr kept={kept}")
main()
