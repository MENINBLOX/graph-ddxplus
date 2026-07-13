"""VERIFY: run the FROZEN v106+R2 IE prompt with gemma-4-12B-it-qat-w4a16-ct on the
canonical benchmark-blind disease texts (v105_sources), and measure per-attribute
fill-rate + distinct values. Purpose: empirically verify the Methods claims (6-attr
schema, aggravating/relieving/radiation actually populate, character values, etc.).
IE prompt is IDENTICAL to frozen v115_R2; only the model changes (E4B -> 12B-QAT)."""
import json, glob, re
from pathlib import Path
from collections import Counter, defaultdict

ATTRS=["location","onset","duration","character","severity","radiation","timing","aggravating","relieving","associated","course","context","prior_episodes"]

HEAD='''You are a clinical information extractor. Read the SOURCE about "{disease}" and extract, USING ONLY facts explicitly stated in the SOURCE, the symptoms/signs a patient experiences and their clinically discriminative attributes.

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
- {sevrule}
- location: a specific body site only (e.g. "cheeks","groin","left chest").
- character: the QUALITY of the symptom (e.g. "sharp","dull","burning","throbbing","productive"). Do NOT define or restate the symptom itself (for "cyanosis" do NOT write "bluish skin"; for "hypoxia" do NOT write "low oxygen").
- radiation/timing/aggravating/relieving/duration: short standard term only if explicitly stated.
- Never put the disease name or a category into any attribute (no context="{disease}", no context="common symptom").
- associated: another specific finding that co-occurs, if stated.

Output exactly:
JSON: {{"findings":[{{"name":"...","location":"","onset":"","duration":"","character":"","severity":"","radiation":"","timing":"","aggravating":"","relieving":"","associated":"","course":"","context":"","prior_episodes":""}}]}}'''

SEV_R2='severity: exactly one word: "mild", "moderate", or "severe" — BUT fill it ONLY when the SOURCE states the severity of THIS SPECIFIC symptom (e.g. "severe headache", "mild cough"). If the SOURCE only describes the severity of the disease/case/episode/infection as a whole (e.g. "in severe cases", "severe disease", "severe pneumonia", "severe infection", "more severe cases may have ..."), leave severity EMPTY. The word must describe the symptom\'s own intensity, not the illness. Confirm the severity word stands directly next to this symptom in the SOURCE (as in "<severity> <symptom>"); if it is not directly attached to the symptom, leave it EMPTY.'

def main():
    icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    cui2name={info["cui"]:dn for dn,info in icd.items() if "cui" in info}
    items=[]
    for fp in sorted(glob.glob("pilot/data/cache/v105_sources/*.txt")):
        c=fp.split("/")[-1][:-4]
        if c in cui2name: items.append((c,cui2name[c],open(fp).read()))
    print(f"loaded {len(items)} disease texts",flush=True)
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-12B-it-qat-w4a16-ct",dtype="auto",max_model_len=8192,
            gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    sp=SamplingParams(temperature=0.0,max_tokens=4096)
    prompts=[[{"role":"user","content":HEAD.format(disease=dn,src=src[:2200],sevrule=SEV_R2)}] for c,dn,src in items]
    outs=llm.chat(prompts,sp,use_tqdm=True)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl):
        ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    outdir="pilot/data/cache/verify_ie_12b"; Path(outdir).mkdir(parents=True,exist_ok=True)
    fill=Counter(); vals=defaultdict(Counter); nfind=0; ndis=0; parsefail=0
    for (c,dn,src),o in zip(items,outs):
        srcl=src.lower(); txt=o.outputs[0].text
        m=re.search(r'JSON:\s*(\{.*\})',txt,re.DOTALL) or re.search(r'(\{.*\})',txt,re.DOTALL)
        finds=[]
        if m:
            try: finds=json.loads(m.group(1)).get("findings",[])
            except Exception: parsefail+=1
        else: parsefail+=1
        clean=[]
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower()
            if len(nm)<2 or nm==dn.lower() or not in_src(nm,srcl): continue
            rec={"name":nm}
            for at in ATTRS:
                v=str(f.get(at,"")).strip().lower()
                rec[at]=v if (v and in_src(v,srcl)) else ""
                if rec[at]: fill[at]+=1; vals[at][rec[at]]+=1
            clean.append(rec); nfind+=1
        ndis+=1
        json.dump({"disease":dn,"cui":c,"findings":clean},open(f"{outdir}/{c}.json","w"))
    stats={"model":"gemma-4-12B-it-qat-w4a16-ct","diseases":ndis,"findings":nfind,
           "parse_fail":parsefail,"findings_per_disease":round(nfind/max(1,ndis),1),
           "fill_count":dict(fill),
           "fill_rate_per_finding":{at:round(fill[at]/max(1,nfind),3) for at in ATTRS},
           "distinct_values":{at:vals[at].most_common(40) for at in
                ["character","aggravating","relieving","radiation","location","timing","onset","severity","duration","course"]}}
    json.dump(stats,open(f"{outdir}/_stats.json","w"),indent=2,ensure_ascii=False)
    print("\n===== FILL RATE per finding =====",flush=True)
    for at in ATTRS: print(f"  {at:14s} {fill[at]:4d}  {fill[at]/max(1,nfind)*100:5.1f}%")
    print(f"\nfindings={nfind} diseases={ndis} per_disease={nfind/max(1,ndis):.1f} parse_fail={parsefail}")
    for at in ["character","aggravating","relieving","radiation"]:
        print(f"\n--- {at} distinct ({len(vals[at])}) ---")
        print(", ".join(f"{v}({n})" for v,n in vals[at].most_common(25)))

if __name__=="__main__": main()
