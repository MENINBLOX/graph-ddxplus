"""Round 2 severity prompt iteration. Base = R2 adjacency (best of round 1, 56%).
R4 = R2 + strict enum/synonym-map + no-repeat-in-name (kills invalid words 'less',
maps 'excruciating'->severe). R5 = strictest immediate-adjacency only."""
import json, glob, re
from pathlib import Path
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
{sevrule}
- location: a specific body site only (e.g. "cheeks","groin","left chest").
- character: the QUALITY of the symptom (e.g. "sharp","dull","burning","throbbing","productive"). Do NOT define or restate the symptom itself (for "cyanosis" do NOT write "bluish skin"; for "hypoxia" do NOT write "low oxygen").
- radiation/timing/aggravating/relieving/duration: short standard term only if explicitly stated.
- Never put the disease name or a category into any attribute (no context="{disease}", no context="common symptom").
- associated: another specific finding that co-occurs, if stated.

Output exactly:
JSON: {{"findings":[{{"name":"...","location":"","onset":"","duration":"","character":"","severity":"","radiation":"","timing":"","aggravating":"","relieving":"","associated":"","course":"","context":"","prior_episodes":""}}]}}'''

R2CORE='- severity: one word: "mild", "moderate", or "severe" — BUT fill ONLY when the SOURCE states the severity of THIS SPECIFIC symptom (e.g. "severe headache", "mild cough"). If the SOURCE only describes the severity of the disease/case/episode/infection as a whole (e.g. "in severe cases", "severe disease", "severe pneumonia", "more severe cases may have ..."), leave severity EMPTY. The word must describe the symptom\'s own intensity, not the illness. Confirm the severity word stands directly next to this symptom in the SOURCE (as in "<severity> <symptom>"); if it is not directly attached to the symptom, leave it EMPTY.'

R4=R2CORE+' Output ONLY one of exactly "mild", "moderate", or "severe". If the SOURCE attaches a synonym directly to this symptom, map it ("excruciating"/"intense"/"extreme"->"severe", "slight"->"mild"); never output any other word (not "less", "noticeable", "acute"). Do NOT output a severity word that already appears inside the symptom name.'

R5='- severity: Output "mild", "moderate", or "severe" ONLY when that exact intensity word (or a clear synonym mapped to it) appears IMMEDIATELY before this symptom in the SOURCE as its direct modifier (e.g. "severe headache" -> headache severity "severe"). In EVERY other case leave severity EMPTY — including when the SOURCE calls the disease/case severe, or lists the symptom with no intensity word attached. Do NOT output a severity that already appears inside the symptom name.'

VARIANTS={"v116_R4":R4,"v116_R5":R5}

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
    for vname,sevrule in VARIANTS.items():
        Path(f"pilot/data/cache/{vname}").mkdir(parents=True,exist_ok=True)
        prompts=[[{"role":"user","content":HEAD.format(disease=dn,src=src[:2200],sevrule=sevrule)}] for c,dn,src in items]
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
        print(f"{vname}: done",flush=True)

if __name__=="__main__": main()
