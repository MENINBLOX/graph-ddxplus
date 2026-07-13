"""Prompt-improvement experiment for attribute IE (gemma-4-12B-it-qat-w4a16-ct).
Diagnosed failure modes of frozen v106 (on 49 texts):
  - multi-factor phrases not atomized (aggravating="exertion, emotional stress, full stomach, cold")
  - character catches non-quality ("blue discoloration","difficulty swallowing")
  - near-duplicate findings (chest pain/discomfort/anginal pains) + associated copied to all
  - onset over-predicts "sudden"
Fixes tested via PROMPT (not model):
  P1 = atomize (multi-valued attrs -> JSON arrays, one atomic concept per element; strip parens)
  P2 = P1 + character quality-lexicon only + onset only-if-explicit + merge duplicates + no assoc copy
Run: python ie_prompt_improve.py --variant p2 --gpu 0
Measures per-attribute fill + atomicity + character-cleanliness, comparable to verify_ie_12b (V0)."""
import os, argparse, json, glob, re
from pathlib import Path
from collections import Counter, defaultdict

ap=argparse.ArgumentParser()
ap.add_argument("--variant",choices=["p1","p2"],required=True)
ap.add_argument("--gpu",default="0")
a=ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=a.gpu

LIST_ATTR=["location","character","radiation","aggravating","relieving","associated"]
SCALAR_ATTR=["onset","duration","severity","timing","course","context","prior_episodes"]
QUALITY=set("sharp dull burning throbbing pulsating stabbing aching cramping colicky pressure pressing tight tightness squeezing pleuritic gnawing shooting tearing ripping stinging".split())

COMMON='''You are a clinical information extractor. Read the SOURCE about "{disease}" and extract, USING ONLY facts explicitly stated in the SOURCE, the distinct symptoms/signs a patient experiences and their clinically discriminative attributes.

SOURCE:
"""{src}"""

Work in two steps.
STEP 1 — Reasoning: list the concrete patient symptoms/signs the SOURCE mentions{merge}; for each, note ONLY the attributes the SOURCE explicitly states about THAT symptom.
STEP 2 — Output one JSON object.

RULES — what counts as a "finding":
- INCLUDE only a specific symptom or sign a patient feels or shows (e.g. fever, productive cough, chest pain, malar rash, groin bulge).
- EXCLUDE headings/categories/disease-names ("general symptoms","signs and symptoms","<disease> presentation").{mergerule}

RULES — attribute values. Fill ONLY what the SOURCE explicitly states about THIS symptom; otherwise use [] (arrays) or "" (scalars). Multi-valued attributes are ARRAYS: put exactly ONE atomic concept per array element — NEVER a comma-joined phrase (write ["exertion","cold"] not ["exertion, cold"]). Strip parentheses.
- location (array): each element ONE body site ("chest","left arm","groin").
- character (array): {charrule}
- radiation (array): each element ONE site the symptom spreads/radiates to.
- aggravating (array): each element ONE factor that worsens or triggers the symptom ("exertion","cold","lying down","after eating").
- relieving (array): each element ONE factor that relieves it ("rest","sitting up","nitroglycerin").
- associated (array): each element ONE distinct co-occurring finding.{assocrule}
- onset (scalar): exactly "sudden" or "gradual"{onsetrule}.
- severity (scalar): exactly "mild","moderate", or "severe" — ONLY when it stands directly next to THIS symptom ("<severity> <symptom>"); never disease-level ("in severe cases","severe disease").
- duration/timing/course (scalar): short standard term only if explicitly stated.
- Never put the disease name or a category into any attribute.

Output exactly:
JSON: {{"findings":[{{"name":"...","location":[],"character":[],"radiation":[],"aggravating":[],"relieving":[],"associated":[],"onset":"","duration":"","severity":"","timing":"","course":"","context":"","prior_episodes":""}}]}}'''

def fill(tmpl, **kw):
    for k,v in kw.items(): tmpl=tmpl.replace("{"+k+"}", v)
    return tmpl
if a.variant=="p1":
    P=fill(COMMON, merge="", mergerule="",
        charrule='each element a short quality of how the symptom feels ("sharp","burning","throbbing").',
        assocrule="", onsetrule="")
else:
    P=fill(COMMON,
        merge="; MERGE near-duplicate symptoms into one canonical finding (e.g. \"chest pain\" and \"chest discomfort\" are ONE finding)",
        mergerule="\n- MERGE near-duplicates: do not output the same symptom under different wordings.",
        charrule='each element EXACTLY ONE quality adjective describing how the symptom FEELS, chosen from: sharp, dull, burning, throbbing, stabbing, aching, cramping, colicky, pressure, tight, squeezing, pleuritic, gnawing, shooting, tearing. Do NOT put another symptom, a sign, or a restatement here (NOT "blue discoloration", NOT "difficulty swallowing", NOT "lump in throat").',
        assocrule=" Do not copy the same associated list onto every finding.",
        onsetrule=" — but ONLY if the SOURCE explicitly states the speed of onset of THIS symptom; otherwise leave \"\" (do not guess)")

def main():
    icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    cui2name={info["cui"]:dn for dn,info in icd.items() if "cui" in info}
    items=[]
    for fp in sorted(glob.glob("pilot/data/cache/v105_sources/*.txt")):
        c=fp.split("/")[-1][:-4]
        if c in cui2name: items.append((c,cui2name[c],open(fp).read()))
    print(f"[{a.variant}] loaded {len(items)} texts",flush=True)
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-12B-it-qat-w4a16-ct",dtype="auto",max_model_len=8192,
            gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    outs=llm.chat([[{"role":"user","content":P.format(disease=dn,src=src[:2200])}] for c,dn,src in items],
                  SamplingParams(temperature=0.0,max_tokens=4096),use_tqdm=True)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',str(s).lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    outdir=f"pilot/data/cache/ie_{a.variant}"; Path(outdir).mkdir(parents=True,exist_ok=True)
    fill=Counter(); vals=defaultdict(Counter); nfind=0; ndis=0; parsefail=0
    atom_ok=Counter(); atom_tot=Counter(); char_clean=0; char_tot=0
    def atomic(x): x=str(x).strip(); return bool(x) and "," not in x and len(x.split())<=4
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
            for at in LIST_ATTR:
                raw=f.get(at,[])
                if isinstance(raw,str): raw=[x.strip() for x in raw.split(",")] if raw else []
                elems=[str(x).strip().lower() for x in raw if isinstance(x,(str,int,float)) and str(x).strip()]
                elems=[e for e in elems if in_src(e,srcl)]
                rec[at]=elems
                if elems:
                    fill[at]+=1
                    for e in elems:
                        vals[at][e]+=1; atom_tot[at]+=1; atom_ok[at]+= 1 if atomic(e) else 0
                        if at=="character":
                            char_tot+=1; char_clean+= 1 if e in QUALITY else 0
            for at in SCALAR_ATTR:
                v=str(f.get(at,"")).strip().lower()
                rec[at]=v if (v and in_src(v,srcl)) else ""
                if rec[at]: fill[at]+=1; vals[at][rec[at]]+=1
            clean.append(rec); nfind+=1
        ndis+=1
        json.dump({"disease":dn,"cui":c,"findings":clean},open(f"{outdir}/{c}.json","w"))
    stats={"variant":a.variant,"model":"gemma-4-12B-qat","diseases":ndis,"findings":nfind,
           "findings_per_disease":round(nfind/max(1,ndis),1),"parse_fail":parsefail,
           "fill_rate":{at:round(fill[at]/max(1,nfind),3) for at in LIST_ATTR+SCALAR_ATTR},
           "atomicity":{at:round(atom_ok[at]/max(1,atom_tot[at]),3) for at in LIST_ATTR},
           "avg_elems_per_filled":{at:round(atom_tot[at]/max(1,fill[at]),2) for at in LIST_ATTR},
           "character_cleanliness":round(char_clean/max(1,char_tot),3),
           "distinct_values":{at:vals[at].most_common(40) for at in LIST_ATTR+["onset","severity","timing","duration"]}}
    json.dump(stats,open(f"{outdir}/_stats.json","w"),indent=2,ensure_ascii=False)
    print(f"\n[{a.variant}] findings={nfind} per_disease={nfind/max(1,ndis):.1f} parse_fail={parsefail}")
    print("fill:",{k:f"{v*100:.0f}%" for k,v in stats['fill_rate'].items()})
    print("atomicity:",stats['atomicity'])
    print("avg_elems:",stats['avg_elems_per_filled'])
    print("char_cleanliness:",stats['character_cleanliness'])
    for at in ["character","aggravating","relieving"]:
        print(f"  {at}:", ", ".join(f"{v}({n})" for v,n in vals[at].most_common(15)))

if __name__=="__main__": main()
