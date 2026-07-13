"""Scale-up IE: run the CHOSEN improved prompt (P2: atomized arrays + character quality-
lexicon + onset discipline + merge) with gemma-4-12B-it-qat on the large crawled disease
set (scaleup_sources/*.json = {disease,title,text}). Outputs arrays for normalization."""
import os, argparse, json, glob, re
from pathlib import Path
from collections import Counter, defaultdict
ap=argparse.ArgumentParser(); ap.add_argument("--gpu",default="2"); ap.add_argument("--out",default="ie_scaleup")
ap.add_argument("--shard",type=int,default=0); ap.add_argument("--nshards",type=int,default=1)
a=ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"]=a.gpu
LIST_ATTR=["location","character","radiation","aggravating","relieving","associated"]
SCALAR_ATTR=["onset","duration","severity","timing","course","context","prior_episodes"]
QUALITY=set("sharp dull burning throbbing pulsating stabbing aching cramping colicky pressure pressing tight tightness squeezing pleuritic gnawing shooting tearing ripping stinging".split())

P='''You are a clinical information extractor. Read the SOURCE about "{disease}" and extract, USING ONLY facts explicitly stated in the SOURCE, the distinct symptoms/signs a patient experiences and their clinically discriminative attributes.

SOURCE:
"""{src}"""

Work in two steps.
STEP 1 — Reasoning: list the concrete patient symptoms/signs the SOURCE mentions; MERGE near-duplicate symptoms into one canonical finding (e.g. "chest pain" and "chest discomfort" are ONE finding); for each, note ONLY the attributes the SOURCE explicitly states about THAT symptom.
STEP 2 — Output one JSON object.

RULES — what counts as a "finding":
- INCLUDE only a specific symptom or sign a patient feels or shows (e.g. fever, productive cough, chest pain, malar rash, groin bulge).
- EXCLUDE headings/categories/disease-names ("general symptoms","signs and symptoms","<disease> presentation").
- MERGE near-duplicates: do not output the same symptom under different wordings.

RULES — attribute values. Fill ONLY what the SOURCE explicitly states about THIS symptom; otherwise use [] (arrays) or "" (scalars). Multi-valued attributes are ARRAYS: put exactly ONE atomic concept per array element — NEVER a comma-joined phrase (write ["exertion","cold"] not ["exertion, cold"]). Strip parentheses.
- location (array): each element ONE body site ("chest","left arm","groin").
- character (array): each element EXACTLY ONE quality adjective describing how the symptom FEELS, chosen from: sharp, dull, burning, throbbing, stabbing, aching, cramping, colicky, pressure, tight, squeezing, pleuritic, gnawing, shooting, tearing. Do NOT put another symptom, a sign, or a restatement here (NOT "blue discoloration", NOT "difficulty swallowing", NOT "lump in throat").
- radiation (array): each element ONE site the symptom spreads/radiates to.
- aggravating (array): each element ONE factor that worsens or triggers the symptom ("exertion","cold","lying down","after eating").
- relieving (array): each element ONE factor that relieves it ("rest","sitting up","nitroglycerin").
- associated (array): each element ONE distinct co-occurring finding. Do not copy the same associated list onto every finding.
- onset (scalar): exactly "sudden" or "gradual" — but ONLY if the SOURCE explicitly states the speed of onset of THIS symptom; otherwise leave "" (do not guess).
- severity (scalar): exactly "mild","moderate", or "severe" — ONLY when it stands directly next to THIS symptom ("<severity> <symptom>"); never disease-level ("in severe cases","severe disease").
- duration/timing/course (scalar): short standard term only if explicitly stated.
- Never put the disease name or a category into any attribute.

Output exactly:
JSON: {{"findings":[{{"name":"...","location":[],"character":[],"radiation":[],"aggravating":[],"relieving":[],"associated":[],"onset":"","duration":"","severity":"","timing":"","course":"","context":"","prior_episodes":""}}]}}'''

def main():
    items=[]
    for fp in sorted(glob.glob("pilot/data/cache/scaleup_sources/*.json")):
        d=json.load(open(fp));
        if d.get("text") and d.get("disease"): items.append((Path(fp).stem,d["disease"],d["text"]))
    items=items[a.shard::a.nshards]
    print(f"[shard {a.shard}/{a.nshards}] {len(items)} crawled disease texts",flush=True)
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-12B-it-qat-w4a16-ct",dtype="auto",max_model_len=8192,
            gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    outs=llm.chat([[{"role":"user","content":P.format(disease=dn,src=src[:2600])}] for i,dn,src in items],
                  SamplingParams(temperature=0.0,max_tokens=4096),use_tqdm=True)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',str(s).lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    outdir=f"pilot/data/cache/{a.out}"; Path(outdir).mkdir(parents=True,exist_ok=True)
    fill=Counter(); vals=defaultdict(Counter); nfind=0; ndis=0; parsefail=0; char_clean=0; char_tot=0
    for (i,dn,src),o in zip(items,outs):
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
                        vals[at][e]+=1
                        if at=="character": char_tot+=1; char_clean+= 1 if e in QUALITY else 0
            for at in SCALAR_ATTR:
                v=str(f.get(at,"")).strip().lower(); rec[at]=v if (v and in_src(v,srcl)) else ""
                if rec[at]: fill[at]+=1; vals[at][rec[at]]+=1
            clean.append(rec); nfind+=1
        ndis+=1
        json.dump({"disease":dn,"findings":clean},open(f"{outdir}/{i}.json","w"))
    print(f"\n[shard {a.shard}] diseases={ndis} findings={nfind} per_disease={nfind/max(1,ndis):.1f} parse_fail={parsefail}",flush=True)
if __name__=="__main__": main()
