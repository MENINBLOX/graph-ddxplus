"""Measure onset yield: run gemma-4-E4B onset IE on ~1000 random PubMed abstracts
to see how many onset (sudden/gradual) instances accumulate. If sufficient -> rerun
with commercial API; here = LOCAL gemma first (cost-free)."""
import json, glob, re, random, argparse
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=1000); ap.add_argument("--cue",action="store_true")
    a=ap.parse_args()
    files=glob.glob("/windows/data/medkg/pubmed/*.jsonl"); random.seed(42); random.shuffle(files)
    docs=[]
    cue=re.compile(r'\b(sudden|gradual|abrupt|insidious|acute onset|rapid onset)\b',re.I)
    for f in files:
        for line in open(f):
            try: r=json.loads(line)
            except: continue
            ab=(r.get("abstract") or "").strip()
            if len(ab)<120: continue
            txt=(r.get("title","")+". "+ab)
            if a.cue and not cue.search(txt): continue
            docs.append({"pmid":r.get("pmid"),"text":txt[:3000]})
            if len(docs)>=a.n: break
        if len(docs)>=a.n: break
    print(f"sampled {len(docs)} abstracts (cue-filter={a.cue})",flush=True)
    PROMPT='''Read the biomedical abstract. Extract symptoms/findings the patient(s) present with, and for each the ONSET only if the text states how it began.

ABSTRACT:
"""{src}"""

For each finding, onset = exactly "sudden" or "gradual" ONLY if the text explicitly says the finding began suddenly/abruptly/acutely (->sudden) or gradually/insidiously/slowly (->gradual). Otherwise onset="".
Output ONLY JSON: {{"findings":[{{"name":"...","onset":""}}]}}'''
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    outs=llm.chat([[{"role":"user","content":PROMPT.format(src=d["text"])}] for d in docs],SamplingParams(temperature=0.0,max_tokens=1024),use_tqdm=True)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    res=[]; nonset=0
    for d,o in zip(docs,outs):
        srcl=d["text"].lower(); txt=o.outputs[0].text
        m=re.search(r'(\{.*\})',txt,re.DOTALL); finds=[]
        if m:
            try: finds=json.loads(m.group(1)).get("findings",[])
            except: pass
        ons=[]
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower(); on=str(f.get("onset","")).strip().lower()
            if on in ("sudden","gradual") and len(nm)>=2 and in_src(nm,srcl): ons.append({"name":nm,"onset":on}); nonset+=1
        if ons: res.append({"pmid":d["pmid"],"text":d["text"],"onset_findings":ons})
    json.dump(res,open("pilot/data/cache/maccrobat/v134_onset_pubmed.json","w"))
    print(f"\n=== onset 수율: {nonset} onset instances in {len(docs)} abstracts ({len(res)} abstracts had >=1) ===")
    print("examples:")
    for r in res[:12]:
        for x in r["onset_findings"][:2]: print(f"   [{r['pmid']}] {x['onset']:8} {x['name'][:40]}")
main()
