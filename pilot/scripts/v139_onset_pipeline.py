"""Onset IE pipeline (converged design): two-pass = V4 stepwise-gate extraction ->
V5 focused per-finding verification, + deterministic non-symptom filter (mortality/
animal-model are definitionally not patient symptoms). Diagnosis from the prompt loop:
single-pass multi-finding extraction plateaus at ~50% precision (tempo bleed-over,
age-of-onset leak); the focused second pass lifts precision to ~75%+, and the stop-list
removes the residual non-patient-finding leak. Runs on a large cue-filtered pool to
accumulate a high-precision onset set for clinician validation. Source-grounded,
benchmark-blind, NO few-shot."""
import json, re, argparse
from v137_onset_loop import V4_RULES  # reuse the validated stepwise-gate extraction rules

EXTRACT='''Read the biomedical abstract. Extract the patient's clinical symptoms/findings and, for each, the ONSET.

ABSTRACT:
"""{src}"""

'''+V4_RULES+'''
Output ONLY JSON: {{"findings":[{{"name":"...","onset":""}}]}}'''

VERIFY='''A biomedical abstract and ONE clinical finding from it are given.

ABSTRACT:
"""{src}"""

FINDING: "{finding}"

Question: does the abstract contain an EXPLICIT temporal-onset-quality phrase that describes
how THIS finding ("{finding}") BEGAN in the patient?
- "sudden" only if sudden/suddenly/abrupt/abruptly/"acute onset of {finding}"/within-minutes-or-hours is attached to THIS finding.
- "gradual" only if gradual/gradually/insidious/"slowly progressive"/over-weeks-months-years is attached to THIS finding.
- "none" if: the tempo word actually modifies a DIFFERENT finding or the disease name; OR only an AGE of onset is given; OR only a sequence word ("developed","presented with") with no sudden/gradual quality; OR this finding is a disease/syndrome/test/lab/imaging result/mortality outcome/animal-model term rather than a patient symptom.
Be strict: if the tempo phrase is not unambiguously bound to THIS finding's beginning, answer "none".
Output ONLY JSON: {{"onset":"sudden|gradual|none"}}'''

# definitional non-symptom stop terms (mortality outcomes / animal-model markers in the FINDING)
NONSYMPTOM=re.compile(r'\b(death|died|mortality|fatal|fatality|survival|necrosis|murine|mouse|mice|rat|knockout|model)\b',re.I)

STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pool",default="pilot/data/cache/maccrobat/onset_pool_large.json")
    ap.add_argument("--out",default="pilot/data/cache/maccrobat/onset_verified_set.json"); a=ap.parse_args()
    pool=json.load(open(a.pool))
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})

    # PASS 1 — stepwise-gate extraction
    outs=llm.chat([[{"role":"user","content":EXTRACT.format(src=d["text"])}] for d in pool],
                  SamplingParams(temperature=0.0,max_tokens=1024),use_tqdm=True)
    cand=[]
    for d,o in zip(pool,outs):
        srcl=d["text"].lower(); m=re.search(r'(\{.*\})',o.outputs[0].text,re.DOTALL); finds=[]
        if m:
            try: finds=json.loads(m.group(1)).get("findings",[])
            except: pass
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower(); on=str(f.get("onset","")).strip().lower()
            if on in ("sudden","gradual") and len(nm)>=2 and in_src(nm,srcl) and not NONSYMPTOM.search(nm):
                cand.append({"pmid":d["pmid"],"finding":nm,"abstract":d["text"]})
    print(f"PASS1 extract: {len(cand)} candidate onset instances on {len(pool)} abstracts",flush=True)

    # PASS 2 — focused per-finding verification
    vouts=llm.chat([[{"role":"user","content":VERIFY.format(src=c["abstract"],finding=c["finding"])}] for c in cand],
                   SamplingParams(temperature=0.0,max_tokens=64),use_tqdm=True)
    items=[]
    for c,o in zip(cand,vouts):
        m=re.search(r'\{[^{}]*\}',o.outputs[0].text,re.DOTALL); on="none"
        if m:
            try: on=str(json.loads(m.group(0)).get("onset","none")).strip().lower()
            except: pass
        if on in ("sudden","gradual") and not NONSYMPTOM.search(c["finding"]):
            items.append({"idx":len(items),"pmid":c["pmid"],"finding":c["finding"],"onset":on,"abstract":c["abstract"]})
    json.dump(items,open(a.out,"w"),ensure_ascii=False,indent=0)
    from collections import Counter
    print(f"PASS2 verify: kept {len(items)}/{len(cand)} -> verified onset set. dist={dict(Counter(x['onset'] for x in items))}")

if __name__=="__main__": main()
