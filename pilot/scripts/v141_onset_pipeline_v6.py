"""V6 onset pipeline = v139 two-pass + strengthened PASS2 verification targeting the two
dominant residual failure themes found by n=30 source-grounded judging of the 254-set:
 (1) GENERIC/DEFINITIONAL statements about a disease class ("Chorea: suddenly appearing
     movements", "Acute eosinophilic pneumonia is a sudden febrile illness", "Symptoms of
     rupture are typically sudden") — not a specific patient's onset.
 (2) tempo word modifies a TRIGGER/MECHANISM ("provoked by sudden movement", "in sudden
     ocular decompression"), not the symptom's own beginning.
PASS1 (stepwise-gate extraction) unchanged. Source-grounded, benchmark-blind, NO few-shot."""
import json, re, argparse
from v137_onset_loop import V4_RULES
from v139_onset_pipeline import EXTRACT, NONSYMPTOM, in_src

VERIFY_V6='''A biomedical abstract and ONE clinical finding from it are given.

ABSTRACT:
"""{src}"""

FINDING: "{finding}"

Question: does the abstract describe, for a SPECIFIC PATIENT (or patients) in THIS report, an
EXPLICIT temporal-onset-quality phrase for how THIS finding ("{finding}") BEGAN?
- "sudden": sudden/suddenly/abrupt/abruptly/"acute onset of {finding}"/within-minutes-or-hours, attached to THIS finding's beginning in a real patient.
- "gradual": gradual/gradually/insidious/"slowly progressive"/over-weeks-months-years, attached to THIS finding's beginning in a real patient.
- "none" in ALL of these:
   * GENERIC / DEFINITIONAL / EPIDEMIOLOGIC statement about a disease class rather than a specific patient — e.g. "X is a sudden febrile illness", "Symptoms of X are typically sudden", "Chorea: suddenly appearing movements", "usually presents acutely". (no individual patient onset)
   * the tempo word modifies a TRIGGER or MECHANISM, not the symptom's own onset — e.g. "provoked by sudden movement", "in sudden ocular decompression", "after sudden exertion".
   * the tempo word actually modifies a DIFFERENT co-listed finding, or the disease name.
   * only an AGE of onset; or only a sequence word ("developed","presented with") with no sudden/gradual quality.
   * the finding is a disease/syndrome/test/lab/imaging result/mortality outcome/animal-model term rather than a patient symptom.
Be strict: if the onset is not unambiguously a specific patient's symptom beginning, answer "none".
Output ONLY JSON: {{"onset":"sudden|gradual|none"}}'''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pool",default="pilot/data/cache/maccrobat/onset_pool_large.json")
    ap.add_argument("--out",default="pilot/data/cache/maccrobat/onset_verified_set_v6.json"); a=ap.parse_args()
    pool=json.load(open(a.pool))
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
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
    print(f"PASS1 extract: {len(cand)} candidates on {len(pool)} abstracts",flush=True)
    vouts=llm.chat([[{"role":"user","content":VERIFY_V6.format(src=c["abstract"],finding=c["finding"])}] for c in cand],
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
    print(f"PASS2 V6 verify: kept {len(items)}/{len(cand)} -> verified set. dist={dict(Counter(x['onset'] for x in items))}")

if __name__=="__main__": main()
