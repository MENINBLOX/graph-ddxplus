"""V5 onset = V4 candidate (finding,onset) instances re-checked by a FOCUSED per-finding
verification pass. Diagnosis (round3): single-pass multi-finding extraction cannot reliably
bind a tempo word to the SPECIFIC finding it modifies -> tempo bleed-over across co-listed
findings + age-of-onset leakage. Fix is architectural, not more prompt text: ask gemma ONE
finding at a time, force it to confirm the tempo phrase attaches to THAT finding's BEGINNING.
Still IE-time, source-grounded, benchmark-blind, NO few-shot. Output = filtered onset set."""
import json, re, argparse

VERIFY='''A biomedical abstract and ONE clinical finding from it are given.

ABSTRACT:
"""{src}"""

FINDING: "{finding}"

Question: does the abstract contain an EXPLICIT temporal-onset-quality phrase that describes
how THIS finding ("{finding}") BEGAN in the patient?
- Answer "sudden" only if a word like sudden/suddenly/abrupt/abruptly/"acute onset of {finding}"/within-minutes-or-hours is attached to THIS finding.
- Answer "gradual" only if a word like gradual/gradually/insidious/"slowly progressive"/over-weeks-months-years is attached to THIS finding.
- Answer "none" if: the tempo word actually modifies a DIFFERENT finding or the disease name (not this one); OR only an AGE of onset is given ("at 13 years", "infantile-onset", "adult onset", "onset at 1.6 years"); OR only a sequence word ("developed", "presented with", "subsequently") with no sudden/gradual quality; OR this finding is a disease/syndrome/test/lab/imaging result/mortality outcome/animal-model term rather than a patient symptom.
Be strict: if the tempo phrase is not unambiguously bound to THIS finding's beginning, answer "none".
Output ONLY JSON: {{"onset":"sudden|gradual|none"}}'''

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--src",default="pilot/data/cache/maccrobat/onset_V4_items.json")
    ap.add_argument("--out",default="pilot/data/cache/maccrobat/onset_V5_items.json"); a=ap.parse_args()
    cand=json.load(open(a.src))
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    prompts=[[{"role":"user","content":VERIFY.format(src=c["abstract"],finding=c["finding"])}] for c in cand]
    outs=llm.chat(prompts,SamplingParams(temperature=0.0,max_tokens=64),use_tqdm=True)
    items=[]; kept=0
    for c,o in zip(cand,outs):
        txt=o.outputs[0].text; m=re.search(r'\{[^{}]*\}',txt,re.DOTALL); on="none"
        if m:
            try: on=str(json.loads(m.group(0)).get("onset","none")).strip().lower()
            except: pass
        if on in ("sudden","gradual"):
            items.append({"idx":len(items),"pmid":c["pmid"],"finding":c["finding"],"onset":on,"abstract":c["abstract"]}); kept+=1
    json.dump(items,open(a.out,"w"),ensure_ascii=False,indent=0)
    print(f"V5 verify: kept {kept}/{len(cand)} candidates after focused per-finding re-check")

if __name__=="__main__": main()
