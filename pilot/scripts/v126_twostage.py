"""Two-stage extract-then-bind. Decouples finding-recall from severity-precision:
Stage 1 (reuse M8's broad findings, 3259) -> high finding recall.
Stage 2: per-finding focused query "does the text give THIS finding an intensity
grading word?" -> precision (extra findings just get NONE and drop out).
Then deterministic measurement filter. DEV-select, test held out. gemma-4-E4B."""
import json, glob, re

Q='''CASE TEXT:
"""{src}"""

Question: In the CASE TEXT above, is there an intensity-GRADING word placed directly on the finding "{finding}" that says HOW SEVERE or MILD that finding is (e.g. mild, moderate, severe, marked, slight, extensive, significant, minimal, minor, small, large, massive, profuse, advanced)?
- If yes, answer with ONLY that single grading word, copied from the text.
- If the text gives no such grading word for this finding, or only describes the overall disease/case severity, answer ONLY: NONE
- Timing/course words (sudden, acute, chronic, recurrent, persistent, progressive, sustained) are NOT severity -> answer NONE.

Answer:'''
BAD=re.compile(r'\d|cm|mm|°|%|kg|mg|ml|grade|fatal|benign')
def clean(ans):
    a=ans.strip().lower().split("\n")[0].strip().strip('.').strip()
    a=re.sub(r'^(answer:?\s*)','',a)
    if not a or a.startswith("none") or len(a)>20 or BAD.search(a): return ""
    # take first word-ish token (allow 'mild to moderate')
    if len(a.split())>3: return ""
    return a

def main():
    docs=json.load(open("pilot/data/cache/maccrobat/MACCROBAT2020-V2.json"))["data"]
    m8=json.load(open("pilot/data/cache/maccrobat/v125_M8_pred.json"))
    pairs=[]  # (doc_idx, finding)
    for i,doc in enumerate(m8):
        for x in doc: pairs.append((i,x["name"]))
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    prompts=[[{"role":"user","content":Q.format(src=docs[i]["full_text"][:3500],finding=f)}] for i,f in pairs]
    outs=llm.chat(prompts,SamplingParams(temperature=0.0,max_tokens=24),use_tqdm=True)
    res=[[] for _ in docs]
    for (i,f),o in zip(pairs,outs):
        sev=clean(o.outputs[0].text)
        res[i].append({"name":f,"severity":sev})
    json.dump(res,open("pilot/data/cache/maccrobat/v126_twostage_pred.json","w"))
    print(f"v126 two-stage: sev-bearing={sum(1 for d in res for x in d if x['severity'])}")
main()
