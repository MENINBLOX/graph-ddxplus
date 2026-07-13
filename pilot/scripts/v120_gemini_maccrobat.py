"""Run gemini-3.1-pro on MACCROBAT held-out TEST docs with baseline vs M2 (best
dev-tuned) severity prompt. Tests: (a) does frontier model raise absolute F1,
(b) does the M2 fix transfer to a stronger model (mechanism generality).
Only TEST docs called (cost); preds aligned to full 200 (dev entries = [])."""
import os, json, re, time
from google import genai

for line in open(".env"):
    if line.startswith("GEMINI_API_KEY="): os.environ["GEMINI_API_KEY"]=line.strip().split("=",1)[1]

HEAD='''You are a clinical information extractor. Read the clinical CASE TEXT and extract, USING ONLY facts explicitly stated in it, every sign or symptom the patient shows, with its severity.

CASE TEXT:
"""{src}"""

STEP 1 — Reasoning: list the concrete patient signs/symptoms the text mentions.
STEP 2 — Output one JSON object.

RULES — sign/symptom: INCLUDE a specific symptom or clinical sign the patient feels or shows.{syminc} EXCLUDE tests/procedures, medications, final diagnoses, lab values, headings.
{sevrule}

Output exactly:
JSON: {{"findings":[{{"name":"...","severity":""}}]}}'''
BASE=('- severity: one word "mild", "moderate", or "severe" if the text states it.', "")
M2=('- severity: the short intensity word the text attaches DIRECTLY to THIS symptom, copied as it appears (e.g. "severe","mild","massive","profuse","marked"). Fill ONLY when the intensity word modifies THIS symptom ("severe headache"->severe). If the text only calls the disease/case severe, or no intensity word is attached to this symptom, leave EMPTY.',
    " INCLUDE physical examination signs too (e.g. murmur, hepatomegaly, edema, rash, pallor, swelling), not only patient-reported symptoms.")
MODEL="gemini-3.1-pro-preview"

def main():
    docs=json.load(open("pilot/data/cache/maccrobat/MACCROBAT2020-V2.json"))["data"]
    split=json.load(open("pilot/data/cache/maccrobat/split.json")); test=set(split["test"])
    import glob
    txt2pmid={open(t).read().strip()[:120]:t.split("/")[-1][:-4] for t in glob.glob("pilot/data/cache/maccrobat/brat/*.txt")}
    idx2pmid=[txt2pmid.get(d["full_text"].strip()[:120]) for d in docs]
    client=genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    for vname,(sevrule,syminc) in {"v120_gem_base":BASE,"v120_gem_M2":M2}.items():
        res=[]
        for i,d in enumerate(docs):
            if idx2pmid[i] not in test: res.append([]); continue
            srcl=d["full_text"].lower(); txt=""
            for att in range(4):
                try:
                    txt=client.models.generate_content(model=MODEL,contents=HEAD.format(src=d["full_text"][:3500],sevrule=sevrule,syminc=syminc)).text; break
                except Exception as e:
                    time.sleep(4*(att+1))
            m=re.search(r'JSON:\s*(\{.*\})',txt,re.DOTALL) or re.search(r'(\{.*\})',txt,re.DOTALL)
            finds=[]
            if m:
                try: finds=json.loads(m.group(1)).get("findings",[])
                except Exception: pass
            out=[]
            for f in finds:
                if not isinstance(f,dict): continue
                nm=str(f.get("name","")).strip().lower()
                if len(nm)<2 or not in_src(nm,srcl): continue
                sev=str(f.get("severity","")).strip().lower()
                out.append({"name":nm,"severity":sev if sev else ""})
            res.append(out)
            if i%20==0: print(f"{vname} {i}/200",flush=True)
        json.dump(res,open(f"pilot/data/cache/maccrobat/{vname}_pred.json","w"))
        print(f"{vname}: done, severity-bearing={sum(1 for doc in res for x in doc if x['severity'])}",flush=True)

if __name__=="__main__": main()
