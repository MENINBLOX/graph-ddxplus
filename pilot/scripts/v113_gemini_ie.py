"""Model-capacity diagnostic: run the IDENTICAL v106 prompt + identical source +
identical post-processing on the 49 DDXPlus disease sources, but with
gemini-3.1-pro-preview instead of gemma-4-E4B. Controlled — only the MODEL changes.
Then v112_attr_faithfulness.py on this output tests whether the severity
mis-binding (gemma faithfulness 18%) is a model-capacity issue or a prompt/task issue.
"""
import os, json, glob, re, time
from pathlib import Path
from google import genai

# load GEMINI_API_KEY from .env
for line in open(".env"):
    if line.startswith("GEMINI_API_KEY="):
        os.environ["GEMINI_API_KEY"]=line.strip().split("=",1)[1]

ATTRS=["location","onset","duration","character","severity","radiation","timing","aggravating","relieving","associated","course","context","prior_episodes"]
# EXACT v106 prompt
PROMPT='''You are a clinical information extractor. Read the SOURCE about "{disease}" and extract, USING ONLY facts explicitly stated in the SOURCE, the symptoms/signs a patient experiences and their clinically discriminative attributes.

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
- severity: exactly one word: "mild" or "moderate" or "severe". Never "severe cases"/"severe disease".
- location: a specific body site only (e.g. "cheeks","groin","left chest").
- character: the QUALITY of the symptom (e.g. "sharp","dull","burning","throbbing","productive"). Do NOT define or restate the symptom itself (for "cyanosis" do NOT write "bluish skin"; for "hypoxia" do NOT write "low oxygen").
- radiation/timing/aggravating/relieving/duration: short standard term only if explicitly stated.
- Never put the disease name or a category into any attribute (no context="{disease}", no context="common symptom").
- associated: another specific finding that co-occurs, if stated.

Output exactly:
JSON: {{"findings":[{{"name":"...","location":"","onset":"","duration":"","character":"","severity":"","radiation":"","timing":"","aggravating":"","relieving":"","associated":"","course":"","context":"","prior_episodes":""}}]}}'''

MODEL="gemini-3.1-pro-preview"
OUT="pilot/data/cache/v106_gemini_ie"; Path(OUT).mkdir(parents=True,exist_ok=True)
icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
cui2name={info["cui"]:dn for dn,info in icd.items() if "cui" in info}
items=[]
for fp in sorted(glob.glob("pilot/data/cache/v105_sources/*.txt")):
    c=fp.split("/")[-1][:-4]
    if c in cui2name: items.append((c,cui2name[c],open(fp).read()))

client=genai.Client(api_key=os.environ["GEMINI_API_KEY"])
STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)

kept=dropped=0
for c,dn,src in items:
    if os.path.exists(f"{OUT}/{c}.json"): continue
    srcl=src.lower()
    for attempt in range(4):
        try:
            resp=client.models.generate_content(model=MODEL,contents=PROMPT.format(disease=dn,src=src[:2200]))
            txt=resp.text; break
        except Exception as e:
            print(f"  retry {dn}: {str(e)[:80]}",flush=True); time.sleep(5*(attempt+1)); txt=""
    m=re.search(r'JSON:\s*(\{.*\})',txt,re.DOTALL) or re.search(r'(\{.*\})',txt,re.DOTALL)
    finds=[]
    if m:
        try: finds=json.loads(m.group(1)).get("findings",[])
        except Exception: pass
    clean=[]
    for f in finds:
        if not isinstance(f,dict): continue
        nm=str(f.get("name","")).strip().lower()
        if len(nm)<2 or nm==dn.lower() or not in_src(nm,srcl): dropped+=1; continue
        rec={"name":nm}
        for at in ATTRS:
            v=str(f.get(at,"")).strip().lower()
            rec[at]=v if (v and in_src(v,srcl)) else ""
            if rec[at]: kept+=1
        clean.append(rec)
    json.dump({"disease":dn,"cui":c,"findings":clean},open(f"{OUT}/{c}.json","w"))
    print(f"  {dn}: {len(clean)} findings",flush=True)
print(f"\nMODEL={MODEL} attr kept={kept} dropped={dropped}")
