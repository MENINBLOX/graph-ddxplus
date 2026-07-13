"""gemini-3.1-pro-preview 4-attribute IE on the fixed 40-doc validation set
(pmids from Label Studio project 2). User explicitly requested gemini for label gen.
Independent of our gemma system -> reduces anchoring bias in professor review.
Output -> v131_gemini_4attr_pred.json (v129 dict format) for Claude 1차 검토 + LS import."""
import os, json, glob, re, time, requests
from google import genai
for line in open(".env"):
    if line.startswith("GEMINI_API_KEY="): os.environ["GEMINI_API_KEY"]=line.strip().split("=",1)[1]
B="http://localhost:8080"

HEAD='''You are a clinical information extractor. Read the clinical CASE TEXT and extract, USING ONLY facts explicitly stated in it, every abnormal clinical finding the patient has, with four attributes.

CASE TEXT:
"""{src}"""

For each finding, fill an attribute ONLY when the text states it directly for THAT finding; otherwise "".
- onset: exactly "sudden" or "gradual" — how the finding began. NOT acute/chronic/recurrent (those are course).
- character: the QUALITY of the finding (e.g. "sharp","dull","burning","throbbing","productive","itchy","bluish"). Do NOT restate/define the finding itself.
- severity: a single intensity-GRADING word directly on this finding (e.g. mild, moderate, severe, marked, slight, extensive, massive). NOT timing/course words.
- location: the specific anatomical body site directly on this finding (e.g. left lung, liver), verbatim. EXCLUDE non-anatomical sites.
INCLUDE any abnormal finding/sign/symptom/lesion/condition. EXCLUDE tests/procedures, medications, normal findings.

Output ONLY one JSON object:
JSON: {{"findings":[{{"name":"...","onset":"","character":"","severity":"","location":""}}]}}'''

def main():
    # get pmids in project 2
    s=requests.Session(); s.get(B+"/user/login/")
    s.post(B+"/user/login/",data={"csrfmiddlewaretoken":s.cookies.get("csrftoken"),"email":"admin@meninblox.com","password":"adminpass123"},headers={"Referer":B+"/user/login/"})
    H={"X-CSRFToken":s.cookies.get("csrftoken")}
    pmids=[t["data"]["pmid"] for t in s.get(B+"/api/tasks?project=2&page_size=100",headers=H).json()["tasks"]]
    docs=json.load(open("pilot/data/cache/maccrobat/MACCROBAT2020-V2.json"))["data"]
    txt2pmid={open(t).read().strip()[:120]:t.split("/")[-1][:-4] for t in glob.glob("pilot/data/cache/maccrobat/brat/*.txt")}
    pmid2text={txt2pmid.get(d["full_text"].strip()[:120]):d["full_text"] for d in docs}
    client=genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    out={}
    for i,pmid in enumerate(pmids):
        text=pmid2text.get(pmid)
        if not text: continue
        srcl=text.lower(); txt=""
        for att in range(4):
            try: txt=client.models.generate_content(model="gemini-3.1-pro-preview",contents=HEAD.format(src=text[:3500])).text; break
            except Exception as e: time.sleep(4*(att+1))
        m=re.search(r'JSON:\s*(\{.*\})',txt,re.DOTALL) or re.search(r'(\{.*\})',txt,re.DOTALL)
        finds=[]
        if m:
            try: finds=json.loads(m.group(1)).get("findings",[])
            except Exception: pass
        clean=[]
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower()
            if len(nm)<2 or not in_src(nm,srcl): continue
            o=str(f.get("onset","")).strip().lower()
            clean.append({"name":nm,"onset":o if o in ("sudden","gradual") else "",
                          "character":str(f.get("character","")).strip().lower(),
                          "severity":str(f.get("severity","")).strip().lower(),
                          "location":str(f.get("location","")).strip().lower()})
        out[pmid]={"text":text,"findings":clean}
        if i%10==0: print(f"{i}/{len(pmids)}",flush=True)
    json.dump(out,open("pilot/data/cache/maccrobat/v131_gemini_4attr_pred.json","w"))
    nf=sum(len(v["findings"]) for v in out.values())
    print(f"\nv131 gemini 4-attr: docs={len(out)} findings={nf}")
    for a in ["onset","character","severity","location"]:
        print(f"  {a}: {sum(1 for v in out.values() for x in v['findings'] if x[a])}")

main()
