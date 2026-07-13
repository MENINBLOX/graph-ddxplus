"""Re-run onset IE with commercial API (gemini-3.1-pro) on the gemma onset-candidate
abstracts (131) — cost-efficient (1/8 of running on all 1000). gemini extracts onset
independently; output = clean onset labels for professor precision validation."""
import os, json, re, time
from google import genai
for line in open(".env"):
    if line.startswith("GEMINI_API_KEY="): os.environ["GEMINI_API_KEY"]=line.strip().split("=",1)[1]
PROMPT='''Read the biomedical abstract. Extract the patient's clinical symptoms/findings and, for each, the ONSET only if the text explicitly states how it began.

ABSTRACT:
"""{src}"""

onset = exactly "sudden" or "gradual": "sudden" if begun suddenly/abruptly/acutely; "gradual" if gradually/insidiously/slowly/progressively-over-time. Otherwise "".
Only real patient clinical findings (not study terms, not substances). Output ONLY JSON: {{"findings":[{{"name":"...","onset":""}}]}}'''
def main():
    docs=json.load(open("pilot/data/cache/maccrobat/v134_onset_pubmed.json"))  # 131 onset-candidate abstracts
    client=genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    out=[]; nonset=0
    for i,d in enumerate(docs):
        srcl=d["text"].lower(); txt=""
        for att in range(4):
            try: txt=client.models.generate_content(model="gemini-3.5-flash",contents=PROMPT.format(src=d["text"])).text; break
            except Exception: time.sleep(4*(att+1))
        m=re.search(r'(\{.*\})',txt,re.DOTALL); finds=[]
        if m:
            try: finds=json.loads(m.group(1)).get("findings",[])
            except: pass
        ons=[]
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower(); on=str(f.get("onset","")).strip().lower()
            if on in ("sudden","gradual") and len(nm)>=2 and in_src(nm,srcl): ons.append({"name":nm,"onset":on}); nonset+=1
        if ons: out.append({"pmid":d["pmid"],"text":d["text"],"onset_findings":ons})
        if i%30==0: print(f"{i}/{len(docs)}",flush=True)
    json.dump(out,open("pilot/data/cache/maccrobat/v135_gemini_onset.json","w"))
    print(f"\n=== gemini onset: {nonset} instances in {len(out)} abstracts (from {len(docs)} candidates) ===")
    for r in out[:12]:
        for x in r["onset_findings"][:2]: print(f"   [{r['pmid']}] {x['onset']:8} {x['name'][:42]}")
main()
