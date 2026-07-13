"""gemma vs gemini matched comparison WITHOUT NLI (robust to env breakage).
Same 14 diseases Gemini finished. Identical prompt/source/post-processing.
Key metric = severity source-binding rate (the exact defect: does the source
literally bind the severity term to THAT symptom, vs case-level severity glued on).
Same heuristic applied to both models -> fair relative comparison."""
import json, glob, os, re
from collections import defaultdict

GEM="pilot/data/cache/v106_gemini_ie"; GMA="pilot/data/cache/v106_grounded_ie"
CUIS=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{GEM}/*.json"))
CORE=["location","severity","onset","character"]
SRC={fp.split("/")[-1][:-4]:open(fp).read().lower() for fp in glob.glob("pilot/data/cache/v105_sources/*.txt")}

def binds(sev,name,src):
    toks=[t for t in re.findall(r'[a-z]+',name.lower()) if len(t)>3]
    if not toks: return False
    # severity adjacent to symptom head/tail within source (<=40 chars)
    for t in toks:
        if re.search(re.escape(sev)+r'\s+\w*\s*'+re.escape(t), src) or re.search(re.escape(t)+r'[^.]{0,40}'+re.escape(sev), src):
            return True
    return False

print(f"Matched diseases: {len(CUIS)}\n")
for label,d in [("gemma-4-E4B",GMA),("gemini-3.1-pro-preview",GEM)]:
    nf=0; fill=defaultdict(int); sev_bound=0; sev_tot=0; char_multi=0; char_tot=0; sev_redundant=0
    for c in CUIS:
        fp=f"{d}/{c}.json"
        if not os.path.exists(fp): continue
        o=json.load(open(fp)); src=SRC.get(c,"")
        for f in o.get("findings",[]):
            nf+=1; nm=f.get("name","")
            for a in CORE:
                v=str(f.get(a,"")).strip()
                if v: fill[a]+=1
            sev=str(f.get("severity","")).strip()
            if sev:
                sev_tot+=1
                if binds(sev,nm,src): sev_bound+=1
                if sev in nm.lower(): sev_redundant+=1   # severity already inside the name
            ch=str(f.get("character","")).strip()
            if ch:
                char_tot+=1
                if "," in ch or " or " in ch: char_multi+=1
    print(f"=== {label} ===  findings={nf}")
    print(f"  fill: " + " ".join(f"{a}={fill[a]}" for a in CORE))
    print(f"  severity: total={sev_tot}  source-bound={sev_bound} ({100*sev_bound/max(sev_tot,1):.0f}%)  redundant-in-name={sev_redundant}")
    print(f"  character: total={char_tot}  multi-value(noisy)={char_multi} ({100*char_multi/max(char_tot,1):.0f}%)\n")

# side-by-side severity examples
print("=== severity examples (symptom | severity | source-binds) ===")
for label,d in [("gemma",GMA),("gemini",GEM)]:
    print(f"-- {label} --")
    shown=0
    for c in CUIS:
        fp=f"{d}/{c}.json"
        if not os.path.exists(fp): continue
        o=json.load(open(fp)); src=SRC.get(c,"")
        for f in o.get("findings",[]):
            sev=str(f.get("severity","")).strip()
            if sev and shown<8:
                print(f"   '{f['name'][:32]:32}' sev={sev:8} bound={binds(sev,f['name'],src)}"); shown+=1
