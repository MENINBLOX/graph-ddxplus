"""Scale-up crawler: fetch Wikipedia clinical/symptom text for a large benchmark-blind
disease set (SymCat 801 + DDXPlus 49), for attribute-IE at scale. Network-only (no GPU).
Output: pilot/data/cache/scaleup_sources/{idx:04d}.json = {"disease":name,"title":t,"text":clinical}
Only disease NAMES are used (benchmark-blind); no benchmark symptom labels touched."""
import json, re, time, glob, requests
from pathlib import Path

OUT=Path("pilot/data/cache/scaleup_sources"); OUT.mkdir(parents=True,exist_ok=True)
HDR={"User-Agent":"GraphDDXPlus-research/1.0 (academic IE; contact max@meninblox.com)"}
API="https://en.wikipedia.org/w/api.php"
SEC_RE=re.compile(r'\n=+\s*(.*?)\s*=+\s*\n')
WANT=["signs and symptoms","symptoms","presentation","clinical features","clinical presentation","signs","characteristics"]

def names():
    s=set()
    d=json.load(open("data/symcat/symcat_parsed_full.json"))
    for n in d.get("diseases",{}): s.add(n.strip())
    icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    for dn in icd: s.add(dn.strip())
    return sorted(x for x in s if x)

def search_title(name):
    try:
        r=requests.get(API,params={"action":"query","list":"search","srsearch":name,"srlimit":1,"format":"json"},headers=HDR,timeout=20)
        h=r.json().get("query",{}).get("search",[])
        return h[0]["title"] if h else None
    except Exception: return None

def get_text(title):
    try:
        r=requests.get(API,params={"action":"query","prop":"extracts","explaintext":1,"redirects":1,"titles":title,"format":"json"},headers=HDR,timeout=25)
        pages=r.json().get("query",{}).get("pages",{})
        for _,p in pages.items(): return p.get("extract","")
    except Exception: return ""
    return ""

def clinical_slice(full):
    # find a symptoms-like section; else use intro
    parts=SEC_RE.split(full)  # [intro, sec1title, sec1body, sec2title, sec2body, ...]
    intro=parts[0]
    secs=[(parts[i].strip().lower(),parts[i+1]) for i in range(1,len(parts)-1,2)]
    for want in WANT:
        for t,body in secs:
            if t==want or t.startswith(want):
                return (body.strip()+"\n"+intro.strip())[:2600]
    # fallback: intro (often lists symptoms)
    return intro.strip()[:2200]

def main():
    ns=names(); print(f"{len(ns)} unique disease names",flush=True)
    done={json.load(open(f))["disease"] for f in glob.glob(str(OUT/"*.json"))} if list(OUT.glob("*.json")) else set()
    idx=len(done); ok=0; fail=0
    for name in ns:
        if name in done: continue
        title=search_title(name); time.sleep(0.15)
        text=get_text(title) if title else ""; time.sleep(0.15)
        cl=clinical_slice(text) if text else ""
        if cl and len(cl)>150:
            json.dump({"disease":name,"title":title,"text":cl},open(OUT/f"{idx:04d}.json","w"))
            idx+=1; ok+=1
        else: fail+=1
        if (ok+fail)%50==0: print(f"  progress ok={ok} fail={fail}",flush=True)
    print(f"DONE crawl: ok={ok} fail={fail} total_files={len(list(OUT.glob('*.json')))}",flush=True)

if __name__=="__main__": main()
