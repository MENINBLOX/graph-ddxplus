"""Build 13-attr channels (disease via scispaCy + patient via DDXPlus) and run a
QUALITATIVE TRACE to verify the attribute eval actually fires (not silent no-op).
Prints, for sample patients: which attrs have patient values, and whether each
matches the GT disease's attribute values (the intersection that drives scoring)."""
import sys, json, csv, ast, glob, re, pickle
from collections import defaultdict,Counter
import spacy
from scispacy.linking import EntityLinker
VC="/windows/data/medkg/kg/ddxplus_evidence_value_cuis.json"
value_cuis=json.load(open(VC)); evmeta=json.load(open("data/ddxplus/release_evidences.json"))
icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json")); cond=json.load(open("data/ddxplus/release_conditions_en.json"))
fr2cui={info.get("cond-name-fr",""):icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
cui2name={info["cui"]:dn for dn,info in icd.items() if "cui" in info}
ALL13=["location","onset","duration","character","severity","radiation","timing","aggravating","relieving","associated","course","context","prior_episodes"]
CUI_ATTR=["location","radiation","character","associated","aggravating","relieving","context"]  # concept-like -> CUI
TOK_ATTR=["onset","duration","severity","timing","course","prior_episodes"]  # categorical -> token
def comps(v):
    v=re.sub(r'\(.*?\)','',v.lower())
    return [p.strip(" .-") for p in re.split(r'[,/;]| and | or | to ',v) if len(p.strip(" .-"))>2]
def tok(a,v):
    v=v.lower()
    if a=="severity": return {"sev_mild"} if "mild" in v else {"sev_severe"} if "sever" in v else {"sev_moderate"} if "moder" in v else set()
    if a=="onset": return {"onset_sudden"} if ("sudden" in v or "acute" in v) else {"onset_gradual"} if ("gradual" in v or "insid" in v) else set()
    if a=="timing":
        out=set()
        for k in ["nocturnal","night","constant","intermittent","morning","post-prandial"]:
            if k in v: out.add("tim_"+k.replace("night","nocturnal"))
        return out
    return {a+"_"+re.sub(r'[^a-z]','',v)[:12]} if v else set()
# --- disease side via scispaCy ---
files=sorted(glob.glob("pilot/data/cache/v104c_attr_ie/*.json")); data=[json.load(open(f)) for f in files]
strs=set()
for o in data:
    for f in o["findings"]:
        strs.add(f["name"])
        for a in CUI_ATTR:
            if f.get(a):
                for c in comps(f[a]): strs.add(c)
strs=sorted(strs)
print(f"scispaCy {len(strs)} strings...",flush=True)
nlp=spacy.load("en_core_sci_lg"); nlp.add_pipe("scispacy_linker",config={"resolve_abbreviations":True,"linker_name":"umls","k":3,"threshold":0.82,"max_entities_per_mention":1})
s2c={}
for doc,s in zip(nlp.pipe(strs,batch_size=256),strs):
    best=None;bs=0.0
    for ent in doc.ents:
        for cui,sc in ent._.kb_ents:
            if sc>bs:bs=sc;best=cui
    if best:s2c[s]=best
def mapcui(v): 
    o=set()
    for c in comps(v):
        if c in s2c: o.add(s2c[c])
    return o
dz={}  # disease -> {base:set, attr:{a:set}}
for o in data:
    dc=o["cui"]; base=set(); at=defaultdict(set)
    for f in o["findings"]:
        pc=s2c.get(f["name"])
        if pc and pc!=dc: base.add(pc)
        for a in ALL13:
            v=f.get(a,"")
            if not v: continue
            at[a]|= mapcui(v) if a in CUI_ATTR else tok(a,v)
    dz[dc]={"base":base,"attr":{a:set(s) for a,s in at.items()}}
# --- patient side from DDXPlus ---
ATTR_SUF={"endroitducorps":"location","precis":"location","irrad":"radiation","intens":"severity","prurit":"severity","sev":"severity","soudain":"onset","carac":"character","aboy":"character","noct":"timing","nuit":"timing"}
def sufof(eid):
    for s,a in ATTR_SUF.items():
        if eid.lower().endswith("_"+s) or eid.lower()==s: return a
    return None
def patient(evs):
    base=set(); at=defaultdict(set)
    for ev in evs:
        eid=ev.split("_@_")[0] if "_@_" in ev else ev
        val=ev.split("_@_")[1] if "_@_" in ev else None
        a=sufof(eid); m=value_cuis.get(eid,{})
        if a is None:
            base|=set(m.get("_question",[]) or [])
            if val: base|=set(m.get(val,[]) or [])
        elif a in ("location","radiation","character"):
            if val: at[a]|=set(m.get(val,[]) or [])
        elif a=="severity":
            try: x=float(val); at["severity"]|={"sev_mild" if x<=3 else "sev_moderate" if x<=6 else "sev_severe"}
            except: pass
        elif a=="onset":
            try: x=float(val); at["onset"]|={"onset_sudden" if x>=5 else "onset_gradual"}
            except: pass
        elif a=="timing": at["timing"]|={"tim_nocturnal"}
    return base,at
pickle.dump({"dz":dz,"s2c":s2c},open("pilot/data/cache/v104c_attr.pkl","wb"))
# === QUALITATIVE TRACE ===
print("\n"+"="*72)
print("정성평가 트레이스: 환자 속성값이 GT 질환 속성값과 실제 매칭되는가?")
print("="*72)
targets={"SLE":2,"Pulmonary embolism":1,"Inguinal hernia":1}
seen=Counter()
with open("data/ddxplus/release_test_patients.csv") as f:
    for row in csv.DictReader(f):
        nm=None
        tc=fr2cui.get(row["PATHOLOGY"])
        nm=cui2name.get(tc)
        if nm not in targets or seen[nm]>=targets[nm]: continue
        seen[nm]+=1
        evs=ast.literal_eval(row["EVIDENCES"])
        pbase,pat=patient(evs)
        d=dz.get(tc,{"base":set(),"attr":{}})
        print(f"\n■ 정답={nm}")
        print(f"  환자가 값을 가진 속성: {[a for a in ALL13 if pat.get(a)]}")
        print(f"  환자가 값 없는 속성(DDXPlus no-op): {[a for a in ALL13 if not pat.get(a)]}")
        for a in ALL13:
            pv=pat.get(a,set()); dv=d['attr'].get(a,set())
            if not pv: continue
            inter=pv&dv
            print(f"    [{a}] 환자값={pv} | GT질환값={list(dv)[:5]} | 매칭={inter if inter else '✗ 불일치'}")
        if sum(seen.values())>=sum(targets.values()): break
print("\n빌드 완료: pilot/data/cache/v104c_attr.pkl")
