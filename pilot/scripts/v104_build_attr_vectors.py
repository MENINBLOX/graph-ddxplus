"""Build per-attribute token channels for patient & disease (DDXPlus active 6:
location, radiation, severity, onset, character, timing). location/radiation/
character -> UMLS CUIs (patient via value_cuis, disease via scispaCy). severity/
onset/timing -> bucketed category tokens (shared vocab). base = bare symptom +
history CUIs (attribute values removed). Saves vectors for the ablation."""
import sys, json, csv, ast, pickle, glob, re
from collections import defaultdict
import spacy
from scispacy.linking import EntityLinker

VC="/windows/data/medkg/kg/ddxplus_evidence_value_cuis.json"
ATTR_SUFFIX={"endroitducorps":"location","precis":"location","irrad":"radiation",
    "intens":"severity","prurit":"severity","sev":"severity","soudain":"onset",
    "carac":"character","aboy":"character","noct":"timing","nuit":"timing"}
def suffix_of(eid):
    for suf,at in ATTR_SUFFIX.items():
        if eid.lower().endswith("_"+suf) or eid.lower()==suf: return at,suf
    return None,None
def bucket_sev(v):
    try: x=float(v)
    except: return None
    return "sev_mild" if x<=3 else ("sev_moderate" if x<=6 else "sev_severe")
def bucket_onset(v):  # DDXPlus soudain: higher = more sudden
    try: x=float(v)
    except: return None
    return "onset_sudden" if x>=5 else "onset_gradual"

def main():
    vc=json.load(open(VC)); evmeta=json.load(open("data/ddxplus/release_evidences.json"))
    icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    cond=json.load(open("data/ddxplus/release_conditions_en.json"))
    fr2cui={info.get("cond-name-fr",""):icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    # ---- patients ----
    patients=[]
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if len(patients)>=3000: break
            tc=fr2cui.get(row["PATHOLOGY"])
            if not tc: continue
            evs=ast.literal_eval(row["EVIDENCES"])
            base=set(); chan=defaultdict(set)
            for ev in evs:
                eid=ev.split("_@_")[0] if "_@_" in ev else ev
                val=ev.split("_@_")[1] if "_@_" in ev else None
                at,suf=suffix_of(eid)
                m=vc.get(eid,{})
                if at is None:
                    base.update(m.get("_question",[]) or [])
                    if val is not None: base.update(m.get(val,[]) or [])
                else:
                    if at in ("location","radiation","character"):
                        if val is not None: chan[at].update(m.get(val,[]) or [])
                    elif at=="severity":
                        b=bucket_sev(val);
                        if b: chan["severity"].add(b)
                    elif at=="onset":
                        b=bucket_onset(val)
                        if b: chan["onset"].add(b)
                    elif at=="timing":
                        chan["timing"].add("timing_nocturnal")
            patients.append({"true":tc,"base":base,"attr":{k:set(v) for k,v in chan.items()}})
    # ---- diseases (v104 IE) ----
    files=sorted(glob.glob("pilot/data/cache/v104_attr13_ie/*.json"))
    # collect strings to scispaCy-map: phenotype names + location/radiation/character values
    str_set=set(); raw=[]
    for fp in files:
        o=json.load(open(fp)); raw.append(o)
        for f in o["findings"]:
            str_set.add(f["name"])
            for at in ("location","radiation","character"):
                v=f.get(at,"")
                if v:
                    for part in re.split(r'[,/;]| and | or ',v):
                        part=re.sub(r'\(.*?\)','',part).strip()
                        if len(part)>2: str_set.add(part)
    strs=sorted(str_set)
    print(f"scispaCy mapping {len(strs)} strings...",flush=True)
    nlp=spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker",config={"resolve_abbreviations":True,"linker_name":"umls","k":3,"threshold":0.82,"max_entities_per_mention":1})
    s2c={}
    for doc,s in zip(nlp.pipe(strs,batch_size=256),strs):
        best=None;bs=0.0
        for ent in doc.ents:
            for cui,sc in ent._.kb_ents:
                if sc>bs: bs=sc;best=cui
        if best: s2c[s]=best
    def mapstr(v):
        out=set()
        for part in re.split(r'[,/;]| and | or ',v):
            part=re.sub(r'\(.*?\)','',part).strip()
            if part in s2c: out.add(s2c[part])
        return out
    diseases={}
    for o in raw:
        dc=o["cui"]; base=set(); chan=defaultdict(set)
        for f in o["findings"]:
            pc=s2c.get(f["name"])
            if pc and pc!=dc: base.add(pc)
            for at in ("location","radiation","character"):
                v=f.get(at,"")
                if v: chan[at]|=mapstr(v)
            sv=f.get("severity","")
            if sv:
                if "mild" in sv: chan["severity"].add("sev_mild")
                elif "moder" in sv: chan["severity"].add("sev_moderate")
                elif "sever" in sv: chan["severity"].add("sev_severe")
            on=f.get("onset","")
            if on:
                if "sudden" in on or "acute" in on: chan["onset"].add("onset_sudden")
                elif "gradual" in on or "insid" in on: chan["onset"].add("onset_gradual")
            tm=f.get("timing","")
            if tm and ("noct" in tm or "night" in tm): chan["timing"].add("timing_nocturnal")
        diseases[dc]={"base":base,"attr":{k:set(v) for k,v in chan.items()}}
    pickle.dump({"patients":patients,"diseases":diseases},open("pilot/data/cache/v104_attr_vectors.pkl","wb"))
    # coverage report
    from collections import Counter
    pcov=Counter(); dcov=Counter()
    for p in patients:
        for at,s in p["attr"].items():
            if s: pcov[at]+=1
    for d in diseases.values():
        for at,s in d["attr"].items():
            if s: dcov[at]+=1
    print("환자 속성 보유(케이스수):",dict(pcov))
    print("질환 속성 보유(질환수):",dict(dcov))
    print(f"saved: {len(patients)} patients, {len(diseases)} diseases")
main()
