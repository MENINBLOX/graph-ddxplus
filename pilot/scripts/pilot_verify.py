#!/usr/bin/env python3
"""파일럿 재현: 동일 데이터(2,217편 NER CUI) + Ollama S2-J.

파일럿 F1=0.793을 vLLM이 아닌 Ollama로 재현할 수 있는지 검증.
체크포인트 지원 (20편마다 저장 + 중간 F1 출력).
"""
import json, requests, re, time
from collections import Counter, defaultdict
from pathlib import Path

UMLS_DIR = Path("data/umls_extracted")
OLLAMA_URL = "http://localhost:11434/api/generate"
ALLOWED = {"T047","T184","T191","T046","T048","T037","T019","T020","T190","T049","T033","T031","T040"}
BLACKLIST = {"C1457887","C3257980","C0012634","C0699748","C3839861"}

parent_map = defaultdict(set)
synonym_map = defaultdict(set)
cui_stys = defaultdict(set)
with open(UMLS_DIR / "MRREL.RRF") as f:
    for line in f:
        p = line.strip().split("|")
        if p[3] in ("PAR","RB"): parent_map[p[0]].add(p[4])
        if p[3] == "SY": synonym_map[p[0]].add(p[4]); synonym_map[p[4]].add(p[0])
with open(UMLS_DIR / "MRSTY.RRF") as f:
    for line in f:
        p = line.strip().split("|"); cui_stys[p[0]].add(p[1])
cui_preferred = {}
with open(UMLS_DIR / "MRCONSO.RRF") as f:
    for line in f:
        p = line.strip().split("|")
        if p[1] == "ENG" and p[2] == "P" and p[0] not in cui_preferred:
            cui_preferred[p[0]] = p[14]

# Gold
with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
with open("data/ddxplus/release_evidences_en.json") as f: ev_en = json.load(f)
with open("data/ddxplus/release_evidences.json") as f: ev_fr = json.load(f)
with open("data/ddxplus/umls_mapping.json") as f: umap = json.load(f)["mapping"]
with open("data/ddxplus/disease_umls_mapping.json") as f: dm = json.load(f)["mapping"]
eid_to_fr = {}
for eid, en in ev_en.items():
    for fn, fr in ev_fr.items():
        if en.get("question_en") == fr.get("question_en") and en.get("question_en"):
            eid_to_fr[eid] = fn; break
gold = set()
for dn, info in cond.items():
    dc = dm.get(dn,{}).get("umls_cui")
    if not dc: continue
    for eid in info.get("symptoms",{}):
        if ev_en.get(eid,{}).get("is_antecedent",False): continue
        fn = eid_to_fr.get(eid)
        if fn and fn in umap:
            cui = umap[fn].get("cui")
            if cui: gold.add(tuple(sorted([dc,cui])))

with open("pilot/data/exp_documents.json") as f:
    pilot = json.load(f)["documents"]

print(f"파일럿 재현: {len(pilot)}편, Gold: {len(gold)}쌍")

ckpt_file = Path("pilot/data/pilot_verify_ckpt.json")
all_cls = []
processed = set()
if ckpt_file.exists():
    with open(ckpt_file) as f:
        ck = json.load(f)
    all_cls = ck.get("cls", [])
    processed = set(ck.get("done", []))
    print(f"체크포인트: {len(processed)}편 완료, {len(all_cls)}건")

def ev(our, gld):
    def cm(a,b):
        if a==b: return True
        return b in parent_map.get(a,set()) or a in parent_map.get(b,set())
    mg,mo=set(),set()
    for op in our:
        for gp in gld:
            if (cm(op[0],gp[0]) and cm(op[1],gp[1])) or (cm(op[0],gp[1]) and cm(op[1],gp[0])):
                mg.add(gp);mo.add(op)
    p=len(mo)/len(our) if our else 0; r=len(mg)/len(gld) if gld else 0
    f1=2*p*r/(p+r) if p+r>0 else 0
    return p,r,f1,len(mg)

start = time.time()
new = 0
for doc in pilot:
    dc = doc["seed_cui"]
    key = f"{dc}_{doc.get('pmid', hash(doc['text'][:50]))}"
    if key in processed: continue

    cuis = [c for c in doc["cuis"] if c != dc][:15]
    if not cuis: processed.add(key); continue

    dn = cui_preferred.get(dc, dc)
    pairs_text = "\n".join(
        f"- ({dn[:40]}, {cui_preferred.get(c,c)[:40]}) [CUI: {dc}, {c}]"
        for c in cuis
    )
    prompt = f"""Extract medical relationships from text. For each concept pair, classify as:
- "present": These concepts have a medical relationship (symptom-disease, cause-effect, complication, co-occurrence, risk factor, treatment indication, diagnostic finding)
- "not_related": No medical relationship described in the text

Text: {doc["text"][:2500]}
Pairs: {pairs_text}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|not_related"}}]"""

    try:
        resp = requests.post(OLLAMA_URL, json={"model":"gemma4:e4b-it-bf16","prompt":prompt,"stream":False,
            "options":{"temperature":0,"num_predict":4096}}, timeout=300)
        text = resp.json().get("response","")
        text = re.sub(r"<think>.*?</think>","",text,flags=re.DOTALL)
        text = re.sub(r"```json\s*","",text); text = re.sub(r"```\s*$","",text)
        m = re.search(r"\[[\s\S]*?\]", text)
        if m:
            for item in json.loads(m.group()):
                cls = item.get("classification","").lower().replace(" ","_")
                if cls == "present":
                    a,b = item.get("cui_a",""), item.get("cui_b","")
                    if a and b:
                        other = b if a == dc else a
                        all_cls.append({"dc":dc,"cui":other})
    except Exception as e:
        print(f"  오류: {e}")

    processed.add(key)
    new += 1
    if new % 20 == 0:
        elapsed = time.time() - start
        rate = new/elapsed
        remain = len(pilot) - len(processed)
        eta = remain/rate if rate>0 else 0
        pc = Counter(tuple(sorted([c["dc"],c["cui"]])) for c in all_cls)
        kg3 = {p for p,cnt in pc.items() if cnt>=3}
        exp3 = set(kg3)
        for (a,b) in list(kg3):
            for pa in parent_map.get(a,set()):
                if cui_stys.get(pa,set())&ALLOWED and pa not in BLACKLIST:
                    exp3.add(tuple(sorted([pa,b])))
            for pb in parent_map.get(b,set()):
                if cui_stys.get(pb,set())&ALLOWED and pb not in BLACKLIST:
                    exp3.add(tuple(sorted([a,pb])))
        p,r,f1,m = ev(exp3, gold)
        print(f"[{len(processed):>5}/{len(pilot)}] rels={len(all_cls):,} {rate:.2f}/s ETA={eta/60:.0f}분 | MC=3 F1={f1:.3f} P={p:.3f} R={r:.3f}")
        with open(ckpt_file,"w") as f:
            json.dump({"cls":all_cls,"done":sorted(processed)},f)

with open(ckpt_file,"w") as f:
    json.dump({"cls":all_cls,"done":sorted(processed)},f)

print(f"\n완료: {len(all_cls):,}건 ({(time.time()-start)/60:.1f}분)")
pc = Counter(tuple(sorted([c["dc"],c["cui"]])) for c in all_cls)
for mc in [1,2,3,5,7,10]:
    kg = {p for p,cnt in pc.items() if cnt>=mc}
    exp = set(kg)
    for (a,b) in list(kg):
        for pa in parent_map.get(a,set()):
            if cui_stys.get(pa,set())&ALLOWED and pa not in BLACKLIST:
                exp.add(tuple(sorted([pa,b])))
        for pb in parent_map.get(b,set()):
            if cui_stys.get(pb,set())&ALLOWED and pb not in BLACKLIST:
                exp.add(tuple(sorted([a,pb])))
    p,r,f1,m = ev(exp, gold)
    print(f"MC={mc:>2} edges={len(exp):>6,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/{len(gold)}")
