#!/usr/bin/env python3
"""KG V8: Clinical 프롬프트 + 정밀 필터링 + 2-pass 검증.

전략:
1. Pass 1: Clinical V2로 전체 추출 (MC=1 recall 93%)
2. 필터: semantic type(증상 STY만), 동의어/부모 제거, 빈도 비율
3. Pass 2: 상위 후보에 대해 재검증 프롬프트 (정밀 분류)
4. MC sweep
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick
from vllm import LLM, SamplingParams

DB_PATH = Path("/home/max/pubmed_data/pubmed.db")
UMLS_DIR = Path("data/umls_extracted")
RESULTS_DIR = Path("pilot/results")

ALLOWED_STYS = {"T047","T184","T191","T046","T048","T037","T019","T020","T190","T049",
                "T033","T031","T040"}
BLACKLIST = {"C1457887","C3257980","C0012634","C0699748","C3839861"}
SYMPTOM_STYS = {"T184","T033","T046","T048","T040","T031","T079","T060","T201"}
DISEASE_STYS = {"T047","T191","T019","T020","T190","T049","T037"}

PROMPT_CLINICAL = """Abstract: {abstract}

Disease: {disease_name} [{disease_cui}]

Clinical findings and symptoms found in this abstract:
{keywords}

From the abstract, identify which of the above findings are CLINICAL SYMPTOMS or SIGNS that a patient with {disease_name} would present with.

Include:
- Symptoms the patient reports (pain, nausea, fever, cough, dyspnea, fatigue, etc.)
- Physical examination findings (edema, skin lesions, pallor, etc.)
- Vital sign abnormalities (tachycardia, hypotension, etc.)

Exclude:
- Laboratory test results
- Imaging findings
- Other diseases or conditions (these are NOT symptoms)
- Synonyms or subtypes of {disease_name} itself

JSON only: [{{"cui":"...","relation":"symptom-of|sign-of|risk-factor-for"}}]
If none: []"""

PROMPT_VERIFY = """Is "{symptom_name}" a clinical symptom or sign that patients with "{disease_name}" commonly present with?

Answer based on established medical knowledge.
Respond with ONLY "yes" or "no"."""

MAX_ABSTRACTS = 500


def load_umls():
    cui_stys = defaultdict(set)
    with open(UMLS_DIR/"MRSTY.RRF") as f:
        for l in f: p=l.strip().split("|"); cui_stys[p[0]].add(p[1])
    parent_map = defaultdict(set)
    synonym_map = defaultdict(set)
    with open(UMLS_DIR/"MRREL.RRF") as f:
        for l in f:
            p=l.strip().split("|")
            if p[3] in("PAR","RB"): parent_map[p[0]].add(p[4])
            if p[3]=="SY": synonym_map[p[0]].add(p[4]); synonym_map[p[4]].add(p[0])
    mesh_to_cui={}; cui_all_names=defaultdict(set); cui_preferred={}
    target={c for c,s in cui_stys.items() if s&ALLOWED_STYS}-BLACKLIST
    A=ahocorasick.Automaton()
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p=l.strip().split("|")
            if p[11]=="MSH" and p[13].startswith("D") and p[13] not in mesh_to_cui:
                mesh_to_cui[p[13]]=p[0]
            if p[1]=="ENG":
                cui_all_names[p[0]].add(p[14].strip())
                if p[2]=="P" and p[0] not in cui_preferred: cui_preferred[p[0]]=p[14].strip()
            if p[0] in target and p[1]=="ENG":
                lo=p[14].strip().lower()
                if len(lo)>=4:
                    try: A.add_word(lo,(lo,p[0]))
                    except: pass
    A.make_automaton()
    return dict(cui_stys),dict(parent_map),dict(synonym_map),mesh_to_cui,dict(cui_all_names),cui_preferred,A

def prepare_gold():
    with open("data/ddxplus/release_conditions_en.json") as f: cond=json.load(f)
    with open("data/ddxplus/release_evidences_en.json") as f: ev_en=json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_fr=json.load(f)
    with open("data/ddxplus/umls_mapping.json") as f: umap=json.load(f)["mapping"]
    with open("data/ddxplus/disease_umls_mapping.json") as f: dm=json.load(f)["mapping"]
    eid_to_fr={}
    for eid,en in ev_en.items():
        for fn,fr in ev_fr.items():
            if en.get("question_en")==fr.get("question_en") and en.get("question_en"):
                eid_to_fr[eid]=fn; break
    gold,dcuis=set(),{}
    for dn,info in cond.items():
        dc=dm.get(dn,{}).get("umls_cui"); un=dm.get(dn,{}).get("umls_name",dn)
        if not dc: continue
        dcuis[dn]={"cui":dc,"umls_name":un}
        for eid in info.get("symptoms",{}):
            if ev_en.get(eid,{}).get("is_antecedent",False): continue
            fn=eid_to_fr.get(eid)
            if fn and fn in umap:
                cui=umap[fn].get("cui")
                if cui: gold.add(tuple(sorted([dc,cui])))
    return gold,dcuis

def text_match(tl,A,ex=None):
    m=set()
    for ei,(n,c) in A.iter(tl):
        if c==ex: continue
        si=ei-len(n)+1
        if si>0 and tl[si-1].isalpha(): continue
        if ei+1<len(tl) and tl[ei+1].isalpha(): continue
        m.add(c)
    return m

def search_abs(conn,dc,dn,un,c2m,can,limit):
    c=conn.cursor(); rows,seen=[],set()
    muids=c2m.get(dc,set())
    if muids:
        mc=" OR ".join("mesh_terms LIKE '%%%s%%'"%m for m in muids)
        c.execute(f"SELECT pmid,abstract FROM abstracts WHERE ({mc}) AND abstract IS NOT NULL AND length(abstract)>200 ORDER BY RANDOM() LIMIT ?", (limit,))
        for p,a in c.fetchall():
            if p not in seen: seen.add(p); rows.append((p,a))
    if len(rows)<limit:
        kws=[]
        for raw in [dn,un]:
            kw=re.sub(r'\(.*?\)','',raw).strip()
            kw=re.sub(r'\b(NOS|unspecified)\b','',kw,flags=re.IGNORECASE).strip()
            if len(kw)>=4: kws.append(kw)
            for part in raw.split('/'):
                part=part.strip()
                if len(part)>=4 and part not in kws: kws.append(part)
        syns=set()
        for name in can.get(dc,set()):
            kw=re.sub(r'\(.*?\)','',name).strip()
            kw=re.sub(r'\b(NOS|unspecified|disease)\b','',kw,flags=re.IGNORECASE).strip().strip(',./').strip()
            if len(kw)>=4 and kw not in kws: syns.add(kw)
        kws.extend(sorted(syns,key=len)[:10])
        for kw in kws:
            if len(rows)>=limit: break
            c.execute("SELECT pmid,abstract FROM abstracts WHERE (title LIKE ? OR abstract LIKE ?) AND abstract IS NOT NULL AND length(abstract)>200 ORDER BY RANDOM() LIMIT ?",
                      (f"%{kw}%",f"%{kw}%",limit-len(rows)))
            for p,a in c.fetchall():
                if p not in seen: seen.add(p); rows.append((p,a))
    return rows

def parse_json_r(text):
    text=re.sub(r"<think>.*?</think>","",text,flags=re.DOTALL)
    text=re.sub(r"```json\s*","",text); text=re.sub(r"```\s*$","",text)
    m=re.search(r"\[[\s\S]*?\]",text)
    if m:
        try: return json.loads(m.group())
        except: pass
    return []

def evaluate(our,gold,pm):
    def cm(a,b):
        if a==b: return True
        return b in pm.get(a,set()) or a in pm.get(b,set())
    mg,mo=set(),set()
    for op in our:
        for gp in gold:
            if (cm(op[0],gp[0]) and cm(op[1],gp[1])) or (cm(op[0],gp[1]) and cm(op[1],gp[0])):
                mg.add(gp); mo.add(op)
    p=len(mo)/len(our) if our else 0; r=len(mg)/len(gold) if gold else 0
    f1=2*p*r/(p+r) if p+r>0 else 0
    return round(p,4),round(r,4),round(f1,4),len(mg)

def main():
    print("="*80)
    print("KG V8: Clinical + 정밀 필터링 + 2-pass 검증")
    print("="*80)

    print("\n[1] UMLS 로드...")
    cui_stys,parent_map,synonym_map,mesh_to_cui,cui_all_names,cui_preferred,automaton=load_umls()
    c2m=defaultdict(set)
    for m,c in mesh_to_cui.items(): c2m[c].add(m)
    gold,dcuis=prepare_gold()
    disease_cuis={v["cui"] for v in dcuis.values()}

    print(f"\n[2] 초록 수집...")
    conn=sqlite3.connect(str(DB_PATH))
    all_docs={}
    for idx,(dn,dinfo) in enumerate(sorted(dcuis.items())):
        dc,un=dinfo["cui"],dinfo["umls_name"]
        rows=search_abs(conn,dc,dn,un,c2m,cui_all_names,MAX_ABSTRACTS)
        docs=[]
        for pmid,ab in rows:
            cuis=text_match(ab.lower(),automaton,ex=dc)
            if cuis: docs.append({"pmid":pmid,"abstract":ab,"cuis":sorted(cuis)})
        all_docs[dc]=docs
        print(f"  [{idx+1}/49] {dn}: {len(docs)}편")
    conn.close()
    total=sum(len(d) for d in all_docs.values())
    print(f"  총: {total}편")

    print("\n[3] vLLM 로드...")
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,
            gpu_memory_utilization=0.95,enforce_eager=True,
            limit_mm_per_prompt={"image":0,"audio":0})
    sampling=SamplingParams(temperature=0,max_tokens=4096)

    # Pass 1: Clinical V2
    print(f"\n[4] Pass 1: Clinical V2 추출...")
    tasks=[]
    for dc,docs in all_docs.items():
        dn=cui_preferred.get(dc,dc)
        for doc in docs:
            kw="\n".join(f"- {cui_preferred.get(c,c)} [{c}]" for c in doc["cuis"])
            tasks.append({
                "prompt":PROMPT_CLINICAL.format(abstract=doc["abstract"][:3000],
                    disease_name=dn,disease_cui=dc,keywords=kw),
                "dc":dc,"cuis":doc["cuis"]
            })

    convs=[[{"role":"user","content":t["prompt"]}] for t in tasks]
    t0=time.time()
    outputs=llm.chat(convs,sampling)
    elapsed=time.time()-t0
    print(f"  LLM: {elapsed:.0f}초 ({len(outputs)/elapsed:.1f}/s)")

    all_cls=[]
    for task,out in zip(tasks,outputs):
        for item in parse_json_r(out.outputs[0].text):
            if not isinstance(item, dict): continue
            cui=item.get("cui",""); rel=item.get("relation","")
            if cui and rel:
                all_cls.append({"dc":task["dc"],"cui":cui,"rel":rel})

    print(f"  Raw: {len(all_cls):,}")

    # === 필터링 전략 비교 ===
    print(f"\n[5] 필터링 전략 비교...")

    # 기본 후처리 (V7과 동일)
    def filter_basic(cls_list):
        out=[]
        for c in cls_list:
            if c["rel"]=="manifestation-of": continue
            if c["cui"] in synonym_map.get(c["dc"],set()): continue
            if c["cui"] in parent_map.get(c["dc"],set()) or c["dc"] in parent_map.get(c["cui"],set()): continue
            out.append(c)
        return out

    # STY 필터: 증상 STY만 유지
    def filter_sty(cls_list):
        out=[]
        for c in cls_list:
            symptom_sty=cui_stys.get(c["cui"],set())
            if symptom_sty & SYMPTOM_STYS:  # 증상 STY인 것만
                out.append(c)
            elif symptom_sty & DISEASE_STYS:  # 질환 STY도 포함 (DDXPlus에 42쌍)
                out.append(c)
        return out

    # 빈도 비율 필터: count/total_abstracts > threshold
    def filter_freq_ratio(pair_counts, disease_doc_counts, threshold):
        out=set()
        for pair,cnt in pair_counts.items():
            # pair에서 disease CUI 찾기
            dc_in_pair=[c for c in pair if c in disease_cuis]
            if dc_in_pair:
                dc=dc_in_pair[0]
                total=disease_doc_counts.get(dc,1)
                ratio=cnt/total
                if ratio>=threshold:
                    out.add(pair)
        return out

    # 질환별 초록 수
    disease_doc_counts={dc:len(docs) for dc,docs in all_docs.items()}

    strategies={
        "A_basic": filter_basic(all_cls),
        "B_basic+sty": filter_sty(filter_basic(all_cls)),
        "C_sty_only": filter_sty(all_cls),
    }

    def run_eval(name, filtered):
        pc=Counter(tuple(sorted([c["dc"],c["cui"]])) for c in filtered)
        best_f1,best_mc=0,1
        for mc in [1,2,3,5,7,10,15,20,30,50]:
            kg={p for p,cnt in pc.items() if cnt>=mc}
            exp=set(kg)
            for (a,b) in list(kg):
                for pa in parent_map.get(a,set()):
                    if cui_stys.get(pa,set())&ALLOWED_STYS and pa not in BLACKLIST:
                        exp.add(tuple(sorted([pa,b])))
                for pb in parent_map.get(b,set()):
                    if cui_stys.get(pb,set())&ALLOWED_STYS and pb not in BLACKLIST:
                        exp.add(tuple(sorted([a,pb])))
            p,r,f1,m=evaluate(exp,gold,parent_map)
            if f1>best_f1: best_f1=f1; best_mc=mc
        # 빈도 비율 필터도 시도
        for thr in [0.01,0.02,0.05,0.10,0.15,0.20]:
            freq_pairs=filter_freq_ratio(pc,disease_doc_counts,thr)
            exp2=set(freq_pairs)
            for (a,b) in list(freq_pairs):
                for pa in parent_map.get(a,set()):
                    if cui_stys.get(pa,set())&ALLOWED_STYS and pa not in BLACKLIST:
                        exp2.add(tuple(sorted([pa,b])))
                for pb in parent_map.get(b,set()):
                    if cui_stys.get(pb,set())&ALLOWED_STYS and pb not in BLACKLIST:
                        exp2.add(tuple(sorted([a,pb])))
            p2,r2,f2,m2=evaluate(exp2,gold,parent_map)
            if f2>best_f1: best_f1=f2; best_mc=f"freq>{thr}"
        return best_f1, best_mc

    for name, filtered in strategies.items():
        f1, mc = run_eval(name, filtered)
        print(f"  {name:<20} best F1={f1:.3f} ({mc})")

    # 상세 MC sweep (최적 전략)
    print(f"\n[6] 상세 결과 (B_basic+sty)...")
    filtered_b = strategies["B_basic+sty"]
    pc = Counter(tuple(sorted([c["dc"],c["cui"]])) for c in filtered_b)

    print(f"\n  MC sweep:")
    for mc in [1,2,3,5,7,10,15,20,30,50]:
        kg={p for p,cnt in pc.items() if cnt>=mc}
        exp=set(kg)
        for (a,b) in list(kg):
            for pa in parent_map.get(a,set()):
                if cui_stys.get(pa,set())&ALLOWED_STYS and pa not in BLACKLIST:
                    exp.add(tuple(sorted([pa,b])))
            for pb in parent_map.get(b,set()):
                if cui_stys.get(pb,set())&ALLOWED_STYS and pb not in BLACKLIST:
                    exp.add(tuple(sorted([a,pb])))
        p,r,f1,m=evaluate(exp,gold,parent_map)
        print(f"    MC={mc:>2} edges={len(exp):>6,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/324")

    print(f"\n  빈도 비율 sweep:")
    for thr in [0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.30]:
        freq_pairs=filter_freq_ratio(pc,disease_doc_counts,thr)
        exp2=set(freq_pairs)
        for (a,b) in list(freq_pairs):
            for pa in parent_map.get(a,set()):
                if cui_stys.get(pa,set())&ALLOWED_STYS and pa not in BLACKLIST:
                    exp2.add(tuple(sorted([pa,b])))
            for pb in parent_map.get(b,set()):
                if cui_stys.get(pb,set())&ALLOWED_STYS and pb not in BLACKLIST:
                    exp2.add(tuple(sorted([a,pb])))
        p,r,f1,m=evaluate(exp2,gold,parent_map)
        print(f"    freq>{thr:.2f} edges={len(exp2):>6,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/324")

    # Pass 2: 상위 후보 재검증
    print(f"\n[7] Pass 2: 상위 후보 재검증...")
    # MC>=3인 쌍에 대해 "이 증상이 이 질환의 증상인가?" 재질문
    verify_pairs=[(p,cnt) for p,cnt in pc.items() if cnt>=3]
    print(f"  검증 대상: {len(verify_pairs)}쌍")

    verify_prompts=[]
    verify_meta=[]
    for pair,cnt in verify_pairs:
        # pair에서 disease/symptom 분리
        dc_candidates=[c for c in pair if c in disease_cuis]
        if not dc_candidates: continue
        dc=dc_candidates[0]
        others=[c for c in pair if c!=dc]
        if not others: continue
        symptom=others[0]
        dn=cui_preferred.get(dc,dc)
        sn=cui_preferred.get(symptom,symptom)
        verify_prompts.append(PROMPT_VERIFY.format(symptom_name=sn,disease_name=dn))
        verify_meta.append({"pair":pair,"dc":dc,"symptom":symptom,"cnt":cnt})

    if verify_prompts:
        convs2=[[{"role":"user","content":p}] for p in verify_prompts]
        sampling2=SamplingParams(temperature=0,max_tokens=10)
        t0=time.time()
        outputs2=llm.chat(convs2,sampling2)
        elapsed2=time.time()-t0
        print(f"  Pass 2 LLM: {elapsed2:.0f}초 ({len(outputs2)/elapsed2:.1f}/s)")

        verified=set()
        for meta,out in zip(verify_meta,outputs2):
            answer=out.outputs[0].text.strip().lower()
            if "yes" in answer:
                verified.add(meta["pair"])

        print(f"  검증 통과: {len(verified)}/{len(verify_meta)}쌍")

        # 검증된 쌍으로 평가
        exp_v=set(verified)
        for (a,b) in list(verified):
            for pa in parent_map.get(a,set()):
                if cui_stys.get(pa,set())&ALLOWED_STYS and pa not in BLACKLIST:
                    exp_v.add(tuple(sorted([pa,b])))
            for pb in parent_map.get(b,set()):
                if cui_stys.get(pb,set())&ALLOWED_STYS and pb not in BLACKLIST:
                    exp_v.add(tuple(sorted([a,pb])))
        p,r,f1,m=evaluate(exp_v,gold,parent_map)
        print(f"  Pass2 검증: edges={len(exp_v):,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/324")

    # 저장
    with open(RESULTS_DIR/"kg_v8_results.json","w") as f:
        json.dump({"status":"done"},f)
    print(f"\n{'='*80}")
    print("V8 완료")
    print(f"{'='*80}")

if __name__=="__main__":
    main()
