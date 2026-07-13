#!/usr/bin/env python3
"""진단 v2: 기존 KG + 개선된 진단 알고리즘.

KG는 이미 구축됨 (kg_diagnose_v1의 pair_counts 재사용 불가 → 재구축).
진단 개선:
  1. 환자 evidence base name → CUI 직접 변환 (umls_mapping.json)
  2. CUI 레벨에서 KG 엣지와 매칭
  3. IDF 가중 스코어링
"""
from __future__ import annotations

import ast
import csv
import json
import math
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

ALLOWED_STYS = {"T047","T184","T191","T046","T048","T037","T019","T020","T190","T049","T033","T031","T040"}
BLACKLIST = {"C1457887","C3257980","C0012634","C0699748","C3839861"}

PROMPT_CLINICAL = """Abstract: {abstract}

Disease: {disease_name} [{disease_cui}]

Clinical findings and symptoms found in this abstract:
{keywords}

From the abstract, identify which of the above findings are CLINICAL SYMPTOMS or SIGNS that a patient with {disease_name} would present with.

Include symptoms patients report and physical examination findings.
Exclude lab results, imaging, other diseases, and synonyms of {disease_name}.

JSON only: [{{"cui":"...","relation":"symptom-of|sign-of"}}]
If none: []"""

MAX_ABSTRACTS = 500


def load_all():
    # UMLS
    cui_stys=defaultdict(set)
    with open(UMLS_DIR/"MRSTY.RRF") as f:
        for l in f: p=l.strip().split("|"); cui_stys[p[0]].add(p[1])
    parent_map=defaultdict(set); synonym_map=defaultdict(set)
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
            if p[11]=="MSH" and p[13].startswith("D") and p[13] not in mesh_to_cui: mesh_to_cui[p[13]]=p[0]
            if p[1]=="ENG":
                cui_all_names[p[0]].add(p[14].strip())
                if p[2]=="P" and p[0] not in cui_preferred: cui_preferred[p[0]]=p[14].strip()
            if p[0] in target and p[1]=="ENG":
                lo=p[14].strip().lower()
                if len(lo)>=4:
                    try: A.add_word(lo,(lo,p[0]))
                    except: pass
    A.make_automaton()
    # DDXPlus
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map=json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond=json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_fr=json.load(f)
    with open("data/ddxplus/umls_mapping.json") as f: umap=json.load(f)["mapping"]

    diseases={}; disease_fr_to_cui={}
    for dn,info in cond.items():
        if dn not in icd_map: continue
        dc=icd_map[dn]["cui"]; fr=info.get("cond-name-fr","")
        diseases[dn]={"cui":dc,"umls_name":icd_map[dn]["umls_name"],"fr":fr}
        disease_fr_to_cui[fr]=dc

    # Evidence base → CUI
    ev_base_to_cui={}
    for fn,info in umap.items():
        if info.get("cui"): ev_base_to_cui[fn]=info["cui"]

    return (dict(cui_stys),dict(parent_map),dict(synonym_map),mesh_to_cui,
            dict(cui_all_names),cui_preferred,A,diseases,disease_fr_to_cui,ev_base_to_cui)


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


def main():
    print("="*80)
    print("진단 v2: KG + CUI 기반 진단")
    print("="*80)

    (cui_stys,parent_map,synonym_map,mesh_to_cui,cui_all_names,
     cui_preferred,automaton,diseases,disease_fr_to_cui,ev_base_to_cui)=load_all()
    c2m=defaultdict(set)
    for m,c in mesh_to_cui.items(): c2m[c].add(m)
    disease_cuis={v["cui"] for v in diseases.values()}

    # KG 구축
    print(f"\n[1] KG 구축 ({MAX_ABSTRACTS}편/질환)...")
    conn=sqlite3.connect(str(DB_PATH))
    tasks=[]
    for idx,(dn,dinfo) in enumerate(sorted(diseases.items())):
        dc,un=dinfo["cui"],dinfo["umls_name"]
        rows=search_abs(conn,dc,dn,un,c2m,cui_all_names,MAX_ABSTRACTS)
        for pmid,ab in rows:
            cuis=text_match(ab.lower(),automaton,ex=dc)
            if cuis:
                kw="\n".join(f"- {cui_preferred.get(c,c)} [{c}]" for c in sorted(cuis))
                tasks.append({"prompt":PROMPT_CLINICAL.format(abstract=ab[:3000],
                    disease_name=cui_preferred.get(dc,dc),disease_cui=dc,keywords=kw),"dc":dc})
        print(f"  [{idx+1}/49] {dn}: {len(rows)}편")
    conn.close()

    print(f"\n[2] vLLM batch ({len(tasks):,}편)...")
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,
            gpu_memory_utilization=0.95,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    sampling=SamplingParams(temperature=0,max_tokens=4096)
    convs=[[{"role":"user","content":t["prompt"]}] for t in tasks]
    t0=time.time()
    outputs=llm.chat(convs,sampling)
    print(f"  완료: {time.time()-t0:.0f}초")

    all_cls=[]
    for task,out in zip(tasks,outputs):
        for item in parse_json_r(out.outputs[0].text):
            if not isinstance(item,dict): continue
            cui=item.get("cui",""); rel=item.get("relation","")
            if cui and rel and rel!="manifestation-of":
                dc=task["dc"]
                if cui not in synonym_map.get(dc,set()) and \
                   cui not in parent_map.get(dc,set()) and dc not in parent_map.get(cui,set()):
                    all_cls.append({"dc":dc,"cui":cui})
    print(f"  관계: {len(all_cls):,}")

    pair_counts=Counter(tuple(sorted([c["dc"],c["cui"]])) for c in all_cls)

    # 질환별 증상 CUI 셋 구축
    def build_disease_symptom_map(mc):
        disease_syms=defaultdict(dict)  # dc -> {symptom_cui: count}
        for (a,b),cnt in pair_counts.items():
            if cnt<mc: continue
            # a,b 중 disease_cui인 것 찾기
            if a in disease_cuis:
                disease_syms[a][b]=cnt
            if b in disease_cuis:
                disease_syms[b][a]=cnt
        return dict(disease_syms)

    # 환자 evidence → CUI 변환
    def patient_to_cuis(evidences):
        """환자 evidence → CUI 셋 (parent 포함)."""
        cuis=set()
        for ev in evidences:
            base=ev.split("_@_")[0]
            cui=ev_base_to_cui.get(base)
            if cui:
                cuis.add(cui)
                # parent CUI도 추가 (CUI 전파)
                for p in parent_map.get(cui,set()):
                    cuis.add(p)
        return cuis

    # 진단 함수들
    def diagnose_coverage(patient_cuis, disease_syms_map):
        """Coverage score: 환자 CUI가 질환 증상을 얼마나 커버하는지."""
        scores={}
        for dc in disease_cuis:
            syms=disease_syms_map.get(dc,{})
            if not syms:
                scores[dc]=0; continue
            matched=sum(1 for s in syms if s in patient_cuis)
            scores[dc]=matched/len(syms)
        return sorted(scores.items(),key=lambda x:-x[1])

    def diagnose_weighted(patient_cuis, disease_syms_map):
        """Weighted score: count 가중."""
        scores={}
        for dc in disease_cuis:
            syms=disease_syms_map.get(dc,{})
            if not syms:
                scores[dc]=0; continue
            total=sum(syms.values())
            matched=sum(cnt for s,cnt in syms.items() if s in patient_cuis)
            scores[dc]=matched/total
        return sorted(scores.items(),key=lambda x:-x[1])

    def diagnose_idf(patient_cuis, disease_syms_map):
        """IDF weighted: 희귀 증상에 높은 가중치."""
        # IDF: 이 증상이 몇 개 질환에 나타나는지
        symptom_df=Counter()
        for dc,syms in disease_syms_map.items():
            for s in syms:
                symptom_df[s]+=1
        n_diseases=len(disease_syms_map)

        scores={}
        for dc in disease_cuis:
            syms=disease_syms_map.get(dc,{})
            if not syms:
                scores[dc]=0; continue
            score=0
            for s,cnt in syms.items():
                if s in patient_cuis:
                    idf=math.log(n_diseases/(symptom_df[s]+1))+1
                    score+=idf*cnt
            scores[dc]=score
        return sorted(scores.items(),key=lambda x:-x[1])

    # 테스트 데이터
    print(f"\n[3] 진단 평가...")
    test_patients=[]
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            test_patients.append({
                "evidences":ast.literal_eval(row["EVIDENCES"]),
                "pathology":row["PATHOLOGY"],
            })
    print(f"  환자: {len(test_patients):,}명")

    # MC × 진단 알고리즘 sweep
    for mc in [1,2,3,5]:
        dsm=build_disease_symptom_map(mc)
        n_diseases_with_syms=sum(1 for d in disease_cuis if d in dsm and dsm[d])

        for algo_name,algo_fn in [("coverage",diagnose_coverage),
                                   ("weighted",diagnose_weighted),
                                   ("idf",diagnose_idf)]:
            top1=top3=top5=0
            n=0
            for patient in test_patients:
                true_dc=disease_fr_to_cui.get(patient["pathology"])
                if not true_dc: continue
                pcuis=patient_to_cuis(patient["evidences"])
                if not pcuis: n+=1; continue
                ranked=algo_fn(pcuis,dsm)
                n+=1
                if ranked and ranked[0][0]==true_dc: top1+=1
                if true_dc in [d for d,s in ranked[:3]]: top3+=1
                if true_dc in [d for d,s in ranked[:5]]: top5+=1

            print(f"  MC={mc} {algo_name:<10}: diseases={n_diseases_with_syms:>2} "
                  f"GTPA@1={100*top1/n:.1f}% @3={100*top3/n:.1f}% @5={100*top5/n:.1f}% (n={n:,})")

    print(f"\n{'='*80}")
    print("완료")
    print(f"{'='*80}")

if __name__=="__main__":
    main()
