#!/usr/bin/env python3
"""KG 구축 + 진단 최적화: GTPA@1 >= 0.80 목표.

1. 49개 질환(ICD-10 매핑) → PubMed → 오픈 어휘 증상 추출 → KG
2. KG로 DDXPlus 134K 환자 감별진단
3. 진단 알고리즘 최적화 (매칭, 스코어링, 가중치)
4. GTPA@1 측정
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

ALLOWED_STYS = {"T047","T184","T191","T046","T048","T037","T019","T020","T190","T049",
                "T033","T031","T040"}
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


def load_umls():
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


def load_ddxplus():
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f:
        icd_map=json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f:
        cond=json.load(f)
    with open("data/ddxplus/release_evidences.json") as f:
        ev_fr=json.load(f)

    diseases={}
    disease_fr_to_cui={}
    for dn,info in cond.items():
        if dn not in icd_map: continue
        dc=icd_map[dn]["cui"]
        fr=info.get("cond-name-fr","")
        diseases[dn]={"cui":dc,"umls_name":icd_map[dn]["umls_name"],"fr":fr}
        disease_fr_to_cui[fr]=dc

    # Evidence FR이름 → 영어 질문 키워드
    ev_keywords={}
    for fr_name,info in ev_fr.items():
        q=info.get("question_en","").lower()
        ev_keywords[fr_name]=q

    return diseases,disease_fr_to_cui,ev_keywords


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


def build_symptom_index(kg_edges, cui_preferred, cui_all_names):
    """KG 증상을 텍스트 검색 가능하도록 인덱스 구축."""
    # 각 질환의 증상 CUI와 이름들
    disease_symptoms = defaultdict(dict)  # dc -> {symptom_cui: {"names": set, "count": int}}
    for (a,b), cnt in kg_edges.items():
        disease_symptoms[a][b] = {"count": cnt}
        disease_symptoms[b][a] = {"count": cnt}

    # 증상 CUI → 검색 가능한 키워드 (소문자)
    symptom_keywords = defaultdict(set)
    all_symptom_cuis = set()
    for dc, syms in disease_symptoms.items():
        for scui in syms:
            all_symptom_cuis.add(scui)

    for scui in all_symptom_cuis:
        pref = cui_preferred.get(scui, "")
        if pref:
            for word in pref.lower().split():
                if len(word) >= 3:
                    symptom_keywords[scui].add(word)
            symptom_keywords[scui].add(pref.lower())
        for name in cui_all_names.get(scui, set()):
            ln = name.lower()
            if len(ln) >= 3:
                symptom_keywords[scui].add(ln)

    return disease_symptoms, symptom_keywords


def match_evidence_to_symptoms(patient_evidences, ev_keywords, symptom_keywords, all_symptom_cuis):
    """환자 evidence → KG 증상 CUI 매칭 (텍스트 기반)."""
    matched_cuis = set()

    for ev in patient_evidences:
        base = ev.split("_@_")[0]
        q = ev_keywords.get(base, "").lower()
        val = ev.split("_@_")[1].lower() if "_@_" in ev else ""

        # 질문 + 값에서 키워드 추출
        ev_words = set()
        for w in (q + " " + val).split():
            w = re.sub(r'[^a-z]', '', w)
            if len(w) >= 3:
                ev_words.add(w)

        # KG 증상과 매칭
        for scui in all_symptom_cuis:
            skw = symptom_keywords.get(scui, set())
            # 키워드 겹침
            overlap = ev_words & skw
            if overlap and len(overlap) >= 1:
                # 너무 일반적인 단어 제외
                generic = {"the","and","you","are","have","does","how","your",
                          "with","for","any","not","that","this","from","was",
                          "been","more","than","other","which","what","where",
                          "when","who","did","its","nos","nec","finding"}
                real_overlap = overlap - generic
                if real_overlap:
                    matched_cuis.add(scui)

    return matched_cuis


def diagnose_patient(patient_cuis, disease_symptoms, disease_cuis):
    """환자 매칭 CUI → 질환 스코어링."""
    scores = {}
    for dc in disease_cuis:
        syms = disease_symptoms.get(dc, {})
        if not syms:
            scores[dc] = 0
            continue

        # 매칭된 증상 수 (가중치: count)
        matched_score = 0
        matched_count = 0
        total_weight = sum(s["count"] for s in syms.values())

        for scui, info in syms.items():
            if scui in patient_cuis:
                matched_score += info["count"]
                matched_count += 1

        # 정규화 점수
        if total_weight > 0:
            scores[dc] = matched_score / total_weight
        else:
            scores[dc] = 0

    return sorted(scores.items(), key=lambda x: -x[1])


def main():
    print("="*80)
    print("KG 구축 + 진단: GTPA@1 >= 0.80 목표")
    print("="*80)

    print("\n[1] 데이터 로드...")
    cui_stys,parent_map,synonym_map,mesh_to_cui,cui_all_names,cui_preferred,automaton=load_umls()
    c2m=defaultdict(set)
    for m,c in mesh_to_cui.items(): c2m[c].add(m)
    diseases,disease_fr_to_cui,ev_keywords=load_ddxplus()
    disease_cuis={v["cui"] for v in diseases.values()}
    print(f"  질환: {len(diseases)}, ICD-10 매핑 완료")

    # === KG 구축 ===
    print(f"\n[2] KG 구축 ({MAX_ABSTRACTS}편/질환)...")
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
    print(f"  총: {len(tasks):,}편")

    print("\n[3] vLLM batch...")
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,
            gpu_memory_utilization=0.95,enforce_eager=True,
            limit_mm_per_prompt={"image":0,"audio":0})
    sampling=SamplingParams(temperature=0,max_tokens=4096)
    convs=[[{"role":"user","content":t["prompt"]}] for t in tasks]
    t0=time.time()
    outputs=llm.chat(convs,sampling)
    elapsed=time.time()-t0
    print(f"  완료: {elapsed:.0f}초 ({len(outputs)/elapsed:.1f}/s)")

    # 파싱
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

    # === 진단 평가 ===
    print(f"\n[4] 진단 평가 (DDXPlus 134K)...")

    # 테스트 데이터
    test_patients=[]
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            test_patients.append({
                "evidences": ast.literal_eval(row["EVIDENCES"]),
                "pathology": row["PATHOLOGY"],
            })
    print(f"  환자: {len(test_patients):,}명")

    # MC sweep으로 다양한 KG 버전 평가
    for mc in [1, 2, 3, 5, 10]:
        kg_edges={p:pair_counts[p] for p,cnt in pair_counts.items() if cnt>=mc}
        if not kg_edges: continue

        disease_symptoms, symptom_keywords = build_symptom_index(
            kg_edges, cui_preferred, cui_all_names)
        all_symptom_cuis = set()
        for dc, syms in disease_symptoms.items():
            all_symptom_cuis.update(syms.keys())

        top1=top3=top5=top10=0
        evaluated=0
        no_match=0

        for patient in test_patients:
            true_dc=disease_fr_to_cui.get(patient["pathology"])
            if not true_dc: continue

            # 환자 evidence → 증상 CUI 매칭
            patient_cuis=match_evidence_to_symptoms(
                patient["evidences"], ev_keywords, symptom_keywords, all_symptom_cuis)

            if not patient_cuis:
                no_match+=1
                evaluated+=1
                continue

            # 진단
            ranked=diagnose_patient(patient_cuis, disease_symptoms, disease_cuis)
            evaluated+=1

            if ranked and ranked[0][0]==true_dc: top1+=1
            if true_dc in [d for d,s in ranked[:3]]: top3+=1
            if true_dc in [d for d,s in ranked[:5]]: top5+=1
            if true_dc in [d for d,s in ranked[:10]]: top10+=1

        if evaluated>0:
            print(f"  MC={mc:>2}: edges={len(kg_edges):>5,} "
                  f"GTPA@1={100*top1/evaluated:.1f}% "
                  f"@3={100*top3/evaluated:.1f}% "
                  f"@5={100*top5/evaluated:.1f}% "
                  f"@10={100*top10/evaluated:.1f}% "
                  f"no_match={no_match} (n={evaluated:,})")

    # 저장
    with open(RESULTS_DIR/"kg_diagnose_v1.json","w") as f:
        json.dump({"status":"done","kg_pairs":len(pair_counts)},f)

    print(f"\n{'='*80}")
    print("완료")
    print(f"{'='*80}")

if __name__=="__main__":
    main()
