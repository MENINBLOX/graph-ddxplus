#!/usr/bin/env python3
"""KG 구축 + 자동 감별진단 평가.

1단계: 49개 질환 → PubMed → 증상 추출 → KG 구축 (오픈 어휘)
2단계: 구축된 KG로 DDXPlus 134K 환자 감별진단 → Top-1/3/5 정확도

KG 구축: Clinical V2 프롬프트 + 텍스트 매칭 + vLLM batch
진단: 환자 증상 → KG 엣지 매칭 → 질환 순위
"""
from __future__ import annotations

import ast
import csv
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


def prepare_ddxplus():
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

    # 질환 매핑: 프랑스어 이름 → CUI
    disease_fr_to_cui = {}
    diseases = {}
    for dn, info in cond.items():
        dc = dm.get(dn,{}).get("umls_cui")
        un = dm.get(dn,{}).get("umls_name",dn)
        fr_name = info.get("cond-name-fr","")
        if dc:
            diseases[dn] = {"cui":dc, "umls_name":un, "fr_name":fr_name}
            disease_fr_to_cui[fr_name] = dc

    # evidence FR 이름 → CUI 매핑
    evidence_fr_to_cui = {}
    for fn, info in umap.items():
        if info.get("cui"):
            evidence_fr_to_cui[fn] = info["cui"]

    # Gold edges
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

    return diseases, disease_fr_to_cui, evidence_fr_to_cui, eid_to_fr, ev_fr, gold


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


def diagnose(patient_evidences, kg_edges, disease_cuis, evidence_fr_to_cui, parent_map):
    """환자 증상 → KG 기반 감별진단."""
    # 환자 evidence를 CUI로 변환
    patient_cuis = set()
    for ev in patient_evidences:
        # evidence 형태: "fievre", "douleurxx_carac_@_lancinante", etc.
        base = ev.split("_@_")[0]  # 값 부분 제거
        cui = evidence_fr_to_cui.get(base)
        if cui:
            patient_cuis.add(cui)
        # 전체 evidence 이름으로도 시도
        cui2 = evidence_fr_to_cui.get(ev)
        if cui2:
            patient_cuis.add(cui2)

    if not patient_cuis:
        return []

    # 각 질환에 대해 매칭 점수 계산
    scores = {}
    for dc in disease_cuis:
        # 이 질환과 연결된 KG 엣지의 증상 CUI들
        disease_symptoms = set()
        for edge in kg_edges:
            if dc in edge:
                other = edge[0] if edge[1] == dc else edge[1]
                disease_symptoms.add(other)

        if not disease_symptoms:
            scores[dc] = 0
            continue

        # 매칭: 환자 CUI와 질환 증상 CUI의 교집합
        def cui_match(a, b):
            if a == b: return True
            return b in parent_map.get(a, set()) or a in parent_map.get(b, set())

        matched = 0
        for pc in patient_cuis:
            for ds in disease_symptoms:
                if cui_match(pc, ds):
                    matched += 1
                    break

        # Jaccard-like score
        scores[dc] = matched / len(disease_symptoms) if disease_symptoms else 0

    # 점수 순으로 정렬
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked


def main():
    print("="*80)
    print("KG 구축 + 자동 감별진단 평가")
    print("="*80)

    print("\n[1] 데이터 로드...")
    cui_stys,parent_map,synonym_map,mesh_to_cui,cui_all_names,cui_preferred,automaton=load_umls()
    c2m=defaultdict(set)
    for m,c in mesh_to_cui.items(): c2m[c].add(m)
    diseases,disease_fr_to_cui,evidence_fr_to_cui,eid_to_fr,ev_fr,gold=prepare_ddxplus()
    disease_cuis={v["cui"] for v in diseases.values()}
    print(f"  질환: {len(diseases)}, Gold edges: {len(gold)}")

    # ========== KG 구축 ==========
    print(f"\n[2] KG 구축 (Clinical V2, {MAX_ABSTRACTS}편/질환)...")
    conn=sqlite3.connect(str(DB_PATH))
    tasks=[]
    for idx,(dn,dinfo) in enumerate(sorted(diseases.items())):
        dc,un=dinfo["cui"],dinfo["umls_name"]
        rows=search_abs(conn,dc,dn,un,c2m,cui_all_names,MAX_ABSTRACTS)
        for pmid,ab in rows:
            cuis=text_match(ab.lower(),automaton,ex=dc)
            if cuis:
                kw="\n".join(f"- {cui_preferred.get(c,c)} [{c}]" for c in sorted(cuis))
                tasks.append({
                    "prompt":PROMPT_CLINICAL.format(abstract=ab[:3000],
                        disease_name=cui_preferred.get(dc,dc),disease_cui=dc,keywords=kw),
                    "dc":dc
                })
        print(f"  [{idx+1}/49] {dn}: {len(rows)}편")
    conn.close()
    print(f"  총 프롬프트: {len(tasks):,}")

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

    # 파싱 + 후처리
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

    # MC sweep → 다양한 KG 버전 생성
    pair_counts=Counter(tuple(sorted([c["dc"],c["cui"]])) for c in all_cls)

    # ========== 감별진단 평가 ==========
    print(f"\n[4] DDXPlus 감별진단 평가 (134K 환자)...")

    # 테스트 데이터 로드
    test_patients=[]
    with open("data/ddxplus/release_test_patients.csv") as f:
        reader=csv.DictReader(f)
        for row in reader:
            evidences=ast.literal_eval(row["EVIDENCES"])
            pathology=row["PATHOLOGY"]  # 프랑스어 질환명
            diff_dx=ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
            test_patients.append({"evidences":evidences,"pathology":pathology,"diff_dx":diff_dx})

    print(f"  테스트 환자: {len(test_patients):,}명")

    # 다양한 MC로 진단 성능 비교
    for mc in [1, 2, 3, 5, 10]:
        kg_edges={p for p,cnt in pair_counts.items() if cnt>=mc}
        if not kg_edges:
            print(f"  MC={mc}: 엣지 없음")
            continue

        top1=top3=top5=0
        evaluated=0
        for patient in test_patients[:10000]:  # 첫 10K만 빠르게
            true_dc=disease_fr_to_cui.get(patient["pathology"])
            if not true_dc: continue

            ranked=diagnose(patient["evidences"],kg_edges,disease_cuis,
                           evidence_fr_to_cui,parent_map)
            if not ranked: continue

            evaluated+=1
            ranked_cuis=[dc for dc,score in ranked if score>0]

            if ranked_cuis and ranked_cuis[0]==true_dc: top1+=1
            if true_dc in ranked_cuis[:3]: top3+=1
            if true_dc in ranked_cuis[:5]: top5+=1

        if evaluated>0:
            print(f"  MC={mc:>2}: edges={len(kg_edges):>5,} "
                  f"Top1={100*top1/evaluated:.1f}% "
                  f"Top3={100*top3/evaluated:.1f}% "
                  f"Top5={100*top5/evaluated:.1f}% "
                  f"(n={evaluated:,})")

    # 전체 환자로 최적 MC 평가
    print(f"\n[5] 전체 134K 환자 평가 (최적 MC)...")
    for mc in [3, 5]:
        kg_edges={p for p,cnt in pair_counts.items() if cnt>=mc}
        top1=top3=top5=top10=0
        evaluated=0

        for patient in test_patients:
            true_dc=disease_fr_to_cui.get(patient["pathology"])
            if not true_dc: continue
            ranked=diagnose(patient["evidences"],kg_edges,disease_cuis,
                           evidence_fr_to_cui,parent_map)
            if not ranked: continue
            evaluated+=1
            ranked_cuis=[dc for dc,score in ranked if score>0]
            if ranked_cuis and ranked_cuis[0]==true_dc: top1+=1
            if true_dc in ranked_cuis[:3]: top3+=1
            if true_dc in ranked_cuis[:5]: top5+=1
            if true_dc in ranked_cuis[:10]: top10+=1

        print(f"  MC={mc}: edges={len(kg_edges):>5,} "
              f"Top1={100*top1/evaluated:.1f}% "
              f"Top3={100*top3/evaluated:.1f}% "
              f"Top5={100*top5/evaluated:.1f}% "
              f"Top10={100*top10/evaluated:.1f}% "
              f"(n={evaluated:,})")

    # 저장
    with open(RESULTS_DIR/"diagnosis_results.json","w") as f:
        json.dump({"status":"done","kg_edges":len(pair_counts)},f)

    print(f"\n{'='*80}")
    print("완료")
    print(f"{'='*80}")


if __name__=="__main__":
    main()
