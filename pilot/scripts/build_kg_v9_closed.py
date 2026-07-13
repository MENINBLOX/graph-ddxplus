#!/usr/bin/env python3
"""KG V9: 폐쇄형 이진 분류.

기존 오픈엔디드 추출 → 폐쇄형 (질환, 증상) 쌍별 이진 분류로 전환.

파이프라인:
  1. UMLS에서 증상 후보 사전 구축 (T184/T033/T046 등)
  2. 각 질환의 PubMed 초록에서 텍스트 매칭으로 어떤 증상 CUI가 등장하는지 확인
  3. 등장하는 (질환, 증상) 쌍에 대해 LLM 이진 분류:
     "이 초록에서 [증상]이 [질환]의 임상 증상으로 기술되어 있는가?"
  4. 여러 초록의 판정을 집계 → KG
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

# 질환-증상 쌍에 대한 엄격한 이진 분류 프롬프트
PROMPT_BINARY = """Abstract: {abstract}

Question: Does this abstract describe "{symptom_name}" as a clinical symptom or presenting sign of "{disease_name}"?

A clinical symptom means something a patient with this disease would experience or complain about (e.g., pain, fever, cough, nausea, shortness of breath, weakness).

Answer ONLY "yes" or "no"."""

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
    print("KG V9: 폐쇄형 이진 분류")
    print("="*80)

    print("\n[1] UMLS 로드...")
    cui_stys,parent_map,synonym_map,mesh_to_cui,cui_all_names,cui_preferred,automaton=load_umls()
    c2m=defaultdict(set)
    for m,c in mesh_to_cui.items(): c2m[c].add(m)
    gold,dcuis=prepare_gold()
    disease_cuis={v["cui"] for v in dcuis.values()}

    # 초록 수집 + CUI 추출
    print(f"\n[2] 초록 수집 ({MAX_ABSTRACTS}편/질환)...")
    conn=sqlite3.connect(str(DB_PATH))
    # 질환별: 초록 목록 + 각 초록에서 발견된 CUI
    disease_data={} # dc -> [{"abstract": str, "cuis": set}, ...]
    for idx,(dn,dinfo) in enumerate(sorted(dcuis.items())):
        dc,un=dinfo["cui"],dinfo["umls_name"]
        rows=search_abs(conn,dc,dn,un,c2m,cui_all_names,MAX_ABSTRACTS)
        docs=[]
        for pmid,ab in rows:
            cuis=text_match(ab.lower(),automaton,ex=dc)
            if cuis: docs.append({"abstract":ab,"cuis":cuis})
        disease_data[dc]=docs
        print(f"  [{idx+1}/49] {dn}: {len(docs)}편")
    conn.close()

    # 질환별 (질환, 증상) 후보 쌍 생성
    print(f"\n[3] 후보 쌍 생성...")
    candidate_pairs=defaultdict(list)  # (dc, symptom_cui) -> [abstract_texts]
    for dc,docs in disease_data.items():
        for doc in docs:
            for cui in doc["cuis"]:
                # 동의어/부모 제외
                if cui in synonym_map.get(dc,set()): continue
                if cui in parent_map.get(dc,set()) or dc in parent_map.get(cui,set()): continue
                pair=tuple(sorted([dc,cui]))
                candidate_pairs[(dc,cui)].append(doc["abstract"])

    print(f"  후보 쌍: {len(candidate_pairs):,}")
    print(f"  총 (쌍, 초록) 조합: {sum(len(v) for v in candidate_pairs.values()):,}")

    # 쌍당 최대 5편만 사용 (효율성)
    MAX_ABS_PER_PAIR = 5
    prompts = []
    prompt_meta = []  # (dc, symptom_cui)

    for (dc, scui), abstracts in candidate_pairs.items():
        dn = cui_preferred.get(dc, dc)
        sn = cui_preferred.get(scui, scui)
        for ab in abstracts[:MAX_ABS_PER_PAIR]:
            prompts.append(PROMPT_BINARY.format(
                abstract=ab[:2000], symptom_name=sn, disease_name=dn))
            prompt_meta.append((dc, scui))

    print(f"  LLM 호출 수: {len(prompts):,}")

    # vLLM batch
    print(f"\n[4] vLLM batch 이진 분류...")
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,
            gpu_memory_utilization=0.95,enforce_eager=True,
            limit_mm_per_prompt={"image":0,"audio":0})
    sampling=SamplingParams(temperature=0,max_tokens=10)

    convs=[[{"role":"user","content":p}] for p in prompts]
    t0=time.time()
    outputs=llm.chat(convs,sampling)
    elapsed=time.time()-t0
    print(f"  완료: {len(outputs):,}건, {elapsed:.0f}초 ({len(outputs)/elapsed:.1f}/s)")

    # 집계: 각 (dc, scui) 쌍에 대해 yes 비율 계산
    print(f"\n[5] 집계...")
    pair_yes = Counter()
    pair_total = Counter()

    for (dc,scui),out in zip(prompt_meta,outputs):
        answer=out.outputs[0].text.strip().lower()
        pair_key=tuple(sorted([dc,scui]))
        pair_total[pair_key]+=1
        if "yes" in answer:
            pair_yes[pair_key]+=1

    print(f"  총 쌍: {len(pair_total):,}")
    print(f"  yes 있는 쌍: {len(pair_yes):,}")

    # 다양한 threshold sweep
    print(f"\n[6] 벤치마크...")

    # MC (yes 횟수) sweep
    print(f"\n  yes 횟수 threshold:")
    for mc in [1,2,3,4,5]:
        kg={p for p,cnt in pair_yes.items() if cnt>=mc}
        exp=set(kg)
        for (a,b) in list(kg):
            for pa in parent_map.get(a,set()):
                if cui_stys.get(pa,set())&ALLOWED_STYS and pa not in BLACKLIST:
                    exp.add(tuple(sorted([pa,b])))
            for pb in parent_map.get(b,set()):
                if cui_stys.get(pb,set())&ALLOWED_STYS and pb not in BLACKLIST:
                    exp.add(tuple(sorted([a,pb])))
        p,r,f1,m=evaluate(exp,gold,parent_map)
        print(f"    yes>={mc}: edges={len(exp):>6,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/324")

    # yes 비율 sweep
    print(f"\n  yes 비율 threshold:")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        kg=set()
        for pair in pair_yes:
            ratio = pair_yes[pair] / pair_total[pair]
            if ratio >= thr:
                kg.add(pair)
        exp=set(kg)
        for (a,b) in list(kg):
            for pa in parent_map.get(a,set()):
                if cui_stys.get(pa,set())&ALLOWED_STYS and pa not in BLACKLIST:
                    exp.add(tuple(sorted([pa,b])))
            for pb in parent_map.get(b,set()):
                if cui_stys.get(pb,set())&ALLOWED_STYS and pb not in BLACKLIST:
                    exp.add(tuple(sorted([a,pb])))
        p,r,f1,m=evaluate(exp,gold,parent_map)
        print(f"    ratio>={thr:.1f}: edges={len(exp):>6,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/324")

    # yes>=2 AND ratio>=0.5 등 조합
    print(f"\n  조합 threshold:")
    for mc in [1,2,3]:
        for thr in [0.3,0.5,0.7]:
            kg=set()
            for pair in pair_yes:
                if pair_yes[pair]>=mc and pair_yes[pair]/pair_total[pair]>=thr:
                    kg.add(pair)
            exp=set(kg)
            for (a,b) in list(kg):
                for pa in parent_map.get(a,set()):
                    if cui_stys.get(pa,set())&ALLOWED_STYS and pa not in BLACKLIST:
                        exp.add(tuple(sorted([pa,b])))
                for pb in parent_map.get(b,set()):
                    if cui_stys.get(pb,set())&ALLOWED_STYS and pb not in BLACKLIST:
                        exp.add(tuple(sorted([a,pb])))
            p,r,f1,m=evaluate(exp,gold,parent_map)
            print(f"    yes>={mc}+ratio>={thr:.1f}: edges={len(exp):>6,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/324")

    # 전파 없이
    print(f"\n  전파 없이 (yes>=2):")
    kg={p for p,cnt in pair_yes.items() if cnt>=2}
    p,r,f1,m=evaluate(kg,gold,parent_map)
    print(f"    edges={len(kg):>6,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/324")

    # 저장
    with open(RESULTS_DIR/"kg_v9_results.json","w") as f:
        json.dump({"pair_yes":[[list(k),v] for k,v in pair_yes.most_common()],
                   "pair_total":[[list(k),v] for k,v in pair_total.most_common()]},f)
    print(f"\n{'='*80}")
    print("V9 완료")
    print(f"{'='*80}")

if __name__=="__main__":
    main()
