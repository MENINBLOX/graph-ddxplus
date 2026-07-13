#!/usr/bin/env python3
"""KG V11: V10 + FN 개선.

V10 F1=0.610 → 0.80 목표.
FN 102개 원인별 대응:
  A. 텍스트 매칭 실패(37): 증상 CUI 검색어 확대 + 초록 2000편
  B. yes=0(35): 프롬프트 개선 (더 관대한 "presenting symptom" 정의)
  C. yes=1(30): threshold=1도 테스트, 초록 수 증가로 자연 해결

핵심 변경:
  1. 초록 2000편/질환 (V10: 500편)
  2. MAX_ABS_PER_PAIR = 20 (V10: 10)
  3. 증상 검색: Aho-Corasick에 UMLS 동의어 + DDXPlus 질문에서 추출한 키워드 추가
  4. 전파 없이 평가 (전파가 FP 81% 원인)
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

PROMPT = """Abstract: {abstract}

Question: Does this abstract describe "{symptom_name}" as a symptom, sign, or clinical finding associated with "{disease_name}"?

Consider "yes" if the abstract mentions that patients with {disease_name} may experience, present with, or develop {symptom_name}, even if it is not the main topic.

Answer ONLY "yes" or "no"."""

MAX_ABSTRACTS = 2000
MAX_ABS_PER_PAIR = 20


def load_umls():
    cui_stys = defaultdict(set)
    with open(UMLS_DIR/"MRSTY.RRF") as f:
        for l in f: p=l.strip().split("|"); cui_stys[p[0]].add(p[1])
    parent_map = defaultdict(set)
    with open(UMLS_DIR/"MRREL.RRF") as f:
        for l in f:
            p=l.strip().split("|")
            if p[3] in("PAR","RB"): parent_map[p[0]].add(p[4])
    cui_preferred = {}
    mesh_to_cui = {}
    cui_all_names = defaultdict(set)
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p=l.strip().split("|")
            if p[11]=="MSH" and p[13].startswith("D") and p[13] not in mesh_to_cui:
                mesh_to_cui[p[13]]=p[0]
            if p[1]=="ENG":
                cui_all_names[p[0]].add(p[14].strip())
                if p[2]=="P" and p[0] not in cui_preferred:
                    cui_preferred[p[0]]=p[14].strip()
    return dict(cui_stys), dict(parent_map), cui_preferred, mesh_to_cui, dict(cui_all_names)


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
    diseases={}
    for dn,info in cond.items():
        dc=dm.get(dn,{}).get("umls_cui"); un=dm.get(dn,{}).get("umls_name",dn)
        if dc: diseases[dn]={"cui":dc,"umls_name":un}
    symptom_cuis=set(); symptom_names={}; gold=set()
    # 질문에서 추가 키워드 추출
    symptom_extra_keywords=defaultdict(set)
    for dn,info in cond.items():
        dc=dm.get(dn,{}).get("umls_cui")
        if not dc: continue
        for eid in info.get("symptoms",{}):
            if ev_en.get(eid,{}).get("is_antecedent",False): continue
            fn=eid_to_fr.get(eid)
            if fn and fn in umap:
                cui=umap[fn].get("cui"); name=umap[fn].get("name",fn)
                if cui:
                    symptom_cuis.add(cui); symptom_names[cui]=name
                    gold.add(tuple(sorted([dc,cui])))
                    # 질문에서 키워드 추출
                    q=ev_en[eid].get("question_en","").lower()
                    for kw in ["fever","pain","cough","breath","nausea","vomit","diarrhea",
                               "fatigue","weakness","swelling","rash","skin","bleed","blood",
                               "weight","dizz","headache","chest","abdom","muscle","joint",
                               "throat","nose","ear","eye","heart","palpit","sweat",
                               "appetite","constipat","seizure","paralys","numbness"]:
                        if kw in q:
                            symptom_extra_keywords[cui].add(kw)
    return diseases, symptom_cuis, symptom_names, gold, symptom_extra_keywords


def build_symptom_automaton(symptom_cuis, cui_all_names, extra_keywords):
    """증상 CUI Aho-Corasick + 추가 키워드."""
    A = ahocorasick.Automaton()
    for cui in symptom_cuis:
        # UMLS 모든 영문 이름
        for name in cui_all_names.get(cui, set()):
            lower = name.lower()
            if len(lower) >= 3:
                try: A.add_word(lower, (lower, cui))
                except: pass
        # DDXPlus 질문에서 추출한 키워드도 추가
        for kw in extra_keywords.get(cui, set()):
            if len(kw) >= 4:
                try: A.add_word(kw, (kw, cui))
                except: pass
    A.make_automaton()
    return A


def search_abs(conn, dc, dn, un, c2m, can, limit):
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
    print(f"KG V11: V10 + FN 개선 ({MAX_ABSTRACTS}편, {MAX_ABS_PER_PAIR}편/쌍)")
    print("="*80)

    print("\n[1] 데이터 로드...")
    cui_stys,parent_map,cui_preferred,mesh_to_cui,cui_all_names=load_umls()
    c2m=defaultdict(set)
    for m,c in mesh_to_cui.items(): c2m[c].add(m)
    diseases,symptom_cuis,symptom_names,gold,extra_kw=prepare_ddxplus()
    symptom_aho=build_symptom_automaton(symptom_cuis,cui_all_names,extra_kw)
    print(f"  질환: {len(diseases)}, 증상: {len(symptom_cuis)}, Gold: {len(gold)}")

    print(f"\n[2] 초록 수집 + 증상 매칭 ({MAX_ABSTRACTS}편/질환)...")
    conn=sqlite3.connect(str(DB_PATH))
    pair_abstracts=defaultdict(list)
    for idx,(dn,dinfo) in enumerate(sorted(diseases.items())):
        dc,un=dinfo["cui"],dinfo["umls_name"]
        rows=search_abs(conn,dc,dn,un,c2m,cui_all_names,MAX_ABSTRACTS)
        for pmid,ab in rows:
            found=set()
            for ei,(n,cui) in symptom_aho.iter(ab.lower()):
                si=ei-len(n)+1
                if si>0 and ab.lower()[si-1].isalpha(): continue
                if ei+1<len(ab) and ab.lower()[ei+1].isalpha(): continue
                found.add(cui)
            for scui in found:
                if len(pair_abstracts[(dc,scui)])<MAX_ABS_PER_PAIR:
                    pair_abstracts[(dc,scui)].append(ab)
        n_pairs=sum(1 for (d,s) in pair_abstracts if d==dc)
        print(f"  [{idx+1}/49] {dn}: {len(rows)}편, {n_pairs}쌍")
    conn.close()

    total_prompts=sum(len(v) for v in pair_abstracts.values())
    print(f"\n  후보 쌍: {len(pair_abstracts):,}, 프롬프트: {total_prompts:,}")

    print(f"\n[3] vLLM 이진 분류...")
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,
            gpu_memory_utilization=0.95,enforce_eager=True,
            limit_mm_per_prompt={"image":0,"audio":0})
    sampling=SamplingParams(temperature=0,max_tokens=10)

    prompts=[]; meta=[]
    for (dc,scui),abstracts in pair_abstracts.items():
        dn=cui_preferred.get(dc,dc)
        sn=symptom_names.get(scui,cui_preferred.get(scui,scui))
        for ab in abstracts:
            prompts.append(PROMPT.format(abstract=ab[:2000],symptom_name=sn,disease_name=dn))
            meta.append((dc,scui))

    convs=[[{"role":"user","content":p}] for p in prompts]
    t0=time.time()
    outputs=llm.chat(convs,sampling)
    elapsed=time.time()-t0
    print(f"  완료: {len(outputs):,}건, {elapsed:.0f}초 ({len(outputs)/elapsed:.1f}/s)")

    pair_yes=Counter(); pair_total=Counter()
    for (dc,scui),out in zip(meta,outputs):
        pk=tuple(sorted([dc,scui]))
        pair_total[pk]+=1
        if "yes" in out.outputs[0].text.strip().lower():
            pair_yes[pk]+=1

    print(f"  총 쌍: {len(pair_total):,}, yes 있는 쌍: {len(pair_yes):,}")

    # 벤치마크 (전파 없이)
    print(f"\n[4] 벤치마크 (전파 없이)...")
    best_f1=0
    for mc in [1,2,3,4,5,6,7,8,10,15,20]:
        kg={p for p,cnt in pair_yes.items() if cnt>=mc}
        p,r,f1,m=evaluate(kg,gold,parent_map)
        marker=" ★" if f1>best_f1 else ""
        if f1>best_f1: best_f1=f1
        print(f"  yes>={mc:>2}: edges={len(kg):>5} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/324{marker}")

    # 비율 threshold
    print(f"\n  비율 threshold (전파 없이):")
    for thr in [0.2,0.3,0.4,0.5]:
        for mc in [1,2,3]:
            kg={p for p in pair_yes if pair_yes[p]>=mc and pair_yes[p]/pair_total[p]>=thr}
            p,r,f1,m=evaluate(kg,gold,parent_map)
            if f1>best_f1-0.05:
                print(f"  yes>={mc}+ratio>={thr}: edges={len(kg):>5} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/324")

    # 저장
    with open(RESULTS_DIR/"kg_v11_results.json","w") as f:
        json.dump({"pair_yes":[[list(k),v] for k,v in pair_yes.most_common()],
                   "pair_total":[[list(k),v] for k,v in pair_total.most_common()],
                   "stats":{"prompts":len(prompts),"time":round(elapsed,1)}},f)

    print(f"\n{'='*80}")
    print(f"V11 최고 F1={best_f1:.3f}")
    print(f"{'='*80}")

if __name__=="__main__":
    main()
