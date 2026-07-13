#!/usr/bin/env python3
"""2×2 Ablation: CUI 추출(NER vs TextMatch) × 프롬프트(S2-J vs V2).

실험 A: TextMatch + V2  (현재 V5 = F1≈0.22)
실험 B: TextMatch + S2-J
실험 C: scispaCy NER + V2
실험 D: scispaCy NER + S2-J (파일럿 재현 = F1≈0.79?)

파일럿 데이터(2,217편, scispaCy CUI)와 동일 초록의 TextMatch CUI를 비교.
vLLM batch로 4개 실험 한 번에 실행.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
RESULTS_DIR = Path("pilot/results")

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049",
                "T033", "T031", "T040"}
# 파일럿은 T033/T031/T040 없이 사용 — 파일럿과 동일 조건도 테스트
PILOT_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

PROMPT_V2 = """Abstract: {abstract}

Disease: {disease_name} [{disease_cui}]

Other medical concepts found in this abstract:
{keywords}

Which of the above concepts does the abstract describe as being related to {disease_name}?
For each related concept, classify the relationship type:
symptom-of, causes, complication-of, risk-factor-for, diagnostic-finding-of, manifestation-of, co-occurs-with.

Rules:
- ONLY include concepts that the abstract EXPLICITLY links to {disease_name}
- Do NOT infer relationships not stated in the text

JSON array only: [{{"cui":"...","relation":"..."}}]
If none related: []"""

PROMPT_S2J = """Extract medical relationships from text. For each concept pair, classify as:
- "present": These concepts have a medical relationship (symptom-disease, cause-effect, complication, co-occurrence, risk factor, treatment indication, diagnostic finding)
- "not_related": No medical relationship described in the text

Text: {text}
Pairs: {pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|not_related"}}]"""


def load_umls():
    cui_stys = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|"); cui_stys[p[0]].add(p[1])
    parent_map = defaultdict(set)
    synonym_map = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] in ("PAR", "RB"): parent_map[p[0]].add(p[4])
            if p[3] == "SY": synonym_map[p[0]].add(p[4]); synonym_map[p[4]].add(p[0])
    cui_preferred = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[1] == "ENG" and p[2] == "P" and p[0] not in cui_preferred:
                cui_preferred[p[0]] = p[14].strip()
    return dict(cui_stys), dict(parent_map), dict(synonym_map), cui_preferred


def build_aho(cui_stys, allowed):
    target = {c for c, s in cui_stys.items() if s & allowed} - BLACKLIST
    A = ahocorasick.Automaton()
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[0] not in target or p[1] != "ENG": continue
            lower = p[14].strip().lower()
            if len(lower) >= 4:
                try: A.add_word(lower, (lower, p[0]))
                except: pass
    A.make_automaton()
    return A


def text_match(text_lower, A, exclude=None):
    matched = set()
    for ei, (n, c) in A.iter(text_lower):
        if c == exclude: continue
        si = ei - len(n) + 1
        if si > 0 and text_lower[si-1].isalpha(): continue
        if ei+1 < len(text_lower) and text_lower[ei+1].isalpha(): continue
        matched.add(c)
    return matched


def prepare_gold():
    with open("data/ddxplus/release_conditions_en.json") as f: conditions = json.load(f)
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
    for dn, info in conditions.items():
        dc = dm.get(dn, {}).get("umls_cui")
        if not dc: continue
        for eid in info.get("symptoms", {}):
            if ev_en.get(eid, {}).get("is_antecedent", False): continue
            fn = eid_to_fr.get(eid)
            if fn and fn in umap:
                cui = umap[fn].get("cui")
                if cui: gold.add(tuple(sorted([dc, cui])))
    return gold


def parse_json_r(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text); text = re.sub(r"```\s*$", "", text)
    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        try: return json.loads(m.group())
        except: pass
    return []


def evaluate(our, gold, pm):
    def cm(a, b):
        if a == b: return True
        return b in pm.get(a, set()) or a in pm.get(b, set())
    mg, mo = set(), set()
    for op in our:
        for gp in gold:
            if ((cm(op[0],gp[0]) and cm(op[1],gp[1])) or (cm(op[0],gp[1]) and cm(op[1],gp[0]))):
                mg.add(gp); mo.add(op)
    p = len(mo)/len(our) if our else 0; r = len(mg)/len(gold) if gold else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    return round(p,4), round(r,4), round(f1,4), len(mg)


def run_eval(name, all_cls, gold, parent_map, cui_stys, synonym_map, allowed):
    """후처리 + MC sweep + 평가."""
    # 후처리
    filtered = []
    for c in all_cls:
        if c.get("rel") == "manifestation-of": continue
        dc, cui = c["dc"], c["cui"]
        if cui in synonym_map.get(dc, set()): continue
        if cui in parent_map.get(dc, set()) or dc in parent_map.get(cui, set()): continue
        filtered.append(c)

    pc = Counter()
    for c in filtered:
        pc[tuple(sorted([c["dc"], c["cui"]]))] += 1

    print(f"\n  {name}: raw={len(all_cls)}, filtered={len(filtered)}, pairs={len(pc)}")
    best_f1, best_mc = 0, 1
    for mc in [1, 2, 3, 5, 7, 10]:
        kg = {p for p, cnt in pc.items() if cnt >= mc}
        exp = set(kg)
        for (a, b) in list(kg):
            for pa in parent_map.get(a, set()):
                if cui_stys.get(pa, set()) & allowed and pa not in BLACKLIST:
                    exp.add(tuple(sorted([pa, b])))
            for pb in parent_map.get(b, set()):
                if cui_stys.get(pb, set()) & allowed and pb not in BLACKLIST:
                    exp.add(tuple(sorted([a, pb])))
        p, r, f1, m = evaluate(exp, gold, parent_map)
        marker = " ★" if f1 > best_f1 else ""
        if f1 > best_f1: best_f1, best_mc = f1, mc
        print(f"    MC={mc:>2} edges={len(exp):>6,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/{len(gold)}{marker}")
    return best_f1, best_mc


def main():
    print("=" * 80)
    print("2×2 Ablation: CUI추출 × 프롬프트")
    print("=" * 80)

    print("\n[1] 데이터 로드...")
    cui_stys, parent_map, synonym_map, cui_preferred = load_umls()
    gold = prepare_gold()
    print(f"  Gold: {len(gold)}쌍")

    # 파일럿 데이터 로드 (scispaCy CUI)
    with open("pilot/data/exp_documents.json") as f:
        pilot_data = json.load(f)
    pilot_docs = pilot_data["documents"]
    print(f"  파일럿 초록: {len(pilot_docs)}편")

    # TextMatch CUI 추출 (동일 초록에 대해)
    print("\n[2] TextMatch CUI 추출 (파일럿 초록)...")
    aho = build_aho(cui_stys, ALLOWED_STYS)
    tm_docs = []
    for doc in pilot_docs:
        cuis = text_match(doc["text"].lower(), aho, exclude=doc["seed_cui"])
        tm_docs.append({"seed_cui": doc["seed_cui"], "text": doc["text"], "cuis": sorted(cuis)})
    avg_ner = sum(d["n_cuis"] for d in pilot_docs) / len(pilot_docs)
    avg_tm = sum(len(d["cuis"]) for d in tm_docs) / len(tm_docs)
    print(f"  NER 평균 CUI: {avg_ner:.1f}, TextMatch 평균 CUI: {avg_tm:.1f}")

    # 4개 실험의 프롬프트 생성
    print("\n[3] 프롬프트 생성...")

    experiments = {}

    # A: TextMatch + V2
    tasks_a = []
    for doc in tm_docs:
        dc = doc["seed_cui"]
        if not doc["cuis"]: continue
        kw = "\n".join(f"- {cui_preferred.get(c, c)} [{c}]" for c in doc["cuis"])
        tasks_a.append({
            "prompt": PROMPT_V2.format(abstract=doc["text"][:3000],
                disease_name=cui_preferred.get(dc, dc), disease_cui=dc, keywords=kw),
            "dc": dc, "type": "v2",
        })
    experiments["A_TM+V2"] = tasks_a

    # B: TextMatch + S2-J
    tasks_b = []
    for doc in tm_docs:
        dc = doc["seed_cui"]
        if not doc["cuis"]: continue
        pairs_text = "\n".join(
            f"- ({cui_preferred.get(dc, dc)[:40]}, {cui_preferred.get(c, c)[:40]}) [CUI: {dc}, {c}]"
            for c in doc["cuis"][:15]  # 최대 15쌍
        )
        tasks_b.append({
            "prompt": PROMPT_S2J.format(text=doc["text"][:2500], pairs=pairs_text),
            "dc": dc, "type": "s2j", "cuis": doc["cuis"][:15],
        })
    experiments["B_TM+S2J"] = tasks_b

    # C: NER + V2
    tasks_c = []
    for doc in pilot_docs:
        dc = doc["seed_cui"]
        cuis = [c for c in doc["cuis"] if c != dc]
        if not cuis: continue
        kw = "\n".join(f"- {cui_preferred.get(c, c)} [{c}]" for c in cuis)
        tasks_c.append({
            "prompt": PROMPT_V2.format(abstract=doc["text"][:3000],
                disease_name=cui_preferred.get(dc, dc), disease_cui=dc, keywords=kw),
            "dc": dc, "type": "v2",
        })
    experiments["C_NER+V2"] = tasks_c

    # D: NER + S2-J
    tasks_d = []
    for doc in pilot_docs:
        dc = doc["seed_cui"]
        cuis = [c for c in doc["cuis"] if c != dc]
        if not cuis: continue
        pairs_text = "\n".join(
            f"- ({cui_preferred.get(dc, dc)[:40]}, {cui_preferred.get(c, c)[:40]}) [CUI: {dc}, {c}]"
            for c in cuis[:15]
        )
        tasks_d.append({
            "prompt": PROMPT_S2J.format(text=doc["text"][:2500], pairs=pairs_text),
            "dc": dc, "type": "s2j", "cuis": cuis[:15],
        })
    experiments["D_NER+S2J"] = tasks_d

    # E: NER+TextMatch 병합 + V2
    tasks_e = []
    for i, doc in enumerate(pilot_docs):
        dc = doc["seed_cui"]
        ner_cuis = set(c for c in doc["cuis"] if c != dc)
        tm_cuis = set(tm_docs[i]["cuis"])
        merged = sorted(ner_cuis | tm_cuis)
        if not merged: continue
        kw = "\n".join(f"- {cui_preferred.get(c, c)} [{c}]" for c in merged)
        tasks_e.append({
            "prompt": PROMPT_V2.format(abstract=doc["text"][:3000],
                disease_name=cui_preferred.get(dc, dc), disease_cui=dc, keywords=kw),
            "dc": dc, "type": "v2",
        })
    experiments["E_Merged+V2"] = tasks_e

    # F: NER+TextMatch 병합 + S2-J
    tasks_f = []
    for i, doc in enumerate(pilot_docs):
        dc = doc["seed_cui"]
        ner_cuis = set(c for c in doc["cuis"] if c != dc)
        tm_cuis = set(tm_docs[i]["cuis"])
        merged = sorted(ner_cuis | tm_cuis)[:15]
        if not merged: continue
        pairs_text = "\n".join(
            f"- ({cui_preferred.get(dc, dc)[:40]}, {cui_preferred.get(c, c)[:40]}) [CUI: {dc}, {c}]"
            for c in merged
        )
        tasks_f.append({
            "prompt": PROMPT_S2J.format(text=doc["text"][:2500], pairs=pairs_text),
            "dc": dc, "type": "s2j", "cuis": merged,
        })
    experiments["F_Merged+S2J"] = tasks_f

    for name, tasks in experiments.items():
        print(f"  {name}: {len(tasks)}편")

    # vLLM 로드
    print("\n[4] vLLM 로드...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=4096)

    # 실험 실행
    results = {}
    for name, tasks in experiments.items():
        print(f"\n{'='*60}")
        print(f"실험 {name}")
        print(f"{'='*60}")

        convs = [[{"role": "user", "content": t["prompt"]}] for t in tasks]
        t0 = time.time()
        outputs = llm.chat(convs, sampling)
        elapsed = time.time() - t0
        print(f"  LLM: {elapsed:.0f}초 ({len(outputs)/elapsed:.1f}/s)")

        # 파싱
        all_cls = []
        for task, out in zip(tasks, outputs):
            parsed = parse_json_r(out.outputs[0].text)
            if task["type"] == "v2":
                for item in parsed:
                    cui = item.get("cui", "")
                    rel = item.get("relation", "")
                    if cui and rel:
                        all_cls.append({"dc": task["dc"], "cui": cui, "rel": rel})
            else:  # s2j
                for item in parsed:
                    cls = item.get("classification", "").lower().replace(" ", "_")
                    if cls == "present":
                        a, b = item.get("cui_a", ""), item.get("cui_b", "")
                        if a and b:
                            # disease_cui가 a 또는 b
                            other = b if a == task["dc"] else a
                            all_cls.append({"dc": task["dc"], "cui": other, "rel": "present"})

        # 평가 (ALLOWED_STYS 사용)
        f1, mc = run_eval(name, all_cls, gold, parent_map, cui_stys, synonym_map, ALLOWED_STYS)
        results[name] = (f1, mc)

    # 요약
    print(f"\n{'='*80}")
    print(f"Ablation 결과 (1-level PAR 매칭)")
    print(f"{'='*80}")
    print(f"\n  2×2 매트릭스:")
    print(f"  {'':>20} {'V2 프롬프트':>15} {'S2-J 프롬프트':>15}")
    print(f"  {'TextMatch CUI':<20} {results.get('A_TM+V2', (0,0))[0]:>15.3f} {results.get('B_TM+S2J', (0,0))[0]:>15.3f}")
    print(f"  {'scispaCy NER CUI':<20} {results.get('C_NER+V2', (0,0))[0]:>15.3f} {results.get('D_NER+S2J', (0,0))[0]:>15.3f}")
    print(f"  {'NER+TM 병합 CUI':<20} {results.get('E_Merged+V2', (0,0))[0]:>15.3f} {results.get('F_Merged+S2J', (0,0))[0]:>15.3f}")
    print(f"\n  파일럿 참고: F1=0.793 (Ollama, NER+S2-J, 1-level)")
    print(f"\n  전체 실험:")
    for name, (f1, mc) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"    {name:<20} F1={f1:.3f} (MC={mc})")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
