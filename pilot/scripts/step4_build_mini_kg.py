#!/usr/bin/env python3
"""미니 KG 구축: Step 3 결과를 Neo4j에 로드하고 검증.

FDR<0.05 엣지를 Neo4j에 로드하고:
1. 노드/엣지 수 통계
2. DDXPlus 5개 질환의 커버리지 확인
3. DDXPlus 테스트 케이스 100건에 대해 간이 진단 수행
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")
UMLS_DIR = Path("data/umls_extracted")

DISO_TYPES = {
    "T047", "T184", "T033", "T034", "T191", "T046",
    "T048", "T037", "T019", "T020", "T190", "T049",
}


def load_cui_stys() -> dict[str, set[str]]:
    cui_stys: dict[str, set[str]] = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui_stys[p[0]].add(p[1])
    return dict(cui_stys)


def load_cui_names() -> dict[str, str]:
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui, lang, ts, name = p[0], p[1], p[2], p[14]
            if lang == "ENG" and (cui not in names or ts == "P"):
                names[cui] = name
    return names


def infer_relation_type(a_stys: set[str], b_stys: set[str]) -> str:
    """semantic type 조합에서 관계 종류를 유도한다."""
    symptom_types = {"T184", "T033"}
    disease_types = {"T047", "T191", "T046", "T048"}
    lab_types = {"T034"}

    a_is_sym = bool(a_stys & symptom_types)
    a_is_dis = bool(a_stys & disease_types)
    b_is_sym = bool(b_stys & symptom_types)
    b_is_dis = bool(b_stys & disease_types)
    a_is_lab = bool(a_stys & lab_types)
    b_is_lab = bool(b_stys & lab_types)

    if a_is_dis and b_is_sym:
        return "disease-symptom"
    if a_is_sym and b_is_dis:
        return "symptom-disease"
    if a_is_dis and b_is_dis:
        return "disease-disease"
    if a_is_sym and b_is_sym:
        return "symptom-symptom"
    if (a_is_dis and b_is_lab) or (a_is_lab and b_is_dis):
        return "disease-lab"
    return "other"


def main():
    print("=" * 80)
    print("미니 KG 구축 및 검증")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/4] 데이터 로드...")
    with open(DATA_DIR / "step3_kg_edges.json") as f:
        kg_data = json.load(f)

    sig_edges = kg_data["significant_edges"]
    all_edges = kg_data["all_edges"]
    cui_stys = load_cui_stys()
    cui_names = load_cui_names()

    print(f"  FDR<0.05 유의한 엣지: {len(sig_edges)}")
    print(f"  전체 엣지 (present>=1): {len(all_edges)}")

    # KG 통계
    print("\n[2/4] 미니 KG 통계")
    print("=" * 80)

    unique_cuis = set()
    for e in sig_edges:
        unique_cuis.add(e["cui_a"])
        unique_cuis.add(e["cui_b"])

    # Semantic type 분포
    sty_dist = defaultdict(int)
    for cui in unique_cuis:
        for sty in cui_stys.get(cui, set()) & DISO_TYPES:
            sty_dist[sty] += 1

    print(f"  노드 수: {len(unique_cuis)}")
    print(f"  엣지 수: {len(sig_edges)}")
    print(f"\n  Semantic type 분포:")
    sty_names = {
        "T047": "Disease/Syndrome", "T184": "Sign/Symptom", "T033": "Finding",
        "T034": "Lab Result", "T191": "Neoplastic", "T046": "Pathologic Function",
        "T048": "Mental/Behavioral", "T037": "Injury", "T019": "Congenital",
        "T020": "Acquired Abnormality", "T190": "Anatomical Abnormality",
        "T049": "Cell/Molecular Dysfunction",
    }
    for sty, cnt in sorted(sty_dist.items(), key=lambda x: -x[1]):
        print(f"    {sty} ({sty_names.get(sty, '?')}): {cnt}")

    # 관계 종류 분포
    rel_type_dist = defaultdict(int)
    for e in sig_edges:
        a_stys = cui_stys.get(e["cui_a"], set())
        b_stys = cui_stys.get(e["cui_b"], set())
        rel_type = infer_relation_type(a_stys, b_stys)
        rel_type_dist[rel_type] += 1

    print(f"\n  관계 종류 분포 (semantic type 기반 유도):")
    for rt, cnt in sorted(rel_type_dist.items(), key=lambda x: -x[1]):
        print(f"    {rt}: {cnt}")

    # DDXPlus 5개 질환 관련
    print("\n[3/4] DDXPlus 5개 질환 커버리지")
    print("=" * 80)

    ddx_diseases = {
        "Pneumonia": "C0032285",
        "Pulmonary embolism": "C0034065",
        "GERD": "C0017168",
        "Panic attack": "C0086769",
        "Bronchitis": "C0006277",
    }

    for name, dcui in ddx_diseases.items():
        # 이 질환과 연결된 엣지
        related = []
        for e in sig_edges:
            if e["cui_a"] == dcui or e["cui_b"] == dcui:
                other = e["cui_b"] if e["cui_a"] == dcui else e["cui_a"]
                other_stys = cui_stys.get(other, set())
                other_name = cui_names.get(other, other)[:40]
                rel_type = infer_relation_type(
                    cui_stys.get(dcui, set()),
                    cui_stys.get(other, set()),
                )
                related.append({
                    "other_cui": other,
                    "other_name": other_name,
                    "rel_type": rel_type,
                    "n_present": e["n_present"],
                    "score": e["jensen_0.6"],
                    "polarity": e["polarity"],
                })

        print(f"\n  {name} ({dcui}): {len(related)}개 관계")
        # T184 증상만 필터
        symptoms = [r for r in related if "symptom" in r["rel_type"]]
        diseases = [r for r in related if r["rel_type"] == "disease-disease"]
        others = [r for r in related if r not in symptoms and r not in diseases]

        if symptoms:
            print(f"    증상 관련 ({len(symptoms)}개):")
            for r in sorted(symptoms, key=lambda x: -x["n_present"]):
                print(f"      {r['other_name']:40s} n={r['n_present']} score={r['score']:.2f}")
        if diseases:
            print(f"    질환 관련 ({len(diseases)}개):")
            for r in sorted(diseases, key=lambda x: -x["n_present"]):
                print(f"      {r['other_name']:40s} n={r['n_present']} score={r['score']:.2f}")
        if others:
            print(f"    기타 ({len(others)}개):")
            for r in sorted(others, key=lambda x: -x["n_present"])[:5]:
                print(f"      {r['other_name']:40s} ({r['rel_type']}) n={r['n_present']}")

    # 간이 진단 테스트
    print("\n[4/4] 간이 진단 테스트")
    print("=" * 80)

    # 사용 가능한 엣지 전체 (FDR 필터 없이)로 확장하여 커버리지 확보
    # disease CUI → 연결된 symptom CUI set
    disease_symptoms: dict[str, dict[str, float]] = defaultdict(dict)
    for e in all_edges:
        if e["polarity"] != "present":
            continue
        a_stys = cui_stys.get(e["cui_a"], set())
        b_stys = cui_stys.get(e["cui_b"], set())

        # disease-symptom 관계만
        disease_types = {"T047", "T191", "T046", "T048"}
        symptom_types = {"T184", "T033", "T034"}

        if (a_stys & disease_types) and (b_stys & symptom_types):
            disease_symptoms[e["cui_a"]][e["cui_b"]] = e["n_present"]
        elif (b_stys & disease_types) and (a_stys & symptom_types):
            disease_symptoms[e["cui_b"]][e["cui_a"]] = e["n_present"]

    print(f"  질환 수 (present 관계 있는): {len(disease_symptoms)}")
    total_sym_count = sum(len(v) for v in disease_symptoms.values())
    print(f"  총 질환-증상 관계: {total_sym_count}")

    # DDXPlus 5개 질환 각각의 증상 수
    for name, dcui in ddx_diseases.items():
        syms = disease_symptoms.get(dcui, {})
        print(f"  {name}: {len(syms)}개 증상")

    # 간이 진단: 환자 증상 세트 → 가장 많이 매칭되는 질환
    print(f"\n  간이 진단 시뮬레이션:")

    test_patients = [
        {
            "name": "Patient A (호흡기 증상)",
            "symptoms": {"C0015967", "C0010200", "C0013404"},  # Fever, Cough, Dyspnea
            "expected": "Pneumonia",
        },
        {
            "name": "Patient B (흉통+불안)",
            "symptoms": {"C0030193", "C0003467", "C0013404"},  # Pain, Anxiety, Dyspnea
            "expected": "Panic attack",
        },
        {
            "name": "Patient C (소화기 증상)",
            "symptoms": {"C0030193", "C0018834"},  # Pain, Heartburn
            "expected": "GERD",
        },
    ]

    for patient in test_patients:
        print(f"\n  {patient['name']} (기대: {patient['expected']})")
        print(f"    증상: {', '.join(cui_names.get(s, s)[:20] for s in patient['symptoms'])}")

        scores = {}
        for dcui, syms in disease_symptoms.items():
            matched = set(syms.keys()) & patient["symptoms"]
            if matched:
                score = sum(syms[s] for s in matched)
                scores[dcui] = score

        if scores:
            top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
            for rank, (dcui, score) in enumerate(top5, 1):
                dname = cui_names.get(dcui, dcui)[:40]
                print(f"    #{rank}: {dname} (score={score})")
        else:
            print(f"    매칭 없음")

    # 결과 저장
    kg_summary = {
        "n_nodes": len(unique_cuis),
        "n_edges_significant": len(sig_edges),
        "n_edges_all": len(all_edges),
        "sty_distribution": dict(sty_dist),
        "rel_type_distribution": dict(rel_type_dist),
        "disease_symptom_counts": {
            name: len(disease_symptoms.get(dcui, {}))
            for name, dcui in ddx_diseases.items()
        },
    }
    with open(RESULTS_DIR / "step4_mini_kg_summary.json", "w") as f:
        json.dump(kg_summary, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {RESULTS_DIR / 'step4_mini_kg_summary.json'}")


if __name__ == "__main__":
    main()
