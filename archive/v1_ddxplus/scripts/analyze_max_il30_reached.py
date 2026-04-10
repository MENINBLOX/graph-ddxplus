#!/usr/bin/env python3
"""max_il=30 도달 케이스 상세 분석 (min_il 없음, Top3_stable_5) - 병렬 처리."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from tqdm import tqdm

from src.data_loader import DDXPlusLoader, Patient
from src.umls_kg import UMLSKG


def get_top_k_ranks(candidates, k):
    return tuple(c.cui for c in candidates[:k])


def process_patient(args):
    """단일 환자 처리 - max_il 도달 여부 및 상세 정보 반환."""
    patient_data, symptom_mapping, disease_mapping, fr_to_eng, evidence_antecedent_map = args

    max_il = 30
    stability_k = 3
    stability_n = 5

    patient = Patient(
        age=patient_data['age'],
        sex=patient_data['sex'],
        pathology=patient_data['pathology'],
        initial_evidence=patient_data['initial_evidence'],
        evidences=patient_data['evidences'],
        differential_diagnosis=patient_data.get('differential_diagnosis', []),
    )

    def get_symptom_cui(code):
        return symptom_mapping.get(code, {}).get('cui')

    def get_disease_cui(name_eng):
        return disease_mapping.get(name_eng, {}).get('umls_cui')

    kg = UMLSKG()

    gt_disease_eng = fr_to_eng.get(patient.pathology, patient.pathology)
    gt_cui = get_disease_cui(gt_disease_eng)

    patient_positive_cuis = set()
    for ev_str in patient.evidences:
        code = ev_str.split('_@_')[0] if '_@_' in ev_str else ev_str
        cui = get_symptom_cui(code)
        if cui:
            patient_positive_cuis.add(cui)

    initial_cui = get_symptom_cui(patient.initial_evidence)
    if not initial_cui:
        kg.close()
        return None

    kg.reset_state()
    kg.state.add_confirmed(initial_cui)

    il = 1
    rank_history = deque(maxlen=stability_n)
    confirmed_count = 1
    denied_count = 0
    max_il_reached = False

    for _ in range(max_il):
        candidates = kg.get_candidate_symptoms(
            initial_cui=initial_cui,
            limit=10,
            confirmed_cuis=kg.state.confirmed_cuis,
            denied_cuis=kg.state.denied_cuis,
        )
        if not candidates:
            break

        next_cui = candidates[0].cui
        if next_cui in patient_positive_cuis:
            kg.state.add_confirmed(next_cui)
            confirmed_count += 1
        else:
            kg.state.add_denied(next_cui)
            denied_count += 1

        il += 1

        if il >= max_il:
            max_il_reached = True
            break

        diagnosis_candidates = kg.get_diagnosis_candidates(top_k=stability_k)
        current_ranks = get_top_k_ranks(diagnosis_candidates, stability_k)
        rank_history.append(current_ranks)

        if len(rank_history) == stability_n:
            if all(r == rank_history[0] for r in rank_history):
                break

    final_candidates = kg.get_diagnosis_candidates(top_k=10)
    correct_at_1 = final_candidates[0].cui == gt_cui if final_candidates else False
    correct_at_10 = any(c.cui == gt_cui for c in final_candidates[:10]) if final_candidates else False

    kg.close()

    if not max_il_reached:
        return {'reached': False, 'correct_at_1': correct_at_1, 'il': il}

    # max_il 도달 케이스만 상세 분석
    total_evidences = len(patient.evidences)
    antecedent_count = 0
    symptom_count = 0
    for ev_str in patient.evidences:
        code = ev_str.split('_@_')[0] if '_@_' in ev_str else ev_str
        if evidence_antecedent_map.get(code, False):
            antecedent_count += 1
        else:
            symptom_count += 1

    return {
        'reached': True,
        'pathology': patient.pathology,
        'pathology_eng': gt_disease_eng,
        'il': il,
        'confirmed': confirmed_count,
        'denied': denied_count,
        'confirm_ratio': confirmed_count / il if il > 0 else 0,
        'correct_at_1': correct_at_1,
        'correct_at_10': correct_at_10,
        'total_evidences': total_evidences,
        'antecedent_count': antecedent_count,
        'symptom_count': symptom_count,
        'antecedent_ratio': antecedent_count / total_evidences if total_evidences > 0 else 0,
    }


def main():
    loader = DDXPlusLoader()
    patients = loader.load_patients(split='test')

    # antecedent 맵 미리 구축
    evidence_antecedent_map = {
        code: ev.is_antecedent for code, ev in loader.evidences.items()
    }

    patient_data_list = [
        {
            'age': p.age,
            'sex': p.sex,
            'pathology': p.pathology,
            'initial_evidence': p.initial_evidence,
            'evidences': p.evidences,
            'differential_diagnosis': p.differential_diagnosis,
        }
        for p in patients
    ]

    tasks = [
        (pd, loader.symptom_mapping, loader.disease_mapping, loader.fr_to_eng, evidence_antecedent_map)
        for pd in patient_data_list
    ]

    n_workers = 8
    print(f"총 {len(tasks):,}건, Workers: {n_workers}")

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_patient, t): i for i, t in enumerate(tasks)}
        with tqdm(total=len(futures), desc="분석") as pbar:
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)
                pbar.update(1)

    reached_cases = [r for r in results if r.get('reached')]
    normal_cases = [r for r in results if not r.get('reached')]

    print(f"\n=== max_il=30 도달 케이스 분석 ===")
    print(f"총 처리: {len(results):,}")
    print(f"정상 수렴: {len(normal_cases):,}")
    print(f"max_il 도달: {len(reached_cases)}건 ({len(reached_cases)/len(results):.2%})")

    if not reached_cases:
        print("max_il 도달 케이스 없음")
        return

    # 질환별 분포
    disease_counter = Counter(r['pathology_eng'] for r in reached_cases)
    print(f"\n--- 질환별 분포 ---")
    disease_stats = {}
    for disease, count in disease_counter.most_common(15):
        cases = [r for r in reached_cases if r['pathology_eng'] == disease]
        gtpa1 = sum(1 for r in cases if r['correct_at_1']) / len(cases)
        gtpa10 = sum(1 for r in cases if r['correct_at_10']) / len(cases)
        avg_ant = sum(r['antecedent_ratio'] for r in cases) / len(cases)
        avg_conf = sum(r['confirmed'] for r in cases) / len(cases)
        print(f"  {disease}: {count}건, GTPA@1={gtpa1:.1%}, GTPA@10={gtpa10:.1%}, "
              f"antecedent_ratio={avg_ant:.1%}, avg_confirmed={avg_conf:.1f}")
        disease_stats[disease] = {
            'count': count,
            'gtpa_1': gtpa1,
            'gtpa_10': gtpa10,
            'avg_antecedent_ratio': avg_ant,
            'avg_confirmed': avg_conf,
        }

    # 전체 통계
    total_correct_1 = sum(1 for r in reached_cases if r['correct_at_1'])
    total_correct_10 = sum(1 for r in reached_cases if r['correct_at_10'])
    avg_ant = sum(r['antecedent_ratio'] for r in reached_cases) / len(reached_cases)
    avg_conf = sum(r['confirmed'] for r in reached_cases) / len(reached_cases)
    avg_denied = sum(r['denied'] for r in reached_cases) / len(reached_cases)

    print(f"\n--- 전체 통계 ---")
    print(f"GTPA@1: {total_correct_1}/{len(reached_cases)} ({total_correct_1/len(reached_cases):.1%})")
    print(f"GTPA@10: {total_correct_10}/{len(reached_cases)} ({total_correct_10/len(reached_cases):.1%})")
    print(f"평균 antecedent 비율: {avg_ant:.1%}")
    print(f"평균 confirmed: {avg_conf:.1f}")
    print(f"평균 denied: {avg_denied:.1f}")

    output = {
        'total_processed': len(results),
        'normal_converged': len(normal_cases),
        'max_il_reached': len(reached_cases),
        'max_il_reached_pct': len(reached_cases) / len(results),
        'reached_gtpa_1': total_correct_1 / len(reached_cases),
        'reached_gtpa_10': total_correct_10 / len(reached_cases),
        'avg_antecedent_ratio': avg_ant,
        'avg_confirmed': avg_conf,
        'avg_denied': avg_denied,
        'disease_distribution': disease_stats,
    }

    output_path = Path(__file__).parent.parent / "results" / "max_il30_reached_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {output_path}")


if __name__ == "__main__":
    main()
