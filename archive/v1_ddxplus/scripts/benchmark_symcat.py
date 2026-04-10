#!/usr/bin/env python3
"""SymCat 벤치마크: KG 구축 → 합성 환자 생성 → GraphTrace 실행.

SymCat 801 diseases + 474 symptoms 데이터로 GraphTrace를 실행하고
DDXPlus 대비 일반화 성능을 측정한다.

Usage:
    uv run python scripts/benchmark_symcat.py --n-patients 10000 --seed 42
    uv run python scripts/benchmark_symcat.py --build-kg-only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# SymCat parse
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "symcat"))

from neo4j import GraphDatabase

# === Configuration ===
SYMCAT_DIR = Path("data/symcat")
NEO4J_URI = os.getenv("SYMCAT_NEO4J_URI", "bolt://localhost:7695")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password123")


# === Data Classes ===

@dataclass
class SymCatPatient:
    """합성 환자."""
    disease_name: str
    disease_cui: str
    initial_symptom_cui: str
    initial_symptom_name: str
    positive_symptom_cuis: Set[str]   # 환자가 가진 증상 CUIs
    all_disease_symptom_cuis: Set[str]  # 해당 질병의 모든 증상 CUIs


@dataclass
class PatientState:
    """환자별 진단 상태."""
    patient: SymCatPatient
    idx: int
    confirmed_cuis: Set[str] = field(default_factory=set)
    denied_cuis: Set[str] = field(default_factory=set)
    asked_cuis: Set[str] = field(default_factory=set)
    il: int = 0
    predicted_cui: Optional[str] = None
    predicted_name: Optional[str] = None
    done: bool = False


# === Step 1: Build SymCat KG ===

def load_symcat_data() -> Tuple[dict, dict, dict, dict]:
    """SymCat 파싱 데이터 + UMLS 매핑 로드."""
    from parse import parse_symcat_conditions, parse_symcat_symptoms

    conditions = parse_symcat_conditions(str(SYMCAT_DIR / "symcat-801-diseases.csv"))
    symptoms_db = parse_symcat_symptoms(str(SYMCAT_DIR / "symcat-474-symptoms.csv"))

    with open(SYMCAT_DIR / "disease_umls_mapping.json") as f:
        disease_mapping = json.load(f)["mapping"]
    with open(SYMCAT_DIR / "symptom_umls_mapping.json") as f:
        symptom_mapping = json.load(f)["mapping"]

    return conditions, symptoms_db, disease_mapping, symptom_mapping


def build_symcat_kg(
    conditions: dict,
    symptoms_db: dict,
    disease_mapping: dict,
    symptom_mapping: dict,
) -> Tuple[int, int, int]:
    """SymCat KG를 Neo4j에 구축. Returns (n_diseases, n_symptoms, n_edges)."""
    print(f"\n[KG 구축] Neo4j: {NEO4J_URI}")

    # slug → symptom name 매핑
    slug_to_name = {slug: data["name"] for slug, data in symptoms_db.items()}

    # 유효한 매핑만 추출 (exact + normalized만 사용)
    valid_diseases = {}
    for d_name, info in disease_mapping.items():
        if info.get("umls_cui") and info.get("match_method") in ("exact", "normalized"):
            valid_diseases[d_name] = info

    valid_symptoms = {}
    for s_name, info in symptom_mapping.items():
        if info.get("umls_cui") and info.get("match_method") == "exact":
            valid_symptoms[s_name] = info

    print(f"  유효 질병: {len(valid_diseases)}, 유효 증상: {len(valid_symptoms)}")

    # Edge 추출
    edges = []  # (symptom_cui, disease_cui)
    disease_nodes = {}  # cui → name
    symptom_nodes = {}  # cui → name

    for cond_slug, cond_data in conditions.items():
        d_name = cond_data["condition_name"]
        if d_name not in valid_diseases:
            continue
        d_cui = valid_diseases[d_name]["umls_cui"]
        d_umls_name = valid_diseases[d_name]["umls_name"]
        disease_nodes[d_cui] = d_umls_name

        for sym_slug, sym_data in cond_data.get("symptoms", {}).items():
            s_name = slug_to_name.get(sym_slug)
            if s_name and s_name in valid_symptoms:
                s_cui = valid_symptoms[s_name]["umls_cui"]
                s_umls_name = valid_symptoms[s_name]["umls_name"]
                symptom_nodes[s_cui] = s_umls_name
                edges.append((s_cui, d_cui))

    print(f"  노드: {len(disease_nodes)} diseases, {len(symptom_nodes)} symptoms")
    print(f"  엣지: {len(edges)} INDICATES")

    # Neo4j import
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run("CREATE INDEX IF NOT EXISTS FOR (s:Symptom) ON (s.cui)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (d:Disease) ON (d.cui)")

        for cui, name in disease_nodes.items():
            session.run("CREATE (d:Disease {cui: $cui, name: $name})", cui=cui, name=name)

        for cui, name in symptom_nodes.items():
            session.run(
                "CREATE (s:Symptom {cui: $cui, name: $name, is_antecedent: false})",
                cui=cui, name=name,
            )

        for s_cui, d_cui in edges:
            session.run(
                "MATCH (s:Symptom {cui: $s}) MATCH (d:Disease {cui: $d}) CREATE (s)-[:INDICATES]->(d)",
                s=s_cui, d=d_cui,
            )

        # 검증
        stats = session.run(
            "MATCH (s:Symptom) WITH count(s) AS sc "
            "MATCH (d:Disease) WITH sc, count(d) AS dc "
            "MATCH ()-[r:INDICATES]->() RETURN sc, dc, count(r) AS rc"
        ).single()
        print(f"  Neo4j: {stats['sc']} symptoms, {stats['dc']} diseases, {stats['rc']} edges")

    driver.close()
    return len(disease_nodes), len(symptom_nodes), len(edges)


# === Step 2: Generate Synthetic Patients ===

def generate_patients(
    conditions: dict,
    symptoms_db: dict,
    disease_mapping: dict,
    symptom_mapping: dict,
    n_patients: int = 10000,
    seed: int = 42,
) -> List[SymCatPatient]:
    """확률 기반 합성 환자 생성."""
    print(f"\n[환자 생성] {n_patients}명, seed={seed}")
    rng = random.Random(seed)

    slug_to_name = {slug: data["name"] for slug, data in symptoms_db.items()}

    valid_diseases = {}
    for d_name, info in disease_mapping.items():
        if info.get("umls_cui") and info.get("match_method") in ("exact", "normalized"):
            valid_diseases[d_name] = info

    valid_symptoms = {}
    for s_name, info in symptom_mapping.items():
        if info.get("umls_cui") and info.get("match_method") == "exact":
            valid_symptoms[s_name] = info

    # 질병별 증상-확률 테이블 (매핑된 것만)
    disease_symptom_probs = {}  # d_name → [(s_cui, s_name, probability)]
    for cond_slug, cond_data in conditions.items():
        d_name = cond_data["condition_name"]
        if d_name not in valid_diseases:
            continue
        syms = []
        for sym_slug, sym_data in cond_data.get("symptoms", {}).items():
            s_name = slug_to_name.get(sym_slug)
            if s_name and s_name in valid_symptoms:
                prob = sym_data["probability"] / 100.0
                s_cui = valid_symptoms[s_name]["umls_cui"]
                syms.append((s_cui, s_name, prob))
        if syms:
            disease_symptom_probs[d_name] = syms

    # 유효 질병 목록
    valid_disease_names = list(disease_symptom_probs.keys())
    print(f"  유효 질병 (증상 있음): {len(valid_disease_names)}")

    # 환자 생성
    patients = []
    per_disease = max(1, n_patients // len(valid_disease_names))
    remainder = n_patients - per_disease * len(valid_disease_names)

    for d_name in valid_disease_names:
        d_info = valid_diseases[d_name]
        d_cui = d_info["umls_cui"]
        sym_probs = disease_symptom_probs[d_name]

        n_for_this = per_disease + (1 if remainder > 0 else 0)
        remainder -= 1

        for _ in range(n_for_this):
            # 확률에 따라 증상 샘플링
            positive_cuis = set()
            positive_names = {}
            all_cuis = set()
            for s_cui, s_name, prob in sym_probs:
                all_cuis.add(s_cui)
                if rng.random() < prob:
                    positive_cuis.add(s_cui)
                    positive_names[s_cui] = s_name

            # 최소 1개 양성 증상 보장
            if not positive_cuis:
                # 가장 확률 높은 증상 1개 강제 추가
                best = max(sym_probs, key=lambda x: x[2])
                positive_cuis.add(best[0])
                positive_names[best[0]] = best[1]

            # 초기 증상: 양성 중 랜덤
            init_cui = rng.choice(list(positive_cuis))

            patients.append(SymCatPatient(
                disease_name=d_name,
                disease_cui=d_cui,
                initial_symptom_cui=init_cui,
                initial_symptom_name=positive_names[init_cui],
                positive_symptom_cuis=positive_cuis,
                all_disease_symptom_cuis=all_cuis,
            ))

    rng.shuffle(patients)
    patients = patients[:n_patients]

    # 통계
    avg_pos = sum(len(p.positive_symptom_cuis) for p in patients) / len(patients)
    print(f"  생성: {len(patients)}명, 평균 양성 증상: {avg_pos:.1f}")

    return patients


# === Step 3: Run GraphTrace Benchmark ===

def run_benchmark(
    patients: List[SymCatPatient],
    scoring: str = "v15_ratio",
    max_il: int = 50,
    min_il: int = 10,
) -> dict:
    """GraphTrace 벤치마크 실행."""
    print(f"\n[벤치마크] {len(patients)}명, scoring={scoring}, max_il={max_il}")

    # UMLSKG import (기존 코드 재사용)
    from src.umls_kg import UMLSKG

    kg = UMLSKG(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS)

    # 질병 CUI → name 역매핑
    with kg.driver.session() as session:
        disease_cuis = {}
        result = session.run("MATCH (d:Disease) RETURN d.cui AS cui, d.name AS name")
        for r in result:
            disease_cuis[r["cui"]] = r["name"]
    print(f"  KG 질병: {len(disease_cuis)}")

    # 유효 disease CUIs (환자의 ground truth가 KG에 있는 것만)
    gt_in_kg = sum(1 for p in patients if p.disease_cui in disease_cuis)
    print(f"  환자 GT가 KG에 있음: {gt_in_kg}/{len(patients)} ({gt_in_kg/len(patients)*100:.1f}%)")

    states = []
    for idx, patient in enumerate(patients):
        state = PatientState(patient=patient, idx=idx)
        state.confirmed_cuis.add(patient.initial_symptom_cui)
        state.asked_cuis.add(patient.initial_symptom_cui)
        state.il = 1
        states.append(state)

    start_time = time.time()
    pbar = tqdm(total=len(states), desc="Diagnosing", unit="patient")

    for round_num in range(1, max_il + 1):
        active = [s for s in states if not s.done]
        if not active:
            break

        for state in active:
            _process_round(state, kg, scoring, max_il, min_il, disease_cuis)

        completed = [s for s in states if s.done]
        pbar.n = len(completed)
        if completed:
            correct = sum(1 for s in completed if s.predicted_cui == s.patient.disease_cui)
            acc = correct / len(completed)
            avg_il = sum(s.il for s in completed) / len(completed)
            pbar.set_postfix({"round": round_num, "GTPA@1": f"{acc:.1%}", "IL": f"{avg_il:.1f}"})
        pbar.refresh()

    # 미완료 강제 진단
    for state in states:
        if not state.done:
            _make_diagnosis(state, kg, scoring, disease_cuis)

    pbar.n = len(states)
    pbar.refresh()
    pbar.close()

    elapsed = time.time() - start_time
    kg.close()

    return _calc_metrics(states, elapsed, scoring, max_il)


def _process_round(
    state: PatientState,
    kg,
    scoring: str,
    max_il: int,
    min_il: int,
    disease_cuis: dict,
):
    """환자 1라운드 처리."""
    # KG 상태 동기화
    kg.state.confirmed_cuis = state.confirmed_cuis.copy()
    kg.state.denied_cuis = state.denied_cuis.copy()
    kg.state.asked_cuis = state.asked_cuis.copy()

    # 중단 조건
    should_stop, _ = kg.should_stop(
        max_il=max_il,
        min_il=min_il,
        confidence_threshold=0.30,
        gap_threshold=0.04,
        relative_gap_threshold=1.5,
    )
    if should_stop:
        _make_diagnosis(state, kg, scoring, disease_cuis)
        return

    # 후보 증상
    candidates = kg.get_candidate_symptoms(
        state.patient.initial_symptom_cui,
        limit=10,
        confirmed_cuis=state.confirmed_cuis.copy(),
        denied_cuis=state.denied_cuis.copy(),
        asked_cuis=state.asked_cuis.copy(),
    )

    if not candidates:
        _make_diagnosis(state, kg, scoring, disease_cuis)
        return

    selected_cui = candidates[0].cui
    state.asked_cuis.add(selected_cui)

    # 환자 응답 시뮬레이션: CUI가 양성 증상에 있으면 yes
    if selected_cui in state.patient.positive_symptom_cuis:
        state.confirmed_cuis.add(selected_cui)
    else:
        state.denied_cuis.add(selected_cui)

    state.il += 1


def _make_diagnosis(state: PatientState, kg, scoring: str, disease_cuis: dict):
    """최종 진단."""
    candidates = kg.get_diagnosis_candidates(
        top_k=100,
        scoring=scoring,
        confirmed_cuis=state.confirmed_cuis.copy(),
        denied_cuis=state.denied_cuis.copy(),
    )

    for c in candidates:
        if c.cui in disease_cuis:
            state.predicted_cui = c.cui
            state.predicted_name = disease_cuis[c.cui]
            break

    if not state.predicted_cui and disease_cuis:
        state.predicted_cui = list(disease_cuis.keys())[0]
        state.predicted_name = disease_cuis[state.predicted_cui]

    state.done = True


def _calc_metrics(states: List[PatientState], elapsed: float, scoring: str, max_il: int) -> dict:
    """메트릭 계산."""
    total = len(states)
    correct_1 = sum(1 for s in states if s.predicted_cui == s.patient.disease_cui)
    total_il = sum(s.il for s in states)

    gtpa1 = correct_1 / total if total > 0 else 0
    avg_il = total_il / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"SYMCAT BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Patients:  {total:,}")
    print(f"GTPA@1:    {gtpa1:.2%}")
    print(f"Avg IL:    {avg_il:.2f}")
    print(f"Scoring:   {scoring}")
    print(f"Max IL:    {max_il}")
    print(f"Time:      {elapsed/60:.1f} min")
    print(f"{'='*70}")

    # 질병별 성능
    disease_stats = {}
    for s in states:
        d = s.patient.disease_name
        if d not in disease_stats:
            disease_stats[d] = {"total": 0, "correct": 0, "il_sum": 0}
        disease_stats[d]["total"] += 1
        disease_stats[d]["il_sum"] += s.il
        if s.predicted_cui == s.patient.disease_cui:
            disease_stats[d]["correct"] += 1

    # Top/Bottom 5
    sorted_diseases = sorted(
        disease_stats.items(),
        key=lambda x: x[1]["correct"] / max(x[1]["total"], 1),
    )

    print(f"\nBottom 5 diseases:")
    for d, stats in sorted_diseases[:5]:
        acc = stats["correct"] / stats["total"]
        avg = stats["il_sum"] / stats["total"]
        print(f"  {d}: {acc:.1%} ({stats['correct']}/{stats['total']}), IL={avg:.1f}")

    print(f"\nTop 5 diseases:")
    for d, stats in sorted_diseases[-5:]:
        acc = stats["correct"] / stats["total"]
        avg = stats["il_sum"] / stats["total"]
        print(f"  {d}: {acc:.1%} ({stats['correct']}/{stats['total']}), IL={avg:.1f}")

    result = {
        "dataset": "symcat",
        "patients": total,
        "gtpa_at_1": gtpa1,
        "avg_il": avg_il,
        "scoring": scoring,
        "max_il": max_il,
        "elapsed_seconds": elapsed,
        "n_diseases": len(disease_stats),
        "disease_stats": {
            d: {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy": s["correct"] / s["total"],
                "avg_il": s["il_sum"] / s["total"],
            }
            for d, s in disease_stats.items()
        },
    }

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "symcat_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    return result


# === Main ===

def main():
    parser = argparse.ArgumentParser(description="SymCat Benchmark for GraphTrace")
    parser.add_argument("--n-patients", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scoring", type=str, default="v15_ratio")
    parser.add_argument("--max-il", type=int, default=50)
    parser.add_argument("--min-il", type=int, default=10)
    parser.add_argument("--build-kg-only", action="store_true")
    args = parser.parse_args()

    conditions, symptoms_db, disease_mapping, symptom_mapping = load_symcat_data()

    # Step 1: Build KG
    build_symcat_kg(conditions, symptoms_db, disease_mapping, symptom_mapping)

    if args.build_kg_only:
        print("\nKG 구축만 완료.")
        return

    # Step 2: Generate patients
    patients = generate_patients(
        conditions, symptoms_db, disease_mapping, symptom_mapping,
        n_patients=args.n_patients,
        seed=args.seed,
    )

    # Step 3: Run benchmark
    run_benchmark(
        patients,
        scoring=args.scoring,
        max_il=args.max_il,
        min_il=args.min_il,
    )


if __name__ == "__main__":
    main()
