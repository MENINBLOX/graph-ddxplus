#!/usr/bin/env python3
"""Gold standard 데이터 준비.

DDXPlus KG, HPO phenotype.hpoa, SemMedDB에서
평가용 disease-symptom CUI 쌍 셋을 생성한다.
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

UMLS_DIR = Path("data/umls_extracted")
DATA_DIR = Path("pilot/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}


def load_cui_stys() -> dict[str, set[str]]:
    r = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            r[p[0]].add(p[1])
    return dict(r)


def load_cui_names() -> dict[str, str]:
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[1] == "ENG" and (p[0] not in names or p[2] == "P"):
                names[p[0]] = p[14]
    return names


def prepare_ddxplus_gold() -> dict:
    """DDXPlus KG를 gold standard CUI 쌍 셋으로 변환."""
    print("[1/3] DDXPlus Gold Standard...")

    # 질환 CUI 매핑
    with open("data/ddxplus/disease_umls_mapping.json") as f:
        disease_map = json.load(f)["mapping"]

    # 증상 CUI 매핑
    with open("data/ddxplus/umls_mapping.json") as f:
        symptom_data = json.load(f)
        symptom_map = symptom_data["mapping"]

    # disease CUI -> name
    disease_cuis = {}
    for name, info in disease_map.items():
        cui = info.get("umls_cui")
        if cui:
            disease_cuis[name] = cui

    # E_xx -> 불어 이름 매핑 (question_en으로 매칭)
    with open("data/ddxplus/release_evidences_en.json") as f:
        ev_en = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f:
        ev_fr = json.load(f)

    eid_to_fr = {}
    for eid, en_info in ev_en.items():
        en_q = en_info.get("question_en", "")
        for fr_name, fr_info in ev_fr.items():
            if en_info.get("question_en") == fr_info.get("question_en") and en_q:
                eid_to_fr[eid] = fr_name
                break

    # E_xx -> CUI
    symptom_cuis = {}
    for eid, fr_name in eid_to_fr.items():
        if fr_name in symptom_map:
            cui = symptom_map[fr_name].get("cui")
            if cui:
                symptom_cuis[eid] = cui

    # release_conditions에서 disease-symptom 쌍 생성
    with open("data/ddxplus/release_conditions_en.json") as f:
        conditions = json.load(f)

    gold_pairs = set()
    disease_symptom_map = defaultdict(set)

    for disease_name, info in conditions.items():
        d_cui = disease_cuis.get(disease_name)
        if not d_cui:
            continue

        # symptoms
        for eid in info.get("symptoms", {}):
            s_cui = symptom_cuis.get(eid)
            if s_cui:
                pair = tuple(sorted([d_cui, s_cui]))
                gold_pairs.add(pair)
                disease_symptom_map[d_cui].add(s_cui)

        # antecedents (과거력/위험인자)
        for eid in info.get("antecedents", {}):
            s_cui = symptom_cuis.get(eid)
            if s_cui:
                pair = tuple(sorted([d_cui, s_cui]))
                gold_pairs.add(pair)
                disease_symptom_map[d_cui].add(s_cui)

    print(f"  질환 수: {len(disease_cuis)}")
    print(f"  증상 CUI 수: {len(set(symptom_cuis.values()))}")
    print(f"  Gold 쌍 수: {len(gold_pairs)}")
    print(f"  질환당 평균 증상: {sum(len(v) for v in disease_symptom_map.values()) / max(len(disease_symptom_map), 1):.1f}")

    return {
        "pairs": sorted([list(p) for p in gold_pairs]),
        "disease_cuis": disease_cuis,
        "symptom_cuis": dict(symptom_cuis),
        "disease_symptom_map": {k: sorted(v) for k, v in disease_symptom_map.items()},
        "n_pairs": len(gold_pairs),
        "n_diseases": len(disease_cuis),
        "n_symptoms": len(set(symptom_cuis.values())),
    }


def prepare_hpo_gold(disease_cuis: dict[str, str]) -> dict:
    """HPO phenotype.hpoa에서 DDXPlus 49 질환 관련 annotation 추출."""
    print("\n[2/3] HPO Gold Standard...")

    # HPO → UMLS CUI 매핑
    with open("data/rarebench/hpo_umls_mapping.json") as f:
        hpo_map = json.load(f)["mapping"]

    hpo_to_cui = {}
    for hpo_id, info in hpo_map.items():
        cui = info.get("umls_cui")
        if cui:
            hpo_to_cui[hpo_id] = cui

    # DDXPlus 질환 CUI → OMIM/MONDO 매핑이 필요
    # MRCONSO에서 DDXPlus CUI의 OMIM ID를 찾기
    target_cuis = set(disease_cuis.values())
    cui_to_omim = {}
    cui_to_orpha = {}

    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui, sab, code = p[0], p[11], p[13]
            if cui not in target_cuis:
                continue
            if sab == "OMIM" and cui not in cui_to_omim:
                cui_to_omim[cui] = f"OMIM:{code}"
            elif sab == "ORPHANET" and cui not in cui_to_orpha:
                cui_to_orpha[cui] = f"ORPHA:{code}"

    print(f"  DDXPlus CUI → OMIM 매핑: {len(cui_to_omim)}/49")
    print(f"  DDXPlus CUI → ORPHANET 매핑: {len(cui_to_orpha)}/49")

    # phenotype.hpoa 파싱
    hpo_file = Path("data/external_kg/phenotype.hpoa")
    if not hpo_file.exists():
        print("  phenotype.hpoa 파일 없음")
        return {"pairs": [], "n_pairs": 0}

    # DDXPlus 질환의 OMIM/ORPHA ID로 HPO annotation 필터
    target_db_ids = set()
    cui_by_dbid = {}
    for cui, omim_id in cui_to_omim.items():
        target_db_ids.add(omim_id)
        cui_by_dbid[omim_id] = cui
    for cui, orpha_id in cui_to_orpha.items():
        target_db_ids.add(orpha_id)
        cui_by_dbid[orpha_id] = cui

    hpo_pairs = set()
    with open(hpo_file) as f:
        for line in f:
            if line.startswith("#") or line.startswith("database_id"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            db_id = parts[0]
            qualifier = parts[2]
            hpo_id = parts[3]

            if db_id not in target_db_ids:
                continue
            if qualifier == "NOT":
                continue

            disease_cui = cui_by_dbid.get(db_id)
            symptom_cui = hpo_to_cui.get(hpo_id)
            if disease_cui and symptom_cui:
                pair = tuple(sorted([disease_cui, symptom_cui]))
                hpo_pairs.add(pair)

    print(f"  HPO Gold 쌍 수: {len(hpo_pairs)}")
    return {
        "pairs": sorted([list(p) for p in hpo_pairs]),
        "n_pairs": len(hpo_pairs),
        "n_diseases_mapped": len(cui_to_omim) + len(cui_to_orpha),
    }


def prepare_semmeddb_baseline(disease_cuis: dict[str, str]) -> dict:
    """SemMedDB에서 DDXPlus 49 질환 관련 predication 추출."""
    print("\n[3/3] SemMedDB Baseline...")

    semmed_file = Path("data/semmeddb/semmedVER43_2024_R_PREDICATION.csv.gz")
    if not semmed_file.exists():
        print("  SemMedDB predication 파일 없음")
        # kg_stats.json에서 간접 정보 사용
        kg_stats = Path("data/semmeddb/kg_stats.json")
        if kg_stats.exists():
            with open(kg_stats) as f:
                stats = json.load(f)
            print(f"  SemMedDB 통계: {stats}")
        return {"pairs": [], "n_pairs": 0}

    target_cuis = set(disease_cuis.values())
    relevant_predicates = {"CAUSES", "MANIFESTATION_OF", "ASSOCIATED_WITH", "COEXISTS_WITH", "PREDISPOSES"}

    semmed_pairs = set()
    count = 0

    with gzip.open(semmed_file, "rt", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            count += 1
            if count % 5_000_000 == 0:
                print(f"  {count:,}행 처리...")
            if len(row) < 9:
                continue
            try:
                s_cui = row[4]  # SUBJECT_CUI
                predicate = row[3]  # PREDICATE
                o_cui = row[8]  # OBJECT_CUI
            except IndexError:
                continue

            if predicate not in relevant_predicates:
                continue

            if s_cui in target_cuis or o_cui in target_cuis:
                pair = tuple(sorted([s_cui, o_cui]))
                semmed_pairs.add(pair)

    print(f"  SemMedDB 관련 쌍: {len(semmed_pairs)}")
    return {
        "pairs": sorted([list(p) for p in semmed_pairs]),
        "n_pairs": len(semmed_pairs),
    }


def main():
    print("=" * 80)
    print("Gold Standard 데이터 준비")
    print("=" * 80)

    cui_names = load_cui_names()

    # DDXPlus
    ddxplus = prepare_ddxplus_gold()

    # HPO
    hpo = prepare_hpo_gold(ddxplus["disease_cuis"])

    # SemMedDB (대용량이므로 시간 걸릴 수 있음)
    semmed = prepare_semmeddb_baseline(ddxplus["disease_cuis"])

    # DDXPlus ∩ HPO 겹침
    ddx_set = set(tuple(p) for p in ddxplus["pairs"])
    hpo_set = set(tuple(p) for p in hpo["pairs"])
    overlap = ddx_set & hpo_set
    print(f"\n=== 겹침 분석 ===")
    print(f"  DDXPlus 쌍: {len(ddx_set)}")
    print(f"  HPO 쌍: {len(hpo_set)}")
    print(f"  DDXPlus ∩ HPO: {len(overlap)}")

    # 저장
    output = {
        "ddxplus": ddxplus,
        "hpo": hpo,
        "semmeddb": {"n_pairs": semmed["n_pairs"]},  # 대용량이므로 쌍 목록은 별도 저장
    }
    with open(DATA_DIR / "gold_standard.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if semmed["pairs"]:
        with open(DATA_DIR / "semmed_baseline_pairs.json", "w") as f:
            json.dump(semmed["pairs"], f)

    print(f"\n저장: {DATA_DIR / 'gold_standard.json'}")
    print("완료!")


if __name__ == "__main__":
    main()
