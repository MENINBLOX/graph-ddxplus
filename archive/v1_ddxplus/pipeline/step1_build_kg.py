#!/usr/bin/env python3
"""Step 1: UMLS 데이터로부터 Neo4j 지식그래프 구축.

UMLS MRCONSO.RRF + MRREL.RRF에서 증상-질환 관계를 추출하여
Neo4j에 Symptom-[INDICATES]->Disease 그래프를 구축한다.

전제조건:
  - data/umls-2025AB-full.zip이 존재
  - Neo4j가 실행 중 (docker compose up -d neo4j)

사용법:
  uv run python pipeline/step1_build_kg.py
"""

import json
import os
import zipfile
from pathlib import Path

from neo4j import GraphDatabase

# === 설정 ===
UMLS_ZIP = Path("data/umls-2025AB-full.zip")
UMLS_EXTRACT_DIR = Path("data/umls_extracted")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password123")

# DDXPlus 매핑에서 사용하는 CUI 목록
MAPPING_DIR = Path("data/ddxplus")

# 증상-질환 관계를 나타내는 UMLS 관계 타입
SYMPTOM_DISEASE_RELAS = {
    "has_finding",
    "finding_site_of",
    "associated_with",
    "manifestation_of",
    "has_manifestation",
    "clinically_associated_with",
}

# 의미 유형 (Semantic Types)
SYMPTOM_STYS = {
    "T184",  # Sign or Symptom
    "T033",  # Finding
    "T034",  # Laboratory or Test Result
    "T048",  # Mental or Behavioral Dysfunction
    "T037",  # Injury or Poisoning
}

DISEASE_STYS = {
    "T047",  # Disease or Syndrome
    "T019",  # Congenital Abnormality
    "T191",  # Neoplastic Process
    "T046",  # Pathologic Function
}


def extract_umls() -> Path:
    """UMLS RRF 파일 위치 확인.

    UMLS full release는 MetamorphoSys로 설치해야 RRF 파일이 생성된다.
    사전 준비:
      1. data/umls-2025AB-full.zip 다운로드
      2. MetamorphoSys 실행: unzip mmsys.zip && ./run_linux.sh
      3. 설치 완료 후 RRF 파일을 data/umls_extracted/에 복사
         또는 UMLS_EXTRACT_DIR 환경변수로 경로 지정
    """
    UMLS_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    needed_files = ["MRCONSO.RRF", "MRREL.RRF", "MRSTY.RRF"]
    existing = [f for f in needed_files if (UMLS_EXTRACT_DIR / f).exists()]

    if len(existing) == len(needed_files):
        print(f"  [ok] UMLS 파일 확인: {UMLS_EXTRACT_DIR}")
        return UMLS_EXTRACT_DIR

    missing = [f for f in needed_files if f not in [Path(e).name for e in existing]]
    raise FileNotFoundError(
        f"UMLS RRF 파일이 없습니다: {missing}\n"
        f"위치: {UMLS_EXTRACT_DIR}\n\n"
        "UMLS full release는 MetamorphoSys로 설치해야 합니다:\n"
        "  1. data/umls-2025AB-full.zip 압축 해제\n"
        "  2. cd 2025AB-full && unzip mmsys.zip\n"
        "  3. ./run_linux.sh (또는 run_mac.sh)\n"
        "  4. MetamorphoSys에서 설치 진행 (기본 설정)\n"
        "  5. 설치 완료 후:\n"
        f"     cp <설치경로>/META/MRCONSO.RRF {UMLS_EXTRACT_DIR}/\n"
        f"     cp <설치경로>/META/MRREL.RRF {UMLS_EXTRACT_DIR}/\n"
        f"     cp <설치경로>/META/MRSTY.RRF {UMLS_EXTRACT_DIR}/\n"
    )


def load_semantic_types(umls_dir: Path) -> dict[str, set[str]]:
    """MRSTY.RRF에서 CUI → Semantic Type 매핑 로드."""
    print("  MRSTY.RRF 로드 중...")
    cui_stys: dict[str, set[str]] = {}
    with open(umls_dir / "MRSTY.RRF", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            cui = parts[0]
            sty = parts[1]
            cui_stys.setdefault(cui, set()).add(sty)
    print(f"    {len(cui_stys):,}개 CUI의 의미 유형 로드")
    return cui_stys


def load_concept_names(umls_dir: Path) -> dict[str, str]:
    """MRCONSO.RRF에서 CUI → 영어 이름 매핑 로드 (ENG, preferred)."""
    print("  MRCONSO.RRF 로드 중 (영어 이름)...")
    cui_names: dict[str, str] = {}
    with open(umls_dir / "MRCONSO.RRF", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            cui = parts[0]
            lang = parts[1]
            ts = parts[2]  # Term Status: P=Preferred
            name = parts[14]
            if lang == "ENG" and cui not in cui_names:
                cui_names[cui] = name
            if lang == "ENG" and ts == "P":
                cui_names[cui] = name  # Preferred가 있으면 교체
    print(f"    {len(cui_names):,}개 CUI 이름 로드")
    return cui_names


def load_ddxplus_cuis() -> tuple[set[str], set[str], dict[str, bool]]:
    """DDXPlus 매핑에서 사용하는 증상/질환 CUI 로드."""
    symptom_cuis = set()
    symptom_antecedent: dict[str, bool] = {}
    disease_cuis = set()

    # 증상 매핑
    mapping_path = MAPPING_DIR / "umls_mapping.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            data = json.load(f)
        for code, info in data.get("mapping", {}).items():
            cui = info.get("cui")
            if cui:
                symptom_cuis.add(cui)

    # 증상의 is_antecedent 정보
    evidences_path = MAPPING_DIR / "release_evidences.json"
    if evidences_path.exists():
        with open(evidences_path) as f:
            evidences = json.load(f)
        with open(mapping_path) as f:
            mapping = json.load(f)
        for code, ev in evidences.items():
            cui = mapping.get("mapping", {}).get(code, {}).get("cui")
            if cui:
                symptom_antecedent[cui] = ev.get("is_antecedent", False)

    # 질환 매핑
    disease_path = MAPPING_DIR / "disease_umls_mapping.json"
    if disease_path.exists():
        with open(disease_path) as f:
            data = json.load(f)
        for name, info in data.get("mapping", {}).items():
            cui = info.get("umls_cui")
            if cui:
                disease_cuis.add(cui)

    print(f"  DDXPlus CUI: 증상 {len(symptom_cuis)}개, 질환 {len(disease_cuis)}개")
    return symptom_cuis, disease_cuis, symptom_antecedent


def build_kg_from_umls(
    umls_dir: Path,
    symptom_cuis: set[str],
    disease_cuis: set[str],
    symptom_antecedent: dict[str, bool],
    cui_names: dict[str, str],
    cui_stys: dict[str, set[str]],
) -> list[tuple[str, str]]:
    """MRREL.RRF에서 증상-질환 관계 추출."""
    print("  MRREL.RRF에서 관계 추출 중...")
    relationships: set[tuple[str, str]] = set()

    # DDXPlus conditions에서 직접 관계 추출 (가장 정확)
    conditions_path = MAPPING_DIR / "release_conditions.json"
    mapping_path = MAPPING_DIR / "umls_mapping.json"
    disease_mapping_path = MAPPING_DIR / "disease_umls_mapping.json"

    if conditions_path.exists() and mapping_path.exists() and disease_mapping_path.exists():
        with open(conditions_path) as f:
            conditions = json.load(f)
        with open(mapping_path) as f:
            sym_mapping = json.load(f).get("mapping", {})
        with open(disease_mapping_path) as f:
            dis_mapping = json.load(f).get("mapping", {})

        for disease_name, cond in conditions.items():
            disease_cui = dis_mapping.get(disease_name, {}).get("umls_cui")
            if not disease_cui:
                continue

            # symptoms + antecedents
            all_symptoms = {}
            all_symptoms.update(cond.get("symptoms", {}))
            all_symptoms.update(cond.get("antecedents", {}))

            for sym_code in all_symptoms:
                sym_cui = sym_mapping.get(sym_code, {}).get("cui")
                if sym_cui:
                    relationships.add((sym_cui, disease_cui))

    print(f"    DDXPlus conditions에서 {len(relationships):,}개 관계 추출")

    # UMLS MRREL에서 추가 관계 보강 (선택적)
    mrrel_path = umls_dir / "MRREL.RRF"
    if mrrel_path.exists():
        umls_rels = 0
        with open(mrrel_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                cui1 = parts[0]
                rel = parts[3]
                rela = parts[7] if len(parts) > 7 else ""
                cui2 = parts[4]

                if rela.lower() not in SYMPTOM_DISEASE_RELAS:
                    continue

                # cui1이 증상이고 cui2가 질환인 경우
                if cui1 in symptom_cuis and cui2 in disease_cuis:
                    relationships.add((cui1, cui2))
                    umls_rels += 1
                elif cui2 in symptom_cuis and cui1 in disease_cuis:
                    relationships.add((cui2, cui1))
                    umls_rels += 1

        print(f"    UMLS MRREL에서 {umls_rels:,}개 추가 관계 보강")

    print(f"    총 관계: {len(relationships):,}개")
    return list(relationships)


def import_to_neo4j(
    relationships: list[tuple[str, str]],
    symptom_cuis: set[str],
    disease_cuis: set[str],
    symptom_antecedent: dict[str, bool],
    cui_names: dict[str, str],
) -> None:
    """Neo4j에 KG 임포트."""
    print(f"\n  Neo4j 연결: {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    with driver.session() as session:
        # 기존 데이터 삭제
        print("  기존 데이터 삭제...")
        session.run("MATCH (n) DETACH DELETE n")

        # 인덱스 생성
        print("  인덱스 생성...")
        session.run("CREATE INDEX IF NOT EXISTS FOR (s:Symptom) ON (s.cui)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (d:Disease) ON (d.cui)")

        # 질환 노드 생성
        used_diseases = set(d for _, d in relationships)
        print(f"  Disease 노드 생성: {len(used_diseases)}개")
        for cui in used_diseases:
            name = cui_names.get(cui, cui)
            session.run(
                "CREATE (d:Disease {cui: $cui, name: $name})",
                cui=cui,
                name=name,
            )

        # 증상 노드 생성
        used_symptoms = set(s for s, _ in relationships)
        print(f"  Symptom 노드 생성: {len(used_symptoms)}개")
        for cui in used_symptoms:
            name = cui_names.get(cui, cui)
            is_ante = symptom_antecedent.get(cui, False)
            session.run(
                "CREATE (s:Symptom {cui: $cui, name: $name, is_antecedent: $is_ante})",
                cui=cui,
                name=name,
                is_ante=is_ante,
            )

        # INDICATES 관계 생성
        print(f"  INDICATES 관계 생성: {len(relationships)}개")
        for i, (sym_cui, dis_cui) in enumerate(relationships):
            session.run(
                """
                MATCH (s:Symptom {cui: $sym_cui})
                MATCH (d:Disease {cui: $dis_cui})
                CREATE (s)-[:INDICATES]->(d)
                """,
                sym_cui=sym_cui,
                dis_cui=dis_cui,
            )
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(relationships)}")

    # 검증
    with driver.session() as session:
        stats = session.run(
            """
            MATCH (s:Symptom) WITH count(s) AS symptoms
            MATCH (d:Disease) WITH symptoms, count(d) AS diseases
            MATCH ()-[r:INDICATES]->() WITH symptoms, diseases, count(r) AS rels
            RETURN symptoms, diseases, rels
            """
        ).single()
        print(f"\n=== KG 통계 ===")
        print(f"  Symptom: {stats['symptoms']}개")
        print(f"  Disease: {stats['diseases']}개")
        print(f"  INDICATES: {stats['rels']}개")

    driver.close()


def main() -> None:
    print("=" * 60)
    print("Step 1: UMLS → Neo4j KG 구축")
    print("=" * 60)

    # 1. UMLS 추출
    print("\n[1/4] UMLS 데이터 추출")
    umls_dir = extract_umls()

    # 2. DDXPlus CUI 로드
    print("\n[2/4] DDXPlus CUI 매핑 로드")
    symptom_cuis, disease_cuis, symptom_antecedent = load_ddxplus_cuis()

    # 3. UMLS 메타데이터 로드
    print("\n[3/4] UMLS 메타데이터 로드")
    cui_names = load_concept_names(umls_dir)
    cui_stys = load_semantic_types(umls_dir)

    # 4. 관계 추출 및 임포트
    print("\n[4/4] KG 구축")
    relationships = build_kg_from_umls(
        umls_dir, symptom_cuis, disease_cuis,
        symptom_antecedent, cui_names, cui_stys,
    )
    import_to_neo4j(
        relationships, symptom_cuis, disease_cuis,
        symptom_antecedent, cui_names,
    )

    print("\n완료!")


if __name__ == "__main__":
    main()
