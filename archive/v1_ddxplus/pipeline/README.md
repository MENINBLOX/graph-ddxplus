# GraphTrace Reproduction Pipeline

논문 재현을 위한 단계별 실행 가이드.

## Prerequisites

- Python 3.11+ with `uv`
- Docker & Docker Compose
- UMLS License (https://www.nlm.nih.gov/research/umls/)
  - `data/umls-2025AB-full.zip` 에 위치

## Steps

```bash
# Step 0: DDXPlus 데이터 다운로드
uv run python pipeline/step0_download_ddxplus.py

# Step 1: UMLS → Neo4j KG 구축
uv run python pipeline/step1_build_kg.py

# Step 2: DDXPlus ↔ UMLS CUI 매핑 생성
uv run python pipeline/step2_build_mappings.py

# Step 3: Neo4j 다중 인스턴스 설정 (병렬 실험용)
bash pipeline/step3_setup_neo4j.sh

# Step 4: 증상 탐색 ANOVA (1단계, 검증셋 1,000건)
uv run python pipeline/step4_exploration_anova.py

# Step 5: 최종 진단 (2단계, 테스트셋 134,529건)
bash pipeline/step5_final_diagnosis.sh

# Step 6: 선행연구 비교 (MEDDxAgent, Complete Profile)
bash pipeline/step6_comparison.sh
```
