# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Entries describe impact on results and reproduction, not commit titles.

## [Unreleased]

### ⚠️ Breaking / 재현 결과 변경
- **frozen config**: 동결 설정을 최종 KG(`kg_merged29`)에서 사전 기준(진단 벤치마크 6종 Top-1 단순 평균, 동률 시 Top-3)으로 재선택했다 — 길이 지수 0.35, 2단계 길이 정규화(verbosity 0.5), 곱셈형 사전 결합, 속성 선택도 가중 off. **영향**: `reproduce.sh eval`/`ablate`의 모든 정확도 수치가 바뀐다(6종 평균 Top-1 35.55→37.23, RareBench 평균 38.9→41.5). `KG_ATTR_SEL=0`은 NLICE 속성 채점기에도 적용되어 NLICE 표가 바뀐다(aggravating +0.77%p, p=0.099; 7채널 전체 14.20→14.97). · **조치**: 이전 설정은 `ablation_suite.FROZEN_DEV`로 보존되며 `config_reselect.py`가 그 설정에서 출발해 선택 과정을 재현한다. 선택 기록은 `pilot/cache/prod/reselect29/reselect.json`.
- **reproduce.sh**: 기본 KG가 `kg_merged27`에서 `kg_merged29`로 바뀌었고, kg 단계가 실제 빌드 경로(전 코퍼스 추가 병합 → 수식어 승격 → 직렬 정규화 → 조립)를 따른다. **영향**: 이전 스크립트로는 논문의 KG가 만들어지지 않았다.

### ✨ Features
- **pipeline**: 질환명만으로 전 코퍼스(34,399개 질환) 문헌을 수집·추출하는 멱등 드라이버(`full_corpus_todo.py`, `full_corpus_resume.sh`). 재부팅 후 같은 명령으로 마지막 체크포인트부터 이어진다.
- **eval**: 구성요소 ablation 17변형 + 환자 부트스트랩 CI + paired 검정(`ablation_suite.py`, `reproduce.sh ablate`). 통계는 순위 덤프에서 재계산되므로 재채점 없이 검증 가능.
- **eval**: 논문의 비정확도 수치(검정 가능성 7개 벤치마크, 큐레이션 그래프 회수율, 오라클 상한, 토폴로지)를 저장소에서 재생성(`paper_measures.py`, `reproduce.sh measures`).
- **eval**: 사전 기준 좌표 탐색으로 단일 설정을 고르는 `config_reselect.py`; 평가한 모든 설정을 기록.
- **scorer**: `KG_ATTR_SEL` 스위치(속성 선택도 가중 on/off), `--unified` 스위치(통합 근거 ablation), 2단계 길이 정규화(`--verb_w`).

### 🐛 Fixes
- **scorer**: 같은 설정에서도 DDXPlus Top-1이 실행마다 환자 1명씩 달라지던 비결정성 → 원인: ARPACK SVD 난수 시작점과 문자열 해시 순서에 따른 부동소수 합산 순서 → SVD 시드 고정, 개념 ID 기준 동점 처리, `PYTHONHASHSEED=0`.
- **sources**: 슬래시로 결합된 RareBench 질환명이 검색 0건이 되어 "문헌 없음"처럼 보이던 문제 → 별칭 분해·숫자 아형 제거 질의 변형(93개 중 86개 회수).
- **reproduce.sh**: ie 단계 출력 파일명과 kg 단계 입력 파일명 불일치, 수집 파라미터가 실제 실행값과 다르던 문제 수정.
- **ablation**: 가중합 결합 변형에 곱셈형용 가중 2.0이 그대로 들어가 근거 채널에 음의 가중이 걸리던 인공적 결과 → 가중합이 실제로 쓰이던 0.4로 평가.

### 📝 Docs
- 국문 논문 초안(`draft/paper_ko/`): 문단별 작성·문장 단위 검토·문단 간 일관성 검토 산출물과 조립 스크립트. 수집 깊이 비대칭, 통제 클래스 사상표의 NLICE 참조 보완, 재선택의 in-sample 편향을 한계로 명시.
- 방법·평가 포렌식·속성 분석 문서(`docs/`).

---
**배포 노트**: DB 마이그레이션 없음. 새 환경 변수 `KG_ATTR_SEL`(동결 설정 = 0), `PYTHONHASHSEED=0`(reproduce.sh가 설정). 평가 프로세스당 메모리 약 4.4GB, 정규화 워커는 약 15GB라 27GB RAM에서는 직렬 실행.

[Unreleased]: https://github.com/MENINBLOX/graph-ddxplus/compare/main...feat/ie-strategy-attr-scope
