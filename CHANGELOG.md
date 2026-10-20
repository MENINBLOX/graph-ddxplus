# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Entries describe impact on results and reproduction, not commit titles.

## [Unreleased]

### ⚠️ Breaking / 재현 결과 변경
- **frozen config**: 동결 설정을 최종 KG(`kg_merged29`)에서 사전 기준(진단 벤치마크 6종 Top-1 단순 평균, 동률 시 Top-3)으로 재선택했다 — 길이 지수 0.35, 2단계 길이 정규화(verbosity 0.5), 곱셈형 사전 결합, 속성 선택도 가중 off. **영향**: `reproduce.sh eval`/`ablate`의 모든 정확도 수치가 바뀐다(6종 평균 Top-1 35.55→37.23, RareBench 평균 38.9→41.5). `KG_ATTR_SEL=0`은 NLICE 속성 채점기에도 적용되어 NLICE 표가 바뀐다(aggravating +0.77%p, p=0.099; 7채널 전체 14.20→14.97). · **조치**: 이전 설정은 `ablation_suite.FROZEN_DEV`로 보존되며 `config_reselect.py`가 그 설정에서 출발해 선택 과정을 재현한다. 선택 기록은 `pilot/cache/prod/reselect29/reselect.json`.
- **reproduce.sh**: 기본 KG가 `kg_merged27`에서 `kg_merged29`로 바뀌었고, kg 단계가 실제 빌드 경로(전 코퍼스 추가 병합 → 수식어 승격 → 직렬 정규화 → 조립)를 따른다. **영향**: 이전 스크립트로는 논문의 KG가 만들어지지 않았다.

### ✨ Features
- **review**: 사전 그래프 대비 혼합 추출의 비용과 기전을 측정했다. 구축 비용은 혼합 약 136시간(GPU 생성 127시간) 대 사전 약 0.45시간(CPU)이다. 혼합 간선의 86.9%가 사전 그래프에도 있고, 희귀질환 개발 세트에서 혼합의 우세는 혼합 전용 간선 없이도 대부분 유지된다(LLM의 역할은 소견 선택 쪽). HPO 주석 일치율은 혼합 12.6%, 사전 7.8%로 둘 다 감사한 구 그래프보다 낮아, 구 감사 결과(29.5%)를 주 시스템에 옮겨 읽을 수 없다. 의료진 판정용 혼합 그래프 간선 100개 블라인드 시트(`docs/audit/smix_audit_*`)를 준비했다(판정 전).
- **recall**: 재현율 손실 분해를 v2+v3 그래프에서 다시 쟀다. 수집 손실은 설계 D 그래프 67.0% → 혼합 37.3%, 문헌 없음 22.4% → 0.3%이고, 추출·연결 손실은 늘지 않았다. `recall_decomp.py`의 기본값은 이전 결과를 그대로 재현한다.
- **confirm-e**: 혼합 추출 그래프(S-mix: LLM이 고른 구절 + 구절 표가 못 붙인 구절 안의 UMLS 사전 이름)를 새 봉인 세트(DDXPlus·SymCat 같은 질환의 새 환자)에서 1회 확증. 설계 D 그래프보다 Top-10 DDXPlus +6.5%p·SymCat +2.4%p, LLM 단독보다 매크로 +1.6%p 높았다. UMLS 사전·합집합 그래프보다는 낮았고(DDXPlus Top-1 −5%p), BM25 결합은 이득이 없었다. 7개 판정이 모두 실행 전에 커밋한 예상과 같다. 결과는 `pilot/cache/v2/confirmE/`와 `docs/research_design.md` E7. 봉인·동결·실행 스크립트: `seal_confirm_e.py`, `freeze_e.py`, `run_confirm_E.sh`, `confirm_e_aux.py`.
- **stats**: 세트 등가중 매크로 대응 부트스트랩(`paired_stats.macro_compare`)과 판정 세트로 한정한 Holm 족(`--require`). 설계 D T4 수치 재현을 테스트로 고정.
- **graph**: 재연결 항목의 근거 언급이 원 LLM 구절(`span`)을 갖고, 근거 문장을 그 구절 중심으로 자른다. 순위화 그래프는 바뀌지 않는다(지문 동일).
- **pipeline**: 질환명만으로 전 코퍼스(34,399개 질환) 문헌을 수집·추출하는 멱등 드라이버(`full_corpus_todo.py`, `full_corpus_resume.sh`). 재부팅 후 같은 명령으로 마지막 체크포인트부터 이어진다.
- **eval**: 구성요소 ablation 17변형 + 환자 부트스트랩 CI + paired 검정(`ablation_suite.py`, `reproduce.sh ablate`). 통계는 순위 덤프에서 재계산되므로 재채점 없이 검증 가능.
- **eval**: 논문의 비정확도 수치(검정 가능성 7개 벤치마크, 큐레이션 그래프 회수율, 오라클 상한, 토폴로지)를 저장소에서 재생성(`paper_measures.py`, `reproduce.sh measures`).
- **eval**: 사전 기준 좌표 탐색으로 단일 설정을 고르는 `config_reselect.py`; 평가한 모든 설정을 기록.
- **scorer**: `KG_ATTR_SEL` 스위치(속성 선택도 가중 on/off), `--unified` 스위치(통합 근거 ablation), 2단계 길이 정규화(`--verb_w`).

### 🐛 Fixes
- **retrieval**: 결합(Fusion) 안의 그래프 순위가 단독 평가(후보 창 40)와 다른 창(800)으로 매겨져, 결합−그래프 차이에 창 차이가 섞이던 문제 → 그래프 목록을 단독 평가와 같은 창으로 매긴다. 설계 D의 T4 수치는 이전 방식의 기록으로 둔다.
- **confirm-e**: 재빌드가 실패해도 이전 빌드 요약이 남아 지문 검사가 통과하던 문제(원인: `.gz.new` 출력이 비압축으로 쓰여 `.gz` 입력이 깨짐) → 확증 스크립트가 요약이 새로 쓰였는지와 적재 간선 수까지 확인한다.
- **scorer**: 같은 설정에서도 DDXPlus Top-1이 실행마다 환자 1명씩 달라지던 비결정성 → 원인: ARPACK SVD 난수 시작점과 문자열 해시 순서에 따른 부동소수 합산 순서 → SVD 시드 고정, 개념 ID 기준 동점 처리, `PYTHONHASHSEED=0`.
- **sources**: 슬래시로 결합된 RareBench 질환명이 검색 0건이 되어 "문헌 없음"처럼 보이던 문제 → 별칭 분해·숫자 아형 제거 질의 변형(93개 중 86개 회수).
- **reproduce.sh**: ie 단계 출력 파일명과 kg 단계 입력 파일명 불일치, 수집 파라미터가 실제 실행값과 다르던 문제 수정.
- **ablation**: 가중합 결합 변형에 곱셈형용 가중 2.0이 그대로 들어가 근거 채널에 음의 가중이 걸리던 인공적 결과 → 가중합이 실제로 쓰이던 0.4로 평가.

### 📝 Docs
- 논문 v3: 리뷰 지적(사전 매칭의 경쟁력, 주 그래프 미감사, DDXPlus 49질환 범위)과 외부 메모 검토에서 채택한 항목(결론 요약, 5.2 후속 연구)을 반영했다. 'threshold-free'로 잘못 적혀 있던 연결기 설명(`normalize_v2.py`)을 실제 동작으로 고쳤다.
- 국문 논문 초안 v3(`draft/paper_ko_v3/`): 혼합 추출과 설계 D·E 기준. 문장별 근거 로그와 절 간 일관성 검토 기록을 함께 둔다.
- 설계 E(`docs/research_design.md` 상단): 판정 기준과 예상값을 동결·실행 전에 커밋했고, 결과를 E7에 그대로 기록했다.
- 국문 논문 초안(`draft/paper_ko/`): 문단별 작성·문장 단위 검토·문단 간 일관성 검토 산출물과 조립 스크립트. 수집 깊이 비대칭, 통제 클래스 사상표의 NLICE 참조 보완, 재선택의 in-sample 편향을 한계로 명시.
- 방법·평가 포렌식·속성 분석 문서(`docs/`).

---
**배포 노트**: DB 마이그레이션 없음. 새 환경 변수 `KG_ATTR_SEL`(동결 설정 = 0), `PYTHONHASHSEED=0`(reproduce.sh가 설정). 평가 프로세스당 메모리 약 4.4GB, 정규화 워커는 약 15GB라 27GB RAM에서는 직렬 실행.

[Unreleased]: https://github.com/MENINBLOX/graph-ddxplus/compare/main...feat/review-dict-audit
