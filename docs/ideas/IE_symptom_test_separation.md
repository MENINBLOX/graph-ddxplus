# IE에서 증상(symptom) vs 징후(sign) 분리 — binary IE-time 태깅  🟡 검증중

**상태**: 2026-06-02 binary IE-time 태깅(`v103d_categorized_ie.py`) 검증 진행. 과거 SOAP/classified는 부분탐색.

## 2026-06-02 핵심 발견 (왜 사후필터가 아니라 IE-time 태깅이어야 하나)
- **사후 UMLS-TUI 필터는 실패**: KG 프로필을 증상 semantic type(T184/T033/...)만 남기면 DDXPlus @1 **33.73→24.88 폭락**. 이유: "노이즈"로 본 T047 Disease(764개)가 사실 **@1 anchor** (DDXPlus 병력 evidence·질환 self-name 고-IDF 변별). TUI는 너무 뭉툭해 *병력-disease(살려야)* 와 *exam-sign/수술합병증(빼야)* 를 못 가림.
- **→ IE 시점 binary 태깅이 정답**: LLM이 각 finding을 `symptom`(환자 self-report 주관 증상) vs `sign`(진찰소견+검사+병력 = 객관/임상의 필요)로 분류. TUI로 불가능한 "병력 살리고 exam-sign만 제거"가 가능.
- 사용자 지시: **바이너리** (증상 / 징후[검사·병력 포함]).
- DDXPlus = 증상 self-report → 증상-태그로 매칭. dual-service: 증상-only vs 증상+징후.
- ⚠ 단 이번 세션 content 레버가 전부 @1 정체(@10만 개선)였으므로, 태깅도 @1 결과는 미지수. 결과 기록 예정.

## 아이디어
IE 시점에 각 phenotype을 **Subjective(환자 자가보고 일반증상: 통증·발열·기침)** 와 **Objective(진찰·검사 소견: 청진음·영상·혈액검사 수치)** 로 태깅. 임상 문서(SOAP)의 S/O 구분과 동형.

## 왜 가치 있나
1. **서비스 분리**: "증상만 입력" 버전(S만) vs "검사 포함" 버전(S+O)으로 KG/평가를 나눠 제공 가능 (사용자 비전).
2. **차등 가중**: Objective 소견은 보통 더 specific·신뢰도 높음 → 변별에 다르게 가중.
3. **평가 분해**: 증상-only 진단 능력과 검사-포함 진단 능력을 분리 보고(임상 현실성).

## 근거
SOAP note 표준(Subjective/Objective/Assessment/Plan). DDXPlus evidence도 자가보고 증상 + 병력 + 진찰 답변이 혼재.

## 과거 시도 (혼재 결과)
- SOAP categorized IE (`docs/old/methodology_soap_categorization.md`), Classified IE(4-boolean) → PR-filtered channel로 SOTA 49.53% 기여한 적 있음.
- patient-reportable filter(v102) 가설검증.
→ 분류 자체는 가능. **서비스 분리 + Objective 차등가중 + 평가분해**가 미완 각도.

## 2026-06-02 binary 검증 결과 (negative for @1)
`v103d_categorized_ie.py`(symptom/sign binary) 발현corpus 49-loop:
- 증상태그 only @1 **15.16** / 징후태그 only @1 24.66 / 증상+징후∪PubMed @1 **34.29**(기준 33.73 +0.56%p, noise수준) @10 80.99.
- **증상-only가 최악**: DDXPlus 변별 evidence(hernia "groin bulge"·위치)가 LLM에서 **"징후"로 분류**됨(만져지는 덩어리=관찰소견). → **임상 증상/징후 경계 ≠ DDXPlus 환자보고 경계.** 태깅 자체가 DDXPlus 평가축과 어긋남.
- 추출 품질도 저하(verbose 문장조각이 phen명 → 증상태그 CUI매핑 64.9%, 징후 39.5%).
- 결론: dual-service 인프라로는 가치 있으나 **@1 개선 아님**. 세션 ~9개 content 레버 전부 @10↑/@1 정체 패턴과 일관.

## 2026-06-02 2-stage CoT 분류 (v103e) — 분류 완성, @1은 여전
사용자 지시로 2-stage CoT IE 구현: Stage1=깨끗한 추출(v103c 재사용), Stage2=`v103e_stage2_classify.py`(CoT reasoning→category, guided_json, few-shot=benchmark무관 generic 임상개념, pydantic에 `class_reasoning`/`required_test`/`test_reason` 저장).
- 분류 품질 ↑: "groin=증상"(위치는 환자가 가리킴) 정확, bulge/tenderness=징후+필요검사(palpation). 증상-only @1 15.16(v103d)→**21.95**(v103e). CUI매핑 72.4%.
- **그래도 @1 벽**: 증상+징후∪PubMed @1 30.62 < 기준 33.73.
- **결론: 증상/징후 IE-time 분류는 고품질로 완성**(dual-service·사유·필요검사 인프라 확보, 사용자 필수요구 충족). 단 @1 개선 레버는 아님 → @1은 매칭층(위치 vocab 정렬) 문제.

## 할 것
IE 스키마에 `evidence_class: subjective|objective` 추가(source-grounded). S-only / S+O 두 평가모드. Objective 가중 sweep. ie_attribute_fix와 함께 진행.
