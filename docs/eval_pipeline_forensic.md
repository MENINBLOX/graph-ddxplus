# Eval-pipeline forensic — low base is disease-CUI resolution, not IE (2026-07-17)

**사용자 지적**: "KG 성능은 벤치마크마다 성능평가 파이프라인을 점검해야 해. 실패 케이스 하나 까보면
왜 내가 이말을 하는지 알거야." → 팠다. 맞았다.

## 실패 케이스 (NLICE, `nlice_forensic.py`)

true = **unilateral mastoiditis(C0024904)**, patient={pain, rhinorrhea, fever}, true-rank=5/30,
승자=otitis externa. pain·fever는 true profile에 **있으나**(in_true_profile=True) 흔한 증상이라
변별 실패. 여기까진 "attribute 필요"의 예시. 그런데 후보 목록을 까보니 **진짜 결함**이 나왔다.

## 근본 원인 — disease→CUI 해석 붕괴

우리 KG는 **rare-disease-inclusive UMLS DISO seed**로 구축돼 질환을 **구체적 subtype CUI**로 인덱싱.
그런데 벤치마크(흔한 질환 umbrella)를 링크하면 **generic/umbrella CUI(profile=0)** 또는 **오링크**에 떨어진다:

| NLICE 질환 | 링크된 CUI (profile) | KG 실제 보유 (profile>0) |
|---|---|---|
| Migraine | C0149931 (**0**) | Hemiplegic migraine(17), Familial Hemiplegic(15)… |
| Appendicitis | C0003615 "symptoms of appendicitis"(**0**) | Acute appendicitis w/ abscess(9), Amebic(5), Chronic(3) |
| Carpal tunnel | C0007286 (**0**) | CARPAL TUNNEL SYNDROME 1(16), 2(8) |
| Rheumatoid arthritis | C0003873 (**0**) | Seropositive RA(6), Pauciarticular JRA(5) |
| Colon cancer | C0007102 (**0**) | HNPCC type2(10), type6(5) |
| Lower UTI | C3714514 "central venous catheter-related infection" (오링크) | Urinary tract infection C0042029(**21**) |
| Oesophageal ca | C0007097 "squamous cell papilloma or carcinoma"(**0**) | — |

**30 후보 중 8개가 profile=0 = 구조적으로 진단 불가** (그 질환 환자 전원 자동 오진). 여러 개는 완전 오링크.
→ base 6%는 IE가 뽑은 phenotype 문제가 **아니라** 평가 파이프라인이 벤치마크 질환을 엉뚱한/빈 CUI에
매핑한 결과.

## 정량 증명 (subtype-cluster 해석, 진단용 proxy)

각 NLICE 질환 → 이름-포함으로 KG subtype CUI 묶음(cluster), profile 합집합(max-IDF)으로 재평가:

| | profile>0 질환 | GTPA@1 | MRR | 후보 수 |
|---|---|---|---|---|
| broken (exact CUI) | 35/43 | 6.08% | 0.208 | 30 |
| **subtype-cluster** | **42/43** | **10.96%** | **0.225** | 43 |

후보가 30→43으로 늘어(**더 어려운데도**) base가 거의 2배. 결함이 정량 확인됨. (name-containment은
proxy — 최종은 MRREL RN/CHD 계층 확장 + non-DISO/‘symptoms of X’ 오링크 거부.)

## 조치 (task #21) — §3 준수

1. **disease 해석기**: 벤치마크 disease CUI → {self ∪ KG에 존재하는 descendant CUI(MRREL RN/CHD)},
   subtype profile 병합. 예측 cluster == true cluster면 hit. UMLS 계층은 벤치마크 데이터 아님 → strict
   zero-shot 위반 아님.
2. **링킹 수리**: DISO semantic type 우선, ‘symptoms of X’·비질환 개념 거부, profile 보유 CUI 선호.
3. RareBench·SymCat·NLICE **전 벤치마크에 적용** 후 재측정. 이어서 속성 ablation 재실행(#23).

## 함의

- diagnosis_baseline.md의 RareBench 9.19%/SymCat 1.52%도 **동일 결함(disease 해석) 재점검 필요** —
  IE/KG 품질 결론을 내리기 전 파이프라인부터 고친다 (verify-first).
- 속성 ablation의 location 유의성(p=0.02)은 relative delta라 유효하나, 교정 substrate에서 재확인(#23).
