# Excluded Sources (사용 불가 — 학술 정체성 보호)

본 연구의 정체성은 **"raw text → 우리 LLM IE → KG"** 입니다. 이미 큐레이션된 disease–phenotype KG 자원은 입력으로 사용하지 않습니다. 이는 우리 연구의 KG 구축 방법론을 명확히 보여주기 위한 결정입니다.

## 제외 — 통합 KG (반칙)

| 자원 | 종류 | 제외 이유 |
|------|------|--------|
| **PrimeKG** (Harvard, 20 sources 통합) | 다종 통합 KG | 17,080 disease × 4M relations 가 이미 구축됨 |
| **Hetionet** (29 sources) | 다종 통합 KG | 11 node type × 24 relation type 이미 구축 |
| **DisGeNET** | disease–gene–symptom KG | 큐레이션 + automatic 통합 KG |
| **SemMedDB** | PubMed 자동 추출 KG | 189K edges 이미 추출 (다른 도구가 추출) |

## 제외 — 큐레이션된 disease–phenotype 매핑

| 자원 | 종류 | 제외 이유 |
|------|------|--------|
| **HPO phenotype.hpoa** | OMIM/Orphanet/DECIPHER × HPO 매핑 | HPO Consortium의 전문가 큐레이션 매핑 — 이미 KG |
| **Orphanet en_product4.xml** | rare disease × HPO 매핑 + frequency | Orphanet 전문가 큐레이션 매핑 — 이미 KG |
| **DR.KNOWS UMLS subset** | UMLS 큐레이션 107 relations | 다른 팀의 큐레이션 결과 |

## 제외 — 통합 ontology 매핑

| 자원 | 종류 | 우리 사용 방식 |
|------|------|------------|
| **MONDO** | disease ontology (OMIM/Orphanet/MeSH/ICD 통합) | vocabulary 표준화에만 사용, KG content 아님 |
| **HPO ontology (hp.obo)** | phenotype ontology 계층 | term ID 정규화에만 사용, KG content 아님 |
| **UMLS MRCONSO** | concept 매핑 | CUI 정규화에만 사용 |
| **SNOMED CT** | clinical terminology | 진단 코드 정규화에만 사용 |
| **ICD-10** | diagnostic codes | benchmark mapping에 사용 |

→ vocabulary/ontology 자체는 *표준 식별자*로만 사용. 그 안의 disease–phenotype 관계 데이터는 사용 안 함.

## 회색지대 — 사용 가능 조건부

| 자원 | 조건 |
|------|------|
| **OMIM Clinical Synopsis** | 자유 텍스트 필드만 IE 입력으로. 구조화된 phenotype 코드 자체 흡수는 ✗ |
| **HPO 자체 description text** | term의 description 텍스트만 IE 입력으로. annotation 매핑은 ✗ |

## 일관성 원칙

- ✓ disease 이름 + raw text → 우리 LLM IE → disease–phenotype edge
- ✓ 표준 vocabulary (UMLS/SNOMED/HPO/ICD)로 식별자 정규화
- ✗ 다른 팀이 만든 disease–phenotype KG 그대로 흡수
- ✗ 큐레이션된 annotation 매핑 직접 사용

이 원칙 하에 우리 연구의 KG 구축 방법론이 *측정 가능한 contribution*이 됩니다. 외부 큐레이션 KG와 우리 KG 의 정확도 비교는 paper의 baseline 비교로 활용 가능 (입력으로는 사용하지 않으면서).
