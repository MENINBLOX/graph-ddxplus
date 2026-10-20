# S-mix 출처 감사 판정 안내

작성 2026-10-20 · 대상 그래프 S-mix(v3llmd_ns, 설계 E 주 시스템) · 표본 생성 `pilot/scripts/smix_audit_sample.py`

## 목적

이전 출처 감사(`docs/research_design.md` §9.17, §9.25)는 구 Qwen3 v2.1 그래프에서만 했다. 주 시스템 S-mix는
추출 모델·문헌·개념 연결 방식이 모두 다르므로 그 결과(supports 29.5%)를 S-mix의 정밀도로 쓸 수 없다. 이 감사는
S-mix 간선의 근거 품질을 같은 세 범주로 직접 잰다.

## 파일

| 파일 | 내용 | 누가 보나 |
|---|---|---|
| `smix_audit_blind.csv` | 판정 시트 100행. 행마다 간선 하나(질환–개념)와 근거 문장 하나 | 판정자 |
| `smix_audit_key.csv` | 행 번호별 근거 언급의 출처 유형, 원래 LLM 구절, 간선의 지지 문헌 수·주제 문헌 수 | 분석자만, 판정이 끝난 뒤 |
| `smix_audit_instructions.md` | 이 문서 | 모두 |

판정자는 판정이 끝날 때까지 `smix_audit_key.csv`를 열지 않는다. 키에는 간선이 어떻게 만들어졌는지와 지지 강도가 들어 있다.

## 판정자

- 이전 감사와 같이 연구팀과 협력 병원 의료진이 판정한다. 결과를 이전 값과 나란히 보고하므로, 가능하면 이전 감사의 판정자가 맡는다.
- 판정자는 LLM 판정이나 다른 판정자의 판정을 보지 않고 혼자 판정한다. 판정자가 여럿이면 각자 시트 사본에 따로 판정한다.
- 이전 감사처럼 LLM 판정을 따로 만들어 일치도(Cohen's κ)를 잴 수 있다. 그 판정은 의료진 판정이 끝날 때까지 의료진에게 보이지 않는다.

## 세 범주 (이전 감사와 같다)

판정 대상은 행의 질환(`disease`)과 개념(`cui`, 연결된 문구는 `phrase`) 쌍이다. 판정 근거는 `sentence` 열의 근거 문장이다.
의학 지식으로 옳은지가 아니라, 그 문장이 그 질환에 대해 그것을 말하는지를 본다.

| 값 | 기준 |
|---|---|
| `supports` | 근거 문장이 그 질환의 소견·징후·검사 이상·합병증·위험인자·병력을 서술한다 |
| `not_supported` | 근거 문장이 그것을 서술하지 않는다 |
| `wrong_concept` | 서술하지만 추출된 개념이 달라졌거나, 소견이 아닌 개념(약물·검사·방법 등)으로 연결되었다 |

## 시트 작성법

1. `smix_audit_blind.csv`를 복사해 `smix_audit_user.csv`로 저장하고 그 사본에 적는다. 인코딩은 UTF-8 CSV를 유지한다.
   Excel에서는 "데이터 > 텍스트/CSV에서"로 열고 인코딩을 UTF-8(65001)로 지정해야 특수 문자(’, – 등)가 깨지지 않는다.
2. 마지막 열 `judgement (supports / not_supported / wrong_concept)`에 세 값 중 하나를 영문 소문자 그대로 적는다. 빈칸으로 두지 않는다.
   판단이 어려우면 가장 가까운 범주를 고른다.
3. 행을 지우거나 순서를 바꾸지 않고, 기존 열의 내용도 고치지 않는다. 메모가 필요하면 맨 오른쪽에 `note` 열을 더해 적는다. 메모는 집계에 쓰지 않는다.
4. 원문이 필요하면 `doc_id`로 찾는다.
   - `PMC…`: `https://pmc.ncbi.nlm.nih.gov/articles/PMC…/`
   - `TBLPMC…`: 같은 PMC 논문의 표다. `sentence`는 `|`로 구분된 표의 한 행이고 표 제목·열 이름은 시트에 없다.
   - `PMID…`: PubMed 초록이다. `https://pubmed.ncbi.nlm.nih.gov/<pmid>/`
   - `statpearls:article-…`: NCBI Bookshelf의 StatPearls 장이다. `title`로 찾는다.
5. `sentence`는 최대 600자다(이 표본에서 20행이 600자). 600자로 잘린 문장에 연결된 문구가 없으면 원문에서 그 문구 주변을 600자로
   다시 잘라 넣었다. 문장 앞뒤가 더 필요하면 원문을 연다.

## 표본

- 그래프: Mention 노드를 넣어 빌드한 S-mix(bolt://localhost:7689). 스크립트가 적재된 HAS_FINDING 간선 5,221,285개와 Disease 43,575개가
  동결 기록(`pilot/cache/v2/devD/frozen_config_E.json`의 S-mix)과 같은지 확인한 뒤 뽑는다. 근거 언급(Mention)은 11,582,169개다.
- 간선: 모든 HAS_FINDING 간선에서 균등 무작위로 100개를 뽑았다(시드 1). Cypher `rand()`가 아니라, (질환 ID, CUI) 순으로 정렬한
  간선 목록의 위치를 Python `random.sample`로 골랐다. 간선 단위 균등이므로 벤치마크 질환에 가중되지 않고, 채점에서 0.5배로 낮추는
  주제 밖 간선도 들어 있다. 이전 감사도 간선 단위 균등 표본이었다.
- 근거 언급: 간선마다 지지 언급 가운데 하나를 시드로 고정해 무작위로 골랐다.
- 출처 불변식: 100개 간선 모두 근거 언급이 1개 이상이었다. 각 간선의 근거 언급 수와 문헌 수는 간선 속성(n_mentions, n_docs)과 같았다.
- 재현: 아래 명령은 같은 빌드에서 같은 파일을 만든다(두 번 실행해 sha256 일치 확인).

```bash
.venv/bin/python pilot/scripts/smix_audit_sample.py --n 100 --seed 1 \
    --blind docs/audit/smix_audit_blind.csv --key docs/audit/smix_audit_key.csv
```

| 파일 | sha256 (판정 전 원본) |
|---|---|
| `smix_audit_blind.csv` | `2b6ab85f47b4d89b46a4d54a53a3d546a21e56d670760e28a54dc3cd57df4934` |
| `smix_audit_key.csv` | `3a7f5d9d9df2e0f6d6dae92c20e360dffc2be5ce4f4b7e4a0d03b346d754add9` |

## 보고 (분석자용, 판정이 끝난 뒤)

- 주 수치: supports 수/100과 비율, Wilson 95% 신뢰구간(점수 구간, z = 1.96). 세 범주 전체 분포도 함께 적는다.
  `smix_audit_sample.wilson()`이 같은 계산이며, 이전 감사의 59/200에 대해 23.6–36.2를 그대로 낸다.
- 비교: 이전 감사 값 supports 29.5%(59/200, Wilson 95% CI 23.6–36.2)를 옆에 적는다. **이전 감사는 다른 그래프였다** —
  구 Qwen3 v2.1 그래프(Qwen3-4B-Instruct-2507 추출, v2 문헌, 구절 표 연결). S-mix는 gemma-4-E4B 구절, v2+v3 문헌, 구절 표 +
  구절 안 UMLS 사전 재연결, 자기 간선 제거다. 두 값의 차이는 이 요소들 전체의 차이이고, 어느 요소 때문인지는 이 감사로 가를 수 없다.
- 정밀도의 폭: 표본 100개의 Wilson 구간은 넓다. 예를 들어 30/100이면 21.9–39.6이다(구간 계산 예시이며 판정 결과가 아니다).
- 보조(기술 통계, 키를 `row`로 결합): 아래 층별 supports 비율을 Wilson 구간과 함께 적는다. 층마다 표본이 작으므로 방향만 읽는다.
  - `mention_origin`: `llm_table`(구절 표가 연결한 LLM 구절) / `dict_relink`(구절 표가 연결하지 못한 LLM 구절 안에서 UMLS 사전 문자열이
    이름을 붙인 항목. `mention_span`이 원래 LLM 구절이고, 시트의 `phrase`는 사전 문자열이다)
  - `edge_origin`: 간선의 모든 근거 언급이 `llm_table`인지, `dict_relink`인지, 둘 다(`both`)인지
  - `n_docs` 1 대 2 이상, `n_topic_docs` 1 이상 대 0(§9.25와 같은 층)
  - `doc_id` 유형(PMC 본문, 표, PubMed 초록, StatPearls)
- 판정 전후로 이 100행을 보고 그래프 규칙을 고치지 않는다. 규칙을 고치면 별도 표본으로 다시 잰다(§9.25의 보류 세트 원칙).

## 이전 감사와 다른 점

| 항목 | 이전 감사(§9.17, §9.25) | 이번 감사 |
|---|---|---|
| 그래프 | Qwen3 v2.1 | S-mix(v3llmd_ns) Mention 빌드 |
| 표본 크기 | 200 | 100 |
| 근거 언급 선택 | 간선의 근거 언급 중 구절 알파벳 순 첫 번째 | 간선의 근거 언급 중 시드 고정 무작위 1개 |
| 시트 열 | `n_docs` 있음 | `n_docs` 없음(지지 강도 맹검), `row` 추가 |
| 근거 문서 구성(표본, `doc_id` 접두어 기준) | PMC 185, StatPearls 13, 기타 2 | PMC 본문 39(Doc.kind review 18, pmc 15, case_report 6), PMC 표 35, PubMed 초록 25, StatPearls 1 |

- 출처 유형 맹검은 열 수준까지다. 시트에는 출처 유형 열이 없지만, `dict_relink` 행의 `phrase`는 LLM 구절보다 짧은 사전 문자열이어서
  형태로 짐작될 수 있다.
