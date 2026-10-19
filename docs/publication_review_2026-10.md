# 게재 가능성·동향 검토 (2026-10-12)

> 조사 에이전트(WebSearch) 결과를 정리. 원문을 직접 확인한 것은 DeepRare·RareBench PDF뿐이며, 나머지 수치·venue 요건은 **인용 전 원문 재확인 필수**.

## 결론
- "진단 성능" 논문으로는 어렵다. **"문장 단위 provenance를 가진 open-set 질환–소견 지식 자원 + 그 평가 방법론"** 논문이면 게재 가능.
- 추론 시점 LLM 없음은 구식으로 보이지 않는다. 최근 흐름은 agentic 시스템이 지식/도구 계층을 결합하는 것(DeepRare, liteOdyssey, LLM+온톨로지 랭커 융합 arXiv 2609.02473).
- 가장 큰 약점은 절대 성능. 예: RAMEDIS Top-1 우리 3.5% vs PhenoBrain API 20.67%, Phenomizer 15.54%(큐레이션 주석 기반).

## Venue 등급
| Tier | Venue | 비고 |
|---|---|---|
| A | JAMIA Open (Application/Database Note) | 데이터·코드 DOI 공개 권장 |
| A | BMC Med Inform Decis Mak, Database (Oxford) | 세부 요건 미확인 |
| A- | AMIA 2027 Annual Symposium | CFP 미확인 |
| A- | BioNLP 2027 @ACL | 2026회 주제가 신뢰할 수 있는 평가 → validate 선택 실패 발견과 부합 |
| B | Journal of Biomedical Informatics | SoA 비교 + 의료진 평가 기대 |
| B | Scientific Data (Data Descriptor) | 공개 예치 필수; UMLS 문자열·GeneReviews·NC 라이선스 재배포 제약 확인 필요 |
| C | npj Digital Medicine, Bioinformatics, KDD ADS | 현재 수치로는 어려움 |

## 가까운 선행·경쟁 연구 (URL은 에이전트 보고 기준)
- MedKGent (arXiv 2508.12393): PubMed 초록 LLM 에이전트 KG 2.97M triple, 전문가 평가 타당도 ~90%, RAG 향상. 가장 직접적 비교 대상.
- iKraph (Nat Mach Intell 2025), PubTator3 (NAR 2024): 대규모 문헌 관계 KG.
- HEG-TKG "The Provenance Gap in Clinical AI" (arXiv 2604.17114): PMID 수준 인용, 소규모(36 시나리오). provenance 서사의 직접 경쟁자.
- Construction-time quality propagation (arXiv 2608.28360): triple별 근거 점수를 하류로 전파 + 맹검 임상의 평가. 우리 출처 가중과 개념이 겹침 → 인용·차별화 필요.
- GraphRareBench (arXiv 2607.24878), DeepRare (Nature 2026), MEDDxAgent (ACL 2025), PhenoBrain (npj DM 2025), SHEPHERD (npj DM 2025).
- MedRAG/MIRAGE (Findings ACL 2024): BM25·MedCPT 검색기 표준 참조. "When Retrieval Doesn't Help" (arXiv 2606.04127): RAG 이득 1~2점 보고(반론 근거).
- 세 요소(LLM 추출 문헌 KG + 문장 단위 provenance + open-set 다중 벤치마크 진단)를 모두 갖춘 연구는 찾지 못함 → 조합이 차별점.

## 리뷰어 예상 요구 (심각도 순)
| # | 요구 | 심각도 | 우리 상태 |
|---|---|---|---|
| 1 | 같은 랭커에 기존 KG(HPOA, PrimeKG, PubTator3/SemMedDB)를 넣은 비교 | 치명 | 없음 |
| 2 | 사람(임상의) provenance 감사, 2인 κ | 치명 | LLM 판정만(37.5%) |
| 3 | 벤치마크 누수 점검(MME·LIRICAL·Phenopacket 출판 증례 vs 코퍼스) | 높음 | Phenopacket만 차단 |
| 4 | 조건을 맞춘 비교(closed-set 재채점, Phenomizer 등 비-LLM 랭커) | 높음 | 참조표만 계획 |
| 5 | 통계(쌍대 검정, 다중비교 보정, 7/9 전이 검정) | 중간 | CI만 |
| 6 | RAG 효용 최소 증거 | 중간 | 없음(범위 밖으로 명시 예정) |
| 7 | 자원 공개·라이선스 | 중간 | 미확인 |

## 참조 수치 (조건이 달라 직접 비교 불가; @k는 같은 k끼리만)
| Benchmark | Method | Metric | % | Conditions |
|---|---|---|---|---|
| DDXPlus | AARLC | GTPA@1 | 99.21 | closed 49, supervised RL |
| DDXPlus | GPT-4o 0-shot → MEDDxAgent | @1 | 69 → 86 | 49-candidate list, 100 cases |
| RareBench (5 sets, 2,185) | GPT-4 0-shot | Top-1/3/10 | 32.3/45.4/58.9 | open generation |
| PUMCH-ADM | GPT-4 | Top-1/3/10 | 52.0/74.7/82.7 | open generation |
| PUMCH_L | PhenoBrain ensemble | Top-1/3/10 | 31.5/46.8/63.0 | 9,260 rare diseases, non-LLM, curated |
| PUMCH-ADM | PhenoBrain ensemble | Top-1/3/10 | 45.3/58.7/69.3 | same |
| MME/HMS/LIRICAL/RAMEDIS | PhenoBrain API | R@1 | 25.0/22.35/27.10/20.67 | HPO input |
| same | DeepRare (DS-V3) | R@1 | 70.0/56.97/55.95/72.60 | agentic; case bank |
| Phenopacket-store (5,213) | Exomiser | Top-1/3/10 | 35.5/46.3/58.5 | HPO only |
| RAMEDIS / Phenopacket | Phenomizer | R@1 | 15.54 / 12.17 | curated annotation |

## 프레이밍
- 주장: provenance 자원(소형 로컬 LLM, GPU 1장 재현), provenance = 품질 신호(13→63%), 출처 가중 전이 vs 점수 형태 비전이, 단일 validate의 IE 모델 선택 역전(방법론 경고), 같은 코퍼스 BM25/MedCPT 대비 검색 계층 비교.
- 회피: SOTA·진단 성능 중심 서술, 실험 없는 RAG 유용성 단정, "추적 가능 = 정확" 혼동.
