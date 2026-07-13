# IE 자료원 (Corpus) — 라이선스 · 다운로드 · 재현 절차

**작성 2026-06-02.** KG 구축의 IE 입력으로 쓰는 raw-text 자료원을 재현 가능하게 명시한다.
배경: PubMed abstract만으로 만든 strict KG의 DDXPlus @1이 33.73%에 정체. 케이스 심층분석(Inguinal hernia) 결과 **PubMed는 수술/연구 내용이라 환자 발현(clinical presentation)을 거의 담지 않음**을 확인(`docs/forensic_v103_*`, memory `project-v103-corpus-mismatch`). 따라서 임상 발현을 기술하는 자료원을 추가한다.

## 자료원 선정 기준
1. **학술적으로 인정** (인용 가능, 신뢰성)
2. **저작권 안전** (재배포·파생 KG 생성이 라이선스상 허용 또는 public domain)
3. **온라인 무료·기계적 접근 가능** (API/덤프)

## 채택 자료원

### 1. MedlinePlus — ★ 1순위 (저작권 청정 + 환자 표현)
- **제공**: U.S. National Library of Medicine (NLM), NIH.
- **라이선스**: **Public Domain** (U.S. 정부 저작물). 등록·라이선스 불필요. NLM 출처 표기는 권고(courtesy).
- **내용**: 질환/증상의 **환자용(lay) 설명** — DDXPlus 환자 자가보고 vocabulary와 정렬.
- **다운로드 방법** (둘 중 택1, 둘 다 무료):
  - **XML 전체 덤프**: <https://medlineplus.gov/xml.html> → `mplus_topics_*.xml` (전 health topic). 일괄 처리에 적합.
  - **Web Service(검색 API)**: `https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term=<disease>` (XML 반환, 키 불필요). 질환명으로 조회.
- **라이선스 근거**: MedlinePlus Web Service / Using Content 페이지(아래 출처).

### 2. Wikipedia — 2순위 (광범위 커버 + 발현)
- **라이선스**: **CC BY-SA 4.0** (저작자표시-동일조건변경허락). 파생물 허용(출처표기+동일 라이선스 유지 조건).
- **내용**: 질환 문서의 **"Signs and symptoms"** 섹션 = 임상 발현.
- **다운로드 방법**:
  - **MediaWiki REST/Action API**: `https://en.wikipedia.org/w/api.php?action=query&prop=extracts&titles=<disease>&format=json` (섹션 추출).
  - 또는 전체 덤프: <https://dumps.wikimedia.org/> (`enwiki-latest-pages-articles.xml.bz2`).
- 프로젝트 기존 IE 이력 있음(`medkg_ie_wikipedia.py`).

### 3. StatPearls — ⚠ 보조 (라이선스 제약 주의)
- **제공**: StatPearls Publishing, NCBI Bookshelf(NBK430685).
- **라이선스**: **CC BY-NC-ND 4.0** — NonCommercial(비영리 OK) + **NoDerivatives(파생물 제한)**.
  - **주의**: ND 조항상 텍스트를 변형해 KG로 재배포하는 것은 법적 회색지대. **연구용 내부 추출은 가능하나 파생 KG 공개 시 제약** → 메인 KG에는 미사용/선택적, 공개 산출물에서 제외 권장. 커버리지 비교 목적의 ablation에만 제한적 사용.
- **다운로드**: NCBI Bookshelf (질환별 NBK ID) / PMC.

### (기존) PubMed — long-tail/희귀질환 보조
- **접근**: NCBI E-utilities `esearch`→`efetch` (`medkg_pubmed_deepcrawl.py`, NCBI_API_KEY 권장). abstract만.
- **역할**: 발현 자료원이 없는 희귀질환의 long-tail 보강. 발현 자료원과 union.

## 채택 결론 (저작권-청정 메인)
**메인 IE corpus = MedlinePlus(public domain) ∪ Wikipedia(CC BY-SA)** + 부족분 PubMed.
StatPearls는 ND 제약으로 공개 KG에서 제외(내부 비교용만). 교수 회신 #3의 "외부 큐레이션 없이 raw text → 우리 IE" 원칙 부합(모두 raw text, 큐레이션 KG 아님).

## DDXPlus-49 커버리지 (2026-06-02 실측)
| 자료원 | 49 중 커버 |
|---|---|
| MedlinePlus (anchored ∪ ddx49) | 37 |
| Wikipedia | 21 |
| StatPearls | 28 |
| **MedlinePlus ∪ Wikipedia (clean)** | **39** |
| + PubMed fallback | 49 |

## 재현 절차 (이 저장소 기준)
1. 자료원 텍스트(청크): `pilot/data/pubmed_ddx_extra/{medlineplus_anchored, medlineplus_ddx49, wikipedia_anchored}.jsonl` — 각 줄 `{cui, pmid, text}`.
2. IE 입력 변환: CUI별 `{cui}.jsonl`(각 줄 `{"abstract": text}`)로 묶음 → `pilot/data/cache/presentation_ddx49/`.
3. IE: `v103_run_shard.py --pubmed_dir presentation_ddx49 --max_abstracts 120 --greedy` (gemma-4-E4B, temperature=0).
4. KG 빌드: `v103_build_kg_cui.py` (MRCONSO/MRSTY 매핑).
5. 평가: DDXPlus @1 (cosine+IDF, top_k=∞, β=0.75, κ=2).

## 라이선스 확인 출처
- MedlinePlus Web Service / 라이선스: <https://medlineplus.gov/about/developers/webservices/>, <https://medlineplus.gov/about/using/usingcontent/>, <https://medlineplus.gov/xml.html>
- StatPearls 라이선스(CC BY-NC-ND): NCBI Bookshelf <https://www.ncbi.nlm.nih.gov/books/NBK430685/>, NCBI 정책 <https://www.ncbi.nlm.nih.gov/home/about/policies/>
- Wikipedia: CC BY-SA 4.0 (Wikimedia Terms of Use), 덤프 <https://dumps.wikimedia.org/>
