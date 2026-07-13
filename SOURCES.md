# Raw Text Medical Sources (KG 구축 입력)

본 연구는 **raw text → LLM IE → KG** 의 일관된 방법론을 따릅니다. 외부에서 이미 큐레이션된 disease–phenotype KG (PrimeKG, Hetionet, SemMedDB, Orphanet en_product4, HPO phenotype.hpoa 등) 는 입력으로 사용하지 않습니다 (→ `EXCLUDED_SOURCES.md` 참조).

## Tier 1 — 핵심 자원 (현재 사용 중 또는 즉시 사용 가능)

| # | 자원 | 라이선스 | 규모 | 접근 방식 | 강점 |
|---|------|--------|-----|---------|------|
| 1 | PubMed 초록 | NLM 무료 | ~36M abstracts | NCBI E-utilities API | 연구 문헌, 전 질환 cover |
| 2 | PMC Open Access full-text | NLM 무료 | ~5M articles | NCBI FTP bulk | abstract 보다 임상 디테일 |
| 3 | StatPearls (NCBI Bookshelf) | CC BY-NC-ND 4.0 | ~10K 질환 chapter | E-utilities + HTML | peer-reviewed textbook, History and Physical 섹션 |
| 4 | GeneReviews (NCBI Bookshelf) | NLM 무료 | ~800 유전질환 | E-utilities + HTML | rare disease textbook 깊이 |
| 5 | MedlinePlus A.D.A.M. Medical Encyclopedia | NLM 무료 | ~4,000 topic | NLM Web Service API | 환자 친화어 lexicon |
| 6 | Wikipedia 의학 article | CC BY-SA 4.0 | 수천 disease + Infobox | Wikipedia API | 광범위 + Infobox structured symptoms |

## Tier 2 — 커버리지·깊이 추가 자원

| # | 자원 | 라이선스 | 규모 | 강점 |
|---|------|--------|-----|------|
| 7 | MSD/Merck Manuals (Consumer + Professional) | 무료 web (ToS 주의) | 수천 질환 | comprehensive clinical textbook |
| 8 | WikiDoc | CC BY-SA | 수천 의학 article | 의사 중심 작성, 임상 디테일 |
| 9 | GARD (Genetic and Rare Diseases Information Center) | NIH public | ~6,000 rare | 환자용 rare disease 텍스트 |
| 10 | NORD Rare Disease Database | 무료 | ~1,300 rare | rare disease 환자 자료 |
| 11 | NICE clinical guidelines (UK) | 무료 | 수백 가이드라인 | evidence-based 진료 지침 |
| 12 | CDC clinical guidelines | US gov public domain | 수백 가이드라인 | 감염병·예방·여행의학 |
| 13 | WHO disease fact sheets + guidelines | 무료 | 수백 질환 | global health, 응급의학 |
| 14 | AAFP clinical guidelines | 무료 | 수백 | 1차 진료 |
| 15 | MSF Clinical Guidelines | 무료 | 응급·자원 부족 환경 | ER 정합 |

## Tier 3 — 특정 도메인 (선택적)

| # | 자원 | 강점 |
|---|------|------|
| 16 | DermNet NZ | 피부과 |
| 17 | LITFL (Life in the Fast Lane) | 응급의학 free wiki |
| 18 | Cochrane Plain Language Summaries (open access 일부) | systematic review 요약 |
| 19 | AAP (American Academy of Pediatrics) | 소아과 |
| 20 | PharmGKB clinical text | 약리유전체 |

## 자원 분류 기준

- **포함 가능 (raw text)**: 자유 텍스트 형태의 의학 문헌·교과서·환자 자료
- **포함 불가 (curated KG)**: 이미 disease–phenotype 매핑이 큐레이션된 구조화 자원 → `EXCLUDED_SOURCES.md`
- **회색지대 (구조화 텍스트)**: OMIM Clinical Synopsis 등 — *텍스트 필드만 IE 입력으로 사용*하면 raw text 처리와 동등

## 데이터 디렉토리 구조

```
data/medkg/
├── pubmed/        # PubMed abstracts (cache)
├── pmc/           # PMC OA full-text (선택)
├── statpearls/    # NCBI Bookshelf chapters
├── genereviews/   # NCBI Bookshelf chapters
├── medlineplus/   # NLM Web Service responses
├── wikipedia/     # Wikipedia API responses
├── msd_manual/    # Tier 2 (예정)
├── wikidoc/       # Tier 2 (예정)
├── gard/          # Tier 2 (예정)
├── nord/          # Tier 2 (예정)
├── nice/          # Tier 2 (예정)
├── who/           # Tier 2 (예정)
├── cdc/           # Tier 2 (예정)
├── raw/manifest.jsonl    # 다운로드 manifest
├── processed/     # section extraction + IE 결과
└── kg/            # 최종 merged KG
```
