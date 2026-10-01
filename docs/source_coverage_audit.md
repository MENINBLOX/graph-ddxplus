# Source Coverage Audit (CLAUDE.md §1) — 2026-07-16

**질문(#3)**: 38,906 seed disease 각각에 대해 *증상을 담은 source* 가 존재하는가?
CLAUDE.md §1 은 KG 구축 전 source coverage 실측을 요구한다 (Disease 100% + Evidence 100%).

**핵심 교훈(verify-first)**: 파일 *존재* ≠ 내용 존재. pubmed 은 seed 의 ~100% 에 대해
CUI-키 파일이 있으나 **88.7% 가 0-byte 스텁**(다운로드 안 됨). 실제 내용 coverage 는 훨씬 낮다.

---

## 1. 실측 (로컬 데이터, 2026-07-16)

| source | 형식 | seed coverage | rare-subset coverage | 비고 |
|--------|------|---------------|----------------------|------|
| pubmed (non-empty abstract) | CUI-keyed text | **12.0%** (4,660) | 10.5% | 파일은 100% 존재, 88.7% 0-byte 스텁 |
| phenotype.hpoa (OMIM+ORPHA→HPO) | 구조화 phenotype | **27.9%** (10,862) | **48.7%** | 증상 보장(HPO term + frequency + onset) |
| medlineplus | name-keyed text | 2.5% (982) | — | 파일 수 자체가 적음(988) |
| wikipedia | name-keyed text | 1.2% (476) | — | 454 파일 |
| statpearls | name-keyed text | 1.1% (415) | — | 596 파일 |
| genereviews | name-keyed text | 0.1% (41) | — | 48 파일 |
| **UNION (전체)** | | **38.0%** (14,788) | **55.3%** | |
| **GAP (source 없음)** | | **62.0%** (24,118) | 44.7% | |

- rare-subset = ORPHANET/OMIM/rarebench 태그 22,322 diseases (seed 의 57.4%).
- name-keyed text 는 MRCONSO 영문 synonym 으로 CUI 매칭(seed-name 동일 매칭 아님).

## 2. GAP 의 성격 = "source 부재" 아니라 "미다운로드"

pubmed 파일 39,181 중:
- **0-byte 스텁: 34,750 (88.7%)** — CUI별 파일만 생성, esearch/efetch 미실행.
- lines-but-no-abstract: 129 (0.3%).
- has abstract: 4,302 (11.0%).

crawl 스크립트(`medkg_pubmed_crawl.py`)는 **abstract 를 찾았을 때만 파일을 쓴다**
(빈 파일 안 씀) → 34,750 스텁은 이 crawl 이 만든 게 아니라 별도 stub 생성 단계 산물.
= **진짜로 크롤한 적 없음.** NCBI_API_KEY(`.env`) 존재 → 9 req/s 재크롤 가능.

> 즉 GAP 62% 의 대부분은 "PubMed 에 자료가 없다"가 아니라 "아직 안 받았다"이다.
> 이것이 fixable 인지 = **pilot 재크롤 200건으로 실측**(§3).

## 3. Pilot 재크롤 (200 empty CUIs, rare 100 + common 100) — 결과

| 그룹 | n | abstract 획득 | symptom-rich (cue≥4) |
|------|---|---------------|----------------------|
| ALL | 200 | 85% | 64% |
| rare | 100 | **87%** | 65% |
| common | 100 | 83% | 64% |

→ 결론: **GAP 은 fixable.** 재크롤 시 rare 조차 87% 가 abstract 를 얻고 65% 가 증상-서술.
"자료 부재"가 아니라 "미다운로드" 였음이 실증됨. → 전건 재크롤 진행.

## 3b. 전건 재크롤 결과 (34,193 empty CUIs, 8-worker 동시, NCBI key)

- ok=26,875 / empty=7,318 / err=0 (3.5 CUI/s, ~2.7h). non-empty pubmed **4,700 → 31,625**.
- 이후 **synonym-retry**(gap CUI 를 MRCONSO 영문 동의어로 재질의) 로 잔여 gap 추가 회수.

### 최종 source coverage (post-crawl)

| source | seed | rare-subset | KG 사용가능? |
|--------|------|-------------|--------------|
| pubmed non-empty | **90.7%** | 92.3% | ✅ (free-text IE) |
| processed/sections (statpearls/wiki/medlineplus) | ~2.5% | — | ✅ (free-text IE) |
| phenotype.hpoa | 27.9% | 48.7% | ❌ **참고용만** |
| **KG-eligible (pubmed∪sections)** | **~91%** | ~92% | ✅ |
| (참고) hpoa 포함 union | 92.5% | 95.4% | — |

> ⚠️ **strict zero-shot 제약(CLAUDE.md §3)**: `phenotype.hpoa`(OMIM/ORPHA→HPO)는 **disease-symptom
> 매핑 데이터**라 KG 구축 source 로 **사용 금지**(§3 "disease-symptom 매핑 데이터 사용 금지").
> hpoa 는 (a) rare-disease phenotype 이 존재함을 **입증**하는 coverage 참고, (b) 잔여 gap 성격 분석에만
> 쓴다. **실제 KG 는 pubmed+sections 자유텍스트 disease-name IE 로만 구축**(v156→v157, hpoa 미사용 — 준수 확인).
> 따라서 KG-eligible source coverage = **pubmed 90.7%**(+sections). Phrank/Phenomizer 등은 hpoa 를
> KB 로 쓰지만 그건 zero-shot 이 아니라 **knowledge-based** 세팅 — 우리 제약과 다름.

- **38% → 90.7%**(KG-eligible, rare 92.3%). CLAUDE.md §1 목표 근접, **strict zero-shot 준수**.
- **잔여 GAP 13.8% 의 성격**: 대부분 ICD 잔여·행정 버킷("Other specified …", "… NOS",
  "Effects of …") = 전용 문헌이 없는 코드. 소수는 희귀 직업병 등(Cannabinosis, Cheese-washers'
  lung). → synonym-retry 로 진성 항목 회수, 잔여는 **free-text/no-source 노드로 보존 + 한계 명시**.

## 4. 정책 결정 (실측 후 확정)

- **재크롤 yield 높으면**: empty 34,750 전건 재크롤 → union coverage 대폭 상승.
- **rare-subset 은 hpoa 로 하한 보장**: 텍스트가 희소해도 phenotype.hpoa 가 rare 48.7% 에
  HPO 증상 + onset(속성 1종!) 을 구조적으로 제공. 단 hpoa 는 **bare phenotype** 이라
  우리 IE 의 free-text 속성(location/character/severity/timing/aggr/reliev)은 못 준다 →
  속성 추출은 **텍스트 source 가 있는 disease 로 상한**이 걸린다.
- **논문 프레이밍**: source coverage 는 (a) disease-existence, (b) 속성-bearing text
  두 층으로 보고. 속성 ablation 은 텍스트-coverage disease 에서, KG phenotype-coverage 는
  hpoa 로 rare 까지 확장.

## 5. 재현성(#9) 메모 — LLM seed 만이 변수가 아니다

IE 디코딩은 `do_sample=False`(greedy/argmax) → **샘플링 seed 자체가 없다**(temperature=0).
따라서 "seed 고정"은 해당 없음. 실제 재현성 변수:
1. **GPU 커널 비결정성** (CUDA matmul/attention) → `torch.use_deterministic_algorithms(True)` +
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` 로 고정 가능.
2. **batch/padding 구성** — left-pad 배치 시 부동소수 차이가 near-tie argmax 를 뒤집을 수 있음
   → batch size 고정으로 재현.
3. **버전 pin**: transformers/torch 버전, 모델 revision, dtype(bf16), 하드웨어(RTX4090).
→ 논문 재현성 절: "greedy decode + 고정 batch + pinned versions + deterministic kernels",
  seed 아님.
