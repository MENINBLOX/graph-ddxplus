# 서버 이전 Handoff (2026-07-13)

Graph-DDXPlus 작업을 다른 서버에서 이어가기 위한 인수인계 문서.
GitHub `MENINBLOX/graph-ddxplus` `main` 최신(PR #1 merge)이 코드/문서 최신 상태.

---

## 1. 현재 연구 위치 (한눈에)

- **속성 스키마 확정**: Bates 6속성(location/character/severity/timing/aggravating/relieving). 교수님(방창석) 검토 완료. → `docs/attribute_rationale_draft_ko.md`
- **검증 방향 확정 (교수님 자문 v260712)**: gold 새로 만들지 않음. **location/severity = 공개 정답 F1**(추출 능력만), 나머지 4속성 = "공개 정답 없음" 논문 명시 + **진단 성능(extrinsic)으로만**. **faithfulness·gold 프로토콜은 논문 미사용(내부 fallback)**.
- **다음 작업 = IE 방법론 정의**: 미결 3항 — (A)source·스코프(rare-disease 포함 대규모), (B)모델(**오픈 패널 비교** gemma/Qwen/Llama), (C)추출 전략(one-turn 원자화 P2 / 속성별 / CoT / 그룹). 선정 지표 = **location/severity 공개 F1 + 구조지표(fill·원자성·character 청결)**.
- **그 다음**: IE 실행 → 속성 KG 구축 → 교수님 요청 **leave-one-out ablation**(SymCat/NLICE 세팅, DDXPlus는 속성이 코드에 녹아 부적합).
- 교수님 다음 자문 = **7월 말**(extrinsic 결과 기반). 스레드 = `email/`.

---

## 2. Git 미추적 데이터 (수동 이전 필요) ⚠️

`.gitignore`가 `data/`(=`pilot/data/` 포함)·`email/`·`CLAUDE.md`·`.venv`·`*.pdf`·`*.log`·`.claude/` 제외.

| 경로 | 크기 | 내용 | 필수도 |
|---|---|---|---|
| `pilot/data/cache/` | 8.3G | IE 산출·KG 캐시·source 텍스트 | **핵심** |
| ├ `pubmed_deep/` | 7.3G | PubMed deep crawl | 재생성 가능(느림) |
| ├ `verify_ie_12b/`, `ie_p2/`, `ie_scaleup/` | ~수십M | 12B IE 산출(P2 원자화) | **핵심** |
| ├ `scaleup_sources/`, `v105_sources/` | ~수십M | benchmark-blind 질환 텍스트 | **핵심** |
| ├ `maccrobat/` | 20M | location/severity F1 평가 gold | **핵심** |
| `data/` | 1.9G | 벤치마크(DDXPlus/SymCat/RareBench/external) | **핵심** |
| `/windows/data/medkg/` | 6.7G | KG 빌드·정규화 edges·source | **핵심** |
| ├ `kg/` 3.3G, `processed/` 2.5G, `pubmed/` 762M, `statpearls/` 65M, `orphanet/` 47M, `seeds/` 34M | | | |
| `email/` | 작음 | 교수님 자문 스레드(로컬 보관) | **핵심** |
| `CLAUDE.md` | 작음 | **프로젝트 지침(gitignore됨!)** | **핵심** |

> **주의**: `CLAUDE.md`는 gitignore라 GitHub에 없음. 반드시 수동 복사(프로젝트 원칙·제약이 여기 담김).
> Claude Code 메모리(`~/.claude/projects/-home-max-Graph-DDXPlus/memory/`)도 이전하면 컨텍스트 보존에 유리(선택).

### 이전 명령 예시
```bash
# 새 서버에서 (경로는 상황에 맞게)
rsync -a pilot/data/cache/  NEWHOST:/path/Graph-DDXPlus/pilot/data/cache/
rsync -a data/             NEWHOST:/path/Graph-DDXPlus/data/
rsync -a /windows/data/medkg/ NEWHOST:/windows/data/medkg/
rsync -a email/ CLAUDE.md  NEWHOST:/path/Graph-DDXPlus/
# pubmed_deep(7.3G)는 급하지 않으면 나중에
```

---

## 3. 실행 환경 (venv 2개, 재생성 필요)

Python 3.12.11. requirements 파일 없음 → 아래 기준으로 재구성.

| venv | 용도 | 핵심 패키지 |
|---|---|---|
| `/home/max/vllm-gemma4-nightly/.venv` | **LLM IE** (vLLM batch) | vllm 0.23.1rc1, transformers 5.12.x, torch(CUDA) |
| `Graph-DDXPlus/.venv` | **정규화·평가** | spacy+scispacy(`en_core_sci_lg`+UMLS linker), scikit-learn, transformers 5.10.x(NLI), networkx |

- **모델**: HF 캐시 `models--google--gemma-4-12B-it-qat-w4a16-ct`(주력), `gemma-4-E4B-it`(frozen v106). 새 서버에서 재다운로드 또는 캐시 복사.
- **GPU**: RTX 4090 ×3 (GPU 0/1/2). vLLM batch·3-shard 병렬로 사용.
- **주의(기록)**: vllm venv(transformers 5.12)는 roberta 토크나이저 로드 실패 → NLI는 `.venv`(5.10)에서. vLLM 좀비 EngineCore는 PID 직접 kill.

---

## 4. 새 서버 첫 스텝 체크리스트

1. `git clone git@github.com:MENINBLOX/graph-ddxplus.git` → `main`
2. §2 데이터 rsync (특히 `CLAUDE.md`·`data/`·`pilot/data/cache`·`medkg`·`email/`)
3. venv 2개 재생성(§3) + gemma-12B-QAT HF 캐시
4. 스모크: `.venv/bin/python -c "import spacy; spacy.load('en_core_sci_lg')"` / vllm venv로 12B 로드
5. 이어서 **IE 방법론 정의** — 추출전략(C)×오픈모델(B) 비교 실험 (지표=location/severity 공개 F1 + 구조). `pilot/scripts/ie_prompt_improve.py`가 P2 프롬프트 출발점.

---

## 5. 핵심 문서 맵

| 주제 | 파일 |
|---|---|
| 속성 스키마·Method 발췌 | `docs/attribute_rationale_draft_ko.md` |
| IE 실측(12B·840·정규화율) | `docs/ie_verification_results.md` |
| IE 데모(10 케이스) | `docs/ie_demo_10cases.md` |
| faithfulness(내부 fallback) | `docs/ie_faithfulness_p2.md` |
| gold 프로토콜(shelved) | `docs/gold_protocol_ko.md` |
| 논문 구조·검증 결정 | `docs/paper_draft.md`, `docs/paper1_necessity_debate.md`, `docs/ie_quality_gold_debate.md` |
| IE 방법론·frozen | `docs/frozen_ie_spec.md`, `docs/methodology_ie_evaluation.md` |
| 벤치마크·source 커버리지 | `BENCHMARK_COVERAGE.md`, `SOURCES.md`, `EXCLUDED_SOURCES.md` |
| 교수님 자문 스레드 | `email/README.md` + `email/email_v2607*.md` (git 미추적) |
