# 벤치마크별 속성 활용 분석 (per-benchmark preprocessing, 3 서브에이전트) — 2026-07-19

사용자 지시: "속성을 적극 활용, 벤치마크마다 생김새 다르니 전처리 다르게." → 벤치마크별 attribute-aware
loader를 서브에이전트로 병렬 구축(`attr_loader_{ddxplus,rarebench,symcat}.py`) + **변별 신호 정량분석** +
discriminative scorer(`attr_score.py`: agree→bonus, controlled-vocab mismatch→penalty) 통합(`--attr`).

## 결론 (3벤치 + 실측 수렴): 공개 벤치마크에서 속성은 진단 lever가 아님. 무대는 NLICE.

verify-first로 각 벤치마크의 속성 인코딩·coverage·변별력을 실측. **세 벤치 모두 속성이 진단 gain을 못 냄**
— 이유는 벤치마크별로 다르나(아래), 공통은 **양측 sparsity + granularity/vocab 불일치**. CLAUDE.md의
"속성 검증은 SymCat/NLICE 세팅" 결정과 정합.

### DDXPlus — 속성 분리 가능하나 신호 미미
- 인코딩: evidence 코드에 carac(character)/endroitducorps·irrad(location)/intens(severity NRS)/soudain(onset).
- coverage: 속성은 pain/rash/edema에만 → location 11.9%·severity 11.1%·character 9.0%·**onset 5.8%**.
- 변별 분석(true-edge agree − distractor-edge agree, 양쪽 edge가 속성 보유한 pair): **onset +7.8%(유일 양성)**,
  severity −0.3%, location −0.9%(환자 89.9% NEITHER: coarse 다부위 broadcast vs KG fine anatomy), character 0 pair.
- 실측: onset-only mismatch penalty → DDXPlus @3 54.2→54.4(flat). **286 true-edge pair뿐이라 metric 불변.**
- 결론: onset만 약양성이나 slice 너무 작음. location/severity/character는 무효/해로움.

### RareBench — compound CUI라 join 0건 (negative)
- HPO term 이름에 속성 녹음: onset/timing 7.2%·severity 1.0%·location 4.3%. 그러나 RareBench는
  modifier+개념을 **단일 compound CUI**("Severe intellectual disability" 자체 CUI)로, 우리 KG는 base CUI +
  edge 속성으로 factor → **exact-CUI join 0건, 변별 0건**. "Neonatal death" 등 fused 토큰은 false-positive.
- 결론: 배선 불가(base-CUI re-linking 필요, 제약 위반). 미사용.

### SymCat — 구조화 속성 없음(이름에만), freq만 미사용
- 원본(474-symptom CSV, parse.py, parsed_full): name/description/common-cause/age·sex·race만. 속성은 symptom
  이름에 lexicalize("Sharp abdominal pain"), 분리 불가.
- 유일 미사용 신호 = **freq(P(symptom|disease))**: ceiling +1.4 @1(sampled). 현재 Bernoulli 포함 후 버림.
- ⚠️ **eval 유효성 이슈**: `load_symcat`이 읽는 `symcat_parsed.json`은 **50개 질환("A…" dev subset)뿐**
  (full=`symcat_parsed_full.json` 801). 즉 현 SymCat eval은 50질환 환자를 798후보에 랭크. full 801로 확장 검토.
- NLICE 겹침: 22 unique SymCat 질환(NLICE 55 CSV). NLICE=SymCat-grounded+속성.

## 함의 / 조치
- **속성 기여축은 NLICE에서 입증**(이미: location ablation p=0.02, permutation-null). 공개 벤치(RareBench/SymCat/
  DDXPlus)는 **속성 미검증 한계 사례**로 정직 보고 — 이유(granularity/sparsity)를 벤치별로 명시.
- 사용 가능한 minor lever: (1) SymCat freq 가중(+1.4 @1), (2) SymCat full-801 확장(eval 완전성).
- loader 3종은 audited 자산으로 보존(`attr_loader_*.py`). `--attr`는 backward-compatible로 유지.
