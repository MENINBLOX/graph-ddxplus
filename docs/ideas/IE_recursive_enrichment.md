# 꼬리물기(recursive) IE — KG 연결 풍부화  🟢 (a)채택 / ⚪ (b)재시도 가능

**상태**: (a) profile 풍부화 채택 / (b) phen-phen edge 과거 negative지만 원칙 제약 삭제로 재시도 가능 (2026-06-01)

## 아이디어
IE 결과가 다시 다음 IE의 seed가 되어 "꼬리에 꼬리를 물고" KG 연결을 풍부하게 만든다. 기존엔 한 evidence에서 관계 하나만 뽑아 노드 연결이 빈약했음.

## 두 갈래
### (a) Recursive 크롤/IE 풍부화 — 🟢 채택
- disease/phenotype seed → 크롤 → IE → 새 phenotype을 다시 seed로 (BFS/depth-K).
- disease당 phenotype 수가 늘어 profile이 풍부 → **점수 상승의 실제 동력**(40%대→60%대는 anatomical/patient-focused IE로 disease당 증상 연결 ↑). 현재 deep-crawl 라인이 이 사상.

### (b) Phenotype↔phenotype 양방향 co-occurrence edge — ⚪ 재시도 가능 (과거 negative)
- 같은 disease/문서에 함께 나온 두 phenotype을 직접 연결.
- 과거 결과: **KG는 촘촘해졌으나 점수 효과 없음**(v18 bidirectional −0.52%p, scispaCy co-occurrence +0.33%p). 단 당시는 얕은 KG(20편)·구 알고리즘.
- **2026-06-01: CLAUDE.md 원칙 #12에서 "2-hop 미사용/phen-phen edge 없음" 제약 삭제됨** → 더 이상 원칙 위반 아님. **deep KG + 신규 알고리즘으로 재검증 여지** 생김 (단 메타결론상 content noise가 선결과제).

## 결론
"꼬리물기로 KG가 풍부해진 것은 사실"이나, **점수를 끌어올린 건 (a) profile 풍부화**이지 (b) phen-phen 연결이 아니었음(과거). (b)는 원칙 제약이 풀려 재시도 가능하나, IE content 교정(`IE_attribute_extraction_fix.md`) 후가 공정. 관련 보관: `docs/old/methodology_recursive_ie.md`, `methodology_exhaustive_recursive_ie.md`.
