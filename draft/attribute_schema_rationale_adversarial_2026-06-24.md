# 속성 스키마 결정 — 학술 명분 + 적대적 공격포인트 (2026-06-24)

대상 결정: **5속성 추출** = location/severity/onset/aggravating(판별 qualified-edge, 정규화) + **character(description, 비정규화·비스코어링, 단 distinct phenotype이면 finding 승격)**.
방법: 두 방향(명분/공격) 병렬 문헌 조사, 인용 기반. 목적 = SCIE 투고용 명분 확보 + 사전 방어.

---

## A. 명분 (Justification) — 인용 기반

1. **5속성 = HPI 프레임워크 교집합(자의 아님)**. OLDCARTS/OPQRST/SOCRATES/LOCATES 모두에 location·severity·onset·aggravating·character 등장(Osmosis, Geeky Medics, Skills TG, Columbia EM).
2. **속성이 진단을 바꾼다 — Panju, JAMA 1998;280:1256** (AAFP 2017 재인용): 양팔 방사 LR+ 7.1, 좌완 2.3 / pleuritic 0.2, sharp 0.3, positional 0.3. 증상 유무 아닌 **속성값**이 posterior를 좌우.
3. **NLICE (arXiv:2401.13756, 2024)**: Nature/Location/Intensity/Chronology/Excitation → SymCat Top-1 **58.8%→82.0%**(NB). Excitation≡aggravating, Nature(dry vs productive cough)=character→phenotype 승격의 선례.
4. **HPO Clinical Modifier(HP:0012823) + GA4GH Phenopacket(Jacobsen, Nat Biotechnol 2022)**: 속성을 phenotype의 **modifier/qualified statement**로 모델(노드 아님). → qualified-edge 설계가 국제표준 정합.
5. **SemEval-2015 Task 14(Elhadad et al.) 9-slot + n2c2**: 통제 target 있는 속성만 정규화(body location/severity/course), quality/character는 slot 없음 → **"target 있으면 정규화, 없으면 문자열 보존"이 주류 관행**.
6. **Semantic qualifiers(Bordage, Acad Med 1994)**: location/timing이 problem representation의 1차 변별 축.

---

## B. 공격포인트 (Adversarial) — 심각도순

### 🔴 FATAL 1 — character 강등이 인용한 Panju와 모순
Panju에서 **character(pleuritic LR 0.2, sharp 0.3, positional 0.3)가 가장 강한 rule-out 신호**. 우리가 "속성이 중요하다"의 근거로 든 바로 그 논문이, 우리가 강등한 character가 (정규화·스코어링하는) severity보다 **독립 LR이 강함**을 보임. 자기모순 → 리뷰어가 그대로 인용.

### 🔴 FATAL 2 — "character ≈ 0"이 단일·결함 벤치마크 산물
character leave-one-out≈0은 **DDXPlus 단독**. 그런데 (a) 팀 스스로 DDXPlus를 "속성 testbed 부적합"으로 선언, (b) character 커버리지 914/3000, base IE @1=12%(저열), (c) 그 IE가 후에 hallucination으로 판명. → "character 비변별" = **artifact 가능성**, 진실 아님. SymCat(NLICE Nature)·RareBench 재검 전엔 강등 주장 불성립.

### 🟠 MAJOR — 프레임워크 합의가 오히려 결정에 反함
3개 mnemonic 모두 만장일치인 건 **Radiation(R/R/R)과 Character(C/Q/C)** — 그런데 스키마는 **radiation 제외 + character 강등**. "합의로 정당화"하면서 합의의 최강 멤버 둘을 버림 = selective.

### 🟠 MAJOR — aggravating 승격이 character보다 근거 약함
aggravating은 **HPO slot 없음 + 공개 gold 없음(SemEval 9-slot에 없음) + 통제어휘 자의적 + DDXPlus 테스트 0케이스**. "gold 없으니 character 강등" 논리면 aggravating은 더 강등 대상. 비대칭.

### 🟠 MAJOR — 정규화 기준 비일관 (onset도 gold 없음)
character 강등 근거가 "공개 gold 부재"라면 **onset도 gold 없음**(SemEval엔 course만, onset-pace 없음) → 같은 기준이면 onset도 강등돼야. 한쪽만 적용.

### 🟠 MAJOR — "distinct phenotype" 예외가 재현 불가
"productive cough는 phenotype, sharp pain은 description" 경계 규칙 없음. "tearing chest pain"(대동맥박리), "burning epigastric"(GERD)은? 2nd 주석자 재현 불가 → IAA·provenance 주장 약화.

### 🟠 MAJOR — NLICE 인용 apples-to-oranges
58.8→82%는 **합성데이터(Synthea)+gold 속성 주입+지도학습**. 우리는 실텍스트+8B zero-shot IE(F1~0.5)+비지도 KG. → NLICE는 **완벽속성 상한(upper bound)**으로 인용해야지 achievable gain 아님.

### 🟠 MAJOR — radiation 제외가 최강 rule-in(LR 7.1) 버림
Panju 최강 rule-in이 radiation. 제외 근거 "DDXPlus 17질환"도 단일벤치 커버리지 artifact(FATAL2와 동일).

### 🟠 MAJOR — baseline 부재 + robustness 미입증
end-to-end LLM(GPT-4 ~55% Top-1; AMIE)·bare-CUI KG·location-only 대조 없음. 팀 @1 표는 오히려 full-6이 @1 regress(16.37→14.73) = 5속성에 反하는 증거. cross-benchmark robustness는 아직 약속어음.

---

## C. 종합 판단 (정직)

- **확실히 지지되는 부분**: aggravating **추가**(NLICE Excitation + 프레임워크 만장일치 + 교수님 흉통 예시). qualified-edge 설계(HPO/Phenopacket 정합). normalize-if-target 관행.
- **학술적으로 위험한 부분**: character **강등을 "비변별"로 정당화**하는 것 = Panju 모순 + DDXPlus artifact (FATAL 2개). radiation 제외 = 최강 LR 버림.

### 권장 재정의 (공격 무력화)
1. **character 강등 근거를 "비변별"이 아니라 "비정규화성"으로 한정**: HPO modifier slot·통제 target 부재(SemEval 선례)라서 **HPO 정규화는 안 하되, 스코어링에서 배제하지 말 것**(string/embedding fuzzy match로 Panju 신호 보존). → FATAL 1/2 무력화.
2. **radiation을 location의 하위로 흡수**(OPQRST가 Region/Radiation 묶음): "pain radiates to X" = location attribute → Panju LR 7.1 신호 회수, "radiation 제외" 공격 제거.
3. **gold-부재 기준을 대칭 적용**: onset·character·aggravating 모두 공개 gold 없음 → 셋 다 **faithfulness(질환텍스트 source-grounded, O2처럼)**로 검증, gold 부재를 강등 근거로 쓰지 말 것.
4. **NLICE를 upper-bound로 인용**(합성·gold·지도 명시) + 우리 IE noise 포함 결과 제시.
5. **결정적 실험**: character·radiation·aggravating을 **SymCat/RareBench에서 production IE로 ablation**(DDXPlus 아님). 4 vs 5 vs +radiation, attribute IE noise 포함. baseline=bare-CUI KG / location-only / 동일 8B end-to-end.

### 결론
"aggravating 추가"는 강함. "character를 비변별로 강등"은 **그대로 쓰면 reject 사유**. 안전한 명분 = **character=비정규화 descriptor(스코어링엔 fuzzy match로 잔존) + radiation=location 흡수 + cross-benchmark ablation으로 변별력 실증**. 단일 DDXPlus·Panju-모순 논리는 폐기.

**출처**: Panju JAMA 1998(280:1256, AAFP 2017 재인용); NLICE arXiv:2401.13756; HPO HP:0012823(Köhler NAR 2019); Phenopacket(Jacobsen Nat Biotechnol 2022;40:817); SemEval-2015 Task14(Elhadad); n2c2(Henry JAMIA 2020); Bordage Acad Med 1994; OLDCARTS/OPQRST/SOCRATES/LOCATES(Geeky Medics 등); end-to-end LLM DDx(Nature 2025 s41586-025-08869-4).
**미검증(투고 전 재확인)**: HPO ID 라벨(HP:0012824/0011009/0011010/0031245 정확 매칭), 한국 CPX 항목 원문, Bruyninckx 2008 LR 표, Panju 원문 CI.
