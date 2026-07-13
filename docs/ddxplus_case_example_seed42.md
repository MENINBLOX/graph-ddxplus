# DDXPlus 케이스 예시 — seed=42

DDXPlus test set에서 `random.seed(42)`로 추출한 환자 한 명의 raw 데이터와 의학적 해석.

생성 스크립트: `pilot/scripts/ddxplus_one_case_medical.py --seed 42`

---

## 1. 환자 기본 정보

| 필드 | 값 | 의미 |
|---|---|---|
| AGE | 62 | 만 62세 |
| SEX | F | 여성 |
| PATHOLOGY | Anaphylaxie | **정답 = Anaphylaxis (아나필락시스, 전신 알레르기 반응)** |
| INITIAL_EVIDENCE | `oedeme` | 환자가 제일 먼저 호소한 증상 = "몸 어딘가에 부종(붓기)이 있어요" |
| EVIDENCES 총 token 수 | 33 | 의사가 추가로 던진 후속 질문 답변 |
| EVIDENCES base 질문 수 | 23 | 33 token이 23개 기본 질문으로 grouping됨 |

DDXPlus 데이터 단위는 한 환자 1행. PATHOLOGY는 정답 단일 disease, DIFFERENTIAL_DIAGNOSIS는 가능 disease별 확률.

---

## 2. EVIDENCES 전체 — 의사 문진 흐름

각 항목: 질문 → 환자 답변 → UMLS CUI 매핑 결과.
질문 type: **B**=Binary Yes/No, **M**=Multi-choice (단일 선택), **C**=Multi-select (복수 선택, numeric scale 포함).

### 2.1 통증 평가 (douleurxx 계열)
| # | Token | Type | 질문 | 환자 답변 | UMLS CUI |
|---|---|---|---|---|---|
| 1 | `douleurxx` | B | 통증 있나요? | YES | C0030193 (Pain) |
| 2 | `douleurxx_carac` | M | 통증 특성 | a cramp / sharp | C0030193 |
| 3 | `douleurxx_endroitducorps` | M | 통증 위치 | L flank, R+L iliac fossa, belly, epigastric (복부 광범위) | C0030193 |
| 4 | `douleurxx_intens` | C | 통증 강도 | **4/10** | C0030193 |
| 5 | `douleurxx_irrad` | M | 통증 방사 | 없음 | C0234254, C1515974 |
| 6 | `douleurxx_precis` | C | 통증 위치 정확도 | **0/10** (확산성) | C0030193 |
| 7 | `douleurxx_soudain` | C | 통증 발현 속도 | **6/10** (비교적 빠름) | C0030193 |

→ **복부 전반 cramping/sharp 통증, 중간 강도(4/10), 확산성 위치, 비교적 빠른 발현**

### 2.2 호흡기 (dyspn / stridor / wheez)
| # | Token | Type | 질문 | 답 | UMLS CUI |
|---|---|---|---|---|---|
| 8 | `dyspn` | B | 호흡 곤란 있나요? | YES | C0013404, C0035203, C1299586 |
| 20 | `stridor` | B | 흡기 시 고음(stridor) 들리나요? | YES | C0035203 |
| 22 | `wheez` | B | 호기 시 천명(wheezing) 들리나요? | YES | C0043144 |

→ **Stridor + Wheezing 동시 = 상기도(인후두) + 하기도(기관지) 동시 협착. 아나필락시스의 가장 위험한 신호.**

### 2.3 피부 병변 (lesions_peau 계열)
| # | Token | Type | 질문 | 답 | UMLS CUI |
|---|---|---|---|---|---|
| 9 | `lesions_peau` | B | 피부 병변/발진 있나요? | YES | C0041834, C0221198 |
| 10 | `lesions_peau_couleur` | C | 발진 색? | pink | [] (CUI 매핑 없음) |
| 11 | `lesions_peau_desquame` | C | 병변 벗겨지나요? | N | C0221198, C0237849 |
| 12 | `lesions_peau_elevee` | C | 발진 부어오름? | **4/10** | C0015230 |
| 13 | `lesions_peau_endroitducorps` | M | 병변 위치 | back of neck, biceps R/L, mouth, thyroid cartilage | [] |
| 14 | `lesions_peau_intens` | C | 발진 통증 강도? | **3/10** | C0015230, C0030193 |
| 15 | `lesions_peau_plusqu1cm` | C | 병변 1cm 초과? | YES | C0221198 |
| 16 | `lesions_peau_prurit` | C | 가려움 정도? | **9/10** | C0033774 (Pruritus) |

→ **분홍색 다부위 두드러기, >1cm 큰 병변, 부어오름 중간, 극심한 가려움(9/10)** = 전형적 urticaria

### 2.4 소화기
| # | Token | Type | 질문 | 답 | UMLS CUI |
|---|---|---|---|---|---|
| 17 | `nausee` | B | 메스꺼움/구토감? | YES | C0027497, C0042963 |

### 2.5 부종 (Chief complaint)
| # | Token | Type | 질문 | 답 | UMLS CUI |
|---|---|---|---|---|---|
| 18 | `oedeme` | B | 어딘가 붓나요? | YES | C0013604 (Edema) |
| 19 | `oedeme_endroitducorps` | M | 부종 위치 | **양쪽 뺨 (joue D + joue G)** | C0013604 |

→ **얼굴 양측 뺨 부종 = Angioedema**, 아나필락시스 핵심 소견

### 2.6 History / Risk factor
| # | Token | Type | 질문 | 답 | UMLS CUI |
|---|---|---|---|---|---|
| 21 | `trav1` | C | 최근 4주 해외 여행? | N | [] |
| 23 | `z84.89` | B | 일반인보다 알레르기 잘 발생하나요? | YES | C0020517 |

→ **아토피 소인 (atopic predisposition)**, 알레르기성 질환 위험인자

---

## 3. DIFFERENTIAL_DIAGNOSIS — 의사의 감별진단 분포

DDXPlus는 정답 단일 disease (PATHOLOGY) 외에, 의사가 고려한 가능 disease들의 확률 분포 (DIFFERENTIAL_DIAGNOSIS)를 함께 제공.

```
 9.1%  Anaphylaxie               ⭐ ← 정답 (top-1)
 8.0%  Possible NSTEMI / STEMI       (심근경색 의증)
 7.6%  OAP/Surcharge pulmonaire      (급성 폐부종)
 7.4%  Angine instable               (불안정 협심증)
 6.8%  Syndrome de Boerhaave         (식도 파열)
 6.6%  Scombroïde                    (생선 알레르기성 식중독)
 6.5%  Hernie inguinale              (사타구니 탈장)
 6.3%  RGO                           (역류성 식도염)
 5.9%  Laryngospasme                 (후두 경련)
 5.6%  Angine stable                 (안정 협심증)
 5.3%  Embolie pulmonaire            (폐색전증)
 5.1%  Syndrome de Guillain-Barré    (길랑-바레 증후군)
 4.8%  Fibrillation auriculaire/Flutter (심방세동/조동)
 3.7%  Attaque de panique            (공황 발작)
 3.6%  Réaction dystonique aïgue     (급성 근긴장증)
 3.5%  Péricardite                   (심낭염)
 2.7%  Chagas                        (샤가스병)
 1.6%  Lupus érythémateux disséminé  (전신 홍반 루푸스)
```

- 18개 후보 (49개 disease 중 18개에 0이 아닌 확률)
- 정답 Anaphylaxis가 top-1이지만 9.1%로 압도적이지 않음
- 의사도 NSTEMI, 폐부종, 불안정 협심증 같은 응급 disease를 일정 확률로 고려 (호흡 곤란 + 부종 → 심부전 가능성)

---

## 4. 의학적 종합 해석

**환자 시나리오**:
> 62세 여성이 응급실에 양쪽 얼굴이 부어 있어 내원. 다부위 두드러기 (목 뒤, 양 이두근, 입, 갑상연골 부위), 극심한 가려움 (9/10), 복부 cramping, 메스꺼움, 호흡 곤란을 호소. 흡기 시 stridor와 호기 시 wheezing이 들림. 평소 알레르기 잘 생기는 체질.

**임상 진단 (Anaphylaxis)** — NIAID/FAAN 기준 충족:
1. **급성 발현 + 피부/점막 침범** (urticaria, angioedema) ✓
2. **호흡 침범** (stridor + wheezing — 상기도+하기도) ✓
3. **추가 시스템** (소화기 nausea/cramps) ✓
4. **알레르기 병력 (atopic)** ✓

→ 최소 2개 시스템 동시 침범 + atopic 소인 = Anaphylaxis. 임상적으로 **epinephrine IM 즉시 + airway 관리** 적응증.

**Differential 다른 후보가 등장한 이유**:
- 호흡 곤란 + 흉부 증상 같음 → NSTEMI/STEMI, 폐부종, 협심증 등 cardiac 응급 disease가 의사 differential에 포함
- 복부 통증 (cramping) → GI 응급 (식도 파열, 탈장, GERD)도 포함
- 의사가 18 disease 모두 일정 확률로 고려한 것 = 임상 현실의 불확실성 반영

---

## 5. 데이터 형식 정리

DDXPlus CSV 한 행의 구조:

```
AGE: integer
SEX: 'M' or 'F'
PATHOLOGY: 정답 disease (French)
INITIAL_EVIDENCE: 환자가 제일 먼저 호소한 evidence ID
EVIDENCES: ["token1", "token2_@_value", ...]  ← 33개 token in 이 케이스
DIFFERENTIAL_DIAGNOSIS: [["disease_fr", probability], ...]  ← 18 entries
```

EVIDENCES token 형식:
- `xxx` (binary): "환자가 xxx에 YES 응답" (B-type)
- `xxx_@_value` (categorical/multi-choice): "xxx 질문에 value 답변" (M/C-type)

CUI 매핑 (`ddxplus_evidence_value_cuis.json`):
- 각 evidence ID → 기본 CUI 리스트 (`_question` 키)
- 일부 value는 추가 CUI 보유 (해부학적 위치, 통증 특성 등)
- Numeric scale 답변 (0-10)은 별도 CUI 매핑 없음

이 환자의 KG matching 결과: 24 unique CUI 중 14개가 Anaphylaxis profile에 존재 → cosine similarity 점수 충분히 높아 v95_full KG가 top-1로 정답 예측 성공한 사례.

---

## 부록: DIFFERENTIAL_DIAGNOSIS 메트릭에 대한 주의

DDXPlus 원논문 (Tchango et al. 2022)이 제안한 DDR/DDP/DDF1 metric은 정답 differential set과 모델 예측 set의 set-recall/precision/F1. 우리 연구에서는 폐기 (CLAUDE.md 원칙 8):
- DDR은 원문 ground truth set 위로 못 올라가는 구조적 한계
- 2024-2026 LLM era 논문 대부분 GTPA@1 (top-1 accuracy)로 수렴 (Field consensus)
- 우리 SOTA = **GTPA@1 only**
