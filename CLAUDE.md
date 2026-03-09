# vllm 사용 이유
- batch를 사용하기 위함
- 10개 이상의 vllm 작업은 batch로 진행 되도록 해야함

# 학술 논문 작성 가이드라인

## 1. 핵심 원칙

### 1.1 객관성 (Objectivity)
- 모든 주장에는 **근거(evidence)**가 필요하다
- 개인적 의견이나 감정이 아닌 **데이터와 사실**에 기반하여 서술한다
- 연구 결과를 있는 그대로 보고하고, 원하는 결론에 맞추어 해석하지 않는다

### 1.2 검증가능성 (Verifiability)
- 인용한 모든 정보의 **출처를 명확히** 밝힌다
- 다른 연구자가 재현할 수 있도록 **방법론을 상세히** 기술한다
- 데이터, 코드, 실험 조건을 공개하거나 명시한다

### 1.3 정확성 (Precision)
- 모호한 표현 대신 **정량적 표현**을 사용한다
- 용어는 일관되게 사용하고, 동의어 남용을 피한다
- 수치는 적절한 유효숫자와 단위를 포함한다

### 1.4 간결성 (Conciseness)
- 불필요한 수식어와 중복 표현을 제거한다
- 한 문장에 하나의 아이디어만 담는다
- 핵심 내용을 직접적으로 전달한다

---

## 2. 피해야 할 표현

### 2.1 주관적/감정적 표현

| 피해야 할 표현 | 대안 |
|--------------|------|
| I think / I believe | The results suggest / The data indicate |
| obviously / clearly | (삭제하거나) As shown in Figure 1 |
| interestingly / surprisingly | (삭제) - 독자가 판단할 문제 |
| very / really / extremely | 정량적 수치로 대체 |
| good / bad / best | effective / ineffective / optimal |

### 2.2 비격식적 표현

| 피해야 할 표현 | 대안 |
|--------------|------|
| a lot of | numerous / substantial / many |
| thing / stuff | factor / element / component |
| get | obtain / acquire / achieve |
| big / small | large / significant / minor / negligible |
| kind of / sort of | (삭제) |
| etc. | 구체적으로 나열하거나 "and others" |
| nowadays | currently / at present |

### 2.3 과장/단정적 표현

| 피해야 할 표현 | 대안 |
|--------------|------|
| prove / demonstrate (결과에 대해) | suggest / indicate / support |
| always / never | typically / rarely / in most cases |
| perfect / ideal | optimal / suitable |
| groundbreaking / revolutionary | novel / significant |
| beats / destroys (비교 시) | outperforms / exceeds |

### 2.4 구어체/비학술적 표현

| 피해야 할 표현 | 대안 |
|--------------|------|
| don't / can't / won't | do not / cannot / will not |
| a bunch of | several / multiple |
| figure out | determine / identify |
| come up with | develop / propose |
| look into | investigate / examine |
| this is no big deal | this has limited impact |

---

## 3. 권장 표현

### 3.1 Hedging (신중한 표현)

적절한 hedging은 학술적 글쓰기의 필수 요소이다. 연구의 한계를 인정하고 과도한 일반화를 피한다.

**동사 hedging:**
- may, might, could, can
- appear to, seem to, tend to
- suggest, indicate, imply

**부사 hedging:**
- possibly, probably, likely
- apparently, presumably
- generally, typically, often

**표현 hedging:**
- "It is possible that..."
- "The evidence suggests that..."
- "This may be attributed to..."
- "One possible explanation is..."

### 3.2 결과 보고

```
✓ "The accuracy improved from 75.4% to 86.0% (p < 0.01)."
✗ "The accuracy improved significantly."

✓ "Our method achieved 86.0% accuracy, compared to 75.4% for the baseline."
✗ "Our method is much better than the baseline."

✓ "These results suggest that the proposed approach is effective for..."
✗ "These results prove that our approach works."
```

### 3.3 비교 표현

```
✓ "Method A outperformed Method B by 10.6 percentage points."
✗ "Method A crushed Method B."

✓ "The proposed approach achieved comparable performance to..."
✗ "Our approach is as good as..."

✓ "The difference was statistically significant (t=3.45, p<0.001)."
✗ "The difference was obviously significant."
```

### 3.4 전환어 (Transition Words)

**대조:** however, nevertheless, in contrast, conversely, on the other hand

**결과:** therefore, thus, consequently, as a result, hence

**추가:** furthermore, moreover, additionally, in addition

**예시:** for example, for instance, specifically, in particular

**요약:** in summary, to summarize, in conclusion, overall

---

## 4. 인용 및 근거 제시

### 4.1 인용이 필요한 경우
- 다른 연구자의 아이디어, 이론, 방법론을 언급할 때
- 통계 수치나 사실적 정보를 제시할 때
- 특정 주장을 뒷받침할 때
- 기존 연구와 비교할 때

### 4.2 인용 방식

**직접 인용:** 원문 그대로 사용 (짧은 구절에만 권장)
```
Kim et al. (2023) stated that "knowledge graphs significantly improve diagnostic accuracy."
```

**간접 인용 (Paraphrase):** 자신의 언어로 재서술 (권장)
```
Previous research has demonstrated the effectiveness of knowledge graphs in medical diagnosis (Kim et al., 2023).
```

**요약:** 핵심 내용만 간략히
```
Several studies have shown improvements in diagnostic accuracy through knowledge graph integration (Kim et al., 2023; Lee et al., 2024).
```

### 4.3 인용 없이 주장하면 안 되는 것들
- "Previous studies have shown that..." → 반드시 인용 필요
- "It is well known that..." → 인용 또는 삭제
- "Research has demonstrated..." → 반드시 인용 필요
- 구체적인 수치나 통계 → 반드시 출처 명시

---

## 5. 논문 섹션별 가이드

### 5.1 Abstract
- 연구 목적, 방법, 주요 결과, 결론을 간결하게
- "In this study, we investigated..."로 시작하지 않기
- 구체적인 수치 포함 (정확도, 개선율 등)
- 참고문헌 인용 피하기

### 5.2 Introduction
- 연구 배경 및 동기 (인용 필수)
- 기존 연구의 한계점 (객관적으로)
- 연구 목적 및 contribution 명시
- 논문 구조 개요

### 5.3 Related Work
- 관련 연구를 체계적으로 분류
- 각 연구의 핵심 기여와 한계 객관적 서술
- 본 연구와의 차별점 명확히

### 5.4 Methodology
- 재현 가능하도록 상세히 기술
- 왜 이 방법을 선택했는지 근거 제시
- 수식, 알고리즘, 다이어그램 활용

### 5.5 Results
- 결과를 객관적으로 보고 (해석은 Discussion에서)
- 표와 그래프로 명확하게 제시
- 통계적 유의성 명시

### 5.6 Discussion
- 결과의 의미와 시사점 해석
- 기존 연구와 비교 분석
- 연구의 한계점 솔직히 인정
- 향후 연구 방향 제시

### 5.7 Conclusion
- 주요 발견 요약
- 연구의 기여도 강조
- 과장 없이 객관적으로 마무리

---

## 6. 자주 하는 실수

### 6.1 근거 없는 주장
```
✗ "This approach is the best solution for medical diagnosis."
✓ "This approach achieved the highest accuracy among the compared methods (Table 2)."
```

### 6.2 모호한 비교
```
✗ "Our method is significantly better."
✓ "Our method achieved 86.0% accuracy, a 10.6 percentage point improvement over the baseline (75.4%)."
```

### 6.3 과도한 자기 홍보
```
✗ "Our groundbreaking method revolutionizes the field."
✓ "The proposed method demonstrates improvements in accuracy and efficiency."
```

### 6.4 한계점 회피
```
✗ (한계점 언급 없음)
✓ "This study has several limitations. First, the evaluation was conducted on a single dataset..."
```

### 6.5 인용 누락
```
✗ "Knowledge graphs have been shown to improve accuracy."
✓ "Knowledge graphs have been shown to improve diagnostic accuracy (Smith et al., 2022; Kim et al., 2023)."
```

---

## 7. 체크리스트

논문 제출 전 확인사항:

### 객관성
- [ ] 모든 주장에 근거(인용 또는 데이터)가 있는가?
- [ ] 주관적 표현(I think, obviously 등)을 제거했는가?
- [ ] 과장된 표현을 수정했는가?

### 정확성
- [ ] 모호한 표현을 정량적 수치로 대체했는가?
- [ ] 용어를 일관되게 사용했는가?
- [ ] 통계적 유의성을 명시했는가?

### 형식
- [ ] 축약형(don't, can't)을 풀어 썼는가?
- [ ] 비격식적 표현을 학술적 표현으로 대체했는가?
- [ ] 적절한 hedging을 사용했는가?

### 인용
- [ ] 모든 인용의 출처를 명시했는가?
- [ ] 참고문헌 형식이 일관적인가?
- [ ] 자기 표절을 피했는가?

### 구조
- [ ] 각 섹션의 목적에 맞는 내용인가?
- [ ] 논리적 흐름이 자연스러운가?
- [ ] 한계점과 향후 연구를 명시했는가?

---

## 참고 자료

- [Academic Phrasebank - University of Manchester](https://www.phrasebank.manchester.ac.uk/)
- [Words to Avoid in Academic Writing](https://proofreading.org/blog/words-to-avoid-in-academic-writing/)
- [Scribbr - Words and Phrases to Avoid](https://www.scribbr.com/academic-writing/taboo-words/)
- [Hedging in Academic Writing](https://writingcenter.gmu.edu/writing-resources/research-based-writing/hedges-softening-claims-in-academic-writing)
- [Using Evidence - Indiana University](https://wts.indiana.edu/writing-guides/using-evidence.html)
- [Guidelines for Scientific Writing](https://conbio.org/images/content_groups/Africa/Guidelines_ScientificWriting.pdf)
- [서울대학교 온라인 글쓰기교실](https://owl.snu.ac.kr/2465/)
