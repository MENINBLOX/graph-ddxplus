# KG: phenotype name → CUI 매핑  🔴 open problem

**상태**: 부분 (현재 ~51%), n-gram 확장은 @1 regression

## 문제
IE가 뽑은 phenotype name의 **51%만 UMLS CUI 매핑**(`v103_build_kg_cui.py` gen_candidates: qualifier strip + v92 multi-substring). 나머지(서술형 구 "swollen neck lymph nodes")는 exact match 실패 → KG에서 소실.

## 시도 결과
- **n-gram subphrase 매칭**: 매핑 47%→57%, 그러나 **DDXPlus @1 34→28 regression**. recall↑이지만 generic·low-IDF phenotype("lymph nodes") 추가로 top-1 희석. 원칙 #8(@1만) 상 revert.
- **scispaCy UMLS linker**: 일부만 정확("middle ear effusion"✓), 일부 부분/오매핑("localized conjunctival edema"→"localized"). noisy.

## 결론
매핑 recall↑가 @1엔 도움 안 됨(@1 병목은 매핑이 아니라 discrimination/content). 정밀 매핑(noise 없이)이 과제. lemma/synonym 정규화 재시도 여지.
