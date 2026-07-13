# 데이터기반 속성 applicability 가중  🔴 (gain은 artifact)

**상태**: 검증됨 — @1 상승했으나 **외부검증 FAILED → artifact** (2026-06-01)

## 아이디어
속성마다 적용가능성이 다름(location은 통증엔 의미·발열엔 무의미). `applicability[증상][속성]=그 증상 edge 중 해당 속성분포 채워진 비율`(KG에서 도출, 하드코딩 X). 적용 안 되는 속성은 scoring에서 중립 처리.

## 근거
HPO clinical modifier(HP:0012823)·Phenopackets: modifier는 표현형마다 선택 적용. SOCRATES.

## 검증
- DDXPlus 5K, deep120 KG: none 33.35 → unif 34.96 → **appl 35.00** (alpha 0.7). appl≥unif 일관.
- **그러나 외부검증(임상 semiology gold 상관) r=−0.27 FAILED.** @1 gain은 "임상 applicability 반영"이 아니라 sparse 노이즈 채널을 죽인 부수효과.

## 결론
개념(HPO 근거)은 타당하나 현재 IE 구현으론 정당화 불가. **ie_attribute_fix.md 선행 필요.** 교정 후 r 회복 시 재검증.
스크립트: `pilot/scripts/v103_eval_applicability.py`
