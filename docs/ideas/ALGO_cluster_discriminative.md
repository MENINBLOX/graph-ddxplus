# 질환-쌍 cluster discriminative scoring  🔴 regression

**상태**: 검증됨 negative/marginal (v68/v96/v98)

## 아이디어
@1 실패 대부분이 유사 질환군(cluster) 혼동(bronchitis vs bronchiolitis). 각 질환 D의 top 유사질환(confuser)을 동적 식별 → **D에 더 흔하고 confuser엔 드문 differential CUI**를 추가 가중.
- `diff(D,c,e)=max(0, prof[D][e]−prof[c][e])`, `score=score_v71 + γ·Σ diff(D,top_confuser,e)·pat_vec[e]`.
- 스크립트: `v98_cluster_aware.py`, `v96_discriminative_score.py`, v68 unique discriminator.

## 검증
- v68 unique discriminator: 56.4% (hurt). v98 per-disease blacklists: slight regression. v65 PMI / v66 specificity: 모두 regression.
- **일관되게 regression/marginal** — 의도적으로 차이를 키우면 noisy KG의 차이를 과신해 역효과.

## 결론
@1 cluster 변별을 직접 노린 정공법이나 전부 실패. 메타결론(KG content noise)과 동일 원인. IE content 신뢰도 선행 후 재시도 가치. ALGO_contrastive_qualifier(속성 차원)와 자매.
