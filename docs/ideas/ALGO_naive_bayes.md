# KG frequency 기반 Bernoulli Naive Bayes  🔴 negative

**상태**: 검증됨 negative (2026-06-01)

## 아이디어
cosine 대신 정식 Bayesian: `score(D)=Σ_E∈vocab [x_E·logP(E|D) + (1-x_E)·log(1-P(E|D))]`, P(E|D)=frequency_in_abstracts. likelihood ratio가 자연히 변별, 부재 증거도 원리적 처리. HPO frequency가 P(E|D) 정당화.

## 검증
deep120 KG, DDXPlus 5K: base 33.33 → nb 27.20(eps=0.01, best), eps↑ 더 하락. 원인: noisy P(E|D)(~120 abstract 추정) + absent-term이 noise 증폭. NB는 calibrated 확률 가정인데 IE frequency가 안 맞음.

## 결론
v71 self-aware negative(감쇠된 부정)는 작동했으나, raw NB(unattenuated)는 noise 증폭. KG content 신뢰도 선행. 과거 v54 KG-NB도 한계.
스크립트: `pilot/scripts/v103_eval_algoideas.py --idea nb`
