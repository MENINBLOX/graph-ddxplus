### 4.5 경계: 출판 증례 희귀 세트, 순환과 미달 판정

이 절은 문헌 그래프가 비교 대상보다 높지 않았던 결과를 모은다. 출판 증례 세트, 순환, 시간 분할, 결합(M1)의 결과는 개발 세트의 결과이거나 탐색 결과다. 그중 설계 D 봉인 세트 Phenopacket-new의 비교군 값만 [봉인 기술]이다. 실행 전에 커밋한 기준으로 판정한 결과는 표 6의 판정을 그대로 옮기고, 봉인 세트의 판정 기준이 없던 PPR 부가 가설의 값은 [봉인 기술]로 적는다. 문헌 그래프와 비교 대상의 차이는 "S-mix(또는 해당 문헌 그래프) − 비교 대상"으로 쓴다. 문헌 그래프 대 비교 대상이 아닌 비교(결합, 랭커, 그래프 판본, 비교 지식원끼리)는 문장에 뺄셈 방향을 적는다(예: 결합 − S-mix, PPR − 1-홉). 이 절에서 출판 증례 세트는 LIRICAL, MME, Phenopacket-store를 뜻한다. M1의 공개 희귀질환 증례층은 여기에 RAMEDIS와 HMS를 더한 다섯 세트다(3.10절).

**출판 증례 희귀 세트.** 출판 증례 세트 세 개에서는 같은 1-홉 랭커의 HPOA가 S-mix보다 크게 높았다 [개발]. S-mix − HPOA의 Top-10은 LIRICAL −36.22%p [−42.16, −30.27], MME −40.00%p [−55.00, −25.00], Phenopacket-store −42.66%p [−43.74, −41.55]였다(표 11a). 세 세트 모두 질환 군집 CI와 질환 등가중 CI도 0보다 낮았다. RAMEDIS(−2.72%p [−7.05, +1.44])와 HMS(0.00%p [−11.36, +11.36])에서는 환자 가중 차이가 0과 구별되지 않았다. 그러나 RAMEDIS의 질환 등가중 차이는 −18.97%p [−30.46, −6.75]로 0보다 낮았다. RAMEDIS는 환자 624명 가운데 148명(23.7%)의 정답이 한 질환이어서, 두 가중의 값이 크게 다를 수 있는 세트다. 설계 D의 봉인 세트 Phenopacket-new에서도 S-D의 Top-10은 HPOA보다 26.8%p 낮았다(9.2% 대 36.0%) [봉인 기술]. 이 세트의 정답 질환 가운데 S-D에 소견 간선이 하나라도 있는 질환은 59.1%였다(정답 정확 일치 기준). LOPO는 개발 세트에만 적용했으므로, 이 세트의 격차에 순환이 얼마나 기여하는지는 재지 않았다. 개발 9개 세트 매크로로는 S-mix − HPOA가 −6.33%p [−8.90, −3.69]였다 [개발].

**출판 증례 순환과 LOPO.** 출판 증례 세트에서는 평가 증례의 원논문이 HPOA 주석의 근거이기도 하다. Jiang 등은 Phenopacket-store 0.1.27에서 정답 질환이 HPOA에 있는 증례 10,348건의 74.6%가 그 증례 논문을 인용한 정답 주석을 가진다고 보고했다 [ref: Jiang et al. 2026, arXiv 2609.02473]. 이 순환이 격차를 얼마나 설명하는지 보려고 LOPO(leave-one-publication-out, 3.9절)를 적용했다. Phenopacket-store 증례의 원논문 PMID 1,745개 가운데 하나라도 인용한 HPOA 행을 모두 지우고, 그 그래프 하나로 모든 개발 세트를 다시 채점했다. 빌드가 읽는 264,245행 가운데 22,261행이 지워졌고, 간선은 393,498개에서 363,488개로 줄었다. 질환 410개는 프로필이 모두 사라졌다. 이 삭제로 Phenopacket-store 환자의 77.9%에서 정답 질환의 간선이 줄었고, 36.0%에서는 정답 질환의 프로필이 모두 사라졌다. HPOA의 채점 설정은 삭제 뒤에 다시 고르지 않았다.

LOPO 뒤 Phenopacket-store에서 HPOA의 Top-10은 73.69%에서 25.45%로 내려갔다 [개발]. S-mix − HPOA는 −42.66%p에서 +5.58%p [+4.75, +6.42]로 부호가 바뀌었고, 군집 CI는 [+1.23, +9.92]였다(표 11a). 질환 등가중으로는 −55.73%p [−58.89, −52.32]에서 +3.28%p [+0.76, +5.79]로 바뀌었다. 같은 수의 행을 무작위로 지운 대조에서 HPOA의 Top-10은 72.22%였고(시드 1·2에서 72.00%, 72.47%), S-mix − HPOA는 −41.19%p [−42.30, −40.08]로 거의 그대로였다. 따라서 격차의 감소는 지운 행의 수로 설명되지 않는다. 다만 지운 주석은 정답 질환에 몰려 있었고, 이 질환들에는 다른 출처의 주석이 거의 없었다. 같은 질환에서 같은 수의 행을 무작위로 지우려 한 대조는 22,261행 가운데 2,631행만 지울 수 있었다. 원논문 PMID를 10겹으로 나눠 각 증례를 자기 겹의 논문만 지운 그래프로 채점해도, HPOA의 Top-10은 25.60%로 전역 삭제(25.45%)와 거의 같았다. 정답 질환 간선의 변화로 나누면, 정답 질환의 HPOA 프로필이 모두 사라진 3,712명(36.0%, 정답 질환 393개)에서 HPOA의 Top-10은 81.68%에서 0.19%로 내려갔고, S-mix는 8.46%였다(S-mix − HPOA-LOPO +8.27%p [+7.38, +9.16], 군집 CI [+5.15, +11.89]) [개발]. 정답 질환의 간선이 일부만 줄어든 4,318명에서는 S-mix − HPOA가 −40.78%p에서 +5.51%p [+3.94, +7.09](군집 CI [−3.46, +14.71])로 바뀌었다 [개발].

LOPO가 완전한 세트는 Phenopacket-store 하나다. LIRICAL은 저장소에 PMID가 없어, Phenopacket-store와 겹치는 논문의 주석만 지워졌다(부분 LOPO). 이 세트에서 S-mix − HPOA는 −36.22%p에서 −4.32%p [−9.73, +1.35]가 되어 0과 구별되지 않았다. MME 정답 질환의 주석은 하나도 지워지지 않았고, S-mix − HPOA는 −40.00%p 그대로였다. PMID가 없는 다른 세트는 다른 세트의 논문을 지운 부수 효과만 받았다(정답 질환의 간선이 줄어든 환자: RAMEDIS 10.1%, PUMCH_L 10.3%, PUMCH-ADM 5.3%, HMS 4.5%). 그래서 희귀질환 7개 세트의 매크로 값은 LOPO 추정치가 아니므로 보고하지 않는다. 전역 삭제는 한 증례에서 보면 다른 증례 논문의 주석까지 지운다. 그러나 경쟁 질환의 주석도 함께 얇아지므로, 증례별 LOPO와의 차이 방향은 보장되지 않는다. 10겹 판이 전역 삭제와 거의 같았으므로, 추가로 지운 주석의 영향은 작았다.

**해석 규칙의 적용.** 이 측정의 해석 규칙은 LOPO를 실행하기 전에 커밋했다(3.10절, 커밋 f676539). 규칙의 기준은 Phenopacket-store의 질환 등가중 차이다. LOPO 뒤 HPOA의 우세가 10%p 이하로 줄고(S-mix − HPOA ≥ −10%p) 무작위 삭제보다 크게 줄었으면, 순환을 걷어내면 격차가 작아졌다고 탐색 수준에서 판단한다. HPOA의 우세가 30%p 이상 남으면 순환을 걷어내도 격차가 컸다고 판단한다. LOPO 뒤 질환 등가중 S-mix − HPOA는 +3.28%p였으므로, HPOA의 우세는 남지 않았다. 무작위 삭제 대조는 환자 가중으로만 계산했다. 그 값(S-mix − HPOA −41.19%p)은 LOPO 값(+5.58%p)과 크게 달랐다. 첫 조건은 규칙의 지표로 충족했고, 둘째 조건은 질환 등가중 무작위 대조가 없어 환자 가중으로 대신 확인했다. 이 범위에서, Phenopacket-store에서 순환을 걷어내면 같은 1-홉 랭커의 HPOA와의 격차가 작아졌다고 판단한다 [탐색]. 다만 지운 주석이 정답 질환에 몰려 있었으므로(환자 36.0%에서 정답 질환 프로필 소멸, 같은 질환 대조는 2,631행만 삭제 가능), 이 대조로는 순환의 효과와 정답 질환 프로필의 소멸(커버리지)을 가르지 못했다. 이 결과는 문헌 그래프가 출판 증례 희귀 세트 전반에서 HPOA와 같은 수준이라는 뜻이 아니다. LOPO는 한 세트에서만 완전하고, MME와 RAMEDIS의 격차는 이 측정으로 검사되지 않았다. 질환 등가중 값은 M1 실행(3.10절)에서 같은 LOPO 그래프로 계산했다. 질환 키는 군집 CI와 같은 정답 질환 키(3.8절)다.

**시간 경계.** 2022년 말까지 출판된 문헌만으로 다시 집계한 그래프를 HPOA 2022-12-15판과 같은 1-홉 랭커로 비교했다(3.9절, 표 11b) [탐색]. 대상은 Phenopacket-store 증례 10,603건 가운데 원논문(9편)이 v2 코퍼스에 들어 있는 32건을 뺀 10,571건이다. 이 증례에는 설계 D 확증에 쓴 Phenopacket-new 증례가 들어 있으므로 확증 결과가 아니다. 문헌 그래프는 γ = 0과 γ = 1에서 모두 평가했다. 본문과 표 11b는 γ = 0의 값을 보고하고, γ = 1의 값은 보충 표에 둔다. 두 값 모두 S-mix와 S-dict가 개발 세트에서 고른 γ(0.5)의 값이 아니다. 2023년 이후 출판된 증례(미래 증례)의 원논문을 인용한 주석은 HPOA 2022-12판에 있을 수 없다. 2022년 말에 이미 HPOA에 등재된 질환의 미래 증례(1,106건, 67개 질환)에서 2022 절단 S-mix 그래프 − HPOA 2022-12의 Top-10은 −7.1%p [−10.1, −3.9]였다. 같은 묶음에서 2022 절단 S-dict 그래프는 −8.1%p [−11.1, −5.0], 2022 절단 S-D는 −17.4%p [−20.3, −14.2]였다. γ = 1에서도 이 묶음의 차이는 비슷했다(보충 표; S-mix −7.0%p [−9.9, −3.8], S-dict −8.3%p [−11.3, −5.2], S-D −18.4%p [−21.3, −15.4]). 이 묶음의 CI는 환자 단위로만 계산했고, 증례는 67개 질환에 몰려 있다. 2022년 말 HPOA에 아직 등재되지 않은 질환의 미래 증례(2,381건, 263개 질환)에서는 2022 절단 문헌 그래프 11개의 Top-10이 두 γ 모두에서 2.2% 이하였고, HPOA 2022-12는 0.1%였다. 어느 지식원도 미래·신규 묶음, 곧 2022년 말 HPOA에 아직 주석되지 않은 질환의 미래 증례를 거의 맞히지 못했다. 다만 이 묶음에는 2022년 말까지 문헌이 없는 질환이 들어 있을 수 있다. 문헌 유무로 층화하지 않았으므로, 문헌 그래프의 낮은 값이 문헌의 부재에서 오는지는 가르지 못한다 [탐색].

**결합의 보완 근거는 얻지 못했다.** M1은 HPOA와 S-mix를 순위 결합(RRF)과 간선 합집합의 두 방식으로 합쳤다(3.7절, 3.10절) [개발]. RRF 가중치 w는 개발 9개 세트에서 D3 규칙으로 골랐고, w = .75가 선택되었다(매크로 Top-10: w = .25 51.43%, .5 48.84%, .75 57.03%). 이 선택은 증례 원논문을 인용한 HPOA 주석이 출판 증례 세트에서 낸 이득에 좌우되었다. HPOA-LOPO 목록으로 같은 규칙을 적용하면 w = .25가 선택되었다(매크로 Top-10: w = .25 50.41%, .5 45.84%, .75 48.63%; 규칙 밖의 관찰) [개발]. 이 HPOA-LOPO는 위의 전역 삭제판이므로, 이 이득을 순환의 효과와 정답 질환 프로필의 커버리지로 가르지 못한다. w를 고른 세트에서 평가했으므로 선택 낙관이 들어 있다. 측정 전에 커밋한 규칙은 비출판 병원층(PUMCH_L, PUMCH-ADM) 매크로에서 결합 − HPOA와 결합 − S-mix의 CI 하한이 둘 다 0보다 클 때만 결합이 보완한다고 쓰게 한다(3.10절). 규칙에 판정 k가 적혀 있지 않아 D3의 주 지표인 Top-10을 썼다. 측정 스크립트는 환자 층화, 질환 군집, 질환 등가중의 세 CI가 모두 조건을 만족해야 통과로 했다. k와 세 CI의 요구는 스크립트에서 정했고, 측정 전 커밋으로는 확인되지 않는다. 병원층에서 결합 − S-mix는 RRF −4.07%p [−10.08, +1.83], 간선 합집합 +0.14%p [−2.26, +2.40]로 둘 다 환자 CI가 0을 포함했다. 결합 − HPOA는 RRF +8.19%p [+5.48, +11.36](군집 [+3.55, +13.10], 질환 등가중 +5.69%p [+2.21, +9.80]), 간선 합집합 +12.41%p [+6.98, +17.95](군집 [+2.70, +21.40], 질환 등가중 +5.52%p [−2.39, +13.77])였다. RRF는 세 CI 모두에서 HPOA보다 높았고, 간선 합집합은 환자 가중 값(환자·군집 CI)에서만 높았다. 참조 비교로, 같은 병원층 매크로의 S-mix − HPOA는 질환 등가중으로 +4.01%p [−3.89, +12.28]였고 0과 구별되지 않았다. 두 결합 모두 규칙을 충족하지 못했다. 결합 − S-mix의 하한을 −1%p까지 허용해도 충족하지 못했다. 선택되지 않은 w에서도 병원층 결합 − S-mix의 CI는 0을 포함하거나 0보다 낮았다(w = .25 +0.29%p [−3.40, +3.88], w = .5 −5.59%p [−10.65, −0.05]). 따라서 결합이 큐레이션 주석을 보완한다는 근거는 얻지 못했다. Top-1에서는 간선 합집합이 병원층 두 비교 모두에서 세 CI 하한이 0보다 컸다 [개발]. 결합 − S-mix는 +3.82%p [+1.05, +6.89](군집 [+0.16, +9.04], 질환 등가중 +5.02%p [+1.12, +10.15])였다. 결합 − HPOA는 +14.39%p [+8.91, +19.79](군집 [+6.86, +21.62], 질환 등가중 +12.11%p [+3.87, +20.34])였다. Top-1은 판정 지표가 아니므로 이 결과를 보완의 근거로 쓰지 않는다.

흔한 질환층(DDXPlus, SymCat)에서 간선 합집합 − S-mix는 +0.17%p [−0.06, +0.38]로 0과 구별되지 않았고, RRF − S-mix는 −14.64%p [−15.58, −13.70]였다 [개발]. 공개 희귀질환 증례층(RAMEDIS, LIRICAL, HMS, MME, Phenopacket-store) 매크로 Top-10에서 RRF − HPOA는 +2.81%p [+0.78, +4.95]였다. 간선 합집합 − HPOA는 −6.08%p [−9.28, −3.04](군집 [−10.10, −1.84], 질환 등가중 −6.92%p [−10.92, −3.13])로, 간선 합집합이 HPOA보다 낮았다. RRF가 HPOA보다 높았던 것은 선택된 w = .75에서이며, w = .25에서는 −14.84%p [−18.68, −11.29]였다. HPOA-LOPO판에서 결합 − HPOA-LOPO는 RRF +4.30%p [+2.27, +6.41], 간선 합집합 +2.96%p [−0.23, +5.96]였다. 규칙상 이 층의 결과는 병원층이 규칙을 충족할 때만 주장을 넓히는 데 쓰므로, 여기서는 수치만 보고한다. Top-1을 포함한 M1 전체 결과는 보충 표에 둔다.

**실행 전에 커밋한 판정의 미달 결과.** 표 6의 미달 판정 가운데 설계 D의 주 목표 T1은 4.2절에서, E-T2의 S-dict·S-union 비교는 4.4절과 4.6절에서 보고한다. 나머지 미달 판정을 표 6에서 그대로 옮긴다. 설계 D의 T2(DDXPlus Top-10 ≥ 40%, Top-1 ≥ 18%)는 미달이었다. Top-10/Top-1은 S-D 35.2/12.4%, 그래프+BM25 결합 37.8/14.9%였다 [확증]. 설계 E의 E-T4(S-mix 단독으로 같은 기준)도 미달이었다. DDXPlus-E에서 S-mix의 Top-10은 41.07% [39.3, 43.0]로 점추정만 기준을 넘었다. Top-1은 12.43% [11.3, 13.6]로 기준보다 5.57%p 낮았다 [확증]. 같은 세트에서 S-dict(17.63%)와 S-union(17.93%)의 Top-1도 18%보다 낮았다(E-T4 기준을 사후에 적용). 두 값은 확증 실행에서 나왔지만, E-T4 기준은 실행 전에 S-mix에만 걸었으므로 이 두 그래프에 대한 기준 적용은 사후다 [사후]. 결합의 이득은 두 설계 사이에서 재현되지 않았다. 설계 D의 T4는 충족이었다(결합 − 그래프 매크로 Top-10 +2.7%p [+1.8, +3.6]) [확증]. 설계 E의 E-T5는 미달이었다(S-comb − S-mix 매크로 Top-10 +0.11%p [−0.39, +0.59]) [확증]. E-T1이 판정한 Top-10과 달리 Top-1은 개선되지 않았다. S-mix − S-D의 Top-1은 DDXPlus-E +0.7%p [−0.6, +2.0]였고, SymCat-E에서는 −0.8%p [−1.2, −0.4](Holm 보정 p = .0012)로 낮았다 [확증-보조]. 개발 세트에서 판정한 설계 D의 T3(PUMCH 두 세트에서 S-D의 Top-10이 같은 랭커의 HPOA 이상)도 미달이었다. 차이는 PUMCH_L −2.5%p [−5.6, +0.5], PUMCH-ADM −1.3%p [−13.3, +10.7]였다 [개발]. 설계 D의 PPR 부가 가설 두 개(PPR-H1, PPR-H4; 표 6 행 5–6)는 실행 전에 적었지만, 봉인 세트에 적용할 판정 기준은 적지 않았다. 그래서 봉인 세트의 PPR 값은 판정 없이 기술한다(3.6절). PPR-H1의 비교에서 S-D 그래프의 PPR은 DDXPlus와 SymCat에서 같은 그래프의 1-홉보다 Top-10이 낮았고, Phenopacket-new에서는 1-홉과 구별되지 않았다. PPR − 1-홉 차이는 DDXPlus −15.8%p [−17.3, −14.3], SymCat −1.6%p [−2.2, −1.0], Phenopacket-new −1.5%p [−4.6, +1.5]였다 [봉인 기술]. 큐레이션 그래프에 같은 PPR을 적용한 PPR-H4의 비교에서는 HPOA의 흔한 질환 세트 값이 소폭 올랐다(SymCat Top-10 +2.8%p, DDXPlus Top-1 +1.8%p). 같은 HPOA에서 Phenopacket-new의 Top-10은 12.3%p 낮아졌다. PrimeKG에서는 DDXPlus(−4.8%p)와 Phenopacket-new(−22.6%p)의 Top-10이 낮아졌다. 이 다섯 차이는 모두 Holm 보정 후 유의했다 [봉인 기술]. 개발 세트에서도 PPR의 매크로 Top-10은 세 지식원 모두에서 1-홉보다 낮았다. 값은 S-D 23.57% 대 30.56%, HPOA 35.8% 대 52.4%, PrimeKG 20.54% 대 33.52%였다 [개발].

Table 11. Publication-derived rare-disease sets: HPOA circularity, leave-one-publication-out (LOPO) and the 2022 time split. Every knowledge source is scored by the same 1-hop Cypher ranker code with settings tuned per source; HPOA keeps its design-D setting and is not re-tuned after deletion, and panel (b) scores the literature graphs at a fixed γ. Lineage Top-10 (%). Differences are S-mix (or the named literature graph) minus HPOA, in percentage points, computed before rounding. Development sets; selection optimism applies.

(a) Development sets. LOPO removes every HPOA row that cites one of the 1,745 PMIDs of the evaluated Phenopacket-store cases (22,261 of the 264,245 rows read by the build; edges 393,498 → 363,488; 410 diseases lose their whole profile) and scores all sets with that single graph. Random = the same number of rows removed at random (seeds 0 / 1 / 2). LOPO coverage: complete = all source papers of the set are removed; partial = only the source papers shared with Phenopacket-store are removed (LIRICAL carries no PMIDs); none = no source paper of the set is known to be removed (the set carries no PMIDs). Gold affected = patients whose gold disease lost HPOA edges (in parentheses: lost its whole profile). Difference cells: [patient-paired bootstrap 95% CI] / [gold-disease cluster bootstrap 95% CI]. Disease-equal = mean over gold diseases (the gold-disease key of the cluster CI, Section 3.8) with diseases resampled, computed in the M1 run with the same LOPO graph; – = not computed. PUMCH_L and PUMCH-ADM are unpublished hospital cases, so their LOPO column shows only the side effect of deleting other sets' papers.

| Set (patients; gold diseases) | LOPO coverage | Gold affected (%) | S-mix | HPOA | HPOA-LOPO | HPOA random s0 / s1 / s2 | S-mix − HPOA | S-mix − HPOA-LOPO | Disease-equal: S-mix − HPOA → S-mix − HPOA-LOPO |
|---|---|---|---|---|---|---|---|---|---|
| Phenopacket-store (10,310; 772) | complete | 77.9 (36.0) | 31.03 | 73.69 | 25.45 | 72.22 / 72.00 / 72.47 | −42.66 [−43.74, −41.55] / [−51.52, −33.68] | +5.58 [+4.75, +6.42] / [+1.23, +9.92] | −55.73 [−58.89, −52.32] → +3.28 [+0.76, +5.79] |
| LIRICAL (370; 252) | partial | 68.1 (5.7) | 38.38 | 74.59 | 42.70 | 72.70 / 74.05 / 74.05 | −36.22 [−42.16, −30.27] / [−44.58, −27.27] | −4.32 [−9.73, +1.35] / [−13.30, +5.12] | −31.03 [−37.95, −23.62] → −1.53 [−7.78, +5.30] |
| MME (40; 17) | none | 0.0 (0.0) | 55.00 | 95.00 | 95.00 | 95.00 / 87.50 / 92.50 | −40.00 [−55.00, −25.00] / [−72.31, −6.45] | −40.00 [−55.00, −25.00] / [−72.31, −6.45] | −24.51 [−47.06, −5.88] → −24.51 [−47.06, −5.88] |
| RAMEDIS (624; 74) | none | 10.1 (0.0) | 57.21 | 59.94 | 59.46 | 60.42 / 59.13 / 59.94 | −2.72 [−7.05, +1.44] / [−18.99, +8.88] | −2.24 [−6.73, +1.92] / [−18.23, +9.58] | −18.97 [−30.46, −6.75] → −18.76 [−30.61, −6.34] |
| HMS (88; 46) | none | 4.5 (0.0) | 44.32 | 44.32 | 43.18 | 44.32 / 42.05 / 40.91 | 0.00 [−11.36, +11.36] / [−13.39, +14.77] | +1.14 [−10.23, +12.50] / [−12.17, +16.13] | +4.52 [−9.52, +18.52] → +5.60 [−7.79, +19.41] |
| PUMCH_L (988; 100) | none (unpublished) | 10.3 (0.0) | 67.11 | 53.24 | 52.83 | 52.73 / 52.73 / 52.02 | +13.87 [+10.83, +16.90] / [+1.86, +25.85] | +14.27 [+11.23, +17.41] / [+2.47, +25.99] | +1.51 [−7.60, +10.95] → – |
| PUMCH-ADM (75; 16) | none (unpublished) | 5.3 (0.0) | 68.00 | 57.33 | 61.33 | 61.33 / 58.67 / 60.00 | +10.67 [−1.33, +22.67] / [−4.69, +26.67] | +6.67 [−4.00, +17.33] / [−6.35, +19.48] | +6.51 [−6.46, +20.26] → – |
| Nine development sets, macro | – | – | 46.11 | 52.45 | (not an LOPO estimate) | – | −6.33 [−8.90, −3.69] (set-stratified) | – | – |

(b) Time split (exploratory). Graphs rebuilt from literature published up to 2022 vs HPOA 2022-12-15. Literature graphs are shown at γ = 0, which is not the γ 0.5 selected for S-mix and S-dict on the development sets; the γ = 1 rows are in the supplementary table. HPOA does not depend on γ. Phenopacket-store cases, n = 10,571 after removing 32 cases whose source paper is in the v2 corpus; these include the Phenopacket-new cases of the design-D confirmation. "New" = disease first annotated in HPOA after 2022-12-31 or not annotated up to the 2026-09 release, so it includes diseases described earlier but annotated late; "future" = case published from 2023-01-01. The future · new group was not stratified by whether any literature on the disease existed before the cut. Past · new cells are not used as evidence for the literature graphs, because HPOA is near 0 there by the group definition. Cells: Top-10 (difference from HPOA 2022-12 [patient-paired bootstrap 95% CI]). Eleven 2022-cut literature graphs were built (three shown); their future · new Top-10 ranged from 0.5% to 2.2% at γ = 0 and from 0.4% to 2.0% at γ = 1.

| Knowledge source (2022 cut, γ = 0) | Future · new, 2,381 (263 diseases) | Future · curated, 1,106 (67) | Past · new, 1,091 (175) | Past · curated, 5,993 (405) |
|---|---|---|---|---|
| HPOA 2022-12 | 0.1 | 49.4 | 0.6 | 52.9 |
| S-D graph (base_nm) | 0.9 | 32.0 (−17.4 [−20.3, −14.2]) | 2.2 | 19.7 (−33.2 [−34.6, −31.7]) |
| S-dict graph (v3umls_ns) | 1.9 | 41.2 (−8.1 [−11.1, −5.0]) | 4.4 | 35.6 (−17.3 [−18.8, −15.8]) |
| S-mix graph (v3llmd_ns) | 1.7 | 42.2 (−7.1 [−10.1, −3.9]) | 4.9 | 37.4 (−15.5 [−16.9, −14.0]) |

### 4.6 재현율 손실 단계와 정확도 요인

이 절은 먼저 설계 E의 주 목표(E0) 판정을 보고한다. 이어서 재현율 손실이 생기는 단계, 수집 깊이의 효과, 추출 방식 비교를 판정 규칙과 입력 구조로 다시 읽은 결과를 보고한다. 확증 판정이 아닌 값에는 [탐색], [개발], [사후], [측정]을 붙인다.

**주 목표 E0.** 설계 E의 주 목표 E0는 E-T1과 E-T2의 S-llm 비교를 모두 충족하는 것이었다(3.6절, 표 6). E-T1에서 S-mix의 Top-10은 S-D보다 DDXPlus-E에서 +6.5%p [+4.6, +8.5](Holm 보정 p = 2.1 × 10⁻¹¹), SymCat-E에서 +2.4%p [+1.7, +3.1](Holm 보정 p = 3.4 × 10⁻¹¹) 높았다 [확증]. E-T2에서 S-mix − S-llm의 두 세트 매크로 Top-10은 +1.64%p [+1.17, +2.18]였다. CI 하한이 비열등 한계 −1%p보다 높아 판정을 충족했다 [확증]. CI 하한은 0보다도 높았으나, 우세는 판정 기준이 아니었다. 세트별로는 DDXPlus-E +1.6%p [+0.8, +2.4], SymCat-E +1.7%p [+1.2, +2.2]였다. 따라서 E0를 충족했다. E-T1의 질환 군집 CI는 DDXPlus-E [−6.1, +19.3], SymCat-E [+1.0, +3.9]였다. S-mix − S-llm의 군집 CI는 DDXPlus-E [−1.3, +4.3], SymCat-E [+0.8, +2.6]였다 [확증-보조]. 같은 E-T2의 S-dict 비교와 S-union 비교는 미달이었다(4.4절, 표 6). S-mix와 S-D는 문헌, 추출, 빌더가 함께 다르므로, E-T1은 수집 깊이의 효과를 따로 확증하지 않는다. S-mix의 Top-10/Top-1은 개발값에서 확증값으로 DDXPlus 40.6/11.8%에서 41.1/12.4%, SymCat 13.3/2.5%에서 12.9/2.5%가 되어, 개발값과 확증값이 비슷했다. 이 비교에는 판정 기준이 없다. 확증 표본의 일부는 입력이 개발 환자와 같으며(D 확증 세트 기준 DDXPlus 26.4%, SymCat 15.4%; E 세트에서는 재지 않음, 3.6절), 질환 수준의 낙관은 이 확증으로 재지 않았다.

**재현율 손실의 단계 분해.** Phenopacket-store 증례 10,603건에서, HPOA가 정답 질환에 붙인 소견 가운데 환자에게 관찰된 소견을 분해 대상으로 삼았다 [탐색]. 대상은 증례×정답×소견 82,751건(질환–소견 쌍 22,626개, 정답 질환 878개)이다. 각 소견은 정답 질환의 근거가 처음 사라진 단계로 분류했다(표 12, 그림 3). 분해 방식은 단계별 재현율 분해의 선례를 따랐다 [ref: Pink 2014]. 수집 손실(쓸 만한 문헌이 없거나, 문헌은 있으나 텍스트에 소견 이름이 없음)은 S-D 67.0%에서 S-mix 37.3%로 줄었다(S-llm 37.7%, S-dict 36.4%). 쓸 만한 문헌이 없는 소견은 22.4%에서 0.3%로 줄었다. 텍스트에 이름이 없는 소견은 44.7%에서 37.0%로 줄었고, S-mix에서도 가장 큰 손실로 남았다. 추출·연결 손실은 8.5%에서 5.1%로 줄었다. 정답 질환의 주제 간선으로 남은 소견은 14.1%에서 41.3%로 늘었다. 단계 분류와 별개로, 각 그래프의 동결 설정으로 랭커가 그 소견에서 정답 질환에 0보다 큰 점수를 주는 비율은 14.9%에서 55.5%로 늘었다. 소견을 랭커가 조회하는 CUI로 대조하면 수집 손실은 S-D 66.6%, S-mix 36.0%였다. 어느 그래프에서든 대조 기준에 따라 수집 손실이 움직인 폭은 3.2%p 이내였다. 이 분해는 문자열 일치 기준이므로 "텍스트에 이름 없음"은 수집 손실의 상한이다. 기준이 HPOA이므로 분해 대상은 희귀질환뿐이다.

**수집 깊이.** 같은 추출 방식과 같은 빌더 규칙으로 v2 문헌 그래프와 v2+v3 문헌 그래프를 짝지은 7쌍을 개발 9개 세트에서 비교했다(표 13) [개발]. 7쌍 모두에서 매크로 Top-10이 +6.95~+12.96%p 올랐다. 자기 간선을 뺀 UMLS 사전 쌍은 34.50%에서 42.88%로, LLM 쌍은 31.94%에서 44.90%로 올랐다. LLM 쌍의 차이에는 구절 표 확장과 γ 변화(0 → 1.0)가 함께 들어 있다. 같은 v2 문헌에서 추출·연결 방식이나 빌더 규칙을 바꾼 변형의 효과는 설계 D 그래프 대비 최대 +4.1%p였다(v2unionu_ns 34.66% 대 30.56%). 누수 의심 환자(한 문서가 환자 개념의 50% 이상이면서 5개 이상을 지지)를 빼고 다시 계산해도, 7쌍 모두 효과의 0.68배 이상이 남았다. 남은 비율이 가장 낮은 쌍은 자기 간선을 둔 UMLS 사전 쌍이었다(+6.95%p → +4.77%p, 0.686배). 이 결과는 측정 전에 커밋한 P0-1 규칙(7쌍 모두 효과의 절반 이상 유지)을 충족했으므로, 수집 깊이의 효과는 개발 수준 주장으로 유지한다. 다만 MME 단독의 효과는 누수 의심 환자를 빼면 +20.0~+27.5%p에서 +4.2~+19.2%p로 줄었다. 일부 쌍의 Top-1 효과도 누수에 민감했다(HPO 사전 쌍 +0.84%p → +0.10%p). 표 13b의 LLM 쌍에서 이득이 가장 작은 세트는 SymCat(+1.06%p [+0.43, +1.71])이었고, UMLS 사전 쌍에서는 PUMCH_L(0.00%p [−2.13, +2.23])과 SymCat(+0.41%p [−0.27, +1.03])이 가장 작았다. DDXPlus는 두 쌍에서 +4.43%p, +3.63%p였다. 희귀질환 세트는 PUMCH_L의 UMLS 사전 쌍(0.00%p)을 빼면 +6.17~+27.50%p였다. 7쌍 전체의 세트별 차이(점추정, 누수 의심 환자 제외 전)는 DDXPlus −1.83~+6.93%p(7쌍 중 6쌍 상승), SymCat −0.31~+1.06%p(6쌍 상승)였다. 희귀질환 세트 가운데 PUMCH_L은 7쌍 중 4쌍에서 내려갔고(−5.16, −3.24, −1.82, −0.51%p), 1쌍에서 0.00%p였으며, 2쌍에서 올랐다(+0.91, +6.17%p). 나머지 희귀질환 6개 세트는 7쌍 모두에서 올랐다(+2.27~+27.50%p). S-dict의 매크로 Top-1에서는 자기 간선 제거의 효과(자기 간선을 둔 v2+v3 문헌 UMLS 사전 그래프 v3umls 대비 +7.56%p)가 수집 깊이의 효과(v2 문헌판 v2umls_ns 대비 +3.87%p)보다 컸다. PMC 본문을 12k 임상 절로 자르지 않고 전부 읽은 UMLS 사전 그래프(v3umlsfull_ns)의 매크로 Top-10은 12k 판인 S-dict와 거의 같았다(43.08% 대 42.88%; 세트별로 높음 4, 낮음 5, 부호검정 p = 1; 선택된 γ .25 대 .5).

**판정 규칙으로 다시 읽은 추출 비교.** E-T2의 S-dict 비교 미달(S-mix − S-dict 매크로 Top-10 −1.80%p [−2.44, −1.12])을 strict 판정으로 다시 계산했다(표 14a) [사후]. strict 판정은 정답 질환 자체가 Top-10에 있을 때만 맞은 것으로 센다(3.5절). strict에서 S-mix − S-dict의 두 세트 매크로는 +1.04%p [+0.39, +1.72]로 부호가 바뀌었고 CI도 0을 배제했다. 측정 전에 커밋한 P0-3 규칙의 판정 단위는 E-T2 비교, 곧 두 세트 매크로 Top-10이다. 이 규칙에 따라, 두 세트 매크로 Top-10의 S-dict 우세는 판정 규칙에 달린 결론이다. 규칙 밖의 관찰로, DDXPlus-E에서도 strict 차이는 +3.00%p [+1.80, +4.13]로 부호가 바뀌고 CI가 0을 배제했다. SymCat-E에서는 strict 판정에서도 S-dict가 높았다(S-mix − S-dict −0.91%p [−1.38, −0.43]). S-mix − S-union은 두 세트 매크로 strict에서 +0.15%p [−0.46, +0.76]로 부호만 바뀌었고 CI가 0을 포함해 규칙의 조건을 충족하지 않았다. 다만 DDXPlus-E만 보면 S-mix − S-union도 strict에서 +1.30%p [+0.20, +2.40]로 부호가 바뀌고 CI가 0을 배제했다. S-mix − S-llm은 strict에서 +4.04%p [+3.53, +4.62]로 lineage보다 컸다. DDXPlus-E의 strict Top-10은 S-mix 31.47%, S-union 30.17%, S-dict 28.47%, S-llm 25.23%로 lineage 판정과 순서가 달랐다. DDXPlus-E의 S-mix − S-dict(−2.43%p)는 strict 차이 +3.00%p와 lineage 가산분 차이 −5.43%p의 합이었다. lineage 가산분은 정답의 IS_A 하위 질환이 Top-10에 들어 맞은 것으로 센 몫이다. Top-10 자리 가운데 정답이 아닌 하위 질환이 차지한 비율은 그래프 간에 비슷했다(DDXPlus-E 3.32–3.74%, SymCat-E 1.17–1.48%; 표 14b). 하위 질환이 들어가는 양이 아니라, 그때 정답이 함께 맞은 것으로 세어지는지에서 차이가 났다. Top-1에서는 strict 판정에서도 S-dict가 높았다(S-mix − S-dict 매크로 −1.14%p [−1.55, −0.77]).

**입력 구조로 다시 읽은 추출 비교.** 환자 입력에 정답 질환 키나 그 직속 상위 질환 키가 이미 들어 있는 환자(입력 내 정답 명시 층)는 DDXPlus-E 717명(23.9%), SymCat-E 275명(3.7%)이었다 [사후]. DDXPlus-E에서 S-mix − S-dict(−2.43%p)의 139.7%가 명시 층에서 나왔다(표 14a). 명시 층에서는 Anemia 한 질환에서 S-dict가 S-mix보다 68명을 더 Top-10에 넣었다. strict 판정에서는 두 그래프 모두 이 질환의 환자를 한 명도 Top-10에 넣지 못했다. 명시 층이 아닌 환자(없음 층, 2,283명)에서 S-mix − S-dict는 +1.27%p [+0.09, +2.54]였다. 환자 CI는 0을 배제했으나 Holm 보정 p = .11이었고, 군집 CI [−4.29, +6.21]는 0을 포함했다 [사후]. SymCat-E에서도 S-mix − S-dict(−1.17%p)의 88.4%가 명시 층에서 나왔고, 없음 층의 차이는 −0.14%p [−0.67, +0.40]였다. SymCat-E에서는 S-dict의 우세가 strict 판정에서도 남았고(앞 문단), 구절 표를 비운 순수 사전 그래프도 S-mix보다 높았다(S-mix − 순수 사전 그래프 −1.09%p [−1.65, −0.48], 표 14a) [사후]. 이 세트에서 두 우세가 0과 구별되지 않게 된 것은 명시 층을 뺐을 때뿐이었다(없음 층 S-mix − S-dict −0.14%p, S-mix − 순수 사전 그래프 +0.17%p [−0.38, +0.71]) [사후]. DDXPlus-E Top-1 차이(−5.20%p)의 89.0%도 명시 층에서 나왔다. 측정 전에 커밋한 보조 규칙(P1-3)은, DDXPlus-E 없음 층의 S-mix − S-dict CI가 0을 포함하거나 양수이면 S-dict 우세가 입력 구조에서 온다는 설명을 봉인 세트 수준으로 쓰게 한다. 그 CI는 [+0.09, +2.54]로 이 조건을 충족했다. 따라서 DDXPlus-E에서 S-dict의 우세가 입력에 정답이 이미 있는 환자에서 나왔다는 설명은 봉인 세트의 사후 분석 수준에서 쓴다 [사후]. 이 규칙이 받치는 것은 없음 층에서 S-dict의 우세가 보이지 않았다는 데까지다. 개발 DDXPlus의 없음 층에서 S-mix − S-dict는 +1.55%p [+0.22, +2.85](Holm 보정 p = .046, 군집 CI [−5.22, +7.53])였다 [개발]. DDXPlus-E 없음 층에서 S-mix − S-union은 −1.23%p [−2.28, −0.09](Holm 보정 p = .083, 군집 CI [−5.56, +2.98])였다. 같은 층에서 S-mix − 순수 사전 그래프는 +3.55%p [+2.37, +4.69](Holm 보정 p = 1.8 × 10⁻⁸, 군집 CI [−0.10, +7.68])였다.

**희귀질환 개발 세트.** 희귀질환 개발 세트 7개에서는 S-mix가 S-dict보다 높았다 [개발]. 7개 세트 매크로 Top-10의 S-mix − S-dict는 +4.55%p [+2.44, +6.73]였고, 누수 의심 환자를 빼면 +4.79%p [+2.21, +7.57]였다(표 14c). 이 결과는 P0-1 규칙(제외 뒤 CI 하한 > 0)을 충족했다. 세트별로는 RAMEDIS +9.29%p, LIRICAL +5.68%p, Phenopacket-store +2.95%p가 Holm 보정 후 유의했고, 세 세트 모두 군집 CI도 0을 배제했다. PUMCH-ADM(+8.00%p)은 Holm 보정 후 유의하지 않았지만(p = .21), 군집 CI [+1.30, +17.33]는 0을 배제했다. 이 결과는 S-mix를 주 시스템으로 둔 근거였다. 추출 경로 비교의 입력은 대부분 HPO 개념이다. 희귀질환 세트의 입력은 HPO 코드이고, 흔한 질환 벤치마크도 평가에 쓰는 CUI로 세면 소견 언급의 70%(DDXPlus)와 65%(SymCat)가 HPO 개념이었다(HPO–UMLS 사상 기준) [측정]. 희귀질환 봉인 세트가 없으므로 이 결과는 확증되지 않았다. 확증된 것은 흔한 질환 두 세트에서 S-mix가 S-dict에 대한 비열등 기준을 충족하지 못했다는 판정이다(E-T2, 매크로 −1.80%p [−2.44, −1.12], lineage 판정) [확증].

**LLM 구절의 측정된 역할.** 두 그래프의 간선을 (질환, 개념) 쌍으로 비교했다 [측정]. S-mix 간선 5,221,285개 가운데 86.9%는 S-dict에도 있었다. S-mix의 재연결 항목은 S-dict와 같은 UMLS 문자열로 이름을 붙이므로, 이 중첩은 구성상 상당 부분 예상되는 값이다. S-dict 간선은 14,506,788개로 S-mix의 2.8배였고, 그 68.7%는 S-mix에 없었다. S-mix에만 있는 간선 685,993개 가운데 97.3%는 재연결되지 않은 LLM 항목, 곧 구절 표 경로에서 나왔다. 그 48.5%(332,817개)는 사전이 이름 붙일 수 없는 개념 위에 있었고, 이 개념에 많은 의미유형은 의약품, 유기화합물, 인구집단이었다. S-dict에만 있는 간선의 HPOA 일치율은 2.31%로, 두 그래프에 함께 있는 간선(13.0%)보다 낮았다 [탐색]. 개발 세트에서 한 그래프만 정답을 lineage Top-10에 넣은 환자를 그 그래프의 승리로 셌다 [개발]. 승리한 그래프에서 정답 계보 질환의 전용 간선 점수를 빼도 Top-10에 남은 환자의 비율은 세트에 따라 달랐다. 희귀 세트에서 S-mix 승리는 RAMEDIS 63/68명(92.6%), LIRICAL 20/26명(76.9%), Phenopacket-store 448/505명(88.7%)이 남았다. 같은 세트에서 S-dict 승리는 9/10명, 3/5명, 99/201명(49.3%)이 남았다. 흔한 질환 세트에서는 DDXPlus의 S-mix 승리 65/159명(40.9%)과 S-dict 승리 94/208명(45.2%), SymCat의 S-mix 승리 166/207명(80.2%)과 S-dict 승리 211/291명(72.5%)이 남았다. 경쟁 질환 쪽의 반사실은 하지 않았다. 이 측정의 해석은 5절에서 다룬다.

**그 밖의 레버.** 자기 간선을 둔 v2+v3 문헌의 UMLS 사전 그래프(v3umls)에서 자기 간선을 빼면(S-dict) 매크로 Top-1이 10.55%에서 18.11%로 올랐다 [개발]. 자기 간선은 질환에서 자기 개념과 직속 상위 질환의 개념으로 가는 간선이다(3.4절). 이 효과와 후보 창(3.5절)의 상호작용은 보충자료에 둔다. HPOA 그래프에 틀린 간선을 가정해 넣은 시뮬레이션에서, 남긴 참 간선이 15%일 때 Top-10은 정밀도를 100%로 올리는 것(+0.1~+2.7%p)보다 남긴 참 간선을 30%로 늘리는 것(+8.0~+11.6%p)에 더 크게 움직였다 [탐색, 가정 의존: 틀린 간선의 구성과 양은 측정값이 아니다].

Table 12. Stage at which a curated finding of the gold disease is first lost (exploratory). Cases: the 10,603 Phenopacket-store cases. Every column uses the same 82,751 case × gold × finding triples (22,626 disease–finding pairs, 878 gold diseases); each graph is decomposed with its own literature, extraction, phrase table and builder rules. Findings are matched by the UMLS cross-references of their HPO terms; string matching makes "name not in text" an upper bound on collection loss. n (%).

| Stage | S-D (v2) | S-llm | S-dict | S-mix |
|---|---|---|---|---|
| No usable document (none, or all generic) | 18,502 (22.4) | 208 (0.3) | 206 (0.2) | 208 (0.3) |
| Documents exist, finding name not in text | 36,963 (44.7) | 30,958 (37.4) | 29,901 (36.1) | 30,656 (37.0) |
| In text, not extracted | 1,906 (2.3) | 2,053 (2.5) | 277 (0.3) | 2,022 (2.4) |
| Extracted, not linked or linked to a different CUI | 5,113 (6.2) | 2,857 (3.5) | 2,439 (2.9) | 2,219 (2.7) |
| Linked, removed by a filter^a^ | 2,881 (3.5) | 3,160 (3.8) | 3,087 (3.7) | 3,113 (3.8) |
| Only in off-topic documents^b^ | 3,569 (4.3) | 3,775 (4.6) | 4,042 (4.9) | 3,627 (4.4) |
| Only a parent or child concept has an edge | 2,132 (2.6) | 6,622 (8.0) | 7,104 (8.6) | 6,755 (8.2) |
| Edge present | 11,685 (14.1) | 33,118 (40.0) | 35,695 (43.1) | 34,151 (41.3) |
| Collection loss (first two stages) | 55,465 (67.0) | 31,166 (37.7) | 30,107 (36.4) | 30,864 (37.3) |
| Collection loss, findings matched by the ranker's CUI | 55,079 (66.6) | 30,678 (37.1) | 28,255 (34.1) | 29,827 (36.0) |
| Extraction and linking loss (third and fourth stages) | 7,019 (8.5) | 4,910 (5.9) | 2,716 (3.3) | 4,241 (5.1) |
| Scored by the frozen ranker for the gold disease^c^ | 12,303 (14.9) | 44,222 (53.4) | 49,090 (59.3) | 45,891 (55.5) |
| Disease–finding pairs with an edge (% of 22,626) | 5.8 | 21.4 | 24.3 | 22.5 |

^a^ Generic document or excluded semantic type; in the v2 + v3 columns also a concept naming the disease or a parent disease (self-edge removal). ^b^ A loss for S-D (γ = 0); scored at the frozen γ for the v2 + v3 graphs. ^c^ Independent of the stage: the frozen ranker gives the gold disease a score above 0 through the finding's CUI or a graph IS_A parent of it.

Table 13. Collection depth: v2 versus v2 + v3 literature with the same extraction and builder rules (development, nine sets, lineage Top-10, %). The off-topic weight γ (Section 3.5) is re-selected per graph by rule D3. Δ = v3 − v2 with set-stratified patient bootstrap 95% CI. Excluding = after removing, from both graphs of a pair, every patient flagged in either graph by the leakage rule (one document supports ≥ 50% and ≥ 5 of the patient's concepts). Retained = Δ excluding / Δ, rounded to two decimals; the smallest unrounded value is 0.686 (UMLS dictionary, self-edges kept).

(a) Macro Top-10.

| Extraction | Self-edges | v2 graph (γ) → v2 + v3 graph (γ) | v2 → v2 + v3 | Δ | Δ excluding flagged | Retained |
|---|---|---|---|---|---|---|
| HPO dictionary | kept | v2dict (0) → v3dict (.25) | 31.64 → 39.20 | +7.57 [+5.17, +9.86] | +6.66 [+4.13, +9.20] | 0.88 |
| UMLS dictionary | kept | v2umls (.5) → v3umls (0) | 34.60 → 41.55 | +6.95 [+4.78, +9.10] | +4.77 [+2.55, +6.96] | 0.69 |
| HPO dictionary | removed | v2dict_ns (.25) → v3dict_ns (0) | 32.65 → 40.51 | +7.86 [+5.59, +10.13] | +7.03 [+4.78, +9.45] | 0.89 |
| UMLS dictionary | removed | v2umls_ns (.5) → v3umls_ns = S-dict (.5) | 34.50 → 42.88 | +8.37 [+6.05, +10.62] | +6.05 [+3.82, +8.31] | 0.72 |
| LLM^a^ | removed | base_split_tg_ns (0) → v3llm_ns = S-llm (1.0) | 31.94 → 44.90 | +12.96 [+10.70, +15.32] | +11.99 [+9.67, +14.57] | 0.93 |
| LLM + HPO dictionary | removed | v2union_ns (.25) → v3union_ns (.25) | 33.36 → 44.00 | +10.63 [+8.36, +12.98] | +9.07 [+6.87, +11.43] | 0.85 |
| LLM + UMLS dictionary | removed | v2unionu_ns (.5) → v3unionu_ns = S-union (.5) | 34.66 → 43.77 | +9.11 [+6.73, +11.42] | +7.59 [+5.46, +9.81] | 0.83 |

^a^ The v2 + v3 LLM graph also uses the phrase table extended to v3 spans, and its selected γ differs.

(b) Per set, two pairs (before leakage exclusion). Δ [patient CI] [gold-disease cluster CI]; * Holm-adjusted exact McNemar p < .05 within the pair.

| Benchmark | n (gold diseases) | UMLS dictionary v2 → v2 + v3 | Δ | LLM v2 → v2 + v3 | Δ |
|---|---|---|---|---|---|
| DDXPlus | 3,000 (48) | 37.83 → 42.27 | +4.43* [3.07, 5.77] [−1.35, 10.43] | 35.13 → 38.77 | +3.63* [1.97, 5.23] [−5.71, 13.13] |
| RAMEDIS | 624 (74) | 39.26 → 47.92 | +8.65* [5.61, 11.54] [0.70, 14.29] | 32.53 → 55.93 | +23.40* [19.23, 27.24] [12.18, 32.18] |
| LIRICAL | 370 (252) | 25.95 → 32.70 | +6.76* [2.70, 11.08] [1.45, 12.57] | 25.41 → 38.92 | +13.51* [9.46, 17.84] [7.10, 19.94] |
| HMS | 88 (46) | 32.95 → 43.18 | +10.23 [2.27, 18.18] [2.70, 18.82] | 29.55 → 44.32 | +14.77* [6.82, 23.86] [4.82, 26.14] |
| MME | 40 (17) | 30.00 → 52.50 | +22.50 [7.50, 37.50] [0.00, 53.85] | 30.00 → 57.50 | +27.50* [15.00, 42.50] [9.23, 59.09] |
| PUMCH_L | 988 (100) | 64.78 → 64.78 | 0.00 [−2.13, 2.23] [−4.08, 4.23] | 57.19 → 63.36 | +6.17* [3.95, 8.40] [0.13, 12.62] |
| PUMCH-ADM | 75 (16) | 45.33 → 60.00 | +14.67 [5.33, 24.00] [1.33, 30.68] | 49.33 → 62.67 | +13.33* [2.67, 24.00] [−2.90, 30.00] |
| Phenopacket-store | 10,310 (772) | 20.37 → 28.08 | +7.71* [7.08, 8.31] [4.78, 10.80] | 17.87 → 31.14 | +13.28* [12.63, 14.00] [9.52, 17.01] |
| SymCat | 7,481 (760) | 14.05 → 14.46 | +0.41 [−0.27, 1.03] [−0.85, 1.71] | 10.43 → 11.48 | +1.06* [0.43, 1.71] [−0.03, 2.15] |

Table 14. Extraction-path comparisons under lineage and strict judgment and by input-gold stratum. Differences are S-mix minus the comparator, Top-10, percentage points, with patient-paired bootstrap 95% CI; macro = equal-weight mean over the two sets with set-stratified CI.

(a) Sealed design-E sets. Only the bold lineage macro cells of S-llm, S-dict and S-union are the E-T2 judgments committed before the run (confirmatory); every other cell is a post hoc analysis of the already used sets. Named = the patient's input already contains the gold disease key or its direct parent key (DDXPlus-E 717 of 3,000; SymCat-E 275 of 7,463); neither = the other patients (2,283; 7,188). – = not computed.

| Comparison | Set | Lineage, all | Strict, all | Lineage, named | Lineage, neither |
|---|---|---|---|---|---|
| S-mix − S-llm | DDXPlus-E | +1.60 [+0.83, +2.40] | +6.23 [+5.30, +7.20] | – | – |
| | SymCat-E | +1.67 [+1.17, +2.21] | +1.85 [+1.43, +2.28] | – | – |
| | Macro | **+1.64 [+1.17, +2.18]** | +4.04 [+3.53, +4.62] | – | – |
| S-mix − S-dict | DDXPlus-E | −2.43 [−3.70, −1.17] | +3.00 [+1.80, +4.13] | −14.23 [−17.15, −11.30] | +1.27 [+0.09, +2.54] |
| | SymCat-E | −1.17 [−1.72, −0.59] | −0.91 [−1.38, −0.43] | −28.00 [−33.45, −21.82] | −0.14 [−0.67, +0.40] |
| | Macro | **−1.80 [−2.44, −1.12]** | +1.04 [+0.39, +1.72] | −21.11 [−24.36, −18.03] | +0.57 [−0.09, +1.28] |
| S-mix − S-union | DDXPlus-E | −4.00 [−5.10, −2.87] | +1.30 [+0.20, +2.40] | −12.83 [−15.76, −9.90] | −1.23 [−2.28, −0.09] |
| | SymCat-E | −1.33 [−1.82, −0.76] | −1.00 [−1.45, −0.55] | −30.91 [−36.36, −25.09] | −0.19 [−0.68, +0.31] |
| | Macro | **−2.66 [−3.27, −2.07]** | +0.15 [−0.46, +0.76] | −21.87 [−25.00, −18.84] | −0.71 [−1.30, −0.08] |
| S-mix − pure dictionary | DDXPlus-E | +0.67 [−0.47, +1.87] | – | −8.51 [−11.16, −5.86] | +3.55 [+2.37, +4.69] |
| | SymCat-E | −1.09 [−1.65, −0.48] | – | −33.82 [−39.64, −28.00] | +0.17 [−0.38, +0.71] |
| | Macro | −0.21 [−0.84, +0.45] | – | −21.16 [−24.52, −18.03] | +1.86 [+1.23, +2.51] |

(b) Lineage and strict Top-10 per graph on the sealed sets. Lineage values are from the confirmation run; strict values and descendant slots are post hoc. Descendant slots = share of all Top-10 positions held by non-gold IS_A descendants of the gold disease.

| Graph | DDXPlus-E lineage / strict | DDXPlus-E descendant slots (%) | SymCat-E lineage / strict | SymCat-E descendant slots (%) |
|---|---|---|---|---|
| S-mix | 41.07 / 31.47 | 3.74 | 12.92 / 8.08 | 1.27 |
| S-llm | 39.47 / 25.23 | 3.32 | 11.24 / 6.23 | 1.17 |
| S-dict | 43.50 / 28.47 | 3.71 | 14.08 / 8.99 | 1.48 |
| S-union | 45.07 / 30.17 | 3.65 | 14.24 / 9.08 | 1.44 |

(c) Development rare-disease sets, S-mix − S-dict, lineage Top-10 [patient CI] [gold-disease cluster CI]. * Holm-adjusted exact McNemar p < .05 within the nine sets. Excluding flagged = after removing patients flagged by the leakage rule in either graph (Table 13); per-set excluded values are given for the three Holm-significant sets.

| Set (patients; gold diseases) | All patients | Excluding flagged |
|---|---|---|
| RAMEDIS (624; 74) | +9.29* [6.89, 12.02] [2.48, 20.24] | +9.2 [6.7, 11.8] [2.3, 20.5] |
| LIRICAL (370; 252) | +5.68* [2.70, 8.65] [2.63, 9.01] | +5.6 [2.5, 9.0] [2.2, 9.4] |
| Phenopacket-store (10,310; 772) | +2.95* [2.43, 3.46] [0.51, 5.48] | +2.8 [2.3, 3.4] [0.5, 5.4] |
| PUMCH-ADM (75; 16) | +8.00 [1.33, 16.00] [1.30, 17.33] | no flagged patients |
| PUMCH_L (988; 100) | +2.33 [0.40, 4.25] [−2.05, 6.49] | no flagged patients |
| HMS (88; 46) | +1.14 [−5.68, 7.95] [−4.21, 7.79] | – |
| MME (40; 17) | +2.50 [−7.50, 12.50] [−8.00, 18.18] | – |
| Seven rare sets, macro | +4.55 [+2.44, +6.73] | +4.79 [+2.21, +7.57] |

### 4.7 출처: 추적 가능성

**구조 검사.** LLM 추출은 원문 창에서 위치가 확인된 구절만 남겼다(3.3절) [구조 검사]. 이 단계에서 v2 문헌의 모델 출력 560만 개 가운데 533만 개, v3 문헌의 1,323만 개 가운데 1,232만 개가 남았다. 버린 비율은 각각 약 5%, 약 7%다 [측정]. 근거 언급(Mention)을 넣어 적재한 S-mix는 동결 전에 출처 불변식 검사를 통과했다(3.4절). 근거 언급은 11,582,169개, 문서 노드는 711,350개, Disease 노드는 43,575개였다. 그래프 지문(15ba677…)은 근거 언급이 없는 빌드와 같았다. 개발 DDXPlus 3,000명 모두에서 정답의 Top-10 내 위치가 근거 언급이 없는 빌드와 같았다(표 15a). 이 검사는 모든 간선을 원문 위치까지 거슬러 올라갈 수 있다는 추적 가능성을 보인다. 간선이 맞다는 증거는 아니다.

**정확성의 하한: HPOA 일치.** 간선 정확성에 대해 이 논문이 보고하는 근거는 HPO 주석(phenotype.hpoa 2026-02-16)과의 정확 개념 일치뿐이다(표 15b) [탐색]. 간선 정확성은 사람이나 LLM의 판정으로 검증하지 않았다. HPOA에 없는 참 간선도 불일치로 세므로 이 일치율은 정밀도의 하한이다. 또 HPOA 주석이 있는 질환(주로 희귀질환)의 간선에만 해당한다. S-mix의 일치율은 12.6%였고, 질환을 무작위로 바꾼 대응(20회 평균)은 2.5%였다. 지지 문서가 1편인 간선은 7.4%, 5편 이상인 간선은 41.7%였다. 같은 구간의 무작위 일치율도 2.0%에서 4.5%로 올랐다. 주제 문서가 1편 이상인 간선은 14.9%, 주제 밖 문서뿐인 간선은 7.1%였다. 지지 문서 수로 나눠도 네 구간 모두 주제 문서가 있는 간선이 높았다(1편 8.3% 대 5.7%, 2편 16.3% 대 11.1%, 3–4편 25.8% 대 18.7%, 5편 이상 42.3% 대 29.0%). 같은 문헌의 S-dict는 7.8%(무작위 1.9%)였고, 주제 문서 유무에 따른 차이(10.6% 대 3.6%)도 네 구간 모두 같은 방향이었다. S-mix의 일치율은 지지 문서 수의 네 구간 모두에서, v2 문헌으로 만든 이전 LLM 그래프 두 개(S-D, 구 Qwen3 v2.1)보다 낮았다(지지 문서 1편 → 5편 이상: S-D 15.0% → 62.8%, 구 Qwen3 v2.1 13.3% → 63.2%). 네 그래프는 문헌, 추출 모델과 방식, 빌더 규칙, 간선 수가 다르므로, 그래프 사이 차이의 원인은 분리하지 않았다. 반대 방향으로, 각 그래프가 비교 가능한 간선을 가진 질환의 큐레이터 주석 쌍 가운데 그래프에 있는 쌍의 비율은 S-mix 29.4%(368,429쌍), S-dict 32.8%(372,112쌍), S-D 10.9%(281,825쌍)였다. 분모의 질환 집합은 그래프마다 다르다(질환 11,207개, 11,385개, 8,132개).

Table 15. Provenance checks and the HPOA agreement lower bound.

(a) Structural checks (no judgment of edge correctness).

| Check | Result |
|---|---|
| Span check, gemma-4-E4B-it output on v2 literature | 5.33 M of 5.60 M model items kept (≈ 5% discarded) |
| Span check, v3 new literature | 12.32 M of 13.23 M kept (≈ 7% discarded) |
| S-mix with Mentions: HAS_FINDING edges / Mentions / Doc / Disease nodes | 5,221,285 / 11,582,169 / 711,350 / 43,575 |
| Provenance invariant (every HAS_FINDING edge has a Mention) and build fingerprint | passed before the freeze; fingerprint 15ba677… unchanged |
| Development DDXPlus: gold position within Top-10 vs the build without Mentions | identical for all 3,000 patients |

(b) Exact-concept agreement with HPO curated annotations (phenotype.hpoa 2026-02-16), %. Agreement is a lower bound on edge correctness and covers only edges of diseases with HPO annotations, mostly rare diseases. HPO-comparable edges = edges of annotated diseases whose concept maps to an HPO term. Random (in parentheses) = the same edges with diseases permuted (mean of 20); per-group random values were recorded for S-mix and S-dict only. Topic document = a supporting document whose title names the disease (or, in the S-mix and S-dict builds, one of its causal-gene symbols). Old Qwen3 = the v2-literature graph extracted with Qwen3-4B-Instruct-2507 (v2.1); S-D = the design-D graph. The four graphs differ in literature, extraction model and method, builder rules and edge count; the causes of their differences were not separated. Edge correctness was not verified by human or LLM judgment. – = not measured.

| Edge group | Old Qwen3 (v2.1) | S-D | S-mix | S-dict |
|---|---|---|---|---|
| HPO-comparable edges | 194,318 | 168,587 | 856,023 | 1,572,570 |
| All, exact (random) | 16.4 (2.6) | 18.2 (2.8) | 12.6 (2.5) | 7.8 (1.9) |
| All, exact or HPO parent (random) | 19.9 (3.5) | 22.1 (3.7) | 16.2 (3.4) | 10.1 (2.7) |
| Supported by 1 document | 13.3 | 15.0 | 7.4 (2.0) | 4.4 (1.6) |
| Supported by 2 documents | 31.7 | 34.0 | 15.2 (2.9) | 8.5 (2.1) |
| Supported by 3–4 documents | 46.3 | 48.7 | 24.9 (3.6) | 14.6 (2.7) |
| Supported by ≥ 5 documents | 63.2 | 62.8 | 41.7 (4.5) | 24.6 (3.1) |
| ≥ 1 topic document | – | – | 14.9 (2.5) | 10.6 (2.0) |
| Off-topic documents only | – | – | 7.1 (2.5) | 3.6 (1.8) |
| Edges in both S-mix and S-dict | – | – | 13.0 (2.5) | 13.0 (2.5) |
| Edges only in this graph (S-mix vs S-dict) | – | – | 7.25 (2.02) | 2.31 (1.31) |
| Curated pairs recovered^a^ | 11.3 | 10.9 | 29.4 | 32.8 |

^a^ Share of the curated (disease, phenotype) pairs that the graph contains, over the annotated diseases on which that graph has at least one HPO-comparable edge. The disease set differs by graph (curated pairs / diseases: Old Qwen3 283,146 / 8,199; S-D 281,825 / 8,132; S-mix 368,429 / 11,207; S-dict 372,112 / 11,385).

<!-- 작성 메모(본문 아님):
반영한 지적과 남은 일.
1) 자기 인용률 72.8%(스크래치패드 SP/adv/hpoa_circ.json, 미커밋)는 본문에서 지웠다. 반대 증거 16.7%(Phenopacket-new, 결과 파일 없음)도 넣지 않았다. 대신 Phenopacket-new 격차에 LOPO를 적용하지 않았다는 사실만 적었다. MET-B 3.9의 "이 연구는 이 비율을 비교군 HPOA(2026-02-16판)에서도 쟀다(4.5절)"는 이제 가리킬 곳이 없으므로 고쳐야 한다.
2) 질환 등가중 LOPO 값은 커밋된 pilot/cache/v3/framing/m1/stats.json(published LOPO, dz-equal)에서 부호를 바꿔 옮겼다(Phenopacket-store HPOA-LOPO − S-mix −3.28 [−5.79, −0.76]). MET-B 3.8과 3.10은 "P0-4의 질환 등가중 값은 계산되지 않았다"고 적는다. M1 실행에서 같은 LOPO 그래프로 계산했다고 고쳐야 한다.
3) 누수 의심 환자를 뺀 LOPO 값 +4.03(n 9,198, p0/summary.md)은 CI가 없어 본문에서 뺐다. CI를 계산해 커밋하면 보충자료에 둔다.
4) 전역 삭제의 방향: hpoa_lopo/summary.md와 MET-B 작성 메모 (6)에 맞춰 "HPOA에 불리하므로 상한" 문장을 지웠다. 근거는 10겹 판(25.60 대 25.45)이다.
5) M1: 공개 희귀질환 증례층의 간선 합집합 − HPOA(−6.08)와 LOPO판 두 결합, w = .25 값을 함께 적었다. 병원층 결합 − HPOA는 질환 등가중을 붙였고, Top-1 관찰은 [개발]로 붙였다([사후]는 3.8절 정의상 봉인 세트 E의 재분석이다). k와 세 CI 요구가 측정 전 커밋으로 확인되지 않는다는 점은 MET-B 3.10 마지막 문단과 맞췄다.
6) 반려한 지적: (a) [구조 검사]는 MET-B 3.8에 정의되어 있고 RES-A도 쓴다. 그대로 둔다. (b) PPR 가설 이름은 MET-B 3.6·3.8과 표 6의 PPR-H1·PPR-H4로 맞췄고, 봉인 세트 값의 지위는 MET-B 3.6에 따라 [봉인 기술]로 고쳤다(설계 D10에 봉인 세트 판정 기준 없음). (c) D-PPR4 수치의 CI는 커밋된 결과 파일에서 찾지 못했다(RES-A 작성 메모 5와 같음). 그래서 CI 대신 docs/research_design.md D9의 "Holm 유의"만 옮겼다.
7) RAMEDIS·HMS의 출판 여부는 데이터 기술 문서로 확인하지 못했다. 그래서 이 절은 출판 증례 세트를 LIRICAL·MME·Phenopacket-store로 정의하고, M1 공개 희귀질환 증례층의 구성(다섯 세트)을 따로 적었다.
8) DIS 55행은 희귀 세트의 S-mix 승리 비율만 인용한다. 이 절에 적은 S-dict 승리와 흔한 질환 세트의 값(DDXPlus 40.9%·45.2%, SymCat 80.2%·72.5%)을 함께 넣어야 한다. DIS는 간선 중첩 수치를 [사후]로 붙였는데, 3.8절 정의로는 [측정]이다.
9) 인용 [ref: Pink 2014]와 Jiang 2026의 74.6%(10,348건)는 원문 대조가 필요하다.

사용자가 정할 것.
- RAMEDIS·HMS를 출판 증례로 볼지(RareBench 원 데이터 기술 문서 기준). 이 결정에 따라 4.5절의 세트 정의와 M1 층 이름이 바뀐다.
- 자기 인용률 계산(72.8%, 16.7%)을 커밋해 원고에 넣을지, 빼고 갈지. 넣으면 둘을 함께 넣는다.
- 누수 제외 LOPO 값(+4.03)과 PPR-H4 차이의 CI를 새로 계산해 커밋할지.
-->
