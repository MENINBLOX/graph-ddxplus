# v103 혼동쌍 forensic — discrimination 병목

KG: `pilot/data/cache/v103deep120m_kg.pkl`  /  patients scored: 2975

- @1 정답: 853 (28.7%)
- top-10엔 있으나 #1 실패 (혼동): 1372 (46.1%)
- top-10 밖: 750 (25.2%)

**개선 여지 = 혼동 1372건 (top-10 안에 정답이 이미 있음).**

## 가장 잦은 혼동쌍 (GT → 잘못 뽑힌 #1), top 15

| # | GT (정답) | 오답 #1 | 건수 |
|---|---|---|---|
| 1 | Anaphylaxis | Scombroid food poisoning | 82 |
| 2 | Pneumonia | Viral pharyngitis | 66 |
| 3 | Viral pharyngitis | URTI | 62 |
| 4 | Viral pharyngitis | Cluster headache | 59 |
| 5 | Viral pharyngitis | Unstable angina | 57 |
| 6 | Acute laryngitis | Viral pharyngitis | 56 |
| 7 | GERD | Unstable angina | 48 |
| 8 | Localized edema | Tuberculosis | 48 |
| 9 | PSVT | Panic attack | 48 |
| 10 | Acute dystonic reactions | Scombroid food poisoning | 47 |
| 11 | Acute COPD exacerbation / infection | Bronchospasm / acute asthma exacerbation | 40 |
| 12 | SLE | Viral pharyngitis | 34 |
| 13 | Unstable angina | Stable angina | 30 |
| 14 | Acute pulmonary edema | Stable angina | 29 |
| 15 | Pancreatic neoplasm | Scombroid food poisoning | 27 |

## 왜 안 갈렸나 — 상위 6 쌍 evidence 분석

### Anaphylaxis → Scombroid food poisoning  (82건)
- 공유(비변별) evidence 4개: C0020517, C0016470, C0015230, C0013404
- GT-only evidence 없음 ⚠️ (환자 증거가 GT 프로필에 안 잡힘 = IE 부족)
- 오답만 가진 환자 evidence 7개(오답으로 끌림): C0033774, C0011991, C0027497, C0042963, C0041834, C0012833, C0043144

### Pneumonia → Viral pharyngitis  (66건)
- 공유(비변별) evidence 6개: C0032285, C0010200, C0015967, C1260880, C0015672, C0013404
- GT만 가진 환자 evidence 2개(정답 쪽이어야): C0004096, C0003123
- 오답만 가진 환자 evidence 7개(오답으로 끌림): C0015230, C0237849, C0030193, C1457887, C0221198, C0041834, C0027424

### Viral pharyngitis → URTI  (62건)
- 공유(비변별) evidence 4개: C0010200, C1260880, C0015967, C0027424
- GT만 가진 환자 evidence 2개(정답 쪽이어야): C0030193, C1457887
- 오답-only evidence 없음

### Viral pharyngitis → Cluster headache  (59건)
- 공유(비변별) evidence 4개: C1260880, C0030193, C1457887, C0027424
- GT만 가진 환자 evidence 1개(정답 쪽이어야): C0010200
- 오답-only evidence 없음

### Viral pharyngitis → Unstable angina  (57건)
- 공유(비변별) evidence 2개: C0015967, C0030193
- GT만 가진 환자 evidence 2개(정답 쪽이어야): C0010200, C1457887
- 오답만 가진 환자 evidence 1개(오답으로 끌림): C0085624

### Acute laryngitis → Viral pharyngitis  (56건)
- 공유(비변별) evidence 4개: C0019825, C0010200, C0015967, C0030193
- GT-only evidence 없음 ⚠️ (환자 증거가 GT 프로필에 안 잡힘 = IE 부족)
- 오답만 가진 환자 evidence 1개(오답으로 끌림): C0009443
