# v82 (N=5 discriminative s=3) Forensic — Random 10/10

- Config: v71 algorithm + v82 KG (PubMed + LLM-aug N=5, scale=3)
- DDXPlus 134K @1=61.90%, MRR=0.7374

## Success cases

### Success #99780 — True: Anemia (`C0002871`)
- Age=77, Sex=F, Rank=**1**, Score=0.1745

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1745 | Anemia (`C0002871`) ← TRUE | 180 | 9 |
| 2 | 0.1274 | Pericarditis (`C0155679`) | 114 | 7 |
| 3 | 0.1051 | Tuberculosis (`C0041327`) | 106 | 8 |
| 4 | 0.0694 | SLE (`C0024141`) | 121 | 6 |
| 5 | 0.0644 | Stable angina (`C0002962`) | 94 | 6 |

---

### Success #73926 — True: Localized edema (`C0013609`)
- Age=68, Sex=M, Rank=**1**, Score=0.2076

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2076 | Localized edema (`C0013609`) ← TRUE | 75 | 4 |
| 2 | 0.0736 | Sarcoidosis (`C0036202`) | 202 | 8 |
| 3 | 0.0728 | SLE (`C0024141`) | 121 | 5 |
| 4 | 0.0721 | Acute pulmonary edema (`C0155919`) | 110 | 5 |
| 5 | 0.0691 | Chagas (`C0041234`) | 186 | 6 |

---

### Success #26580 — True: Atrial fibrillation (`C3264374`)
- Age=8, Sex=F, Rank=**1**, Score=0.3036

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3036 | Atrial fibrillation (`C3264374`) ← TRUE | 77 | 8 |
| 2 | 0.2216 | Possible NSTEMI / STEMI (`C0010072`) | 114 | 9 |
| 3 | 0.1947 | Panic attack (`C0349232`) | 112 | 9 |
| 4 | 0.1884 | Pulmonary embolism (`C0034065`) | 125 | 11 |
| 5 | 0.1576 | PSVT (`C0039240`) | 82 | 6 |

---

### Success #75363 — True: Viral pharyngitis (`C0001344`)
- Age=21, Sex=F, Rank=**1**, Score=0.2421

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2421 | Viral pharyngitis (`C0001344`) ← TRUE | 86 | 7 |
| 2 | 0.1905 | Acute rhinosinusitis (`C0149512`) | 153 | 9 |
| 3 | 0.1606 | URTI (`C0041912`) | 90 | 7 |
| 4 | 0.1599 | Chronic rhinosinusitis (`C0037199`) | 129 | 8 |
| 5 | 0.1483 | Acute laryngitis (`C0001327`) | 116 | 8 |

---

### Success #108247 — True: Pericarditis (`C0155679`)
- Age=24, Sex=M, Rank=**1**, Score=0.2622

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2622 | Pericarditis (`C0155679`) ← TRUE | 114 | 7 |
| 2 | 0.1355 | Epiglottitis (`C0155814`) | 127 | 6 |
| 3 | 0.1012 | Acute pulmonary edema (`C0155919`) | 110 | 6 |
| 4 | 0.0757 | Croup (`C0010380`) | 99 | 4 |
| 5 | 0.0570 | Myocarditis (`C0027059`) | 158 | 5 |

---

### Success #37387 — True: Anemia (`C0002871`)
- Age=37, Sex=M, Rank=**1**, Score=0.1723

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1723 | Anemia (`C0002871`) ← TRUE | 180 | 8 |
| 2 | 0.1102 | Atrial fibrillation (`C3264374`) | 77 | 6 |
| 3 | 0.0979 | Panic attack (`C0349232`) | 112 | 7 |
| 4 | 0.0974 | Tuberculosis (`C0041327`) | 106 | 6 |
| 5 | 0.0888 | PSVT (`C0039240`) | 82 | 4 |

---

### Success #49218 — True: HIV (initial infection) (`C0001175`)
- Age=106, Sex=M, Rank=**1**, Score=0.1742

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1742 | HIV (initial infection) (`C0001175`) ← TRUE | 69 | 9 |
| 2 | 0.1515 | SLE (`C0024141`) | 121 | 13 |
| 3 | 0.1501 | Acute laryngitis (`C0001327`) | 116 | 11 |
| 4 | 0.1471 | Chagas (`C0041234`) | 186 | 15 |
| 5 | 0.1460 | Viral pharyngitis (`C0001344`) | 86 | 8 |

---

### Success #97340 — True: Allergic sinusitis (`C0018621`)
- Age=1, Sex=F, Rank=**1**, Score=0.3174

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3174 | Allergic sinusitis (`C0018621`) ← TRUE | 117 | 8 |
| 2 | 0.2178 | Chronic rhinosinusitis (`C0037199`) | 129 | 6 |
| 3 | 0.2020 | Acute rhinosinusitis (`C0149512`) | 153 | 6 |
| 4 | 0.1909 | Bronchospasm / acute asthma exacerb (`C0004096`) | 190 | 6 |
| 5 | 0.1652 | URTI (`C0041912`) | 90 | 6 |

---

### Success #44920 — True: Pancreatic neoplasm (`C0346647`)
- Age=31, Sex=M, Rank=**1**, Score=0.3204

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3204 | Pancreatic neoplasm (`C0346647`) ← TRUE | 70 | 7 |
| 2 | 0.0981 | Acute laryngitis (`C0001327`) | 116 | 8 |
| 3 | 0.0947 | Anaphylaxis (`C0685898`) | 177 | 7 |
| 4 | 0.0726 | Pulmonary neoplasm (`C0348343`) | 75 | 4 |
| 5 | 0.0693 | Acute otitis media (`C0029882`) | 118 | 7 |

---

### Success #89815 — True: Scombroid food poisoning (`C0275143`)
- Age=6, Sex=M, Rank=**1**, Score=0.2550

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2550 | Scombroid food poisoning (`C0275143`) ← TRUE | 105 | 13 |
| 2 | 0.2111 | Anaphylaxis (`C0685898`) | 177 | 14 |
| 3 | 0.2017 | Chagas (`C0041234`) | 186 | 15 |
| 4 | 0.1707 | Boerhaave (`C0014860`) | 82 | 9 |
| 5 | 0.1337 | Anemia (`C0002871`) | 180 | 13 |

---


## Failure cases

### Failure #32912 — True: Acute COPD exacerbation / infection (`C0340044`)
- Age=55, Sex=F, Rank=**16**, Score=0.1045

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2790 | Bronchiolitis (`C0001311`) | 128 | 8 |
| 2 | 0.2690 | Bronchitis (`C0006277`) | 156 | 7 |
| 3 | 0.2568 | Bronchiectasis (`C0006267`) | 148 | 8 |
| 4 | 0.2534 | Bronchospasm / acute asthma exacerb (`C0004096`) | 190 | 8 |
| 5 | 0.1857 | Croup (`C0010380`) | 99 | 8 |

**Diff:**
- Only-in-top1 (2):
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.269, idf=2.43)
  - `C0017168` Gastroesophageal reflux disease (w_t1=0.088, idf=2.83)
- Only-in-true (0):

---

### Failure #48362 — True: Acute pulmonary edema (`C0155919`)
- Age=44, Sex=M, Rank=**7**, Score=0.1107

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1550 | Larygospasm (`C0023066`) | 88 | 6 |
| 2 | 0.1461 | Localized edema (`C0013609`) | 75 | 4 |
| 3 | 0.1432 | Pulmonary embolism (`C0034065`) | 125 | 10 |
| 4 | 0.1350 | Possible NSTEMI / STEMI (`C0010072`) | 114 | 8 |
| 5 | 0.1274 | Pericarditis (`C0155679`) | 114 | 9 |

**Diff:**
- Only-in-top1 (1):
  - `C0008301` Choking (w_t1=1.023, idf=2.83)
- Only-in-true (3):
  - `C0018801` Heart failure (w_tr=0.371, idf=2.20)
  - `C0030193` Pain (w_tr=0.106, idf=1.45)
  - `C0003086` Ankle (w_tr=0.046, idf=3.81)

---

### Failure #121844 — True: Epiglottitis (`C0155814`)
- Age=76, Sex=F, Rank=**4**, Score=0.1405

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1777 | Viral pharyngitis (`C0001344`) | 86 | 7 |
| 2 | 0.1592 | Croup (`C0010380`) | 99 | 6 |
| 3 | 0.1506 | Myasthenia gravis (`C0026896`) | 123 | 10 |
| 4 | 0.1405 | Epiglottitis (`C0155814`) ← TRUE | 127 | 9 |
| 5 | 0.1185 | Acute laryngitis (`C0001327`) | 116 | 7 |

**Diff:**
- Only-in-top1 (2):
  - `C0040421` Palatine Tonsil (w_t1=1.065, idf=4.22)
  - `C0740170` Does swallow (w_t1=0.454, idf=3.81)
- Only-in-true (4):
  - `C0035203` Respiration (w_tr=0.668, idf=1.82)
  - `C0013404` Dyspnea (w_tr=0.354, idf=1.45)
  - `C0011847` Diabetes (w_tr=0.214, idf=2.71)
  - `C0020538` Hypertensive disease (w_tr=0.044, idf=2.83)

---

### Failure #109072 — True: Influenza (`C0021400`)
- Age=47, Sex=F, Rank=**10**, Score=0.0785

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2043 | Ebola (`C0282687`) | 111 | 7 |
| 2 | 0.1619 | Acute laryngitis (`C0001327`) | 116 | 10 |
| 3 | 0.1110 | Tuberculosis (`C0041327`) | 106 | 7 |
| 4 | 0.1087 | Sarcoidosis (`C0036202`) | 202 | 13 |
| 5 | 0.1020 | Acute otitis media (`C0029882`) | 118 | 8 |

**Diff:**
- Only-in-top1 (1):
  - `C0015230` Exanthema (w_t1=0.610, idf=2.83)
- Only-in-true (2):
  - `C0237849` Peeling of skin (w_tr=0.146, idf=3.53)
  - `C0041834` Erythema (w_tr=0.036, idf=2.71)

---

### Failure #2125 — True: Myocarditis (`C0027059`)
- Age=0, Sex=M, Rank=**15**, Score=0.0607

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2023 | Pericarditis (`C0155679`) | 114 | 9 |
| 2 | 0.1561 | Ebola (`C0282687`) | 111 | 7 |
| 3 | 0.1089 | Croup (`C0010380`) | 99 | 5 |
| 4 | 0.1079 | Acute pulmonary edema (`C0155919`) | 110 | 9 |
| 5 | 0.1073 | Epiglottitis (`C0155814`) | 127 | 7 |

**Diff:**
- Only-in-top1 (2):
  - `C0277814` Sitting position (w_t1=0.862, idf=4.22)
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.168, idf=2.43)
- Only-in-true (0):

---

### Failure #18688 — True: Sarcoidosis (`C0036202`)
- Age=54, Sex=F, Rank=**10**, Score=0.0340

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.0919 | Acute dystonic reactions (`C0236832`) | 94 | 5 |
| 2 | 0.0845 | Anaphylaxis (`C0685898`) | 177 | 7 |
| 3 | 0.0667 | Larygospasm (`C0023066`) | 88 | 4 |
| 4 | 0.0639 | Acute laryngitis (`C0001327`) | 116 | 7 |
| 5 | 0.0586 | Epiglottitis (`C0155814`) | 127 | 5 |

**Diff:**
- Only-in-top1 (2):
  - `C0026820` Muscle Contraction (w_t1=1.036, idf=3.81)
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.070, idf=2.43)
- Only-in-true (7):
  - `C0221198` Lesion (w_tr=0.625, idf=2.83)
  - `C0015230` Exanthema (w_tr=0.569, idf=2.83)
  - `C0013404` Dyspnea (w_tr=0.444, idf=1.45)
  - `C0085624` Burning sensation (w_tr=0.330, idf=3.30)
  - `C1515974` Anatomic Site (w_tr=0.108, idf=1.00)

---

### Failure #5322 — True: Myocarditis (`C0027059`)
- Age=78, Sex=F, Rank=**22**, Score=0.0060

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1640 | Croup (`C0010380`) | 99 | 6 |
| 2 | 0.1513 | Pericarditis (`C0155679`) | 114 | 7 |
| 3 | 0.1131 | Epiglottitis (`C0155814`) | 127 | 6 |
| 4 | 0.0867 | Acute pulmonary edema (`C0155919`) | 110 | 7 |
| 5 | 0.0716 | Bronchiolitis (`C0001311`) | 128 | 5 |

**Diff:**
- Only-in-top1 (2):
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.415, idf=2.43)
  - `C5236002` Increased (finding) (w_t1=0.362, idf=4.22)
- Only-in-true (1):
  - `C0030193` Pain (w_tr=0.133, idf=1.45)

---

### Failure #94960 — True: Influenza (`C0021400`)
- Age=9, Sex=M, Rank=**24**, Score=0.0216

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1563 | Acute laryngitis (`C0001327`) | 116 | 10 |
| 2 | 0.1414 | Ebola (`C0282687`) | 111 | 7 |
| 3 | 0.1404 | Viral pharyngitis (`C0001344`) | 86 | 7 |
| 4 | 0.1388 | HIV (initial infection) (`C0001175`) | 69 | 8 |
| 5 | 0.1223 | Acute otitis media (`C0029882`) | 118 | 8 |

**Diff:**
- Only-in-top1 (3):
  - `C0221198` Lesion (w_t1=0.477, idf=2.83)
  - `C0033774` Pruritus (w_t1=0.378, idf=3.30)
  - `C0027530` Neck (w_t1=0.241, idf=2.20)
- Only-in-true (1):
  - `C0003123` Anorexia (w_tr=0.450, idf=2.61)

---

### Failure #114203 — True: Acute COPD exacerbation / infection (`C0340044`)
- Age=41, Sex=F, Rank=**10**, Score=0.0651

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3293 | Bronchitis (`C0006277`) | 156 | 5 |
| 2 | 0.2193 | Bronchiectasis (`C0006267`) | 148 | 5 |
| 3 | 0.1521 | Bronchiolitis (`C0001311`) | 128 | 4 |
| 4 | 0.1365 | Bronchospasm / acute asthma exacerb (`C0004096`) | 190 | 4 |
| 5 | 0.1022 | Acute laryngitis (`C0001327`) | 116 | 3 |

**Diff:**
- Only-in-top1 (2):
  - `C0038056` Sputum (w_t1=1.148, idf=3.12)
  - `C0017168` Gastroesophageal reflux disease (w_t1=0.716, idf=2.83)
- Only-in-true (0):

---

### Failure #56667 — True: Influenza (`C0021400`)
- Age=19, Sex=F, Rank=**8**, Score=0.0984

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2011 | Acute laryngitis (`C0001327`) | 116 | 12 |
| 2 | 0.1406 | Allergic sinusitis (`C0018621`) | 117 | 8 |
| 3 | 0.1373 | HIV (initial infection) (`C0001175`) | 69 | 7 |
| 4 | 0.1305 | Ebola (`C0282687`) | 111 | 6 |
| 5 | 0.1280 | Acute rhinosinusitis (`C0149512`) | 153 | 10 |

**Diff:**
- Only-in-top1 (3):
  - `C0221198` Lesion (w_t1=0.477, idf=2.83)
  - `C0033774` Pruritus (w_t1=0.378, idf=3.30)
  - `C0027530` Neck (w_t1=0.241, idf=2.20)
- Only-in-true (0):

---


## Confusion pairs
- Influenza → Acute laryngitis (2x)
- Acute COPD exacerbation / infection → Bronchiolitis (1x)
- Acute pulmonary edema → Larygospasm (1x)
- Epiglottitis → Viral pharyngitis (1x)
- Influenza → Ebola (1x)
- Myocarditis → Pericarditis (1x)
- Sarcoidosis → Acute dystonic reactions (1x)
- Myocarditis → Croup (1x)
- Acute COPD exacerbation / infection → Bronchitis (1x)
