# v84 (multi-prompt merge s=3) Forensic — Random 10/10

- Config: v71 algorithm + v84 KG (generic+discriminative merge scale=3)
- DDXPlus 134K @1=62.53%, MRR=0.7426

## Success cases

### Success #44834 — True: Viral pharyngitis (`C0001344`)
- Age=50, Sex=F, Rank=**1**, Score=0.2295

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2295 | Viral pharyngitis (`C0001344`) ← TRUE | 98 | 7 |
| 2 | 0.1907 | Acute rhinosinusitis (`C0149512`) | 177 | 9 |
| 3 | 0.1779 | URTI (`C0041912`) | 117 | 8 |
| 4 | 0.1715 | Chronic rhinosinusitis (`C0037199`) | 145 | 8 |
| 5 | 0.1240 | Acute laryngitis (`C0001327`) | 139 | 7 |

---

### Success #25849 — True: Localized edema (`C0013609`)
- Age=26, Sex=M, Rank=**1**, Score=0.2584

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2584 | Localized edema (`C0013609`) ← TRUE | 106 | 8 |
| 2 | 0.0838 | Atrial fibrillation (`C3264374`) | 111 | 5 |
| 3 | 0.0561 | Bronchitis (`C0006277`) | 176 | 7 |
| 4 | 0.0495 | Inguinal hernia (`C0019294`) | 89 | 4 |
| 5 | 0.0430 | Acute pulmonary edema (`C0155919`) | 135 | 5 |

---

### Success #45274 — True: Viral pharyngitis (`C0001344`)
- Age=24, Sex=F, Rank=**1**, Score=0.2095

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2095 | Viral pharyngitis (`C0001344`) ← TRUE | 98 | 6 |
| 2 | 0.1767 | Acute rhinosinusitis (`C0149512`) | 177 | 8 |
| 3 | 0.1518 | Chronic rhinosinusitis (`C0037199`) | 145 | 7 |
| 4 | 0.1480 | URTI (`C0041912`) | 117 | 7 |
| 5 | 0.1097 | Acute laryngitis (`C0001327`) | 139 | 6 |

---

### Success #34026 — True: Pancreatic neoplasm (`C0346647`)
- Age=28, Sex=M, Rank=**1**, Score=0.2449

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2449 | Pancreatic neoplasm (`C0346647`) ← TRUE | 94 | 4 |
| 2 | 0.1072 | Boerhaave (`C0014860`) | 115 | 4 |
| 3 | 0.1038 | Chagas (`C0041234`) | 216 | 9 |
| 4 | 0.1019 | Scombroid food poisoning (`C0275143`) | 122 | 5 |
| 5 | 0.0995 | Anaphylaxis (`C0685898`) | 202 | 5 |

---

### Success #47750 — True: Panic attack (`C0349232`)
- Age=64, Sex=M, Rank=**1**, Score=0.3255

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3255 | Panic attack (`C0349232`) ← TRUE | 132 | 17 |
| 2 | 0.1922 | Scombroid food poisoning (`C0275143`) | 122 | 16 |
| 3 | 0.1837 | Anaphylaxis (`C0685898`) | 202 | 21 |
| 4 | 0.1781 | Larygospasm (`C0023066`) | 127 | 9 |
| 5 | 0.1781 | Possible NSTEMI / STEMI (`C0010072`) | 135 | 16 |

---

### Success #37387 — True: Anemia (`C0002871`)
- Age=37, Sex=M, Rank=**1**, Score=0.1848

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1848 | Anemia (`C0002871`) ← TRUE | 233 | 8 |
| 2 | 0.1164 | Atrial fibrillation (`C3264374`) | 111 | 6 |
| 3 | 0.1070 | Tuberculosis (`C0041327`) | 127 | 6 |
| 4 | 0.1040 | PSVT (`C0039240`) | 101 | 4 |
| 5 | 0.0882 | Panic attack (`C0349232`) | 132 | 7 |

---

### Success #78951 — True: Anemia (`C0002871`)
- Age=54, Sex=F, Rank=**1**, Score=0.1734

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1734 | Anemia (`C0002871`) ← TRUE | 233 | 8 |
| 2 | 0.0986 | Pericarditis (`C0155679`) | 137 | 5 |
| 3 | 0.0845 | Atrial fibrillation (`C3264374`) | 111 | 6 |
| 4 | 0.0803 | PSVT (`C0039240`) | 101 | 4 |
| 5 | 0.0634 | Anaphylaxis (`C0685898`) | 202 | 5 |

---

### Success #74235 — True: Chronic rhinosinusitis (`C0037199`)
- Age=61, Sex=M, Rank=**1**, Score=0.1976

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1976 | Chronic rhinosinusitis (`C0037199`) ← TRUE | 145 | 6 |
| 2 | 0.1953 | Acute rhinosinusitis (`C0149512`) | 177 | 6 |
| 3 | 0.0968 | Allergic sinusitis (`C0018621`) | 148 | 4 |
| 4 | 0.0914 | GERD (`C0017168`) | 177 | 6 |
| 5 | 0.0666 | Bronchospasm / acute asthma exacerb (`C0004096`) | 225 | 4 |

---

### Success #58801 — True: Bronchospasm / acute asthma exacerbation (`C0004096`)
- Age=25, Sex=F, Rank=**1**, Score=0.3011

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3011 | Bronchospasm / acute asthma exacerb (`C0004096`) ← TRUE | 225 | 9 |
| 2 | 0.2899 | Bronchiolitis (`C0001311`) | 165 | 8 |
| 3 | 0.2141 | Acute COPD exacerbation / infection (`C0340044`) | 104 | 5 |
| 4 | 0.1677 | Bronchitis (`C0006277`) | 176 | 7 |
| 5 | 0.1595 | Bronchiectasis (`C0006267`) | 160 | 8 |

---

### Success #113582 — True: Viral pharyngitis (`C0001344`)
- Age=12, Sex=M, Rank=**1**, Score=0.2282

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2282 | Viral pharyngitis (`C0001344`) ← TRUE | 98 | 6 |
| 2 | 0.1631 | URTI (`C0041912`) | 117 | 7 |
| 3 | 0.1331 | Chronic rhinosinusitis (`C0037199`) | 145 | 6 |
| 4 | 0.1324 | Acute rhinosinusitis (`C0149512`) | 177 | 7 |
| 5 | 0.1236 | Acute laryngitis (`C0001327`) | 139 | 6 |

---


## Failure cases

### Failure #105261 — True: HIV (initial infection) (`C0001175`)
- Age=7, Sex=M, Rank=**4**, Score=0.1573

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2148 | Chagas (`C0041234`) | 216 | 17 |
| 2 | 0.1847 | Ebola (`C0282687`) | 122 | 9 |
| 3 | 0.1655 | Anaphylaxis (`C0685898`) | 202 | 9 |
| 4 | 0.1573 | HIV (initial infection) (`C0001175`) ← TRUE | 93 | 9 |
| 5 | 0.1434 | Boerhaave (`C0014860`) | 115 | 8 |

**Diff:**
- Only-in-top1 (8):
  - `C0042963` Vomiting (w_t1=0.762, idf=2.35)
  - `C0027497` Nausea (w_t1=0.503, idf=2.27)
  - `C0036916` Sexually Transmitted Diseases (w_t1=0.309, idf=2.97)
  - `C0426642` Frequency of bowel action (w_t1=0.308, idf=4.22)
  - `C0030232`  (w_t1=0.253, idf=3.12)
- Only-in-true (0):

---

### Failure #69817 — True: Acute pulmonary edema (`C0155919`)
- Age=59, Sex=M, Rank=**9**, Score=0.1130

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1620 | Localized edema (`C0013609`) | 106 | 6 |
| 2 | 0.1568 | Pericarditis (`C0155679`) | 137 | 11 |
| 3 | 0.1544 | Stable angina (`C0002962`) | 118 | 9 |
| 4 | 0.1520 | Pulmonary embolism (`C0034065`) | 152 | 12 |
| 5 | 0.1408 | Pneumonia (`C0694504`) | 188 | 12 |

**Diff:**
- Only-in-top1 (1):
  - `C0040184` Bone structure of tibia (w_t1=0.368, idf=4.22)
- Only-in-true (5):
  - `C0013404` Dyspnea (w_tr=0.462, idf=1.42)
  - `C0817096` Chest (w_tr=0.312, idf=1.69)
  - `C0035203` Respiration (w_tr=0.220, idf=1.82)
  - `C1299586` Has difficulty doing (qualifier value) (w_tr=0.033, idf=2.43)
  - `C0015672` Fatigue (w_tr=0.014, idf=1.54)

---

### Failure #74212 — True: Acute pulmonary edema (`C0155919`)
- Age=55, Sex=F, Rank=**6**, Score=0.1298

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1696 | Localized edema (`C0013609`) | 106 | 7 |
| 2 | 0.1689 | Stable angina (`C0002962`) | 118 | 10 |
| 3 | 0.1603 | Possible NSTEMI / STEMI (`C0010072`) | 135 | 12 |
| 4 | 0.1524 | Pericarditis (`C0155679`) | 137 | 12 |
| 5 | 0.1452 | Pneumonia (`C0694504`) | 188 | 13 |

**Diff:**
- Only-in-top1 (1):
  - `C0040184` Bone structure of tibia (w_t1=0.368, idf=4.22)
- Only-in-true (5):
  - `C0013404` Dyspnea (w_tr=0.462, idf=1.42)
  - `C0817096` Chest (w_tr=0.312, idf=1.69)
  - `C0035203` Respiration (w_tr=0.220, idf=1.82)
  - `C1299586` Has difficulty doing (qualifier value) (w_tr=0.033, idf=2.43)
  - `C0015672` Fatigue (w_tr=0.014, idf=1.54)

---

### Failure #9883 — True: Myocarditis (`C0027059`)
- Age=23, Sex=M, Rank=**6**, Score=0.1578

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2103 | Pericarditis (`C0155679`) | 137 | 7 |
| 2 | 0.1852 | PSVT (`C0039240`) | 101 | 4 |
| 3 | 0.1685 | Spontaneous pneumothorax (`C0032326`) | 155 | 7 |
| 4 | 0.1624 | Stable angina (`C0002962`) | 118 | 6 |
| 5 | 0.1604 | Unstable angina (`C0002965`) | 135 | 5 |

**Diff:**
- Only-in-top1 (1):
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.168, idf=2.43)
- Only-in-true (0):

---

### Failure #34013 — True: HIV (initial infection) (`C0001175`)
- Age=58, Sex=F, Rank=**3**, Score=0.1438

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2118 | Ebola (`C0282687`) | 122 | 9 |
| 2 | 0.1449 | Acute laryngitis (`C0001327`) | 139 | 12 |
| 3 | 0.1438 | HIV (initial infection) (`C0001175`) ← TRUE | 93 | 8 |
| 4 | 0.1394 | Chagas (`C0041234`) | 216 | 15 |
| 5 | 0.1318 | Anemia (`C0002871`) | 233 | 13 |

**Diff:**
- Only-in-top1 (2):
  - `C0042963` Vomiting (w_t1=0.426, idf=2.35)
  - `C0027497` Nausea (w_t1=0.241, idf=2.27)
- Only-in-true (1):
  - `C0221198` Lesion (w_tr=0.074, idf=2.83)

---

### Failure #114658 — True: Acute pulmonary edema (`C0155919`)
- Age=33, Sex=M, Rank=**10**, Score=0.0933

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1650 | Localized edema (`C0013609`) | 106 | 6 |
| 2 | 0.1543 | Possible NSTEMI / STEMI (`C0010072`) | 135 | 11 |
| 3 | 0.1423 | Stable angina (`C0002962`) | 118 | 9 |
| 4 | 0.1117 | Pulmonary embolism (`C0034065`) | 152 | 11 |
| 5 | 0.1108 | Pneumonia (`C0694504`) | 188 | 12 |

**Diff:**
- Only-in-top1 (1):
  - `C0040184` Bone structure of tibia (w_t1=0.368, idf=4.22)
- Only-in-true (5):
  - `C0013404` Dyspnea (w_tr=0.462, idf=1.42)
  - `C0817096` Chest (w_tr=0.312, idf=1.69)
  - `C0035203` Respiration (w_tr=0.220, idf=1.82)
  - `C1299586` Has difficulty doing (qualifier value) (w_tr=0.033, idf=2.43)
  - `C0015672` Fatigue (w_tr=0.014, idf=1.54)

---

### Failure #6167 — True: Bronchitis (`C0006277`)
- Age=20, Sex=M, Rank=**9**, Score=0.1086

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1893 | Bronchiolitis (`C0001311`) | 165 | 8 |
| 2 | 0.1842 | URTI (`C0041912`) | 117 | 9 |
| 3 | 0.1664 | Acute laryngitis (`C0001327`) | 139 | 9 |
| 4 | 0.1404 | Chronic rhinosinusitis (`C0037199`) | 145 | 7 |
| 5 | 0.1393 | GERD (`C0017168`) | 177 | 7 |

**Diff:**
- Only-in-top1 (1):
  - `C0027424` Nasal congestion (finding) (w_t1=0.322, idf=2.61)
- Only-in-true (1):
  - `C0030193` Pain (w_tr=0.093, idf=1.42)

---

### Failure #41388 — True: Possible NSTEMI / STEMI (`C0010072`)
- Age=54, Sex=F, Rank=**3**, Score=0.1692

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2743 | Unstable angina (`C0002965`) | 135 | 12 |
| 2 | 0.1706 | Stable angina (`C0002962`) | 118 | 11 |
| 3 | 0.1692 | Possible NSTEMI / STEMI (`C0010072`) ← TRUE | 135 | 9 |
| 4 | 0.1579 | Pericarditis (`C0155679`) | 137 | 11 |
| 5 | 0.1201 | Pancreatic neoplasm (`C0346647`) | 94 | 5 |

**Diff:**
- Only-in-top1 (5):
  - `C0020443` Hypercholesterolemia (w_t1=0.836, idf=4.22)
  - `C0019693` HIV Infections (w_t1=0.648, idf=2.71)
  - `C0011847` Diabetes (w_t1=0.631, idf=2.71)
  - `C0425043` Death of relative (w_t1=0.055, idf=1.00)
  - `C0037004` Shoulder (w_t1=0.044, idf=3.53)
- Only-in-true (2):
  - `C0035203` Respiration (w_tr=0.198, idf=1.82)
  - `C0234254` Radiating pain (w_tr=0.095, idf=1.00)

---

### Failure #23787 — True: Sarcoidosis (`C0036202`)
- Age=66, Sex=F, Rank=**8**, Score=0.0114

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.0670 | Anaphylaxis (`C0685898`) | 202 | 3 |
| 2 | 0.0428 | Allergic sinusitis (`C0018621`) | 148 | 3 |
| 3 | 0.0394 | Scombroid food poisoning (`C0275143`) | 122 | 3 |
| 4 | 0.0388 | Acute laryngitis (`C0001327`) | 139 | 4 |
| 5 | 0.0317 | Boerhaave (`C0014860`) | 115 | 2 |

**Diff:**
- Only-in-top1 (0):
- Only-in-true (4):
  - `C0221198` Lesion (w_tr=0.681, idf=2.83)
  - `C1515974` Anatomic Site (w_tr=0.108, idf=1.00)
  - `C0237849` Peeling of skin (w_tr=0.087, idf=3.53)
  - `C0497406` Overweight (w_tr=0.040, idf=3.12)

---

### Failure #109366 — True: Acute pulmonary edema (`C0155919`)
- Age=49, Sex=F, Rank=**3**, Score=0.1610

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1895 | Localized edema (`C0013609`) | 106 | 7 |
| 2 | 0.1627 | Possible NSTEMI / STEMI (`C0010072`) | 135 | 12 |
| 3 | 0.1610 | Acute pulmonary edema (`C0155919`) ← TRUE | 135 | 12 |
| 4 | 0.1481 | Pulmonary embolism (`C0034065`) | 152 | 12 |
| 5 | 0.1426 | Sarcoidosis (`C0036202`) | 220 | 16 |

**Diff:**
- Only-in-top1 (1):
  - `C0040184` Bone structure of tibia (w_t1=0.368, idf=4.22)
- Only-in-true (6):
  - `C0013404` Dyspnea (w_tr=0.462, idf=1.42)
  - `C0817096` Chest (w_tr=0.312, idf=1.69)
  - `C0035203` Respiration (w_tr=0.220, idf=1.82)
  - `C1457887` Symptoms (w_tr=0.115, idf=1.42)
  - `C1299586` Has difficulty doing (qualifier value) (w_tr=0.033, idf=2.43)

---


## Confusion pairs
- Acute pulmonary edema → Localized edema (4x)
- HIV (initial infection) → Chagas (1x)
- Myocarditis → Pericarditis (1x)
- HIV (initial infection) → Ebola (1x)
- Bronchitis → Bronchiolitis (1x)
- Possible NSTEMI / STEMI → Unstable angina (1x)
- Sarcoidosis → Anaphylaxis (1x)
