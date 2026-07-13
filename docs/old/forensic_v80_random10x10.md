# v80 (scale=5) Forensic — Random 10/10

- Config: v71 algorithm + v80 KG (PubMed + LLM-aug, scale=5)
- DDXPlus 134K @1=61.10%, MRR=0.7217

## Success cases

### Success #26342 — True: Pulmonary neoplasm (`C0348343`)
- Age=35, Sex=F, Rank=**1**, Score=0.2889

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2889 | Pulmonary neoplasm (`C0348343`) ← TRUE | 68 | 8 |
| 2 | 0.1301 | Bronchiectasis (`C0006267`) | 137 | 9 |
| 3 | 0.1281 | Tuberculosis (`C0041327`) | 95 | 7 |
| 4 | 0.1033 | Pneumonia (`C0694504`) | 150 | 8 |
| 5 | 0.0985 | Bronchitis (`C0006277`) | 134 | 8 |

---

### Success #6863 — True: Viral pharyngitis (`C0001344`)
- Age=24, Sex=M, Rank=**1**, Score=0.2092

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2092 | Viral pharyngitis (`C0001344`) ← TRUE | 76 | 7 |
| 2 | 0.1823 | Acute rhinosinusitis (`C0149512`) | 132 | 9 |
| 3 | 0.1476 | URTI (`C0041912`) | 70 | 7 |
| 4 | 0.1394 | Chronic rhinosinusitis (`C0037199`) | 110 | 7 |
| 5 | 0.1183 | Acute laryngitis (`C0001327`) | 102 | 7 |

---

### Success #93742 — True: Allergic sinusitis (`C0018621`)
- Age=48, Sex=M, Rank=**1**, Score=0.3477

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3477 | Allergic sinusitis (`C0018621`) ← TRUE | 79 | 9 |
| 2 | 0.2166 | Bronchospasm / acute asthma exacerb (`C0004096`) | 171 | 8 |
| 3 | 0.2156 | Chronic rhinosinusitis (`C0037199`) | 110 | 7 |
| 4 | 0.1561 | Acute rhinosinusitis (`C0149512`) | 132 | 7 |
| 5 | 0.1468 | URTI (`C0041912`) | 70 | 6 |

---

### Success #5704 — True: Allergic sinusitis (`C0018621`)
- Age=17, Sex=F, Rank=**1**, Score=0.3430

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3430 | Allergic sinusitis (`C0018621`) ← TRUE | 79 | 8 |
| 2 | 0.2366 | Chronic rhinosinusitis (`C0037199`) | 110 | 6 |
| 3 | 0.2222 | Acute rhinosinusitis (`C0149512`) | 132 | 6 |
| 4 | 0.2204 | URTI (`C0041912`) | 70 | 7 |
| 5 | 0.1943 | Bronchospasm / acute asthma exacerb (`C0004096`) | 171 | 6 |

---

### Success #53582 — True: GERD (`C0017168`)
- Age=18, Sex=F, Rank=**1**, Score=0.0854

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.0854 | GERD (`C0017168`) ← TRUE | 126 | 5 |
| 2 | 0.0653 | Bronchiectasis (`C0006267`) | 137 | 5 |
| 3 | 0.0585 | Stable angina (`C0002962`) | 74 | 5 |
| 4 | 0.0470 | Acute rhinosinusitis (`C0149512`) | 132 | 5 |
| 5 | 0.0403 | Chronic rhinosinusitis (`C0037199`) | 110 | 4 |

---

### Success #4202 — True: Localized edema (`C0013609`)
- Age=35, Sex=M, Rank=**1**, Score=0.1827

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1827 | Localized edema (`C0013609`) ← TRUE | 77 | 6 |
| 2 | 0.0548 | Inguinal hernia (`C0019294`) | 62 | 3 |
| 3 | 0.0483 | Sarcoidosis (`C0036202`) | 183 | 9 |
| 4 | 0.0408 | Possible NSTEMI / STEMI (`C0010072`) | 94 | 5 |
| 5 | 0.0363 | Chagas (`C0041234`) | 155 | 7 |

---

### Success #67990 — True: Viral pharyngitis (`C0001344`)
- Age=17, Sex=M, Rank=**1**, Score=0.2037

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2037 | Viral pharyngitis (`C0001344`) ← TRUE | 76 | 6 |
| 2 | 0.1555 | Acute rhinosinusitis (`C0149512`) | 132 | 7 |
| 3 | 0.1417 | URTI (`C0041912`) | 70 | 6 |
| 4 | 0.1350 | Chronic rhinosinusitis (`C0037199`) | 110 | 6 |
| 5 | 0.1163 | Acute laryngitis (`C0001327`) | 102 | 6 |

---

### Success #94186 — True: URTI (`C0041912`)
- Age=58, Sex=F, Rank=**1**, Score=0.2473

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2473 | URTI (`C0041912`) ← TRUE | 70 | 9 |
| 2 | 0.1701 | Influenza (`C0021400`) | 170 | 10 |
| 3 | 0.1539 | Chronic rhinosinusitis (`C0037199`) | 110 | 8 |
| 4 | 0.1529 | Ebola (`C0282687`) | 96 | 7 |
| 5 | 0.1471 | Viral pharyngitis (`C0001344`) | 76 | 8 |

---

### Success #41530 — True: Unstable angina (`C0002965`)
- Age=56, Sex=M, Rank=**1**, Score=0.2384

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2384 | Unstable angina (`C0002965`) ← TRUE | 96 | 12 |
| 2 | 0.1321 | Stable angina (`C0002962`) | 74 | 10 |
| 3 | 0.1215 | Atrial fibrillation (`C3264374`) | 66 | 7 |
| 4 | 0.1062 | Possible NSTEMI / STEMI (`C0010072`) | 94 | 8 |
| 5 | 0.0761 | Pancreatic neoplasm (`C0346647`) | 59 | 6 |

---

### Success #21199 — True: Guillain-Barré syndrome (`C0018378`)
- Age=52, Sex=F, Rank=**1**, Score=0.2976

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2976 | Guillain-Barré syndrome (`C0018378`) ← TRUE | 127 | 14 |
| 2 | 0.1763 | Myasthenia gravis (`C0026896`) | 117 | 11 |
| 3 | 0.1739 | Localized edema (`C0013609`) | 77 | 7 |
| 4 | 0.1177 | Anaphylaxis (`C0685898`) | 170 | 9 |
| 5 | 0.0916 | Sarcoidosis (`C0036202`) | 183 | 11 |

---


## Failure cases

### Failure #96088 — True: HIV (initial infection) (`C0001175`)
- Age=58, Sex=F, Rank=**11**, Score=0.0771

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1658 | Boerhaave (`C0014860`) | 75 | 7 |
| 2 | 0.1593 | Viral pharyngitis (`C0001344`) | 76 | 7 |
| 3 | 0.1121 | Acute laryngitis (`C0001327`) | 102 | 9 |
| 4 | 0.1089 | Anaphylaxis (`C0685898`) | 170 | 7 |
| 5 | 0.1065 | Ebola (`C0282687`) | 96 | 6 |

**Diff:**
- Only-in-top1 (4):
  - `C0027530` Neck (w_t1=0.575, idf=2.27)
  - `C0221198` Lesion (w_t1=0.519, idf=2.83)
  - `C0426642` Frequency of bowel action (w_t1=0.466, idf=4.22)
  - `C0030232`  (w_t1=0.237, idf=3.12)
- Only-in-true (3):
  - `C0031350` Pharyngitis (w_tr=0.230, idf=2.27)
  - `C0015230` Exanthema (w_tr=0.151, idf=2.83)
  - `C0018670` Head (w_tr=0.073, idf=1.87)

---

### Failure #117497 — True: Chagas (`C0041234`)
- Age=65, Sex=M, Rank=**4**, Score=0.1379

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1604 | Anaphylaxis (`C0685898`) | 170 | 9 |
| 2 | 0.1494 | Croup (`C0010380`) | 86 | 7 |
| 3 | 0.1404 | Epiglottitis (`C0155814`) | 116 | 9 |
| 4 | 0.1379 | Chagas (`C0041234`) ← TRUE | 155 | 11 |
| 5 | 0.1190 | Larygospasm (`C0023066`) | 83 | 7 |

**Diff:**
- Only-in-top1 (2):
  - `C0026821` Muscle Cramp (w_t1=0.632, idf=3.30)
  - `C0035203` Respiration (w_t1=0.205, idf=1.82)
- Only-in-true (4):
  - `C0003123` Anorexia (w_tr=0.513, idf=2.61)
  - `C0015967` Fever (w_tr=0.436, idf=1.69)
  - `C0024204` lymph nodes (w_tr=0.436, idf=2.83)
  - `C1515974` Anatomic Site (w_tr=0.108, idf=1.00)

---

### Failure #98353 — True: Larygospasm (`C0023066`)
- Age=34, Sex=F, Rank=**15**, Score=0.0044

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1290 | Acute laryngitis (`C0001327`) | 102 | 3 |
| 2 | 0.1254 | Bronchitis (`C0006277`) | 134 | 3 |
| 3 | 0.0804 | Bronchospasm / acute asthma exacerb (`C0004096`) | 171 | 3 |
| 4 | 0.0797 | Chagas (`C0041234`) | 155 | 2 |
| 5 | 0.0749 | Croup (`C0010380`) | 86 | 3 |

**Diff:**
- Only-in-top1 (1):
  - `C0009443` Common Cold (w_t1=0.325, idf=2.14)
- Only-in-true (0):

---

### Failure #76197 — True: Inguinal hernia (`C0019294`)
- Age=11, Sex=M, Rank=**9**, Score=0.0502

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1168 | Anaphylaxis (`C0685898`) | 170 | 7 |
| 2 | 0.1023 | Acute rhinosinusitis (`C0149512`) | 132 | 6 |
| 3 | 0.0930 | Boerhaave (`C0014860`) | 75 | 4 |
| 4 | 0.0774 | Allergic sinusitis (`C0018621`) | 79 | 6 |
| 5 | 0.0721 | Acute laryngitis (`C0001327`) | 102 | 6 |

**Diff:**
- Only-in-top1 (4):
  - `C0033774` Pruritus (w_t1=0.886, idf=3.30)
  - `C0015230` Exanthema (w_t1=0.786, idf=2.83)
  - `C0030232`  (w_t1=0.432, idf=3.12)
  - `C0041834` Erythema (w_t1=0.303, idf=2.83)
- Only-in-true (1):
  - `C1515974` Anatomic Site (w_tr=0.108, idf=1.00)

---

### Failure #7827 — True: HIV (initial infection) (`C0001175`)
- Age=38, Sex=F, Rank=**3**, Score=0.1719

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1985 | Ebola (`C0282687`) | 96 | 10 |
| 2 | 0.1952 | Chagas (`C0041234`) | 155 | 18 |
| 3 | 0.1719 | HIV (initial infection) (`C0001175`) ← TRUE | 57 | 10 |
| 4 | 0.1635 | Anaphylaxis (`C0685898`) | 170 | 11 |
| 5 | 0.1332 | Scombroid food poisoning (`C0275143`) | 96 | 7 |

**Diff:**
- Only-in-top1 (2):
  - `C0042963` Vomiting (w_t1=0.426, idf=2.35)
  - `C0027497` Nausea (w_t1=0.236, idf=2.20)
- Only-in-true (2):
  - `C0024204` lymph nodes (w_tr=0.181, idf=2.83)
  - `C0038990` Sweating (w_tr=0.068, idf=2.27)

---

### Failure #110679 — True: Inguinal hernia (`C0019294`)
- Age=10, Sex=F, Rank=**3**, Score=0.0895

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1268 | Boerhaave (`C0014860`) | 75 | 6 |
| 2 | 0.0977 | Anaphylaxis (`C0685898`) | 170 | 8 |
| 3 | 0.0895 | Inguinal hernia (`C0019294`) ← TRUE | 62 | 6 |
| 4 | 0.0873 | Acute rhinosinusitis (`C0149512`) | 132 | 7 |
| 5 | 0.0797 | Chagas (`C0041234`) | 155 | 13 |

**Diff:**
- Only-in-top1 (3):
  - `C0221198` Lesion (w_t1=0.519, idf=2.83)
  - `C0015733` Feces (w_t1=0.417, idf=3.81)
  - `C0030232`  (w_t1=0.237, idf=3.12)
- Only-in-true (3):
  - `C0021853` Intestines (w_tr=0.459, idf=3.12)
  - `C0010200` Coughing (w_tr=0.168, idf=1.73)
  - `C1515974` Anatomic Site (w_tr=0.108, idf=1.00)

---

### Failure #28704 — True: Boerhaave (`C0014860`)
- Age=33, Sex=M, Rank=**16**, Score=0.0284

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1185 | Larygospasm (`C0023066`) | 83 | 5 |
| 2 | 0.0915 | Possible NSTEMI / STEMI (`C0010072`) | 94 | 6 |
| 3 | 0.0557 | Anaphylaxis (`C0685898`) | 170 | 6 |
| 4 | 0.0533 | Scombroid food poisoning (`C0275143`) | 96 | 5 |
| 5 | 0.0532 | Cluster headache (`C0009088`) | 120 | 4 |

**Diff:**
- Only-in-top1 (1):
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.274, idf=2.43)
- Only-in-true (1):
  - `C0030193` Pain (w_tr=0.219, idf=1.45)

---

### Failure #48933 — True: Bronchitis (`C0006277`)
- Age=21, Sex=F, Rank=**3**, Score=0.1600

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1818 | Viral pharyngitis (`C0001344`) | 76 | 6 |
| 2 | 0.1633 | Acute laryngitis (`C0001327`) | 102 | 6 |
| 3 | 0.1600 | Bronchitis (`C0006277`) ← TRUE | 134 | 6 |
| 4 | 0.1258 | GERD (`C0017168`) | 126 | 6 |
| 5 | 0.1186 | URTI (`C0041912`) | 70 | 7 |

**Diff:**
- Only-in-top1 (2):
  - `C0031350` Pharyngitis (w_t1=1.139, idf=2.27)
  - `C0031354` Pharyngeal structure (w_t1=0.897, idf=2.61)
- Only-in-true (2):
  - `C0038056` Sputum (w_tr=1.218, idf=2.97)
  - `C0043144` Wheezing (w_tr=0.759, idf=2.14)

---

### Failure #115113 — True: Sarcoidosis (`C0036202`)
- Age=35, Sex=F, Rank=**4**, Score=0.0763

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1427 | Anaphylaxis (`C0685898`) | 170 | 8 |
| 2 | 0.0936 | Acute laryngitis (`C0001327`) | 102 | 8 |
| 3 | 0.0770 | Epiglottitis (`C0155814`) | 116 | 7 |
| 4 | 0.0763 | Sarcoidosis (`C0036202`) ← TRUE | 183 | 12 |
| 5 | 0.0639 | Inguinal hernia (`C0019294`) | 62 | 4 |

**Diff:**
- Only-in-top1 (1):
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.262, idf=2.43)
- Only-in-true (5):
  - `C0221198` Lesion (w_tr=0.625, idf=2.83)
  - `C0024204` lymph nodes (w_tr=0.545, idf=2.83)
  - `C1515974` Anatomic Site (w_tr=0.108, idf=1.00)
  - `C0237849` Peeling of skin (w_tr=0.087, idf=3.53)
  - `C0497406` Overweight (w_tr=0.040, idf=3.12)

---

### Failure #54648 — True: Bronchiectasis (`C0006267`)
- Age=13, Sex=F, Rank=**3**, Score=0.2379

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3097 | Pneumonia (`C0694504`) | 150 | 9 |
| 2 | 0.2938 | Bronchitis (`C0006277`) | 134 | 8 |
| 3 | 0.2379 | Bronchiectasis (`C0006267`) ← TRUE | 137 | 8 |
| 4 | 0.1626 | Tuberculosis (`C0041327`) | 95 | 7 |
| 5 | 0.1478 | Bronchospasm / acute asthma exacerb (`C0004096`) | 171 | 8 |

**Diff:**
- Only-in-top1 (1):
  - `C0085393` Immunocompromised Host (w_t1=0.728, idf=2.83)
- Only-in-true (0):

---


## Confusion pairs
- HIV (initial infection) → Boerhaave (1x)
- Chagas → Anaphylaxis (1x)
- Larygospasm → Acute laryngitis (1x)
- Inguinal hernia → Anaphylaxis (1x)
- HIV (initial infection) → Ebola (1x)
- Inguinal hernia → Boerhaave (1x)
- Boerhaave → Larygospasm (1x)
- Bronchitis → Viral pharyngitis (1x)
- Sarcoidosis → Anaphylaxis (1x)
- Bronchiectasis → Pneumonia (1x)
