# v64 Forensic — top_K=80, beta=0.75

- Config: cosine + IDF(df_thr=0.12, alpha=1.0, beta=0.75) + top-K=80
- |diseases|=49, |all_evs|=807

## Success cases

### Success #48 — True: Pericarditis (`C0155679`)
- Rank: **1**, Score: 0.3437

**Evidence (19):** Have you recently had a viral infection? (yes); Have you ever had a pericarditis? (yes); Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → un_coup_de_couteau; Characterize your pain: → vive; Do you feel pain somewhere? → haut_du_thorax ... +13 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3437 | Pericarditis (`C0155679`) ← TRUE | 80 | 9 |
| 2 | 0.2533 | Myocarditis (`C0027059`) | 80 | 8 |
| 3 | 0.2339 | Croup (`C0010380`) | 78 | 7 |
| 4 | 0.2116 | Atrial fibrillation (`C3264374`) | 52 | 6 |
| 5 | 0.2003 | Anemia (`C0002871`) | 80 | 7 |

---

### Success #74 — True: PSVT (`C0039240`)
- Rank: **1**, Score: 0.2705

**Evidence (16):** Do you regularly drink coffee or tea? (yes); Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → lancinante_/_choc_électrique; Characterize your pain: → une_crampe; Characterize your pain: → une_lourdeur_ou_serrement; Do you feel pain somewhere? → arrière_de_tête ... +10 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2705 | PSVT (`C0039240`) ← TRUE | 48 | 4 |
| 2 | 0.2677 | Atrial fibrillation (`C3264374`) | 52 | 5 |
| 3 | 0.2082 | Scombroid food poisoning (`C0275143`) | 80 | 5 |
| 4 | 0.1925 | Panic attack (`C0349232`) | 80 | 5 |
| 5 | 0.1884 | Anaphylaxis (`C0685898`) | 80 | 4 |

---

### Success #75 — True: URTI (`C0041912`)
- Rank: **1**, Score: 0.2621

**Evidence (19):** Do you live with 4 or more people? (yes); Do you attend or work in a daycare? (yes); Have you had significantly increased sweating? (yes); Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → pénible; Do you feel pain somewhere? → front ... +13 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2621 | URTI (`C0041912`) ← TRUE | 63 | 6 |
| 2 | 0.2289 | Influenza (`C0021400`) | 80 | 6 |
| 3 | 0.2197 | Chronic rhinosinusitis (`C0037199`) | 80 | 5 |
| 4 | 0.2072 | Acute laryngitis (`C0001327`) | 80 | 6 |
| 5 | 0.2024 | Ebola (`C0282687`) | 80 | 4 |

---

### Success #86 — True: Panic attack (`C0349232`)
- Rank: **1**, Score: 0.2907

**Evidence (27):** Do any members of your immediate family have a psychiatric illness? (yes); Have you had significantly increased sweating? (yes); Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → une_crampe; Do you feel pain somewhere? → côté_du_thorax_G_; Do you feel pain somewhere? → fosse_iliaque_G_ ... +21 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2907 | Panic attack (`C0349232`) ← TRUE | 80 | 14 |
| 2 | 0.2713 | Guillain-Barré syndrome (`C0018378`) | 80 | 13 |
| 3 | 0.2651 | Anaphylaxis (`C0685898`) | 80 | 13 |
| 4 | 0.2626 | Scombroid food poisoning (`C0275143`) | 80 | 13 |
| 5 | 0.2086 | Sarcoidosis (`C0036202`) | 80 | 12 |

---

### Success #258 — True: Guillain-Barré syndrome (`C0018378`)
- Rank: **1**, Score: 0.4528

**Evidence (6):** Have you recently had a viral infection? (yes); Do you feel weakness in both arms and/or both legs? (yes); Do you have numbness, loss of sensation or tingling in the feet? (yes); Have you had weakness or paralysis on one side of the face, which may still be present or completely resolved? (yes); Have you recently had numbness, loss of sensation or tingling, in both arms and legs and around your mouth? (yes); Have you traveled out of the country in the last 4 weeks? → N

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.4528 | Guillain-Barré syndrome (`C0018378`) ← TRUE | 80 | 8 |
| 2 | 0.2366 | Localized edema (`C0013609`) | 51 | 6 |
| 3 | 0.2190 | Myasthenia gravis (`C0026896`) | 80 | 6 |
| 4 | 0.1994 | Sarcoidosis (`C0036202`) | 80 | 7 |
| 5 | 0.1949 | HIV (initial infection) (`C0001175`) | 48 | 4 |

---

### Success #283 — True: Tuberculosis (`C0041327`)
- Rank: **1**, Score: 0.2892

**Evidence (5):** Are you infected with the human immunodeficiency virus (HIV)? (yes); Are you currently using intravenous drugs? (yes); Do you have a fever (either felt or measured with a thermometer)? (yes); Do you have a cough? (yes); Have you traveled out of the country in the last 4 weeks? → N

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2892 | Tuberculosis (`C0041327`) ← TRUE | 80 | 3 |
| 2 | 0.2619 | Whooping cough (`C0043168`) | 43 | 4 |
| 3 | 0.1939 | Myocarditis (`C0027059`) | 80 | 3 |
| 4 | 0.1827 | Acute laryngitis (`C0001327`) | 80 | 3 |
| 5 | 0.1776 | Bronchitis (`C0006277`) | 80 | 4 |

---

### Success #350 — True: Allergic sinusitis (`C0018621`)
- Rank: **1**, Score: 0.4729

**Evidence (10):** Do you have any close family members who suffer from allergies (any type), hay fever or eczema? (yes); Do you have any family members who have asthma? (yes); Do you have asthma or have you ever had to use a bronchodilator in the past? (yes); Is your nose or the back of your throat itchy? (yes); Do you have severe itching in one or both eyes? (yes); Do you have nasal congestion or a clear runny nose? (yes) ... +4 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.4729 | Allergic sinusitis (`C0018621`) ← TRUE | 60 | 11 |
| 2 | 0.3429 | Bronchospasm / acute asthma exacerb (`C0004096`) | 80 | 8 |
| 3 | 0.3423 | Chronic rhinosinusitis (`C0037199`) | 80 | 8 |
| 4 | 0.3035 | Acute rhinosinusitis (`C0149512`) | 80 | 8 |
| 5 | 0.2313 | URTI (`C0041912`) | 63 | 7 |

---

### Success #373 — True: Chronic rhinosinusitis (`C0037199`)
- Rank: **1**, Score: 0.3882

**Evidence (24):** Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → vive; Do you feel pain somewhere? → bouche; Do you feel pain somewhere? → front; Do you feel pain somewhere? → joue_D_; Do you feel pain somewhere? → oeil_G_ ... +18 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3882 | Chronic rhinosinusitis (`C0037199`) ← TRUE | 80 | 9 |
| 2 | 0.3558 | Acute rhinosinusitis (`C0149512`) | 80 | 8 |
| 3 | 0.2910 | Bronchospasm / acute asthma exacerb (`C0004096`) | 80 | 7 |
| 4 | 0.2595 | Allergic sinusitis (`C0018621`) | 60 | 7 |
| 5 | 0.2155 | URTI (`C0041912`) | 63 | 8 |

---

### Success #457 — True: Acute dystonic reactions (`C0236832`)
- Rank: **1**, Score: 0.2951

**Evidence (7):** Are you experiencing shortness of breath or difficulty breathing in a significant way? (yes); Have you ever felt like you were suffocating for a very short time associated with inability to breathe or speak? (yes); Have you been treated in hospital recently for nausea, agitation, intoxication or aggressive behavior and received medication via an intravenous or intramuscular route? (yes); Do you have trouble keeping your tongue in your mouth? (yes); Are you unable to control the direction of your eyes? (yes); Have you traveled out of the country in the last 4 weeks? → N ... +1 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2951 | Acute dystonic reactions (`C0236832`) ← TRUE | 60 | 4 |
| 2 | 0.1949 | Anaphylaxis (`C0685898`) | 80 | 5 |
| 3 | 0.1856 | Myasthenia gravis (`C0026896`) | 80 | 5 |
| 4 | 0.1828 | HIV (initial infection) (`C0001175`) | 48 | 3 |
| 5 | 0.1774 | Scombroid food poisoning (`C0275143`) | 80 | 5 |

---

### Success #504 — True: Pneumonia (`C0694504`)
- Rank: **1**, Score: 0.3003

**Evidence (34):** Have you been coughing up blood? (yes); Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → vive; Do you feel pain somewhere? → côté_du_thorax_D_; Do you feel pain somewhere? → côté_du_thorax_G_; Do you feel pain somewhere? → haut_du_thorax ... +28 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3003 | Pneumonia (`C0694504`) ← TRUE | 80 | 12 |
| 2 | 0.2591 | Bronchiectasis (`C0006267`) | 80 | 11 |
| 3 | 0.2333 | Tuberculosis (`C0041327`) | 80 | 11 |
| 4 | 0.2126 | Ebola (`C0282687`) | 80 | 8 |
| 5 | 0.2084 | Bronchitis (`C0006277`) | 80 | 7 |

---


## Failure cases

### Failure #2 — True: Bronchitis (`C0006277`)
- Rank: **6**, Score: 0.2537

**Evidence (14):** Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → une_brûlure_ou_chaleur; Do you feel pain somewhere? → côté_du_thorax_D_; Do you feel pain somewhere? → pharynx; How intense is the pain? → 5; Does the pain radiate to another location? → nulle_part ... +8 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3837 | Bronchiolitis (`C0001311`) | 80 | 9 |
| 2 | 0.2973 | Acute laryngitis (`C0001327`) | 80 | 9 |
| 3 | 0.2727 | Bronchospasm / acute asthma exacerb (`C0004096`) | 80 | 8 |
| 4 | 0.2571 | GERD (`C0017168`) | 80 | 8 |
| 5 | 0.2562 | Bronchiectasis (`C0006267`) | 80 | 8 |

**Diff:**
- Only-in-top1 (3):
  - `C1260880` Rhinorrhea (w_t1=0.463, idf=2.27)
  - `C0027424` Nasal congestion (finding) (w_t1=0.322, idf=2.61)
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.269, idf=2.43)
- Only-in-true (0):

---

### Failure #7 — True: Inguinal hernia (`C0019294`)
- Rank: **9**, Score: 0.1266

**Evidence (23):** Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → une_lourdeur_ou_serrement; Do you feel pain somewhere? → fosse_iliaque_D_; Do you feel pain somewhere? → fosse_iliaque_G_; Do you feel pain somewhere? → hanche_D_; Do you feel pain somewhere? → hanche_G_ ... +17 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1819 | Acute laryngitis (`C0001327`) | 80 | 7 |
| 2 | 0.1788 | Anaphylaxis (`C0685898`) | 80 | 6 |
| 3 | 0.1672 | Acute rhinosinusitis (`C0149512`) | 80 | 5 |
| 4 | 0.1611 | Allergic sinusitis (`C0018621`) | 60 | 7 |
| 5 | 0.1516 | Chagas (`C0041234`) | 80 | 7 |

**Diff:**
- Only-in-top1 (4):
  - `C0221198` Lesion (w_t1=0.477, idf=2.83)
  - `C0033774` Pruritus (w_t1=0.378, idf=3.30)
  - `C0041834` Erythema (w_t1=0.377, idf=2.83)
  - `C0237849` Peeling of skin (w_t1=0.130, idf=3.53)
- Only-in-true (2):
  - `C0021853` Intestines (w_tr=0.459, idf=3.12)
  - `C1515974` Anatomic Site (w_tr=0.108, idf=1.00)

---

### Failure #9 — True: Bronchitis (`C0006277`)
- Rank: **4**, Score: 0.2878

**Evidence (20):** Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → sensible; Characterize your pain: → une_brûlure_ou_chaleur; Do you feel pain somewhere? → côté_du_thorax_G_; Do you feel pain somewhere? → haut_du_thorax; Do you feel pain somewhere? → pharynx ... +14 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3223 | URTI (`C0041912`) | 63 | 9 |
| 2 | 0.3206 | Acute laryngitis (`C0001327`) | 80 | 9 |
| 3 | 0.3129 | Bronchiolitis (`C0001311`) | 80 | 8 |
| 4 | 0.2878 | Bronchitis (`C0006277`) ← TRUE | 80 | 6 |
| 5 | 0.2875 | Viral pharyngitis (`C0001344`) | 66 | 7 |

**Diff:**
- Only-in-top1 (5):
  - `C0031350` Pharyngitis (w_t1=0.931, idf=2.35)
  - `C0027424` Nasal congestion (finding) (w_t1=0.658, idf=2.61)
  - `C1260880` Rhinorrhea (w_t1=0.420, idf=2.27)
  - `C0031354` Pharyngeal structure (w_t1=0.400, idf=2.61)
  - `C0030193` Pain (w_t1=0.239, idf=1.51)
- Only-in-true (2):
  - `C0038056` Sputum (w_tr=1.148, idf=3.12)
  - `C0817096` Chest (w_tr=0.482, idf=1.69)

---

### Failure #11 — True: Bronchitis (`C0006277`)
- Rank: **4**, Score: 0.2246

**Evidence (17):** Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → une_brûlure_ou_chaleur; Do you feel pain somewhere? → bas_du_thorax; Do you feel pain somewhere? → haut_du_thorax; How intense is the pain? → 1; Does the pain radiate to another location? → nulle_part ... +11 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3507 | Bronchiolitis (`C0001311`) | 80 | 9 |
| 2 | 0.2405 | Bronchospasm / acute asthma exacerb (`C0004096`) | 80 | 8 |
| 3 | 0.2292 | Bronchiectasis (`C0006267`) | 80 | 8 |
| 4 | 0.2246 | Bronchitis (`C0006277`) ← TRUE | 80 | 6 |
| 5 | 0.2214 | Croup (`C0010380`) | 78 | 8 |

**Diff:**
- Only-in-top1 (3):
  - `C1260880` Rhinorrhea (w_t1=0.463, idf=2.27)
  - `C0027424` Nasal congestion (finding) (w_t1=0.322, idf=2.61)
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.269, idf=2.43)
- Only-in-true (0):

---

### Failure #12 — True: Bronchitis (`C0006277`)
- Rank: **13**, Score: 0.2034

**Evidence (20):** Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → sensible; Characterize your pain: → une_brûlure_ou_chaleur; Do you feel pain somewhere? → haut_du_thorax; Do you feel pain somewhere? → pharynx; Do you feel pain somewhere? → sein_D_ ... +14 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3250 | Bronchiolitis (`C0001311`) | 80 | 9 |
| 2 | 0.3073 | Acute laryngitis (`C0001327`) | 80 | 9 |
| 3 | 0.2671 | URTI (`C0041912`) | 63 | 9 |
| 4 | 0.2535 | Viral pharyngitis (`C0001344`) | 66 | 7 |
| 5 | 0.2290 | Bronchiectasis (`C0006267`) | 80 | 8 |

**Diff:**
- Only-in-top1 (4):
  - `C1260880` Rhinorrhea (w_t1=0.463, idf=2.27)
  - `C0027424` Nasal congestion (finding) (w_t1=0.322, idf=2.61)
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.269, idf=2.43)
  - `C0031350` Pharyngitis (w_t1=0.136, idf=2.35)
- Only-in-true (0):

---

### Failure #15 — True: Larygospasm (`C0023066`)
- Rank: **17**, Score: 0.0874

**Evidence (3):** Have you noticed a high pitched sound when breathing in? (yes); Have you traveled out of the country in the last 4 weeks? → AmerN; Are you exposed to secondhand cigarette smoke on a daily basis? (yes)

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2042 | Bronchitis (`C0006277`) | 80 | 1 |
| 2 | 0.2000 | Croup (`C0010380`) | 78 | 1 |
| 3 | 0.1946 | Epiglottitis (`C0155814`) | 80 | 1 |
| 4 | 0.1945 | Bronchiolitis (`C0001311`) | 80 | 1 |
| 5 | 0.1484 | Whooping cough (`C0043168`) | 43 | 1 |

**Diff:**
- Only-in-top1 (0):
- Only-in-true (0):

---

### Failure #21 — True: Pulmonary embolism (`C0034065`)
- Rank: **3**, Score: 0.2363

**Evidence (25):** Have you been coughing up blood? (yes); Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → sensible; Characterize your pain: → vive; Do you feel pain somewhere? → côté_du_thorax_D_; Do you feel pain somewhere? → côté_du_thorax_G_ ... +19 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2672 | Croup (`C0010380`) | 78 | 7 |
| 2 | 0.2483 | Pneumonia (`C0694504`) | 80 | 9 |
| 3 | 0.2363 | Pulmonary embolism (`C0034065`) ← TRUE | 80 | 8 |
| 4 | 0.2170 | Bronchiectasis (`C0006267`) | 80 | 7 |
| 5 | 0.2150 | Acute pulmonary edema (`C0155919`) | 76 | 8 |

**Diff:**
- Only-in-top1 (2):
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.415, idf=2.43)
  - `C5236002` Increased (finding) (w_t1=0.362, idf=4.22)
- Only-in-true (3):
  - `C0005767` Blood (w_tr=0.464, idf=3.12)
  - `C0817096` Chest (w_tr=0.444, idf=1.69)
  - `C0030193` Pain (w_tr=0.268, idf=1.51)

---

### Failure #23 — True: Influenza (`C0021400`)
- Rank: **8**, Score: 0.1878

**Evidence (32):** Have you had significantly increased sweating? (yes); Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → une_lourdeur_ou_serrement; Characterize your pain: → épuisante; Do you feel pain somewhere? → arrière_de_tête; Do you feel pain somewhere? → arrière_du_cou ... +26 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3267 | Ebola (`C0282687`) | 80 | 9 |
| 2 | 0.2886 | Acute laryngitis (`C0001327`) | 80 | 12 |
| 3 | 0.2317 | Viral pharyngitis (`C0001344`) | 66 | 8 |
| 4 | 0.2169 | Tuberculosis (`C0041327`) | 80 | 8 |
| 5 | 0.2141 | Sarcoidosis (`C0036202`) | 80 | 10 |

**Diff:**
- Only-in-top1 (1):
  - `C0015230` Exanthema (w_t1=0.610, idf=2.83)
- Only-in-true (0):

---

### Failure #25 — True: Influenza (`C0021400`)
- Rank: **8**, Score: 0.2339

**Evidence (29):** Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → sensible; Do you feel pain somewhere? → côté_du_cou_D_; Do you feel pain somewhere? → côté_du_cou_G_; Do you feel pain somewhere? → dessus_de_tête; Do you feel pain somewhere? → front ... +23 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3435 | Acute laryngitis (`C0001327`) | 80 | 14 |
| 2 | 0.3113 | Ebola (`C0282687`) | 80 | 8 |
| 3 | 0.2831 | Acute rhinosinusitis (`C0149512`) | 80 | 11 |
| 4 | 0.2775 | URTI (`C0041912`) | 63 | 8 |
| 5 | 0.2633 | Chronic rhinosinusitis (`C0037199`) | 80 | 9 |

**Diff:**
- Only-in-top1 (5):
  - `C0221198` Lesion (w_t1=0.477, idf=2.83)
  - `C0033774` Pruritus (w_t1=0.378, idf=3.30)
  - `C0041834` Erythema (w_t1=0.377, idf=2.83)
  - `C0027530` Neck (w_t1=0.247, idf=2.27)
  - `C0237849` Peeling of skin (w_t1=0.130, idf=3.53)
- Only-in-true (0):

---

### Failure #26 — True: Acute dystonic reactions (`C0236832`)
- Rank: **6**, Score: 0.1137

**Evidence (5):** Do you regularly take stimulant drugs? (yes); Have you ever felt like you were suffocating for a very short time associated with inability to breathe or speak? (yes); Do you have a hard time opening/raising one or both eyelids? (yes); Have you traveled out of the country in the last 4 weeks? → N; Do you suddenly have difficulty or an inability to open your mouth or have jaw pain when opening it? (yes)

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2425 | Myasthenia gravis (`C0026896`) | 80 | 3 |
| 2 | 0.2130 | HIV (initial infection) (`C0001175`) | 48 | 2 |
| 3 | 0.1782 | Scombroid food poisoning (`C0275143`) | 80 | 3 |
| 4 | 0.1561 | Anemia (`C0002871`) | 80 | 4 |
| 5 | 0.1299 | SLE (`C0024141`) | 80 | 2 |

**Diff:**
- Only-in-top1 (2):
  - `C0015426` Eyelid structure (w_t1=0.808, idf=3.12)
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.497, idf=2.43)
- Only-in-true (0):

---

### Failure #36 — True: Influenza (`C0021400`)
- Rank: **11**, Score: 0.1513

**Evidence (26):** Have you had significantly increased sweating? (yes); Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → sensible; Characterize your pain: → une_lourdeur_ou_serrement; Do you feel pain somewhere? → arrière_du_cou; Do you feel pain somewhere? → côté_du_cou_G_ ... +20 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2833 | Ebola (`C0282687`) | 80 | 7 |
| 2 | 0.2351 | Sarcoidosis (`C0036202`) | 80 | 10 |
| 3 | 0.2329 | Acute laryngitis (`C0001327`) | 80 | 10 |
| 4 | 0.1799 | Boerhaave (`C0014860`) | 56 | 5 |
| 5 | 0.1789 | HIV (initial infection) (`C0001175`) | 48 | 7 |

**Diff:**
- Only-in-top1 (1):
  - `C0015230` Exanthema (w_t1=0.610, idf=2.83)
- Only-in-true (0):

---

### Failure #43 — True: Inguinal hernia (`C0019294`)
- Rank: **4**, Score: 0.1621

**Evidence (21):** Do you feel your abdomen is bloated or distended (swollen due to pressure from inside)? (yes); Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → une_lourdeur_ou_serrement; Do you feel pain somewhere? → fosse_iliaque_D_; Do you feel pain somewhere? → fosse_iliaque_G_; Do you feel pain somewhere? → hanche_D_ ... +15 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2059 | Boerhaave (`C0014860`) | 56 | 3 |
| 2 | 0.1807 | Acute laryngitis (`C0001327`) | 80 | 5 |
| 3 | 0.1669 | Anaphylaxis (`C0685898`) | 80 | 4 |
| 4 | 0.1621 | Inguinal hernia (`C0019294`) ← TRUE | 44 | 4 |
| 5 | 0.1555 | Scombroid food poisoning (`C0275143`) | 80 | 3 |

**Diff:**
- Only-in-top1 (1):
  - `C0221198` Lesion (w_t1=0.519, idf=2.83)
- Only-in-true (2):
  - `C0010200` Coughing (w_tr=0.171, idf=1.78)
  - `C1515974` Anatomic Site (w_tr=0.108, idf=1.00)

---

### Failure #45 — True: Epiglottitis (`C0155814`)
- Rank: **5**, Score: 0.2400

**Evidence (18):** Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → un_coup_de_couteau; Characterize your pain: → vive; Do you feel pain somewhere? → amygdale_G_; Do you feel pain somewhere? → côté_du_cou_D_; Do you feel pain somewhere? → palais ... +12 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3074 | Croup (`C0010380`) | 78 | 7 |
| 2 | 0.2972 | Viral pharyngitis (`C0001344`) | 66 | 6 |
| 3 | 0.2917 | Acute laryngitis (`C0001327`) | 80 | 7 |
| 4 | 0.2519 | Larygospasm (`C0023066`) | 57 | 6 |
| 5 | 0.2400 | Epiglottitis (`C0155814`) ← TRUE | 80 | 8 |

**Diff:**
- Only-in-top1 (1):
  - `C0040578` Trachea (w_t1=0.718, idf=2.83)
- Only-in-true (2):
  - `C0030193` Pain (w_tr=0.449, idf=1.51)
  - `C0011847` Diabetes (w_tr=0.214, idf=2.71)

---

### Failure #47 — True: HIV (initial infection) (`C0001175`)
- Rank: **6**, Score: 0.2057

**Evidence (33):** Do you have swollen or painful lymph nodes? (yes); Have you ever had a sexually transmitted infection? (yes); Have you had diarrhea or an increase in stool frequency? (yes); Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → sensible; Characterize your pain: → une_pulsation ... +27 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2474 | Ebola (`C0282687`) | 80 | 7 |
| 2 | 0.2449 | SLE (`C0024141`) | 80 | 11 |
| 3 | 0.2398 | Boerhaave (`C0014860`) | 56 | 6 |
| 4 | 0.2303 | Acute laryngitis (`C0001327`) | 80 | 12 |
| 5 | 0.2184 | Chagas (`C0041234`) | 80 | 11 |

**Diff:**
- Only-in-top1 (0):
- Only-in-true (1):
  - `C0024204` lymph nodes (w_tr=0.181, idf=2.83)

---

### Failure #56 — True: Sarcoidosis (`C0036202`)
- Rank: **3**, Score: 0.1751

**Evidence (24):** Do you have swollen or painful lymph nodes? (yes); Have you lost consciousness associated with violent and sustained muscle contractions or had an absence episode? (yes); Do you have pain somewhere, related to your reason for consulting? (yes); Characterize your pain: → sensible; Do you feel pain somewhere? → doigt_annulaire__D_; Do you feel pain somewhere? → doigt_auriculaire__D_ ... +18 more

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1903 | Acute laryngitis (`C0001327`) | 80 | 8 |
| 2 | 0.1842 | Anaphylaxis (`C0685898`) | 80 | 6 |
| 3 | 0.1751 | Sarcoidosis (`C0036202`) ← TRUE | 80 | 7 |
| 4 | 0.1666 | Epiglottitis (`C0155814`) | 80 | 7 |
| 5 | 0.1595 | SLE (`C0024141`) | 80 | 8 |

**Diff:**
- Only-in-top1 (2):
  - `C0033774` Pruritus (w_t1=0.378, idf=3.30)
  - `C0237849` Peeling of skin (w_t1=0.130, idf=3.53)
- Only-in-true (1):
  - `C0015230` Exanthema (w_tr=0.569, idf=2.83)

---


## Aggregate

- Success: avg pcuis=12.8
- Failure: avg pcuis=13.0, gap=0.0862

### Confusion pairs (true → predicted)
- Bronchitis → Bronchiolitis (3x)
- Influenza → Ebola (2x)
- Inguinal hernia → Acute laryngitis (1x)
- Bronchitis → URTI (1x)
- Larygospasm → Bronchitis (1x)
- Pulmonary embolism → Croup (1x)
- Influenza → Acute laryngitis (1x)
- Acute dystonic reactions → Myasthenia gravis (1x)
- Inguinal hernia → Boerhaave (1x)
- Epiglottitis → Croup (1x)
- HIV (initial infection) → Ebola (1x)
- Sarcoidosis → Acute laryngitis (1x)
