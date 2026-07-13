# v63 IDF Forensic — DDXPlus residual failures

- KG: `pilot/data/onlykg_graph_v49_v5_full.pkl`
- Config: cosine + IDF (df_threshold=0.12, alpha=1.0, beta=0.75)
- |diseases|=49, |all_evs|=1061

## Success cases

### Success #1 — True: GERD (`C0017168`)
- Rank: **1** / 49, Score: 0.2764

**Evidence (22 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → lancinante_/_choc_électrique
- Characterize your pain: → sensible
- Characterize your pain: → un_tiraillement
- Characterize your pain: → une_brûlure_ou_chaleur
- Do you feel pain somewhere? → bas_du_thorax
- Do you feel pain somewhere? → haut_du_thorax
- Do you feel pain somewhere? → hypochondre_D_
- How intense is the pain? → 6
- Does the pain radiate to another location? → bas_du_thorax
- ... +12 more

**Patient CUIs in profile (14):**
- `C0085281` Addictive Behavior [idf=4.22]
- `C0277814` Sitting position [idf=4.22]
- `C0015733` Feces [idf=3.81]
- `C0038351` Stomach [idf=3.53]
- `C0085624` Burning sensation [idf=3.30]
- `C0497406` Overweight [idf=3.12]
- `C0226896` Oral cavity [idf=2.71]
- `C0031354` Pharyngeal structure [idf=2.61]
- `C0010200` Coughing [idf=1.78]
- `C0030193` Pain [idf=1.51]
- `C1457887` Symptoms [idf=1.45]
- `C0549206` Patient currently pregnant [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]
- `C0446469` Surface region of upper chest [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2764 | GERD (`C0017168`) ← TRUE | 111 | 9 |
| 2 | 0.1580 | Tuberculosis (`C0041327`) | 81 | 6 |
| 3 | 0.1567 | Stable angina (`C0002962`) | 48 | 4 |
| 4 | 0.1463 | Viral pharyngitis (`C0001344`) | 66 | 5 |
| 5 | 0.1456 | Inguinal hernia (`C0019294`) | 44 | 7 |

---

### Success #19 — True: Bronchitis (`C0006277`)
- Rank: **1** / 49, Score: 0.3507

**Evidence (15 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → une_brûlure_ou_chaleur
- Do you feel pain somewhere? → côté_du_thorax_D_
- Do you feel pain somewhere? → sein_G_
- Do you feel pain somewhere? → thorax_postérieur_G_
- How intense is the pain? → 2
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 3
- How fast did the pain appear? → 2
- Are you experiencing shortness of breath or difficulty breathing in a significant way? (yes)
- ... +5 more

**Patient CUIs in profile (12):**
- `C0085624` Burning sensation [idf=3.30]
- `C0038056` Sputum [idf=3.12]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C0043144` Wheezing [idf=2.27]
- `C0035203` Respiration [idf=1.82]
- `C0010200` Coughing [idf=1.78]
- `C0015967` Fever [idf=1.78]
- `C0817096` Chest [idf=1.69]
- `C0013404` Dyspnea [idf=1.51]
- `C0030193` Pain [idf=1.51]
- `C0006141` Breast [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3507 | Bronchitis (`C0006277`) ← TRUE | 122 | 8 |
| 2 | 0.3109 | Bronchiolitis (`C0001311`) | 88 | 7 |
| 3 | 0.2917 | Bronchiectasis (`C0006267`) | 125 | 9 |
| 4 | 0.2879 | Pneumonia (`C0694504`) | 138 | 9 |
| 5 | 0.2758 | Acute pulmonary edema (`C0155919`) | 76 | 9 |

---

### Success #46 — True: Pulmonary embolism (`C0034065`)
- Rank: **1** / 49, Score: 0.2772

**Evidence (26 items):**
- Do you have an active cancer? (yes)
- Have you been coughing up blood? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → vive
- Do you feel pain somewhere? → côté_du_thorax_D_
- Do you feel pain somewhere? → côté_du_thorax_G_
- Do you feel pain somewhere? → omoplate_D_
- Do you feel pain somewhere? → omoplate_G_
- Do you feel pain somewhere? → thorax_postérieur_G_
- ... +16 more

**Patient CUIs in profile (13):**
- `C5236002` Increased (finding) [idf=4.22]
- `C0005767` Blood [idf=3.12]
- `C0041657` Unconscious State [idf=2.97]
- `C0006826` Malignant Neoplasms [idf=2.43]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C0013604` Edema [idf=1.87]
- `C0035203` Respiration [idf=1.82]
- `C0010200` Coughing [idf=1.78]
- `C0817096` Chest [idf=1.69]
- `C0013404` Dyspnea [idf=1.51]
- `C0030193` Pain [idf=1.51]
- `C0149871` Deep Vein Thrombosis [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2772 | Pulmonary embolism (`C0034065`) ← TRUE | 106 | 10 |
| 2 | 0.2516 | Croup (`C0010380`) | 78 | 7 |
| 3 | 0.2437 | Pneumonia (`C0694504`) | 138 | 10 |
| 4 | 0.2335 | Pulmonary neoplasm (`C0348343`) | 49 | 6 |
| 5 | 0.2251 | Spontaneous pneumothorax (`C0032326`) | 84 | 9 |

---

### Success #98 — True: Panic attack (`C0349232`)
- Rank: **1** / 49, Score: 0.3425

**Evidence (27 items):**
- Do you feel anxious? (yes)
- Have you had significantly increased sweating? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → une_crampe
- Characterize your pain: → vive
- Do you feel pain somewhere? → flanc_G_
- Do you feel pain somewhere? → fosse_iliaque_D_
- Do you feel pain somewhere? → hypochondre_D_
- Do you feel pain somewhere? → pubis
- Do you feel pain somewhere? → ventre
- ... +17 more

**Patient CUIs in profile (24):**
- `C0444584` Whole body [idf=4.22]
- `C0085281` Addictive Behavior [idf=4.22]
- `C5236002` Increased (finding) [idf=4.22]
- `C0026821` Muscle Cramp [idf=3.53]
- `C1140621` Leg [idf=3.12]
- `C0220870` Lightheadedness [idf=3.12]
- `C0030554` Paresthesia [idf=3.12]
- `C0020580` Hypesthesia [idf=3.12]
- `C0446516` Upper arm [idf=2.97]
- `C0003467` Anxiety [idf=2.83]
- `C0226896` Oral cavity [idf=2.71]
- `C0012833` Dizziness [idf=2.61]
- `C0011570` Mental Depression [idf=2.61]
- `C0027497` Nausea [idf=2.43]
- `C0030252` Palpitations [idf=2.35]
- `C0042963` Vomiting [idf=2.35]
- `C0038990` Sweating [idf=2.27]
- `C0004096` Asthma [idf=2.14]
- `C0030193` Pain [idf=1.51]
- `C0018787` Heart [idf=1.48]
- `C0018674` Craniocerebral Trauma [idf=1.00]
- `C0149931` Migraine Disorders [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]
- `C0224086` Belly of skeletal muscle [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3425 | Panic attack (`C0349232`) ← TRUE | 82 | 12 |
| 2 | 0.2865 | Guillain-Barré syndrome (`C0018378`) | 103 | 14 |
| 3 | 0.2681 | Scombroid food poisoning (`C0275143`) | 87 | 11 |
| 4 | 0.2412 | Anaphylaxis (`C0685898`) | 159 | 14 |
| 5 | 0.2321 | Sarcoidosis (`C0036202`) | 168 | 16 |

---

### Success #113 — True: URTI (`C0041912`)
- Rank: **1** / 49, Score: 0.3241

**Evidence (19 items):**
- Have you been in contact with a person with similar symptoms in the past 2 weeks? (yes)
- Have you had significantly increased sweating? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → pénible
- Characterize your pain: → une_lourdeur_ou_serrement
- Do you feel pain somewhere? → joue_D_
- Do you feel pain somewhere? → tempe_G_
- How intense is the pain? → 7
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 1
- ... +9 more

**Patient CUIs in profile (13):**
- `C5236002` Increased (finding) [idf=4.22]
- `C0231528` Myalgia [idf=3.53]
- `C0027424` Nasal congestion (finding) [idf=2.61]
- `C0031350` Pharyngitis [idf=2.35]
- `C1260880` Rhinorrhea [idf=2.27]
- `C0038990` Sweating [idf=2.27]
- `C0010200` Coughing [idf=1.78]
- `C0015967` Fever [idf=1.78]
- `C0030193` Pain [idf=1.51]
- `C1457887` Symptoms [idf=1.45]
- `C0007966` Cheek structure [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]
- `C0230007` Temporal region [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3241 | URTI (`C0041912`) ← TRUE | 63 | 8 |
| 2 | 0.2763 | Ebola (`C0282687`) | 87 | 6 |
| 3 | 0.2714 | Chronic rhinosinusitis (`C0037199`) | 99 | 7 |
| 4 | 0.2708 | Influenza (`C0021400`) | 158 | 8 |
| 5 | 0.2456 | Acute rhinosinusitis (`C0149512`) | 114 | 7 |

---

### Success #114 — True: Cluster headache (`C0009088`)
- Rank: **1** / 49, Score: 0.3810

**Evidence (18 items):**
- Have any of your family members been diagnosed with cluster headaches? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → lancinante_/_choc_électrique
- Characterize your pain: → un_tiraillement
- Characterize your pain: → violente
- Do you feel pain somewhere? → front
- Do you feel pain somewhere? → joue_G_
- Do you feel pain somewhere? → oeil_G_
- Do you feel pain somewhere? → tempe_D_
- Do you feel pain somewhere? → tempe_G_
- ... +8 more

**Patient CUIs in profile (12):**
- `C0700124` Dilated [idf=4.22]
- `C0039409` Tears (substance) [idf=3.81]
- `C0009088` Cluster Headache [idf=2.71]
- `C0027424` Nasal congestion (finding) [idf=2.61]
- `C1260880` Rhinorrhea [idf=2.27]
- `C0005847` Blood Vessel [idf=2.08]
- `C0015392` Eye [idf=1.87]
- `C0030193` Pain [idf=1.51]
- `C0007966` Cheek structure [idf=1.00]
- `C0425043` Death of relative [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]
- `C0230007` Temporal region [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3810 | Cluster headache (`C0009088`) ← TRUE | 101 | 6 |
| 2 | 0.2614 | URTI (`C0041912`) | 63 | 6 |
| 3 | 0.2164 | Chronic rhinosinusitis (`C0037199`) | 99 | 5 |
| 4 | 0.2117 | Allergic sinusitis (`C0018621`) | 60 | 5 |
| 5 | 0.2017 | Acute rhinosinusitis (`C0149512`) | 114 | 6 |

---

### Success #191 — True: Anaphylaxis (`C0685898`)
- Rank: **1** / 49, Score: 0.4006

**Evidence (36 items):**
- Do you have a known severe food allergy? (yes)
- Have you been in contact with or ate something that you have an allergy to? (yes)
- Have you had diarrhea or an increase in stool frequency? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → une_crampe
- Do you feel pain somewhere? → flanc_G_
- Do you feel pain somewhere? → hypochondre_D_
- Do you feel pain somewhere? → hypochondre_G_
- Do you feel pain somewhere? → pubis
- How intense is the pain? → 4
- ... +26 more

**Patient CUIs in profile (27):**
- `C0426642` Frequency of bowel action [idf=4.22]
- `C0003086` Ankle [idf=3.81]
- `C0016470` Food Allergy [idf=3.81]
- `C0237849` Peeling of skin [idf=3.53]
- `C0026821` Muscle Cramp [idf=3.53]
- `C0033774` Pruritus [idf=3.30]
- `C0220870` Lightheadedness [idf=3.12]
- `C0041657` Unconscious State [idf=2.97]
- `C0015230` Exanthema [idf=2.83]
- `C0041834` Erythema [idf=2.83]
- `C0221198` Lesion [idf=2.83]
- `C0226896` Oral cavity [idf=2.71]
- `C0012833` Dizziness [idf=2.61]
- `C0027497` Nausea [idf=2.43]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C0011991` Diarrhea [idf=2.35]
- `C0042963` Vomiting [idf=2.35]
- `C0027530` Neck [idf=2.27]
- `C0020517` Hypersensitivity [idf=2.08]
- `C0013604` Edema [idf=1.87]
- `C0035203` Respiration [idf=1.82]
- `C0205082` Severe (severity modifier) [idf=1.82]
- `C0030193` Pain [idf=1.51]
- `C0013404` Dyspnea [idf=1.51]
- `C1515974` Anatomic Site [idf=1.00]
- `C0559499` Biceps brachii muscle structure [idf=1.00]
- `C0007966` Cheek structure [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.4006 | Anaphylaxis (`C0685898`) ← TRUE | 159 | 19 |
| 2 | 0.3578 | Scombroid food poisoning (`C0275143`) | 87 | 15 |
| 3 | 0.2765 | Chagas (`C0041234`) | 133 | 18 |
| 4 | 0.2510 | Boerhaave (`C0014860`) | 56 | 10 |
| 5 | 0.1968 | SLE (`C0024141`) | 83 | 10 |

---

### Success #210 — True: Localized edema (`C0013609`)
- Rank: **1** / 49, Score: 0.2524

**Evidence (25 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → un_tiraillement
- Characterize your pain: → vive
- Do you feel pain somewhere? → coté_lateral_du_pied_D_
- Do you feel pain somewhere? → coté_lateral_du_pied_G_
- Do you feel pain somewhere? → cuisse_D_
- Do you feel pain somewhere? → face_dorsale_du_pied_G_
- Do you feel pain somewhere? → plante_du_pied_G_
- How intense is the pain? → 3
- ... +15 more

**Patient CUIs in profile (9):**
- `C0003086` Ankle [idf=3.81]
- `C0023890` Liver Cirrhosis [idf=3.12]
- `C0016504` Foot [idf=2.97]
- `C0022646` Kidney [idf=2.71]
- `C0014130` Endocrine System Diseases [idf=2.43]
- `C0013604` Edema [idf=1.87]
- `C0030193` Pain [idf=1.51]
- `C0039866` Thigh structure [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2524 | Localized edema (`C0013609`) ← TRUE | 51 | 4 |
| 2 | 0.2007 | SLE (`C0024141`) | 83 | 5 |
| 3 | 0.1909 | Sarcoidosis (`C0036202`) | 168 | 8 |
| 4 | 0.1517 | Ebola (`C0282687`) | 87 | 5 |
| 5 | 0.1310 | Pericarditis (`C0155679`) | 89 | 4 |

---

### Success #219 — True: Bronchiectasis (`C0006267`)
- Rank: **1** / 49, Score: 0.3823

**Evidence (9 items):**
- Do you have cystic fibrosis? (yes)
- Do you have Rheumatoid Arthritis? (yes)
- Have you been coughing up blood? (yes)
- Do you suffer from Crohn’s disease or ulcerative colitis (UC)? (yes)
- Do you have a cough that produces colored or more abundant sputum than usual? (yes)
- Have you ever had pneumonia? (yes)
- Do you have a cough? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Are you immunosuppressed? (yes)

**Patient CUIs in profile (9):**
- `C0009324` Ulcerative Colitis [idf=3.81]
- `C0010346` Crohn Disease [idf=3.81]
- `C0005767` Blood [idf=3.12]
- `C0038056` Sputum [idf=3.12]
- `C0003873` Rheumatoid Arthritis [idf=2.83]
- `C0085393` Immunocompromised Host [idf=2.83]
- `C0010674` Cystic Fibrosis [idf=2.35]
- `C0032285` Pneumonia [idf=2.02]
- `C0010200` Coughing [idf=1.78]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3823 | Bronchiectasis (`C0006267`) ← TRUE | 125 | 7 |
| 2 | 0.2887 | Pneumonia (`C0694504`) | 138 | 7 |
| 3 | 0.2617 | Bronchitis (`C0006277`) | 122 | 6 |
| 4 | 0.2311 | Chagas (`C0041234`) | 133 | 5 |
| 5 | 0.1831 | Acute rhinosinusitis (`C0149512`) | 114 | 5 |

---

### Success #237 — True: Anaphylaxis (`C0685898`)
- Rank: **1** / 49, Score: 0.3459

**Evidence (34 items):**
- Do you have a known severe food allergy? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → un_coup_de_couteau
- Do you feel pain somewhere? → flanc_G_
- Do you feel pain somewhere? → fosse_iliaque_D_
- Do you feel pain somewhere? → fosse_iliaque_G_
- Do you feel pain somewhere? → hypochondre_D_
- Do you feel pain somewhere? → épigastre
- How intense is the pain? → 4
- Does the pain radiate to another location? → nulle_part
- ... +24 more

**Patient CUIs in profile (20):**
- `C0003086` Ankle [idf=3.81]
- `C0016470` Food Allergy [idf=3.81]
- `C0237849` Peeling of skin [idf=3.53]
- `C0033774` Pruritus [idf=3.30]
- `C0220870` Lightheadedness [idf=3.12]
- `C0041657` Unconscious State [idf=2.97]
- `C0015230` Exanthema [idf=2.83]
- `C0041834` Erythema [idf=2.83]
- `C0221198` Lesion [idf=2.83]
- `C0226896` Oral cavity [idf=2.71]
- `C0012833` Dizziness [idf=2.61]
- `C0043144` Wheezing [idf=2.27]
- `C0027530` Neck [idf=2.27]
- `C0020517` Hypersensitivity [idf=2.08]
- `C0013604` Edema [idf=1.87]
- `C0205082` Severe (severity modifier) [idf=1.82]
- `C0030193` Pain [idf=1.51]
- `C1515974` Anatomic Site [idf=1.00]
- `C0559499` Biceps brachii muscle structure [idf=1.00]
- `C0007966` Cheek structure [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3459 | Anaphylaxis (`C0685898`) ← TRUE | 159 | 13 |
| 2 | 0.2702 | Scombroid food poisoning (`C0275143`) | 87 | 10 |
| 3 | 0.2314 | SLE (`C0024141`) | 83 | 10 |
| 4 | 0.2033 | Sarcoidosis (`C0036202`) | 168 | 16 |
| 5 | 0.1918 | Localized edema (`C0013609`) | 51 | 7 |

---


## Failure cases (residual)

### Failure #2 — True: Bronchitis (`C0006277`)
- Rank: **6** / 49, Score: 0.2562

**Evidence (14 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → une_brûlure_ou_chaleur
- Do you feel pain somewhere? → côté_du_thorax_D_
- Do you feel pain somewhere? → pharynx
- How intense is the pain? → 5
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 7
- How fast did the pain appear? → 4
- Are you experiencing shortness of breath or difficulty breathing in a significant way? (yes)
- Do you have a chronic obstructive pulmonary disease (COPD)? (yes)
- ... +4 more

**Patient CUIs in profile (13):**
- `C0085624` Burning sensation [idf=3.30]
- `C0031354` Pharyngeal structure [idf=2.61]
- `C0027424` Nasal congestion (finding) [idf=2.61]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C1260880` Rhinorrhea [idf=2.27]
- `C0043144` Wheezing [idf=2.27]
- `C0024117` Chronic Obstructive Airway Disease [idf=2.20]
- `C0035203` Respiration [idf=1.82]
- `C0010200` Coughing [idf=1.78]
- `C0817096` Chest [idf=1.69]
- `C0013404` Dyspnea [idf=1.51]
- `C0030193` Pain [idf=1.51]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3836 | Bronchiolitis (`C0001311`) | 88 | 9 |
| 2 | 0.2970 | Acute laryngitis (`C0001327`) | 94 | 9 |
| 3 | 0.2719 | Bronchospasm / acute asthma exacerb (`C0004096`) | 143 | 8 |
| 4 | 0.2588 | Bronchiectasis (`C0006267`) | 125 | 9 |
| 5 | 0.2568 | GERD (`C0017168`) | 111 | 8 |

**Diff:**
- Only-in-top1 (3):
  - `C1260880` Rhinorrhea (w_top1=0.463, idf=2.27)
  - `C0027424` Nasal congestion (finding) (w_top1=0.322, idf=2.61)
  - `C1299586` Has difficulty doing (qualifier value) (w_top1=0.269, idf=2.43)
- Only-in-true (1):
  - `C0030193` Pain (w_true=0.097, idf=1.51)

---

### Failure #7 — True: Inguinal hernia (`C0019294`)
- Rank: **9** / 49, Score: 0.1266

**Evidence (23 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → une_lourdeur_ou_serrement
- Do you feel pain somewhere? → fosse_iliaque_D_
- Do you feel pain somewhere? → fosse_iliaque_G_
- Do you feel pain somewhere? → hanche_D_
- Do you feel pain somewhere? → hanche_G_
- Do you feel pain somewhere? → testicule_G_
- How intense is the pain? → 1
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 6
- ... +13 more

**Patient CUIs in profile (14):**
- `C5236002` Increased (finding) [idf=4.22]
- `C0009566` Complication [idf=4.22]
- `C0237849` Peeling of skin [idf=3.53]
- `C0033774` Pruritus [idf=3.30]
- `C0030232`  [idf=3.12]
- `C0021853` Intestines [idf=3.12]
- `C0015230` Exanthema [idf=2.83]
- `C0041834` Erythema [idf=2.83]
- `C0221198` Lesion [idf=2.83]
- `C0010200` Coughing [idf=1.78]
- `C0030193` Pain [idf=1.51]
- `C1457887` Symptoms [idf=1.45]
- `C0019552` Hip structure [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1817 | Acute laryngitis (`C0001327`) | 94 | 7 |
| 2 | 0.1804 | Anaphylaxis (`C0685898`) | 159 | 7 |
| 3 | 0.1787 | Chagas (`C0041234`) | 133 | 10 |
| 4 | 0.1735 | Acute rhinosinusitis (`C0149512`) | 114 | 6 |
| 5 | 0.1611 | Allergic sinusitis (`C0018621`) | 60 | 7 |

**Diff:**
- Only-in-top1 (4):
  - `C0221198` Lesion (w_top1=0.477, idf=2.83)
  - `C0033774` Pruritus (w_top1=0.378, idf=3.30)
  - `C0041834` Erythema (w_top1=0.377, idf=2.83)
  - `C0237849` Peeling of skin (w_top1=0.130, idf=3.53)
- Only-in-true (2):
  - `C0021853` Intestines (w_true=0.459, idf=3.12)
  - `C1515974` Anatomic Site (w_true=0.108, idf=1.00)

---

### Failure #9 — True: Bronchitis (`C0006277`)
- Rank: **4** / 49, Score: 0.2834

**Evidence (20 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → une_brûlure_ou_chaleur
- Do you feel pain somewhere? → côté_du_thorax_G_
- Do you feel pain somewhere? → haut_du_thorax
- Do you feel pain somewhere? → pharynx
- Do you feel pain somewhere? → sein_D_
- Do you feel pain somewhere? → thorax_postérieur_D_
- How intense is the pain? → 1
- Does the pain radiate to another location? → nulle_part
- ... +10 more

**Patient CUIs in profile (15):**
- `C0085624` Burning sensation [idf=3.30]
- `C0038056` Sputum [idf=3.12]
- `C0031354` Pharyngeal structure [idf=2.61]
- `C0027424` Nasal congestion (finding) [idf=2.61]
- `C0031350` Pharyngitis [idf=2.35]
- `C1260880` Rhinorrhea [idf=2.27]
- `C0043144` Wheezing [idf=2.27]
- `C0024117` Chronic Obstructive Airway Disease [idf=2.20]
- `C0010200` Coughing [idf=1.78]
- `C0015967` Fever [idf=1.78]
- `C0817096` Chest [idf=1.69]
- `C0030193` Pain [idf=1.51]
- `C0006141` Breast [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]
- `C0446469` Surface region of upper chest [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3151 | URTI (`C0041912`) | 63 | 9 |
| 2 | 0.3133 | Acute laryngitis (`C0001327`) | 94 | 9 |
| 3 | 0.3059 | Bronchiolitis (`C0001311`) | 88 | 8 |
| 4 | 0.2834 | Bronchitis (`C0006277`) ← TRUE | 122 | 7 |
| 5 | 0.2811 | Viral pharyngitis (`C0001344`) | 66 | 7 |

**Diff:**
- Only-in-top1 (4):
  - `C0031350` Pharyngitis (w_top1=0.931, idf=2.35)
  - `C0027424` Nasal congestion (finding) (w_top1=0.658, idf=2.61)
  - `C1260880` Rhinorrhea (w_top1=0.420, idf=2.27)
  - `C0031354` Pharyngeal structure (w_top1=0.400, idf=2.61)
- Only-in-true (2):
  - `C0038056` Sputum (w_true=1.148, idf=3.12)
  - `C0817096` Chest (w_true=0.482, idf=1.69)

---

### Failure #11 — True: Bronchitis (`C0006277`)
- Rank: **4** / 49, Score: 0.2247

**Evidence (17 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → une_brûlure_ou_chaleur
- Do you feel pain somewhere? → bas_du_thorax
- Do you feel pain somewhere? → haut_du_thorax
- How intense is the pain? → 1
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 6
- How fast did the pain appear? → 1
- Are you experiencing shortness of breath or difficulty breathing in a significant way? (yes)
- Do you smoke cigarettes? (yes)
- ... +7 more

**Patient CUIs in profile (14):**
- `C0042196` Vaccination [idf=4.22]
- `C0085624` Burning sensation [idf=3.30]
- `C0027424` Nasal congestion (finding) [idf=2.61]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C1260880` Rhinorrhea [idf=2.27]
- `C0043144` Wheezing [idf=2.27]
- `C0024117` Chronic Obstructive Airway Disease [idf=2.20]
- `C0035203` Respiration [idf=1.82]
- `C0010200` Coughing [idf=1.78]
- `C0013404` Dyspnea [idf=1.51]
- `C0030193` Pain [idf=1.51]
- `C1457887` Symptoms [idf=1.45]
- `C1515974` Anatomic Site [idf=1.00]
- `C0446469` Surface region of upper chest [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3467 | Bronchiolitis (`C0001311`) | 88 | 9 |
| 2 | 0.2372 | Bronchospasm / acute asthma exacerb (`C0004096`) | 143 | 8 |
| 3 | 0.2291 | Bronchiectasis (`C0006267`) | 125 | 9 |
| 4 | 0.2247 | Bronchitis (`C0006277`) ← TRUE | 122 | 7 |
| 5 | 0.2189 | Croup (`C0010380`) | 78 | 8 |

**Diff:**
- Only-in-top1 (3):
  - `C1260880` Rhinorrhea (w_top1=0.463, idf=2.27)
  - `C0027424` Nasal congestion (finding) (w_top1=0.322, idf=2.61)
  - `C1299586` Has difficulty doing (qualifier value) (w_top1=0.269, idf=2.43)
- Only-in-true (1):
  - `C0030193` Pain (w_true=0.097, idf=1.51)

---

### Failure #12 — True: Bronchitis (`C0006277`)
- Rank: **13** / 49, Score: 0.2019

**Evidence (20 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → une_brûlure_ou_chaleur
- Do you feel pain somewhere? → haut_du_thorax
- Do you feel pain somewhere? → pharynx
- Do you feel pain somewhere? → sein_D_
- Do you feel pain somewhere? → thorax_postérieur_D_
- Do you feel pain somewhere? → thorax_postérieur_G_
- How intense is the pain? → 1
- Does the pain radiate to another location? → nulle_part
- ... +10 more

**Patient CUIs in profile (16):**
- `C0042196` Vaccination [idf=4.22]
- `C0085624` Burning sensation [idf=3.30]
- `C0031354` Pharyngeal structure [idf=2.61]
- `C0027424` Nasal congestion (finding) [idf=2.61]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C0031350` Pharyngitis [idf=2.35]
- `C1260880` Rhinorrhea [idf=2.27]
- `C0043144` Wheezing [idf=2.27]
- `C0024117` Chronic Obstructive Airway Disease [idf=2.20]
- `C0035203` Respiration [idf=1.82]
- `C0010200` Coughing [idf=1.78]
- `C0013404` Dyspnea [idf=1.51]
- `C0030193` Pain [idf=1.51]
- `C0006141` Breast [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]
- `C0446469` Surface region of upper chest [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3187 | Bronchiolitis (`C0001311`) | 88 | 9 |
| 2 | 0.3011 | Acute laryngitis (`C0001327`) | 94 | 9 |
| 3 | 0.2620 | URTI (`C0041912`) | 63 | 9 |
| 4 | 0.2486 | Viral pharyngitis (`C0001344`) | 66 | 7 |
| 5 | 0.2269 | Bronchiectasis (`C0006267`) | 125 | 9 |

**Diff:**
- Only-in-top1 (4):
  - `C1260880` Rhinorrhea (w_top1=0.463, idf=2.27)
  - `C0027424` Nasal congestion (finding) (w_top1=0.322, idf=2.61)
  - `C1299586` Has difficulty doing (qualifier value) (w_top1=0.269, idf=2.43)
  - `C0031350` Pharyngitis (w_top1=0.136, idf=2.35)
- Only-in-true (1):
  - `C0030193` Pain (w_true=0.097, idf=1.51)

---

### Failure #15 — True: Larygospasm (`C0023066`)
- Rank: **17** / 49, Score: 0.0874

**Evidence (3 items):**
- Have you noticed a high pitched sound when breathing in? (yes)
- Have you traveled out of the country in the last 4 weeks? → AmerN
- Are you exposed to secondhand cigarette smoke on a daily basis? (yes)

**Patient CUIs in profile (1):**
- `C0035203` Respiration [idf=1.82]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2024 | Bronchitis (`C0006277`) | 122 | 1 |
| 2 | 0.2000 | Croup (`C0010380`) | 78 | 1 |
| 3 | 0.1945 | Epiglottitis (`C0155814`) | 104 | 1 |
| 4 | 0.1944 | Bronchiolitis (`C0001311`) | 88 | 1 |
| 5 | 0.1484 | Whooping cough (`C0043168`) | 43 | 1 |

**Diff:**
- Only-in-top1 (0):
- Only-in-true (0):

---

### Failure #21 — True: Pulmonary embolism (`C0034065`)
- Rank: **3** / 49, Score: 0.2337

**Evidence (25 items):**
- Have you been coughing up blood? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → vive
- Do you feel pain somewhere? → côté_du_thorax_D_
- Do you feel pain somewhere? → côté_du_thorax_G_
- Do you feel pain somewhere? → omoplate_D_
- Do you feel pain somewhere? → omoplate_G_
- Do you feel pain somewhere? → thorax_postérieur_D_
- How intense is the pain? → 5
- ... +15 more

**Patient CUIs in profile (12):**
- `C5236002` Increased (finding) [idf=4.22]
- `C0005767` Blood [idf=3.12]
- `C0041657` Unconscious State [idf=2.97]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C0013604` Edema [idf=1.87]
- `C0035203` Respiration [idf=1.82]
- `C0010200` Coughing [idf=1.78]
- `C0817096` Chest [idf=1.69]
- `C0013404` Dyspnea [idf=1.51]
- `C0030193` Pain [idf=1.51]
- `C0149871` Deep Vein Thrombosis [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2637 | Croup (`C0010380`) | 78 | 7 |
| 2 | 0.2400 | Pneumonia (`C0694504`) | 138 | 9 |
| 3 | 0.2337 | Pulmonary embolism (`C0034065`) ← TRUE | 106 | 9 |
| 4 | 0.2150 | Bronchiectasis (`C0006267`) | 125 | 8 |
| 5 | 0.2122 | Acute pulmonary edema (`C0155919`) | 76 | 8 |

**Diff:**
- Only-in-top1 (2):
  - `C1299586` Has difficulty doing (qualifier value) (w_top1=0.415, idf=2.43)
  - `C5236002` Increased (finding) (w_top1=0.362, idf=4.22)
- Only-in-true (4):
  - `C0005767` Blood (w_true=0.464, idf=3.12)
  - `C0817096` Chest (w_true=0.444, idf=1.69)
  - `C0030193` Pain (w_true=0.268, idf=1.51)
  - `C0149871` Deep Vein Thrombosis (w_true=0.017, idf=1.00)

---

### Failure #23 — True: Influenza (`C0021400`)
- Rank: **8** / 49, Score: 0.1915

**Evidence (32 items):**
- Have you had significantly increased sweating? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → une_lourdeur_ou_serrement
- Characterize your pain: → épuisante
- Do you feel pain somewhere? → arrière_de_tête
- Do you feel pain somewhere? → arrière_du_cou
- Do you feel pain somewhere? → côté_du_cou_D_
- Do you feel pain somewhere? → dessus_de_tête
- Do you feel pain somewhere? → front
- How intense is the pain? → 7
- ... +22 more

**Patient CUIs in profile (18):**
- `C5236002` Increased (finding) [idf=4.22]
- `C0237849` Peeling of skin [idf=3.53]
- `C0231528` Myalgia [idf=3.53]
- `C0033774` Pruritus [idf=3.30]
- `C0221198` Lesion [idf=2.83]
- `C0015230` Exanthema [idf=2.83]
- `C0041834` Erythema [idf=2.83]
- `C0085393` Immunocompromised Host [idf=2.83]
- `C0003123` Anorexia [idf=2.61]
- `C0031350` Pharyngitis [idf=2.35]
- `C0038990` Sweating [idf=2.27]
- `C0027530` Neck [idf=2.27]
- `C0018670` Head [idf=1.87]
- `C0010200` Coughing [idf=1.78]
- `C0015967` Fever [idf=1.78]
- `C0015672` Fatigue [idf=1.65]
- `C0030193` Pain [idf=1.51]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3266 | Ebola (`C0282687`) | 87 | 9 |
| 2 | 0.2884 | Acute laryngitis (`C0001327`) | 94 | 12 |
| 3 | 0.2415 | Sarcoidosis (`C0036202`) | 168 | 15 |
| 4 | 0.2317 | Viral pharyngitis (`C0001344`) | 66 | 8 |
| 5 | 0.2169 | Tuberculosis (`C0041327`) | 81 | 8 |

**Diff:**
- Only-in-top1 (1):
  - `C0015230` Exanthema (w_top1=0.610, idf=2.83)
- Only-in-true (2):
  - `C0237849` Peeling of skin (w_true=0.146, idf=3.53)
  - `C0041834` Erythema (w_true=0.037, idf=2.83)

---

### Failure #25 — True: Influenza (`C0021400`)
- Rank: **8** / 49, Score: 0.2363

**Evidence (29 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Do you feel pain somewhere? → côté_du_cou_D_
- Do you feel pain somewhere? → côté_du_cou_G_
- Do you feel pain somewhere? → dessus_de_tête
- Do you feel pain somewhere? → front
- Do you feel pain somewhere? → occiput
- How intense is the pain? → 7
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 6
- ... +19 more

**Patient CUIs in profile (17):**
- `C0237849` Peeling of skin [idf=3.53]
- `C0231528` Myalgia [idf=3.53]
- `C0033774` Pruritus [idf=3.30]
- `C0015230` Exanthema [idf=2.83]
- `C0041834` Erythema [idf=2.83]
- `C0085393` Immunocompromised Host [idf=2.83]
- `C0221198` Lesion [idf=2.83]
- `C0027424` Nasal congestion (finding) [idf=2.61]
- `C0031350` Pharyngitis [idf=2.35]
- `C1260880` Rhinorrhea [idf=2.27]
- `C0027530` Neck [idf=2.27]
- `C0018670` Head [idf=1.87]
- `C0010200` Coughing [idf=1.78]
- `C0015967` Fever [idf=1.78]
- `C0015672` Fatigue [idf=1.65]
- `C0030193` Pain [idf=1.51]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3433 | Acute laryngitis (`C0001327`) | 94 | 14 |
| 2 | 0.3112 | Ebola (`C0282687`) | 87 | 8 |
| 3 | 0.2891 | Acute rhinosinusitis (`C0149512`) | 114 | 12 |
| 4 | 0.2775 | URTI (`C0041912`) | 63 | 8 |
| 5 | 0.2632 | Chronic rhinosinusitis (`C0037199`) | 99 | 9 |

**Diff:**
- Only-in-top1 (3):
  - `C0221198` Lesion (w_top1=0.477, idf=2.83)
  - `C0033774` Pruritus (w_top1=0.378, idf=3.30)
  - `C0027530` Neck (w_top1=0.247, idf=2.27)
- Only-in-true (0):

---

### Failure #26 — True: Acute dystonic reactions (`C0236832`)
- Rank: **7** / 49, Score: 0.1137

**Evidence (5 items):**
- Do you regularly take stimulant drugs? (yes)
- Have you ever felt like you were suffocating for a very short time associated with inability to breathe or speak? (yes)
- Do you have a hard time opening/raising one or both eyelids? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Do you suddenly have difficulty or an inability to open your mouth or have jaw pain when opening it? (yes)

**Patient CUIs in profile (4):**
- `C0015426` Eyelid structure [idf=3.12]
- `C0236000` Jaw pain [idf=2.83]
- `C0226896` Oral cavity [idf=2.71]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2424 | Myasthenia gravis (`C0026896`) | 91 | 3 |
| 2 | 0.2091 | HIV (initial infection) (`C0001175`) | 48 | 2 |
| 3 | 0.1782 | Scombroid food poisoning (`C0275143`) | 87 | 3 |
| 4 | 0.1556 | Anemia (`C0002871`) | 121 | 4 |
| 5 | 0.1292 | SLE (`C0024141`) | 83 | 2 |

**Diff:**
- Only-in-top1 (2):
  - `C0015426` Eyelid structure (w_top1=0.808, idf=3.12)
  - `C1299586` Has difficulty doing (qualifier value) (w_top1=0.497, idf=2.43)
- Only-in-true (0):

---


## Aggregate

- Success: avg pcuis=15.3, score=0.3333
- Failure: avg pcuis=12.4, score=0.1955, top1-true gap=0.0969

### Confusion pairs (true → predicted)
- Bronchitis → Bronchiolitis (3x)
- Inguinal hernia → Acute laryngitis (1x)
- Bronchitis → URTI (1x)
- Larygospasm → Bronchitis (1x)
- Pulmonary embolism → Croup (1x)
- Influenza → Ebola (1x)
- Influenza → Acute laryngitis (1x)
- Acute dystonic reactions → Myasthenia gravis (1x)
