# v71 Forensic — Random 10 Success + 10 Failure

- Config: strict zero-shot KG-only (cosine + IDF + self-aware neg)
- Sampling: random 5000 from 134K, then random 10 from each pool
- Pool sizes: 3024 success (rank=1), 1265 failure (rank≥3)

## Success cases (rank=1)

### Success #59833 — True: Scombroid food poisoning (`C0275143`)
- Age=9, Sex=M, Rank=**1**, Score=0.1867

**Evidence (17 items):**
- Do you feel lightheaded and dizzy or do you feel like you are about to faint? (yes)
- Did your cheeks suddenly turn red? (yes)
- Do you have any lesions, redness or problems on your skin that you believe are related to the condition you are consulting for? (yes)
- What color is the rash? → rose
- Do your lesions peel off? → N
- Is the rash swollen? → 6
- Where is the affected region located? → arrière_du_cou
- Where is the affected region located? → biceps_G_
- Where is the affected region located? → bouche
- Where is the affected region located? → cartilage_thyroidien
- Where is the affected region located? → cheville_G_
- How intense is the pain caused by the rash? → 2
- ... +5 more

**Patient CUIs (15):**
- `C0003086` Ankle [idf=3.81]
- `C0237849` Peeling of skin [idf=3.53]
- `C0033774` Pruritus [idf=3.30]
- `C0220870` Lightheadedness [idf=3.12]
- `C0015230` Exanthema [idf=2.83]
- `C0041834` Erythema [idf=2.83]
- `C0221198` Lesion [idf=2.83]
- `C0226896` Oral cavity [idf=2.71]
- `C0012833` Dizziness [idf=2.61]
- `C0027497` Nausea [idf=2.43]
- `C0042963` Vomiting [idf=2.35]
- `C0027530` Neck [idf=2.27]
- `C0030193` Pain [idf=1.51]
- `C0559499` Biceps brachii muscle structure [idf=1.00]
- `C0007966` Cheek structure [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1867 | Scombroid food poisoning (`C0275143`) ← TRUE | 87 | 8 |
| 2 | 0.1845 | Anaphylaxis (`C0685898`) | 159 | 9 |
| 3 | 0.1177 | Boerhaave (`C0014860`) | 56 | 4 |
| 4 | 0.1127 | HIV (initial infection) (`C0001175`) | 48 | 3 |
| 5 | 0.1077 | Acute dystonic reactions (`C0236832`) | 60 | 5 |

---

### Success #70982 — True: GERD (`C0017168`)
- Age=52, Sex=M, Rank=**1**, Score=0.1779

**Evidence (21 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → un_coup_de_couteau
- Characterize your pain: → un_tiraillement
- Characterize your pain: → écoeurante
- Characterize your pain: → épeurante
- Do you feel pain somewhere? → bas_du_thorax
- Do you feel pain somewhere? → haut_du_thorax
- Do you feel pain somewhere? → hypochondre_D_
- Do you feel pain somewhere? → hypochondre_G_
- Do you feel pain somewhere? → épigastre
- How intense is the pain? → 9
- ... +9 more

**Patient CUIs (10):**
- `C0085281` Addictive Behavior [idf=4.22]
- `C0015733` Feces [idf=3.81]
- `C0038351` Stomach [idf=3.53]
- `C0085624` Burning sensation [idf=3.30]
- `C0226896` Oral cavity [idf=2.71]
- `C0031354` Pharyngeal structure [idf=2.61]
- `C0010200` Coughing [idf=1.78]
- `C0030193` Pain [idf=1.51]
- `C1515974` Anatomic Site [idf=1.00]
- `C0446469` Surface region of upper chest [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1779 | GERD (`C0017168`) ← TRUE | 111 | 6 |
| 2 | 0.0901 | HIV (initial infection) (`C0001175`) | 48 | 2 |
| 3 | 0.0865 | Stable angina (`C0002962`) | 48 | 3 |
| 4 | 0.0836 | Localized edema (`C0013609`) | 51 | 4 |
| 5 | 0.0722 | Tuberculosis (`C0041327`) | 81 | 5 |

---

### Success #72081 — True: Panic attack (`C0349232`)
- Age=73, Sex=M, Rank=**1**, Score=0.2723

**Evidence (26 items):**
- Do you feel anxious? (yes)
- Have you had significantly increased sweating? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → vive
- Do you feel pain somewhere? → bas_du_thorax
- Do you feel pain somewhere? → côté_du_thorax_D_
- Do you feel pain somewhere? → flanc_D_
- Do you feel pain somewhere? → hypochondre_G_
- Do you feel pain somewhere? → pubis
- How intense is the pain? → 7
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 7
- ... +14 more

**Patient CUIs (24):**
- `C0444584` Whole body [idf=4.22]
- `C5236002` Increased (finding) [idf=4.22]
- `C0020580` Hypesthesia [idf=3.12]
- `C1140621` Leg [idf=3.12]
- `C0030554` Paresthesia [idf=3.12]
- `C0446516` Upper arm [idf=2.97]
- `C0003467` Anxiety [idf=2.83]
- `C0008301` Choking [idf=2.83]
- `C0226896` Oral cavity [idf=2.71]
- `C0011570` Mental Depression [idf=2.61]
- `C0027497` Nausea [idf=2.43]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C0042963` Vomiting [idf=2.35]
- `C0004044` Asphyxia [idf=2.35]
- `C0038990` Sweating [idf=2.27]
- `C0004096` Asthma [idf=2.14]
- `C0020517` Hypersensitivity [idf=2.08]
- `C0035203` Respiration [idf=1.82]
- `C0817096` Chest [idf=1.69]
- `C0013404` Dyspnea [idf=1.51]
- `C0030193` Pain [idf=1.51]
- `C0149931` Migraine Disorders [idf=1.00]
- `C0018674` Craniocerebral Trauma [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2723 | Panic attack (`C0349232`) ← TRUE | 82 | 13 |
| 2 | 0.2138 | Larygospasm (`C0023066`) | 57 | 8 |
| 3 | 0.2105 | Guillain-Barré syndrome (`C0018378`) | 103 | 14 |
| 4 | 0.1604 | Anaphylaxis (`C0685898`) | 159 | 17 |
| 5 | 0.1410 | Acute pulmonary edema (`C0155919`) | 76 | 13 |

---

### Success #14416 — True: URTI (`C0041912`)
- Age=11, Sex=F, Rank=**1**, Score=0.2911

**Evidence (17 items):**
- Do you live with 4 or more people? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → une_lourdeur_ou_serrement
- Do you feel pain somewhere? → dessus_de_tête
- Do you feel pain somewhere? → front
- Do you feel pain somewhere? → joue_D_
- Do you feel pain somewhere? → occiput
- Do you feel pain somewhere? → tempe_D_
- How intense is the pain? → 4
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 5
- ... +5 more

**Patient CUIs (8):**
- `C0027424` Nasal congestion (finding) [idf=2.61]
- `C0031350` Pharyngitis [idf=2.35]
- `C1260880` Rhinorrhea [idf=2.27]
- `C0018670` Head [idf=1.87]
- `C0030193` Pain [idf=1.51]
- `C0230007` Temporal region [idf=1.00]
- `C0007966` Cheek structure [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2911 | URTI (`C0041912`) ← TRUE | 63 | 5 |
| 2 | 0.2190 | Chronic rhinosinusitis (`C0037199`) | 99 | 5 |
| 3 | 0.1954 | Cluster headache (`C0009088`) | 101 | 5 |
| 4 | 0.1849 | Acute rhinosinusitis (`C0149512`) | 114 | 5 |
| 5 | 0.1820 | Viral pharyngitis (`C0001344`) | 66 | 4 |

---

### Success #112186 — True: Bronchiectasis (`C0006267`)
- Age=55, Sex=F, Rank=**1**, Score=0.2808

**Evidence (8 items):**
- Do you have cystic fibrosis? (yes)
- Do you have Rheumatoid Arthritis? (yes)
- Do you suffer from Crohn’s disease or ulcerative colitis (UC)? (yes)
- Do you have a cough that produces colored or more abundant sputum than usual? (yes)
- Have you ever had pneumonia? (yes)
- Do you have asthma or have you ever had to use a bronchodilator in the past? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Are you immunosuppressed? (yes)

**Patient CUIs (9):**
- `C0009324` Ulcerative Colitis [idf=3.81]
- `C0010346` Crohn Disease [idf=3.81]
- `C0038056` Sputum [idf=3.12]
- `C0003873` Rheumatoid Arthritis [idf=2.83]
- `C0085393` Immunocompromised Host [idf=2.83]
- `C0010674` Cystic Fibrosis [idf=2.35]
- `C0004096` Asthma [idf=2.14]
- `C0032285` Pneumonia [idf=2.02]
- `C0010200` Coughing [idf=1.78]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2808 | Bronchiectasis (`C0006267`) ← TRUE | 125 | 7 |
| 2 | 0.2163 | Bronchitis (`C0006277`) | 122 | 6 |
| 3 | 0.1962 | Pneumonia (`C0694504`) | 138 | 7 |
| 4 | 0.1134 | Chagas (`C0041234`) | 133 | 4 |
| 5 | 0.0890 | Chronic rhinosinusitis (`C0037199`) | 99 | 5 |

---

### Success #52506 — True: Acute dystonic reactions (`C0236832`)
- Age=15, Sex=F, Rank=**1**, Score=0.3937

**Evidence (10 items):**
- Have you started or taken any antipsychotic medication within the last 7 days? (yes)
- Are you experiencing shortness of breath or difficulty breathing in a significant way? (yes)
- Have you ever felt like you were suffocating for a very short time associated with inability to breathe or speak? (yes)
- Have you been treated in hospital recently for nausea, agitation, intoxication or aggressive behavior and received medication via an intravenous or intramuscular route? (yes)
- Do you have trouble keeping your tongue in your mouth? (yes)
- Are you unable to control the direction of your eyes? (yes)
- Do you feel that muscle spasms or soreness in your neck are keeping you from turning your head to one side? (yes)
- Do you have annoying muscle spasms in your face, neck or any other part of your body? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Do you suddenly have difficulty or an inability to open your mouth or have jaw pain when opening it? (yes)

**Patient CUIs (13):**
- `C0037763` Spasm [idf=4.22]
- `C0040408` Tongue [idf=3.30]
- `C0728899` Intoxication [idf=2.97]
- `C0236000` Jaw pain [idf=2.83]
- `C0226896` Oral cavity [idf=2.71]
- `C0234233` Sore to touch [idf=2.43]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C0027530` Neck [idf=2.27]
- `C0018670` Head [idf=1.87]
- `C0015392` Eye [idf=1.87]
- `C0035203` Respiration [idf=1.82]
- `C0013404` Dyspnea [idf=1.51]
- `C0040615` Antipsychotic Agents [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3937 | Acute dystonic reactions (`C0236832`) ← TRUE | 60 | 7 |
| 2 | 0.1323 | HIV (initial infection) (`C0001175`) | 48 | 5 |
| 3 | 0.1194 | Myasthenia gravis (`C0026896`) | 91 | 8 |
| 4 | 0.1108 | Epiglottitis (`C0155814`) | 104 | 6 |
| 5 | 0.1104 | Scombroid food poisoning (`C0275143`) | 87 | 7 |

---

### Success #76641 — True: Viral pharyngitis (`C0001344`)
- Age=6, Sex=F, Rank=**1**, Score=0.2252

**Evidence (17 items):**
- Have you been in contact with a person with similar symptoms in the past 2 weeks? (yes)
- Have you been coughing up blood? (yes)
- Do you attend or work in a daycare? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → une_brûlure_ou_chaleur
- Do you feel pain somewhere? → amygdale_G_
- Do you feel pain somewhere? → palais
- Do you feel pain somewhere? → pharynx
- Do you feel pain somewhere? → sous_la_machoire
- How intense is the pain? → 8
- Does the pain radiate to another location? → nulle_part
- ... +5 more

**Patient CUIs (10):**
- `C0040421` Palatine Tonsil [idf=4.22]
- `C0085624` Burning sensation [idf=3.30]
- `C0005767` Blood [idf=3.12]
- `C0031354` Pharyngeal structure [idf=2.61]
- `C0027424` Nasal congestion (finding) [idf=2.61]
- `C1260880` Rhinorrhea [idf=2.27]
- `C0010200` Coughing [idf=1.78]
- `C0030193` Pain [idf=1.51]
- `C1457887` Symptoms [idf=1.45]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2252 | Viral pharyngitis (`C0001344`) ← TRUE | 66 | 6 |
| 2 | 0.1465 | Acute rhinosinusitis (`C0149512`) | 114 | 7 |
| 3 | 0.1319 | URTI (`C0041912`) | 63 | 6 |
| 4 | 0.1257 | Acute laryngitis (`C0001327`) | 94 | 6 |
| 5 | 0.1142 | Chronic rhinosinusitis (`C0037199`) | 99 | 5 |

---

### Success #131257 — True: Inguinal hernia (`C0019294`)
- Age=5, Sex=M, Rank=**1**, Score=0.1093

**Evidence (21 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → une_lourdeur_ou_serrement
- Do you feel pain somewhere? → fosse_iliaque_D_
- Do you feel pain somewhere? → hanche_D_
- Do you feel pain somewhere? → testicule_G_
- How intense is the pain? → 7
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 3
- How fast did the pain appear? → 4
- Are you significantly overweight compared to people of the same height as you? (yes)
- Do you have any lesions, redness or problems on your skin that you believe are related to the condition you are consulting for? (yes)
- What color is the rash? → pale
- ... +9 more

**Patient CUIs (14):**
- `C5236002` Increased (finding) [idf=4.22]
- `C0237849` Peeling of skin [idf=3.53]
- `C0033774` Pruritus [idf=3.30]
- `C0030232`  [idf=3.12]
- `C0021853` Intestines [idf=3.12]
- `C0497406` Overweight [idf=3.12]
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
| 1 | 0.1093 | Inguinal hernia (`C0019294`) ← TRUE | 44 | 6 |
| 2 | 0.1056 | Anaphylaxis (`C0685898`) | 159 | 7 |
| 3 | 0.1033 | Acute laryngitis (`C0001327`) | 94 | 7 |
| 4 | 0.0925 | Allergic sinusitis (`C0018621`) | 60 | 7 |
| 5 | 0.0857 | Chagas (`C0041234`) | 133 | 10 |

---

### Success #21136 — True: URTI (`C0041912`)
- Age=40, Sex=M, Rank=**1**, Score=0.3493

**Evidence (22 items):**
- Have you been in contact with a person with similar symptoms in the past 2 weeks? (yes)
- Do you attend or work in a daycare? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → pénible
- Characterize your pain: → sensible
- Characterize your pain: → une_lourdeur_ou_serrement
- Do you feel pain somewhere? → dessus_de_tête
- Do you feel pain somewhere? → joue_D_
- Do you feel pain somewhere? → tempe_D_
- Do you feel pain somewhere? → tempe_G_
- How intense is the pain? → 3
- Does the pain radiate to another location? → nulle_part
- ... +10 more

**Patient CUIs (12):**
- `C0231528` Myalgia [idf=3.53]
- `C0027424` Nasal congestion (finding) [idf=2.61]
- `C0031350` Pharyngitis [idf=2.35]
- `C1260880` Rhinorrhea [idf=2.27]
- `C0018670` Head [idf=1.87]
- `C0015967` Fever [idf=1.78]
- `C0010200` Coughing [idf=1.78]
- `C0030193` Pain [idf=1.51]
- `C1457887` Symptoms [idf=1.45]
- `C0230007` Temporal region [idf=1.00]
- `C0007966` Cheek structure [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3493 | URTI (`C0041912`) ← TRUE | 63 | 9 |
| 2 | 0.2709 | Ebola (`C0282687`) | 87 | 7 |
| 3 | 0.2424 | Chronic rhinosinusitis (`C0037199`) | 99 | 8 |
| 4 | 0.2318 | Influenza (`C0021400`) | 158 | 9 |
| 5 | 0.2178 | Acute laryngitis (`C0001327`) | 94 | 9 |

---

### Success #34919 — True: GERD (`C0017168`)
- Age=16, Sex=F, Rank=**1**, Score=0.2105

**Evidence (21 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → un_coup_de_couteau
- Characterize your pain: → un_tiraillement
- Do you feel pain somewhere? → bas_du_thorax
- Do you feel pain somewhere? → haut_du_thorax
- Do you feel pain somewhere? → hypochondre_G_
- Do you feel pain somewhere? → ventre
- How intense is the pain? → 9
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 8
- How fast did the pain appear? → 2
- ... +9 more

**Patient CUIs (13):**
- `C0085281` Addictive Behavior [idf=4.22]
- `C0277814` Sitting position [idf=4.22]
- `C0038351` Stomach [idf=3.53]
- `C0085624` Burning sensation [idf=3.30]
- `C0226896` Oral cavity [idf=2.71]
- `C0031354` Pharyngeal structure [idf=2.61]
- `C0004096` Asthma [idf=2.14]
- `C0010200` Coughing [idf=1.78]
- `C0030193` Pain [idf=1.51]
- `C1457887` Symptoms [idf=1.45]
- `C0224086` Belly of skeletal muscle [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]
- `C0446469` Surface region of upper chest [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2105 | GERD (`C0017168`) ← TRUE | 111 | 9 |
| 2 | 0.1020 | Viral pharyngitis (`C0001344`) | 66 | 6 |
| 3 | 0.0955 | Stable angina (`C0002962`) | 48 | 5 |
| 4 | 0.0855 | HIV (initial infection) (`C0001175`) | 48 | 3 |
| 5 | 0.0855 | Larygospasm (`C0023066`) | 57 | 2 |

---


## Failure cases (rank≥3)

### Failure #15022 — True: HIV (initial infection) (`C0001175`)
- Age=39, Sex=F, Rank=**3**, Score=0.1973

**Evidence (37 items):**
- Do you have swollen or painful lymph nodes? (yes)
- Have you ever had a sexually transmitted infection? (yes)
- Have you had significantly increased sweating? (yes)
- Have you had diarrhea or an increase in stool frequency? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → pénible
- Characterize your pain: → sensible
- Characterize your pain: → épuisante
- Do you feel pain somewhere? → arrière_du_cou
- Do you feel pain somewhere? → dessus_de_tête
- Do you feel pain somewhere? → front
- Do you feel pain somewhere? → tempe_D_
- ... +25 more

**Patient CUIs (26):**
- `C5236002` Increased (finding) [idf=4.22]
- `C0426642` Frequency of bowel action [idf=4.22]
- `C1384606` Dyspareunia [idf=3.81]
- `C0231528` Myalgia [idf=3.53]
- `C0237849` Peeling of skin [idf=3.53]
- `C0033774` Pruritus [idf=3.30]
- `C0030232`  [idf=3.12]
- `C0036916` Sexually Transmitted Diseases [idf=2.97]
- `C0024204` lymph nodes [idf=2.83]
- `C0015230` Exanthema [idf=2.83]
- `C0221198` Lesion [idf=2.83]
- `C0041834` Erythema [idf=2.83]
- `C0038999` Swelling [idf=2.43]
- `C0027497` Nausea [idf=2.43]
- `C0031350` Pharyngitis [idf=2.35]
- `C0042963` Vomiting [idf=2.35]
- `C0011991` Diarrhea [idf=2.35]
- `C0027530` Neck [idf=2.27]
- `C0038990` Sweating [idf=2.27]
- `C0018670` Head [idf=1.87]
- `C0015967` Fever [idf=1.78]
- `C0015672` Fatigue [idf=1.65]
- `C0030193` Pain [idf=1.51]
- `C0230007` Temporal region [idf=1.00]
- `C1578559` Buccal mucosa [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2253 | Ebola (`C0282687`) | 87 | 11 |
| 2 | 0.2043 | Chagas (`C0041234`) | 133 | 19 |
| 3 | 0.1973 | HIV (initial infection) (`C0001175`) ← TRUE | 48 | 11 |
| 4 | 0.1865 | Boerhaave (`C0014860`) | 56 | 8 |
| 5 | 0.1708 | Acute laryngitis (`C0001327`) | 94 | 14 |

**Diff (failure case):**
- Only-in-top1 (2):
  - `C0042963` Vomiting (w_t1=0.426, idf=2.35)
  - `C0027497` Nausea (w_t1=0.254, idf=2.43)
- Only-in-true (2):
  - `C0024204` lymph nodes (w_tr=0.181, idf=2.83)
  - `C0038990` Sweating (w_tr=0.068, idf=2.27)

---

### Failure #94826 — True: Tuberculosis (`C0041327`)
- Age=67, Sex=F, Rank=**3**, Score=0.1468

**Evidence (8 items):**
- Are you infected with the human immunodeficiency virus (HIV)? (yes)
- Do you take corticosteroids? (yes)
- Have you been coughing up blood? (yes)
- Are you experiencing shortness of breath or difficulty breathing in a significant way? (yes)
- Have you had an involuntary weight loss over the last 3 months? (yes)
- Do you have a cough? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Is your BMI less than 18.5, or are you underweight? (yes)

**Patient CUIs (10):**
- `C0005767` Blood [idf=3.12]
- `C0019693` HIV Infections [idf=2.71]
- `C0439663` Infected [idf=2.51]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C0035203` Respiration [idf=1.82]
- `C0010200` Coughing [idf=1.78]
- `C0013404` Dyspnea [idf=1.51]
- `C0041667` Underweight [idf=1.00]
- `C2363736` Unintentional weight loss [idf=1.00]
- `C0005893`  [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1653 | Whooping cough (`C0043168`) | 43 | 5 |
| 2 | 0.1543 | Bronchiectasis (`C0006267`) | 125 | 7 |
| 3 | 0.1468 | Tuberculosis (`C0041327`) ← TRUE | 81 | 5 |
| 4 | 0.1377 | Croup (`C0010380`) | 78 | 4 |
| 5 | 0.1134 | Acute COPD exacerbation / infection (`C0340044`) | 63 | 5 |

**Diff (failure case):**
- Only-in-top1 (1):
  - `C0439663` Infected (w_t1=0.247, idf=2.51)
- Only-in-true (1):
  - `C0005767` Blood (w_tr=0.241, idf=3.12)

---

### Failure #53077 — True: Pulmonary embolism (`C0034065`)
- Age=26, Sex=F, Rank=**3**, Score=0.1448

**Evidence (27 items):**
- Do you have an active cancer? (yes)
- Have you been coughing up blood? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → un_coup_de_couteau
- Characterize your pain: → vive
- Do you feel pain somewhere? → côté_du_thorax_D_
- Do you feel pain somewhere? → côté_du_thorax_G_
- Do you feel pain somewhere? → omoplate_G_
- Do you feel pain somewhere? → thorax_postérieur_D_
- Do you feel pain somewhere? → thorax_postérieur_G_
- How intense is the pain? → 3
- Does the pain radiate to another location? → côté_du_thorax_D_
- ... +15 more

**Patient CUIs (13):**
- `C5236002` Increased (finding) [idf=4.22]
- `C0005767` Blood [idf=3.12]
- `C0006826` Malignant Neoplasms [idf=2.43]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C0013604` Edema [idf=1.87]
- `C0035203` Respiration [idf=1.82]
- `C0010200` Coughing [idf=1.78]
- `C0817096` Chest [idf=1.69]
- `C0030193` Pain [idf=1.51]
- `C0013404` Dyspnea [idf=1.51]
- `C0149871` Deep Vein Thrombosis [idf=1.00]
- `C0230445` Structure of calf of leg [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1923 | Croup (`C0010380`) | 78 | 6 |
| 2 | 0.1898 | Pulmonary neoplasm (`C0348343`) | 49 | 6 |
| 3 | 0.1448 | Pulmonary embolism (`C0034065`) ← TRUE | 106 | 9 |
| 4 | 0.1332 | Epiglottitis (`C0155814`) | 104 | 7 |
| 5 | 0.1285 | Acute pulmonary edema (`C0155919`) | 76 | 8 |

**Diff (failure case):**
- Only-in-top1 (2):
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.415, idf=2.43)
  - `C5236002` Increased (finding) (w_t1=0.362, idf=4.22)
- Only-in-true (5):
  - `C0006826` Malignant Neoplasms (w_tr=0.657, idf=2.43)
  - `C0005767` Blood (w_tr=0.464, idf=3.12)
  - `C0817096` Chest (w_tr=0.444, idf=1.69)
  - `C0030193` Pain (w_tr=0.268, idf=1.51)
  - `C0149871` Deep Vein Thrombosis (w_tr=0.017, idf=1.00)

---

### Failure #57050 — True: HIV (initial infection) (`C0001175`)
- Age=49, Sex=M, Rank=**3**, Score=0.1906

**Evidence (38 items):**
- Do you have swollen or painful lymph nodes? (yes)
- Have you ever had a sexually transmitted infection? (yes)
- Have you had significantly increased sweating? (yes)
- Have you had diarrhea or an increase in stool frequency? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → pénible
- Characterize your pain: → sensible
- Characterize your pain: → une_pulsation
- Characterize your pain: → épuisante
- Do you feel pain somewhere? → colonne_cervicale
- Do you feel pain somewhere? → dessus_de_tête
- Do you feel pain somewhere? → front
- ... +26 more

**Patient CUIs (25):**
- `C0030851` penis [idf=4.22]
- `C5236002` Increased (finding) [idf=4.22]
- `C0426642` Frequency of bowel action [idf=4.22]
- `C0231528` Myalgia [idf=3.53]
- `C0237849` Peeling of skin [idf=3.53]
- `C0033774` Pruritus [idf=3.30]
- `C0030232`  [idf=3.12]
- `C0036916` Sexually Transmitted Diseases [idf=2.97]
- `C0024204` lymph nodes [idf=2.83]
- `C0015230` Exanthema [idf=2.83]
- `C0221198` Lesion [idf=2.83]
- `C0041834` Erythema [idf=2.83]
- `C0038999` Swelling [idf=2.43]
- `C0027497` Nausea [idf=2.43]
- `C0031350` Pharyngitis [idf=2.35]
- `C0042963` Vomiting [idf=2.35]
- `C0011991` Diarrhea [idf=2.35]
- `C0038990` Sweating [idf=2.27]
- `C0018670` Head [idf=1.87]
- `C0015967` Fever [idf=1.78]
- `C0030193` Pain [idf=1.51]
- `C0230007` Temporal region [idf=1.00]
- `C1578559` Buccal mucosa [idf=1.00]
- `C2363736` Unintentional weight loss [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2050 | Ebola (`C0282687`) | 87 | 10 |
| 2 | 0.1968 | Chagas (`C0041234`) | 133 | 18 |
| 3 | 0.1906 | HIV (initial infection) (`C0001175`) ← TRUE | 48 | 10 |
| 4 | 0.1466 | Anaphylaxis (`C0685898`) | 159 | 11 |
| 5 | 0.1419 | Viral pharyngitis (`C0001344`) | 66 | 8 |

**Diff (failure case):**
- Only-in-top1 (2):
  - `C0042963` Vomiting (w_t1=0.426, idf=2.35)
  - `C0027497` Nausea (w_t1=0.254, idf=2.43)
- Only-in-true (2):
  - `C0024204` lymph nodes (w_tr=0.181, idf=2.83)
  - `C0038990` Sweating (w_tr=0.068, idf=2.27)

---

### Failure #124298 — True: Boerhaave (`C0014860`)
- Age=65, Sex=M, Rank=**4**, Score=0.0700

**Evidence (18 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → déchirante
- Characterize your pain: → violente
- Characterize your pain: → écoeurante
- Do you feel pain somewhere? → flanc_D_
- Do you feel pain somewhere? → flanc_G_
- Do you feel pain somewhere? → haut_du_thorax
- Do you feel pain somewhere? → hypochondre_G_
- Do you feel pain somewhere? → épigastre
- How intense is the pain? → 10
- Does the pain radiate to another location? → colonne_dorsale
- Does the pain radiate to another location? → omoplate_D_
- ... +6 more

**Patient CUIs (7):**
- `C0085281` Addictive Behavior [idf=4.22]
- `C0005767` Blood [idf=3.12]
- `C0027497` Nausea [idf=2.43]
- `C0042963` Vomiting [idf=2.35]
- `C0030193` Pain [idf=1.51]
- `C1515974` Anatomic Site [idf=1.00]
- `C0446469` Surface region of upper chest [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1056 | Cluster headache (`C0009088`) | 101 | 4 |
| 2 | 0.0923 | Chagas (`C0041234`) | 133 | 5 |
| 3 | 0.0717 | Larygospasm (`C0023066`) | 57 | 2 |
| 4 | 0.0700 | Boerhaave (`C0014860`) ← TRUE | 56 | 3 |
| 5 | 0.0522 | Scombroid food poisoning (`C0275143`) | 87 | 3 |

**Diff (failure case):**
- Only-in-top1 (2):
  - `C0027497` Nausea (w_t1=0.404, idf=2.43)
  - `C0085281` Addictive Behavior (w_t1=0.199, idf=4.22)
- Only-in-true (1):
  - `C0005767` Blood (w_tr=0.231, idf=3.12)

---

### Failure #971 — True: Stable angina (`C0002962`)
- Age=50, Sex=M, Rank=**3**, Score=0.1355

**Evidence (24 items):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → épuisante
- Do you feel pain somewhere? → bas_du_thorax
- Do you feel pain somewhere? → biceps_D_
- Do you feel pain somewhere? → côté_du_thorax_G_
- Do you feel pain somewhere? → haut_du_thorax
- Do you feel pain somewhere? → sein_G_
- How intense is the pain? → 8
- Does the pain radiate to another location? → biceps_D_
- Does the pain radiate to another location? → biceps_G_
- Does the pain radiate to another location? → colonne_dorsale
- Does the pain radiate to another location? → trachée
- ... +12 more

**Patient CUIs (21):**
- `C0085281` Addictive Behavior [idf=4.22]
- `C5236002` Increased (finding) [idf=4.22]
- `C0020443` Hypercholesterolemia [idf=4.22]
- `C0037004` Shoulder [idf=3.81]
- `C0027051` Myocardial Infarction [idf=3.12]
- `C0497406` Overweight [idf=3.12]
- `C0040578` Trachea [idf=2.83]
- `C0011847` Diabetes [idf=2.71]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C0035203` Respiration [idf=1.82]
- `C0817096` Chest [idf=1.69]
- `C0007222` Cardiovascular Diseases [idf=1.51]
- `C0013404` Dyspnea [idf=1.51]
- `C0030193` Pain [idf=1.51]
- `C1457887` Symptoms [idf=1.45]
- `C0425043` Death of relative [idf=1.00]
- `C0006141` Breast [idf=1.00]
- `C0446469` Surface region of upper chest [idf=1.00]
- `C0559499` Biceps brachii muscle structure [idf=1.00]
- `C0033213` Problem [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2142 | Unstable angina (`C0002965`) | 83 | 11 |
| 2 | 0.1374 | Spontaneous pneumothorax (`C0032326`) | 84 | 8 |
| 3 | 0.1355 | Stable angina (`C0002962`) ← TRUE | 48 | 10 |
| 4 | 0.1242 | Possible NSTEMI / STEMI (`C0010072`) | 83 | 7 |
| 5 | 0.1233 | Croup (`C0010380`) | 78 | 8 |

**Diff (failure case):**
- Only-in-top1 (2):
  - `C0497406` Overweight (w_t1=0.282, idf=3.12)
  - `C0425043` Death of relative (w_t1=0.055, idf=1.00)
- Only-in-true (1):
  - `C0035203` Respiration (w_tr=0.368, idf=1.82)

---

### Failure #14878 — True: Acute pulmonary edema (`C0155919`)
- Age=55, Sex=F, Rank=**9**, Score=0.1174

**Evidence (32 items):**
- Have you ever had fluid in your lungs? (yes)
- Do you currently undergo dialysis? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → un_tiraillement
- Characterize your pain: → une_lourdeur_ou_serrement
- Do you feel pain somewhere? → côté_du_thorax_D_
- Do you feel pain somewhere? → haut_du_thorax
- Do you feel pain somewhere? → sein_D_
- Do you feel pain somewhere? → sein_G_
- Do you feel pain somewhere? → thorax_postérieur_D_
- How intense is the pain? → 8
- Does the pain radiate to another location? → biceps_G_
- ... +20 more

**Patient CUIs (26):**
- `C0277814` Sitting position [idf=4.22]
- `C5236002` Increased (finding) [idf=4.22]
- `C0040184` Bone structure of tibia [idf=4.22]
- `C0011946` Dialysis procedure [idf=3.81]
- `C0003086` Ankle [idf=3.81]
- `C0037004` Shoulder [idf=3.81]
- `C0231528` Myalgia [idf=3.53]
- `C0027051` Myocardial Infarction [idf=3.12]
- `C0020538` Hypertensive disease [idf=2.97]
- `C0008301` Choking [idf=2.83]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C0018801` Heart failure [idf=2.20]
- `C0013604` Edema [idf=1.87]
- `C0035203` Respiration [idf=1.82]
- `C0817096` Chest [idf=1.69]
- `C0015672` Fatigue [idf=1.65]
- `C0024109` Lung [idf=1.58]
- `C0013404` Dyspnea [idf=1.51]
- `C0030193` Pain [idf=1.51]
- `C1457887` Symptoms [idf=1.45]
- `C0230445` Structure of calf of leg [idf=1.00]
- `C0006141` Breast [idf=1.00]
- `C0446469` Surface region of upper chest [idf=1.00]
- `C0559499` Biceps brachii muscle structure [idf=1.00]
- `C0008114` Chin [idf=1.00]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1915 | Pericarditis (`C0155679`) | 89 | 13 |
| 2 | 0.1570 | Stable angina (`C0002962`) | 48 | 10 |
| 3 | 0.1434 | Pulmonary embolism (`C0034065`) | 106 | 14 |
| 4 | 0.1360 | Larygospasm (`C0023066`) | 57 | 6 |
| 5 | 0.1295 | Pneumonia (`C0694504`) | 138 | 12 |

**Diff (failure case):**
- Only-in-top1 (4):
  - `C0277814` Sitting position (w_t1=0.682, idf=4.22)
  - `C0027051` Myocardial Infarction (w_t1=0.625, idf=3.12)
  - `C0037004` Shoulder (w_t1=0.299, idf=3.81)
  - `C0020538` Hypertensive disease (w_t1=0.195, idf=2.97)
- Only-in-true (2):
  - `C0018801` Heart failure (w_tr=0.371, idf=2.20)
  - `C0003086` Ankle (w_tr=0.046, idf=3.81)

---

### Failure #72042 — True: PSVT (`C0039240`)
- Age=6, Sex=M, Rank=**3**, Score=0.1414

**Evidence (12 items):**
- Characterize your pain: → NA
- Do you feel pain somewhere? → nulle_part
- How intense is the pain? → 0
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 0
- How fast did the pain appear? → 0
- Do you consume energy drinks regularly? (yes)
- Do you regularly take stimulant drugs? (yes)
- Are you experiencing shortness of breath or difficulty breathing in a significant way? (yes)
- Do you feel your heart is beating fast (racing), irregularly (missing a beat) or do you feel palpitations? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Have you recently taken decongestants or other substances that may have stimulant effects? (yes)

**Patient CUIs (8):**
- `C0543419` Sequela of disorder [idf=2.97]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C0030252` Palpitations [idf=2.35]
- `C0035203` Respiration [idf=1.82]
- `C0030193` Pain [idf=1.51]
- `C0013404` Dyspnea [idf=1.51]
- `C0018787` Heart [idf=1.48]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.1585 | Pericarditis (`C0155679`) | 89 | 6 |
| 2 | 0.1483 | Chagas (`C0041234`) | 133 | 7 |
| 3 | 0.1414 | PSVT (`C0039240`) ← TRUE | 48 | 3 |
| 4 | 0.1293 | Myocarditis (`C0027059`) | 123 | 5 |
| 5 | 0.1253 | Atrial fibrillation (`C3264374`) | 52 | 4 |

**Diff (failure case):**
- Only-in-top1 (3):
  - `C0035203` Respiration (w_t1=0.424, idf=1.82)
  - `C0030193` Pain (w_t1=0.197, idf=1.51)
  - `C1299586` Has difficulty doing (qualifier value) (w_t1=0.168, idf=2.43)
- Only-in-true (0):

---

### Failure #59557 — True: Acute COPD exacerbation / infection (`C0340044`)
- Age=66, Sex=F, Rank=**11**, Score=0.0572

**Evidence (10 items):**
- Have you had one or several flare ups of chronic obstructive pulmonary disease (COPD) in the past year? (yes)
- Do you have a cough that produces colored or more abundant sputum than usual? (yes)
- Do you have a chronic obstructive pulmonary disease (COPD)? (yes)
- Have you ever been diagnosed with gastroesophageal reflux? (yes)
- Do you work in agriculture? (yes)
- Do you work in construction? (yes)
- Do you work in the mining sector? (yes)
- Do you have a cough? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Have you noticed a wheezing sound when you exhale? (yes)

**Patient CUIs (6):**
- `C0038056` Sputum [idf=3.12]
- `C0017168` Gastroesophageal reflux disease [idf=2.83]
- `C0043144` Wheezing [idf=2.27]
- `C0024117` Chronic Obstructive Airway Disease [idf=2.20]
- `C0010200` Coughing [idf=1.78]
- `C1517205` Flare [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.3392 | Bronchitis (`C0006277`) | 122 | 5 |
| 2 | 0.2297 | Bronchiectasis (`C0006267`) | 125 | 5 |
| 3 | 0.1679 | Bronchiolitis (`C0001311`) | 88 | 4 |
| 4 | 0.1352 | Bronchospasm / acute asthma exacerb (`C0004096`) | 143 | 4 |
| 5 | 0.1116 | Acute laryngitis (`C0001327`) | 94 | 3 |

**Diff (failure case):**
- Only-in-top1 (2):
  - `C0038056` Sputum (w_t1=1.148, idf=3.12)
  - `C0017168` Gastroesophageal reflux disease (w_t1=0.716, idf=2.83)
- Only-in-true (0):

---

### Failure #25187 — True: Epiglottitis (`C0155814`)
- Age=21, Sex=F, Rank=**3**, Score=0.1561

**Evidence (20 items):**
- Do you have pain that improves when you lean forward? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → un_coup_de_couteau
- Characterize your pain: → vive
- Do you feel pain somewhere? → amygdale_D_
- Do you feel pain somewhere? → amygdale_G_
- Do you feel pain somewhere? → arrière_du_cou
- Do you feel pain somewhere? → côté_du_cou_D_
- Do you feel pain somewhere? → palais
- How intense is the pain? → 7
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 7
- ... +8 more

**Patient CUIs (11):**
- `C0040421` Palatine Tonsil [idf=4.22]
- `C0740170` Does swallow [idf=3.81]
- `C0184511` Improved [idf=3.53]
- `C1299586` Has difficulty doing (qualifier value) [idf=2.43]
- `C0027530` Neck [idf=2.27]
- `C0019825` Hoarseness [idf=2.27]
- `C0035203` Respiration [idf=1.82]
- `C0015967` Fever [idf=1.78]
- `C0030193` Pain [idf=1.51]
- `C0013404` Dyspnea [idf=1.51]
- `C1515974` Anatomic Site [idf=1.00]

**Top-5:**
| Rank | Score | Disease | Profile | Overlap |
|---|---|---|---|---|
| 1 | 0.2338 | Viral pharyngitis (`C0001344`) | 66 | 7 |
| 2 | 0.2247 | Croup (`C0010380`) | 78 | 6 |
| 3 | 0.1561 | Epiglottitis (`C0155814`) ← TRUE | 104 | 7 |
| 4 | 0.1542 | Myasthenia gravis (`C0026896`) | 91 | 8 |
| 5 | 0.1225 | Acute laryngitis (`C0001327`) | 94 | 6 |

**Diff (failure case):**
- Only-in-top1 (2):
  - `C0040421` Palatine Tonsil (w_t1=0.990, idf=4.22)
  - `C0740170` Does swallow (w_t1=0.454, idf=3.81)
- Only-in-true (2):
  - `C0035203` Respiration (w_tr=0.668, idf=1.82)
  - `C0013404` Dyspnea (w_tr=0.366, idf=1.51)

---


## Aggregate

- Success: avg pcuis=12.8, avg true_score=0.2497
- Failure: avg pcuis=15.3, avg true_score=0.1357

### Confusion pairs (true → predicted)
- HIV (initial infection) → Ebola (2x)
- Tuberculosis → Whooping cough (1x)
- Pulmonary embolism → Croup (1x)
- Boerhaave → Cluster headache (1x)
- Stable angina → Unstable angina (1x)
- Acute pulmonary edema → Pericarditis (1x)
- PSVT → Pericarditis (1x)
- Acute COPD exacerbation / infection → Bronchitis (1x)
- Epiglottitis → Viral pharyngitis (1x)
