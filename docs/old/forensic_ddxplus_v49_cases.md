# v49 (cosine + v5 prevalence) Forensic Cases — 10 Success / 10 Failure

- KG: pilot/data/onlykg_graph_v49_v5_full.pkl
- Mode: lay (cosine)
- |dcs|=49, |all_evs|=1061

## Success cases

### Success #1 — True: Acute dystonic reactions (`C0236832`)
- Rank: **1** / 49, Score: 0.4978

**Patient evidence:**
- Do you regularly take stimulant drugs? (yes)
- Have you been treated in hospital recently for nausea, agitation, intoxication or aggressive behavior and received medication via an intravenous or intramuscular route? (yes)
- Do you have trouble keeping your tongue in your mouth? (yes)
- Do you have a hard time opening/raising one or both eyelids? (yes)
- Are you unable to control the direction of your eyes? (yes)
- Do you feel that muscle spasms or soreness in your neck are keeping you from turning your head to one side? (yes)
- Do you have annoying muscle spasms in your face, neck or any other part of your body? (yes)
- Have you traveled out of the country in the last 4 weeks? → N

**Patient CUIs in profile (9):**
- `C0015392` Eye
- `C0037763` Spasm
- `C0728899` Intoxication
- `C0015426` Eyelid structure
- `C0226896` Oral cavity
- `C0040408` Tongue
- `C0234233` Sore to touch
- `C0018670` Head
- `C0027530` Neck

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.4978 | Acute dystonic reactions (`C0236832`) ← TRUE | 60 | 5 |
| 2 | 0.2900 | Myasthenia gravis (`C0026896`) | 91 | 6 |
| 3 | 0.2758 | Cluster headache (`C0009088`) | 101 | 4 |
| 4 | 0.2274 | Localized edema (`C0013609`) | 51 | 6 |
| 5 | 0.2239 | HIV (initial infection) (`C0001175`) | 48 | 4 |

---

### Success #2 — True: Viral pharyngitis (`C0001344`)
- Rank: **1** / 49, Score: 0.4103

**Patient evidence:**
- Have you been in contact with a person with similar symptoms in the past 2 weeks? (yes)
- Do you live with 4 or more people? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → une_brûlure_ou_chaleur
- Do you feel pain somewhere? → amygdale_D_
- Do you feel pain somewhere? → amygdale_G_
- Do you feel pain somewhere? → cartilage_thyroidien
- Do you feel pain somewhere? → palais
- Do you feel pain somewhere? → pharynx
- How intense is the pain? → 5
- ... +5 more

**Patient CUIs in profile (7):**
- `C0040421` Palatine Tonsil
- `C0015967` Fever
- `C0030193` Pain
- `C1457887` Symptoms
- `C0085624` Burning sensation
- `C0031354` Pharyngeal structure
- `C1515974` Anatomic Site

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.4103 | Viral pharyngitis (`C0001344`) ← TRUE | 66 | 5 |
| 2 | 0.2603 | GERD (`C0017168`) | 111 | 4 |
| 3 | 0.2447 | Acute laryngitis (`C0001327`) | 94 | 4 |
| 4 | 0.2360 | URTI (`C0041912`) | 63 | 4 |
| 5 | 0.2355 | Spontaneous rib fracture (`C0478237`) | 10 | 2 |

---

### Success #3 — True: Localized edema (`C0013609`)
- Rank: **1** / 49, Score: 0.2955

**Patient evidence:**
- Are you currently taking or have you recently taken anti-inflammatory drugs (NSAIDs)? (yes)
- Do you take corticosteroids? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → un_tiraillement
- Characterize your pain: → vive
- Do you feel pain somewhere? → cheville_D_
- Do you feel pain somewhere? → coté_lateral_du_pied_D_
- Do you feel pain somewhere? → face_dorsale_du_pied_D_
- Do you feel pain somewhere? → plante_du_pied_D_
- ... +16 more

**Patient CUIs in profile (10):**
- `C0022646` Kidney
- `C0013604` Edema
- `C0030193` Pain
- `C0039866` Thigh structure
- `C0016504` Foot
- `C0520679` Sleep Apnea, Obstructive
- `C0023890` Liver Cirrhosis
- `C0024204` lymph nodes
- `C0003086` Ankle
- `C1515974` Anatomic Site

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.2955 | Localized edema (`C0013609`) ← TRUE | 51 | 5 |
| 2 | 0.2465 | SLE (`C0024141`) | 83 | 6 |
| 3 | 0.2272 | Sarcoidosis (`C0036202`) | 168 | 9 |
| 4 | 0.1558 | Chagas (`C0041234`) | 133 | 6 |
| 5 | 0.1498 | Cluster headache (`C0009088`) | 101 | 3 |

---

### Success #5 — True: Anaphylaxis (`C0685898`)
- Rank: **1** / 49, Score: 0.3943

**Patient evidence:**
- Do you have a known severe food allergy? (yes)
- Have you been in contact with or ate something that you have an allergy to? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → vive
- Do you feel pain somewhere? → hypochondre_D_
- Do you feel pain somewhere? → hypochondre_G_
- Do you feel pain somewhere? → ventre
- How intense is the pain? → 9
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 4
- ... +24 more

**Patient CUIs in profile (23):**
- `C0205082` Severe (severity modifier)
- `C0013404` Dyspnea
- `C0030193` Pain
- `C0007966` Cheek structure
- `C0221198` Lesion
- `C0016470` Food Allergy
- `C0041834` Erythema
- `C0015230` Exanthema
- `C0020517` Hypersensitivity
- `C0559499` Biceps brachii muscle structure
- `C0003086` Ankle
- `C1515974` Anatomic Site
- `C0033774` Pruritus
- `C0220870` Lightheadedness
- `C0042963` Vomiting
- `C0013604` Edema
- `C1299586` Has difficulty doing (qualifier value)
- `C0237849` Peeling of skin
- `C0012833` Dizziness
- `C0226896` Oral cavity
- `C0224086` Belly of skeletal muscle
- `C0035203` Respiration
- `C0027497` Nausea

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.3943 | Anaphylaxis (`C0685898`) ← TRUE | 159 | 16 |
| 2 | 0.3048 | Scombroid food poisoning (`C0275143`) | 87 | 12 |
| 3 | 0.2565 | Chagas (`C0041234`) | 133 | 15 |
| 4 | 0.2559 | Epiglottitis (`C0155814`) | 104 | 7 |
| 5 | 0.2371 | Acute pulmonary edema (`C0155919`) | 76 | 11 |

---

### Success #6 — True: URTI (`C0041912`)
- Rank: **1** / 49, Score: 0.4453

**Patient evidence:**
- Have you been in contact with a person with similar symptoms in the past 2 weeks? (yes)
- Do you live with 4 or more people? (yes)
- Do you attend or work in a daycare? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → pénible
- Characterize your pain: → sensible
- Do you feel pain somewhere? → dessus_de_tête
- Do you feel pain somewhere? → front
- Do you feel pain somewhere? → joue_G_
- Do you feel pain somewhere? → occiput
- ... +10 more

**Patient CUIs in profile (11):**
- `C0015967` Fever
- `C0030193` Pain
- `C1457887` Symptoms
- `C0007966` Cheek structure
- `C0231528` Myalgia
- `C0230007` Temporal region
- `C0018670` Head
- `C0031350` Pharyngitis
- `C0027424` Nasal congestion (finding)
- `C1260880` Rhinorrhea
- `C1515974` Anatomic Site

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.4453 | URTI (`C0041912`) ← TRUE | 63 | 8 |
| 2 | 0.3746 | Chronic rhinosinusitis (`C0037199`) | 99 | 7 |
| 3 | 0.3728 | Viral pharyngitis (`C0001344`) | 66 | 6 |
| 4 | 0.3680 | Ebola (`C0282687`) | 87 | 6 |
| 5 | 0.3327 | Acute rhinosinusitis (`C0149512`) | 114 | 7 |

---

### Success #7 — True: Panic attack (`C0349232`)
- Rank: **1** / 49, Score: 0.3376

**Patient evidence:**
- Do any members of your immediate family have a psychiatric illness? (yes)
- Have you had significantly increased sweating? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → une_crampe
- Characterize your pain: → vive
- Do you feel pain somewhere? → côté_du_thorax_G_
- Do you feel pain somewhere? → fosse_iliaque_D_
- Do you feel pain somewhere? → fosse_iliaque_G_
- Do you feel pain somewhere? → hypochondre_D_
- Do you feel pain somewhere? → hypochondre_G_
- ... +13 more

**Patient CUIs in profile (20):**
- `C0004936` Mental disorders
- `C0030193` Pain
- `C0446516` Upper arm
- `C0020580` Hypesthesia
- `C0026821` Muscle Cramp
- `C0038990` Sweating
- `C0016053` Fibromyalgia
- `C0018787` Heart
- `C0004096` Asthma
- `C1515974` Anatomic Site
- `C0817096` Chest
- `C0042963` Vomiting
- `C0220870` Lightheadedness
- `C1140621` Leg
- `C0012833` Dizziness
- `C0030252` Palpitations
- `C0030554` Paresthesia
- `C0226896` Oral cavity
- `C5236002` Increased (finding)
- `C0027497` Nausea

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.3376 | Panic attack (`C0349232`) ← TRUE | 82 | 11 |
| 2 | 0.3051 | PSVT (`C0039240`) | 48 | 7 |
| 3 | 0.3048 | Scombroid food poisoning (`C0275143`) | 87 | 11 |
| 4 | 0.3017 | Sarcoidosis (`C0036202`) | 168 | 14 |
| 5 | 0.3002 | Guillain-Barré syndrome (`C0018378`) | 103 | 10 |

---

### Success #8 — True: Anaphylaxis (`C0685898`)
- Rank: **1** / 49, Score: 0.3937

**Patient evidence:**
- Do you have a known severe food allergy? (yes)
- Have you been in contact with or ate something that you have an allergy to? (yes)
- Have you had diarrhea or an increase in stool frequency? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → vive
- Do you feel pain somewhere? → flanc_G_
- Do you feel pain somewhere? → fosse_iliaque_G_
- Do you feel pain somewhere? → hypochondre_G_
- Do you feel pain somewhere? → pubis
- Do you feel pain somewhere? → ventre
- ... +26 more

**Patient CUIs in profile (25):**
- `C0205082` Severe (severity modifier)
- `C0013404` Dyspnea
- `C0030193` Pain
- `C0007966` Cheek structure
- `C0221198` Lesion
- `C0016470` Food Allergy
- `C0041834` Erythema
- `C0027530` Neck
- `C0015230` Exanthema
- `C0020517` Hypersensitivity
- `C0559499` Biceps brachii muscle structure
- `C0011991` Diarrhea
- `C0003086` Ankle
- `C1515974` Anatomic Site
- `C0033774` Pruritus
- `C0042963` Vomiting
- `C0013604` Edema
- `C0043144` Wheezing
- `C1299586` Has difficulty doing (qualifier value)
- `C0237849` Peeling of skin
- `C0426642` Frequency of bowel action
- `C0224086` Belly of skeletal muscle
- `C0041657` Unconscious State
- `C0035203` Respiration
- `C0027497` Nausea

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.3937 | Anaphylaxis (`C0685898`) ← TRUE | 159 | 16 |
| 2 | 0.3162 | Scombroid food poisoning (`C0275143`) | 87 | 13 |
| 3 | 0.3116 | Chagas (`C0041234`) | 133 | 17 |
| 4 | 0.2761 | Boerhaave (`C0014860`) | 56 | 11 |
| 5 | 0.2621 | Epiglottitis (`C0155814`) | 104 | 8 |

---

### Success #9 — True: GERD (`C0017168`)
- Rank: **1** / 49, Score: 0.2480

**Patient evidence:**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → lancinante_/_choc_électrique
- Characterize your pain: → sensible
- Characterize your pain: → un_tiraillement
- Do you feel pain somewhere? → bas_du_thorax
- Do you feel pain somewhere? → hypochondre_G_
- How intense is the pain? → 8
- Does the pain radiate to another location? → bas_du_thorax
- Does the pain radiate to another location? → haut_du_thorax
- How precisely is the pain located? → 7
- ... +11 more

**Patient CUIs in profile (11):**
- `C0030193` Pain
- `C0446469` Surface region of upper chest
- `C0277814` Sitting position
- `C0005767` Blood
- `C1457887` Symptoms
- `C0497406` Overweight
- `C0010200` Coughing
- `C0085281` Addictive Behavior
- `C0004096` Asthma
- `C0015733` Feces
- `C1515974` Anatomic Site

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.2480 | GERD (`C0017168`) ← TRUE | 111 | 6 |
| 2 | 0.2316 | Inguinal hernia (`C0019294`) | 44 | 6 |
| 3 | 0.2300 | Chronic rhinosinusitis (`C0037199`) | 99 | 4 |
| 4 | 0.2170 | Acute rhinosinusitis (`C0149512`) | 114 | 5 |
| 5 | 0.2124 | Viral pharyngitis (`C0001344`) | 66 | 4 |

---

### Success #12 — True: GERD (`C0017168`)
- Rank: **1** / 49, Score: 0.3711

**Patient evidence:**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → lancinante_/_choc_électrique
- Characterize your pain: → un_tiraillement
- Characterize your pain: → une_brûlure_ou_chaleur
- Characterize your pain: → écoeurante
- Characterize your pain: → épeurante
- Do you feel pain somewhere? → épigastre
- How intense is the pain? → 6
- Does the pain radiate to another location? → bas_du_thorax
- How precisely is the pain located? → 7
- ... +11 more

**Patient CUIs in profile (12):**
- `C0030193` Pain
- `C0038351` Stomach
- `C1457887` Symptoms
- `C0277814` Sitting position
- `C0226896` Oral cavity
- `C0497406` Overweight
- `C0010200` Coughing
- `C0085624` Burning sensation
- `C0549206` Patient currently pregnant
- `C0031354` Pharyngeal structure
- `C0015733` Feces
- `C1515974` Anatomic Site

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.3711 | GERD (`C0017168`) ← TRUE | 111 | 9 |
| 2 | 0.2621 | Viral pharyngitis (`C0001344`) | 66 | 5 |
| 3 | 0.2210 | Acute laryngitis (`C0001327`) | 94 | 4 |
| 4 | 0.2094 | Inguinal hernia (`C0019294`) | 44 | 6 |
| 5 | 0.1911 | Stable angina (`C0002962`) | 48 | 4 |

---

### Success #13 — True: Pneumonia (`C0694504`)
- Rank: **1** / 49, Score: 0.3339

**Patient evidence:**
- Have you been coughing up blood? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → un_coup_de_couteau
- Do you feel pain somewhere? → côté_du_thorax_G_
- Do you feel pain somewhere? → sein_D_
- Do you feel pain somewhere? → thorax_postérieur_D_
- How intense is the pain? → 2
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 7
- How fast did the pain appear? → 4
- ... +25 more

**Patient CUIs in profile (25):**
- `C0038056` Sputum
- `C0042196` Vaccination
- `C0030193` Pain
- `C0005893` 
- `C0221198` Lesion
- `C0041834` Erythema
- `C0231528` Myalgia
- `C0027530` Neck
- `C0015230` Exanthema
- `C0015672` Fatigue
- `C0038454` Cerebrovascular accident
- `C0006141` Breast
- `C0004096` Asthma
- `C0036973` Shivering
- `C0018801` Heart failure
- `C1515974` Anatomic Site
- `C0817096` Chest
- `C0015967` Fever
- `C0033774` Pruritus
- `C0041667` Underweight
- `C0237849` Peeling of skin
- `C0005767` Blood
- `C0010200` Coughing
- `C5236002` Increased (finding)
- `C0024117` Chronic Obstructive Airway Disease

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.3339 | Pneumonia (`C0694504`) ← TRUE | 138 | 15 |
| 2 | 0.2952 | Bronchitis (`C0006277`) | 122 | 12 |
| 3 | 0.2895 | Bronchiectasis (`C0006267`) | 125 | 13 |
| 4 | 0.2671 | Sarcoidosis (`C0036202`) | 168 | 20 |
| 5 | 0.2570 | Boerhaave (`C0014860`) | 56 | 6 |

---


## Failure cases

### Failure #4 — True: Possible NSTEMI / STEMI (`C0010072`)
- Rank: **11** / 49, Score: 0.2096

**Patient evidence:**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → pénible
- Characterize your pain: → une_lourdeur_ou_serrement
- Characterize your pain: → vive
- Characterize your pain: → écoeurante
- Characterize your pain: → épeurante
- Do you feel pain somewhere? → bas_du_thorax
- Do you feel pain somewhere? → côté_du_thorax_D_
- Do you feel pain somewhere? → haut_du_thorax
- Do you feel pain somewhere? → sein_D_
- ... +19 more

**Patient CUIs in profile (18):**
- `C0011847` Diabetes
- `C0027051` Myocardial Infarction
- `C0817096` Chest
- `C0042963` Vomiting
- `C0027497` Nausea
- `C0030193` Pain
- `C0013404` Dyspnea
- `C0446469` Surface region of upper chest
- `C0037004` Shoulder
- `C0015672` Fatigue
- `C0003123` Anorexia
- `C0497406` Overweight
- `C0006141` Breast
- `C1299586` Has difficulty doing (qualifier value)
- `C0020443` Hypercholesterolemia
- `C0035203` Respiration
- `C0559499` Biceps brachii muscle structure
- `C1515974` Anatomic Site

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.3053 | Stable angina (`C0002962`) | 48 | 10 |
| 2 | 0.2708 | Pericarditis (`C0155679`) | 89 | 11 |
| 3 | 0.2616 | Chagas (`C0041234`) | 133 | 10 |
| 4 | 0.2553 | Unstable angina (`C0002965`) | 83 | 11 |
| 5 | 0.2417 | Pancreatic neoplasm (`C0346647`) | 44 | 6 |

**Diff vs True:**
- Only in top-1 profile (3):
  - `C0011847` Diabetes (P=0.03)
  - `C0020443` Hypercholesterolemia (P=0.06)
  - `C0037004` Shoulder (P=0.17)
- Only in true profile (1):
  - `C0042963` Vomiting (P=0.07)

---

### Failure #11 — True: Acute rhinosinusitis (`C0149512`)
- Rank: **8** / 49, Score: 0.2429

**Patient evidence:**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → une_brûlure_ou_chaleur
- Do you feel pain somewhere? → bouche
- Do you feel pain somewhere? → joue_D_
- Do you feel pain somewhere? → joue_G_
- Do you feel pain somewhere? → nez
- Do you feel pain somewhere? → oeil_G_
- How intense is the pain? → 2
- Does the pain radiate to another location? → arrière_de_tête
- Does the pain radiate to another location? → front
- ... +12 more

**Patient CUIs in profile (12):**
- `C0015392` Eye
- `C0017168` Gastroesophageal reflux disease
- `C0030193` Pain
- `C0226896` Oral cavity
- `C0007966` Cheek structure
- `C0032285` Pneumonia
- `C0020517` Hypersensitivity
- `C0010200` Coughing
- `C0085624` Burning sensation
- `C0004096` Asthma
- `C0230007` Temporal region
- `C1515974` Anatomic Site

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.3118 | GERD (`C0017168`) | 111 | 7 |
| 2 | 0.3097 | Chronic rhinosinusitis (`C0037199`) | 99 | 7 |
| 3 | 0.2987 | Bronchospasm / acute asthma exacerbation (`C0004096`) | 143 | 6 |
| 4 | 0.2653 | Bronchiectasis (`C0006267`) | 125 | 7 |
| 5 | 0.2531 | Pneumonia (`C0694504`) | 138 | 6 |

**Diff vs True:**
- Only in top-1 profile (3):
  - `C0017168` Gastroesophageal reflux disease (P=0.38)
  - `C0085624` Burning sensation (P=0.29)
  - `C0226896` Oral cavity (P=0.42)
- Only in true profile (2):
  - `C0015392` Eye (P=0.19)
  - `C0032285` Pneumonia (P=0.23)

---

### Failure #21 — True: Influenza (`C0021400`)
- Rank: **20** / 49, Score: 0.1744

**Patient evidence:**
- Have you had significantly increased sweating? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → épuisante
- Do you feel pain somewhere? → arrière_du_cou
- Do you feel pain somewhere? → côté_du_cou_D_
- Do you feel pain somewhere? → côté_du_cou_G_
- Do you feel pain somewhere? → dessus_de_tête
- Do you feel pain somewhere? → front
- How intense is the pain? → 8
- ... +19 more

**Patient CUIs in profile (15):**
- `C0038990` Sweating
- `C0015967` Fever
- `C0033774` Pruritus
- `C0030193` Pain
- `C0015230` Exanthema
- `C0015672` Fatigue
- `C0221198` Lesion
- `C0003123` Anorexia
- `C0041834` Erythema
- `C5236002` Increased (finding)
- `C0237849` Peeling of skin
- `C0018670` Head
- `C0031350` Pharyngitis
- `C0027530` Neck
- `C1515974` Anatomic Site

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.3360 | Viral pharyngitis (`C0001344`) | 66 | 7 |
| 2 | 0.3356 | Ebola (`C0282687`) | 87 | 7 |
| 3 | 0.3065 | Acute laryngitis (`C0001327`) | 94 | 10 |
| 4 | 0.2479 | Tuberculosis (`C0041327`) | 81 | 7 |
| 5 | 0.2465 | Sarcoidosis (`C0036202`) | 168 | 12 |

**Diff vs True:**
- Only in top-1 profile (1):
  - `C0027530` Neck (P=0.31)
- Only in true profile (2):
  - `C0003123` Anorexia (P=0.17)
  - `C0237849` Peeling of skin (P=0.06)

---

### Failure #22 — True: HIV (initial infection) (`C0001175`)
- Rank: **7** / 49, Score: 0.1821

**Patient evidence:**
- Do you have swollen or painful lymph nodes? (yes)
- Have you ever had a sexually transmitted infection? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → pénible
- Characterize your pain: → sensible
- Characterize your pain: → une_pulsation
- Characterize your pain: → épuisante
- Do you feel pain somewhere? → arrière_de_tête
- Do you feel pain somewhere? → arrière_du_cou
- Do you feel pain somewhere? → front
- ... +24 more

**Patient CUIs in profile (19):**
- `C0030193` Pain
- `C0221198` Lesion
- `C0041834` Erythema
- `C0027530` Neck
- `C0227121` Gum of maxilla
- `C0038999` Swelling
- `C0015230` Exanthema
- `C0015672` Fatigue
- `C1515974` Anatomic Site
- `C0030232` 
- `C0015967` Fever
- `C0033774` Pruritus
- `C1578559` Buccal mucosa
- `C0024204` lymph nodes
- `C0237849` Peeling of skin
- `C0227123` Gum of mandible
- `C2363736` Unintentional weight loss
- `C0036916` Sexually Transmitted Diseases
- `C0230007` Temporal region

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.3076 | SLE (`C0024141`) | 83 | 11 |
| 2 | 0.2374 | Acute laryngitis (`C0001327`) | 94 | 10 |
| 3 | 0.2259 | Viral pharyngitis (`C0001344`) | 66 | 7 |
| 4 | 0.2055 | Sarcoidosis (`C0036202`) | 168 | 12 |
| 5 | 0.1992 | Chagas (`C0041234`) | 133 | 12 |

**Diff vs True:**
- Only in top-1 profile (5):
  - `C0030232`  (P=0.17)
  - `C0227121` Gum of maxilla (P=0.04)
  - `C0227123` Gum of mandible (P=0.04)
- Only in true profile (0):

---

### Failure #43 — True: Pulmonary embolism (`C0034065`)
- Rank: **13** / 49, Score: 0.2360

**Patient evidence:**
- Do you have an active cancer? (yes)
- Have you been coughing up blood? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → vive
- Do you feel pain somewhere? → côté_du_thorax_G_
- Do you feel pain somewhere? → omoplate_D_
- Do you feel pain somewhere? → omoplate_G_
- Do you feel pain somewhere? → thorax_postérieur_D_
- Do you feel pain somewhere? → thorax_postérieur_G_
- ... +17 more

**Patient CUIs in profile (16):**
- `C0230445` Structure of calf of leg
- `C0817096` Chest
- `C0149871` Deep Vein Thrombosis
- `C0013604` Edema
- `C0030193` Pain
- `C0013404` Dyspnea
- `C0005767` Blood
- `C0006826` Malignant Neoplasms
- `C0006141` Breast
- `C1299586` Has difficulty doing (qualifier value)
- `C0010200` Coughing
- `C0751438` Posterior pituitary disease
- `C5236002` Increased (finding)
- `C0035203` Respiration
- `C0003086` Ankle
- `C1515974` Anatomic Site

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.2846 | Pulmonary neoplasm (`C0348343`) | 49 | 6 |
| 2 | 0.2766 | Croup (`C0010380`) | 78 | 6 |
| 3 | 0.2750 | Acute laryngitis (`C0001327`) | 94 | 7 |
| 4 | 0.2686 | Acute pulmonary edema (`C0155919`) | 76 | 9 |
| 5 | 0.2567 | Spontaneous pneumothorax (`C0032326`) | 84 | 8 |

**Diff vs True:**
- Only in top-1 profile (0):
- Only in true profile (3):
  - `C0005767` Blood (P=0.20)
  - `C0030193` Pain (P=0.20)
  - `C0149871` Deep Vein Thrombosis (P=0.02)

---

### Failure #50 — True: Pulmonary neoplasm (`C0348343`)
- Rank: **12** / 49, Score: 0.2275

**Patient evidence:**
- Have you been coughing up blood? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → un_coup_de_couteau
- Characterize your pain: → épuisante
- Do you feel pain somewhere? → thorax_postérieur_D_
- How intense is the pain? → 4
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 2
- How fast did the pain appear? → 2
- ... +7 more

**Patient CUIs in profile (10):**
- `C1515974` Anatomic Site
- `C0030193` Pain
- `C0013404` Dyspnea
- `C0005767` Blood
- `C0015672` Fatigue
- `C0337671` Former smoker
- `C0010200` Coughing
- `C1299586` Has difficulty doing (qualifier value)
- `C0035203` Respiration
- `C2363736` Unintentional weight loss

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.3118 | Bronchiectasis (`C0006267`) | 125 | 8 |
| 2 | 0.3115 | Whooping cough (`C0043168`) | 43 | 4 |
| 3 | 0.2778 | Acute laryngitis (`C0001327`) | 94 | 6 |
| 4 | 0.2748 | Croup (`C0010380`) | 78 | 4 |
| 5 | 0.2676 | Bronchitis (`C0006277`) | 122 | 7 |

**Diff vs True:**
- Only in top-1 profile (3):
  - `C0005767` Blood (P=0.23)
  - `C0030193` Pain (P=0.07)
  - `C1299586` Has difficulty doing (qualifier value) (P=0.17)
- Only in true profile (0):

---

### Failure #76 — True: HIV (initial infection) (`C0001175`)
- Rank: **7** / 49, Score: 0.2060

**Patient evidence:**
- Do you have swollen or painful lymph nodes? (yes)
- Have you ever had a sexually transmitted infection? (yes)
- Have you had significantly increased sweating? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → pénible
- Characterize your pain: → sensible
- Characterize your pain: → une_pulsation
- Characterize your pain: → épuisante
- Do you feel pain somewhere? → arrière_de_tête
- Do you feel pain somewhere? → dessus_de_tête
- ... +24 more

**Patient CUIs in profile (24):**
- `C0030193` Pain
- `C1384606` Dyspareunia
- `C0221198` Lesion
- `C0041834` Erythema
- `C5236002` Increased (finding)
- `C0227121` Gum of maxilla
- `C0038999` Swelling
- `C0015230` Exanthema
- `C0015672` Fatigue
- `C1515974` Anatomic Site
- `C0030232` 
- `C0015967` Fever
- `C0033774` Pruritus
- `C0042963` Vomiting
- `C1578559` Buccal mucosa
- `C0024204` lymph nodes
- `C0237849` Peeling of skin
- `C0227123` Gum of mandible
- `C2363736` Unintentional weight loss
- `C0036916` Sexually Transmitted Diseases
- `C0018670` Head
- `C0038990` Sweating
- `C0230007` Temporal region
- `C0027497` Nausea

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.2865 | SLE (`C0024141`) | 83 | 12 |
| 2 | 0.2605 | Ebola (`C0282687`) | 87 | 8 |
| 3 | 0.2604 | Chagas (`C0041234`) | 133 | 15 |
| 4 | 0.2199 | Acute laryngitis (`C0001327`) | 94 | 11 |
| 5 | 0.2146 | Sarcoidosis (`C0036202`) | 168 | 14 |

**Diff vs True:**
- Only in top-1 profile (5):
  - `C0030232`  (P=0.17)
  - `C0227121` Gum of maxilla (P=0.04)
  - `C0227123` Gum of mandible (P=0.04)
- Only in true profile (1):
  - `C0038990` Sweating (P=0.04)

---

### Failure #80 — True: Influenza (`C0021400`)
- Rank: **12** / 49, Score: 0.2083

**Patient evidence:**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Do you feel pain somewhere? → côté_du_cou_D_
- Do you feel pain somewhere? → côté_du_cou_G_
- Do you feel pain somewhere? → front
- Do you feel pain somewhere? → occiput
- Do you feel pain somewhere? → pharynx
- How intense is the pain? → 3
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 8
- ... +15 more

**Patient CUIs in profile (14):**
- `C0027424` Nasal congestion (finding)
- `C0015967` Fever
- `C0033774` Pruritus
- `C0030193` Pain
- `C0015230` Exanthema
- `C0015672` Fatigue
- `C0221198` Lesion
- `C0237849` Peeling of skin
- `C0031354` Pharyngeal structure
- `C0031350` Pharyngitis
- `C0036973` Shivering
- `C0027530` Neck
- `C1260880` Rhinorrhea
- `C1515974` Anatomic Site

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.3721 | Acute laryngitis (`C0001327`) | 94 | 11 |
| 2 | 0.3591 | Viral pharyngitis (`C0001344`) | 66 | 8 |
| 3 | 0.3035 | URTI (`C0041912`) | 63 | 6 |
| 4 | 0.2746 | Chronic rhinosinusitis (`C0037199`) | 99 | 6 |
| 5 | 0.2659 | Ebola (`C0282687`) | 87 | 6 |

**Diff vs True:**
- Only in top-1 profile (3):
  - `C0027530` Neck (P=0.13)
  - `C0033774` Pruritus (P=0.15)
  - `C0221198` Lesion (P=0.22)
- Only in true profile (1):
  - `C0036973` Shivering (P=0.18)

---

### Failure #81 — True: Pulmonary embolism (`C0034065`)
- Rank: **8** / 49, Score: 0.2411

**Patient evidence:**
- Do you have an active cancer? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → un_coup_de_couteau
- Characterize your pain: → vive
- Do you feel pain somewhere? → omoplate_D_
- Do you feel pain somewhere? → sein_D_
- Do you feel pain somewhere? → sein_G_
- Do you feel pain somewhere? → thorax_postérieur_D_
- Do you feel pain somewhere? → thorax_postérieur_G_
- ... +16 more

**Patient CUIs in profile (13):**
- `C0230445` Structure of calf of leg
- `C0817096` Chest
- `C0013604` Edema
- `C0030193` Pain
- `C0013404` Dyspnea
- `C0041657` Unconscious State
- `C0006141` Breast
- `C1299586` Has difficulty doing (qualifier value)
- `C0149871` Deep Vein Thrombosis
- `C5236002` Increased (finding)
- `C0035203` Respiration
- `C0006826` Malignant Neoplasms
- `C1515974` Anatomic Site

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.2837 | Spontaneous pneumothorax (`C0032326`) | 84 | 7 |
| 2 | 0.2637 | Stable angina (`C0002962`) | 48 | 4 |
| 3 | 0.2595 | Pericarditis (`C0155679`) | 89 | 8 |
| 4 | 0.2589 | Epiglottitis (`C0155814`) | 104 | 6 |
| 5 | 0.2569 | Croup (`C0010380`) | 78 | 6 |

**Diff vs True:**
- Only in top-1 profile (0):
- Only in true profile (1):
  - `C0149871` Deep Vein Thrombosis (P=0.02)

---

### Failure #84 — True: Spontaneous rib fracture (`C0478237`)
- Rank: **11** / 49, Score: 0.1621

**Patient evidence:**
- Do you have metastatic cancer? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → déchirante
- Characterize your pain: → lancinante_/_choc_électrique
- Characterize your pain: → violente
- Characterize your pain: → vive
- Do you feel pain somewhere? → côté_du_thorax_D_
- Do you feel pain somewhere? → sein_G_
- Do you feel pain somewhere? → thorax_postérieur_D_
- How intense is the pain? → 7
- ... +9 more

**Patient CUIs in profile (8):**
- `C0817096` Chest
- `C0030193` Pain
- `C0006141` Breast
- `C0010200` Coughing
- `C0085281` Addictive Behavior
- `C5236002` Increased (finding)
- `C0027627` Neoplasm Metastasis
- `C1515974` Anatomic Site

**Top-5:**
| Rank | Score | Disease | Profile size | Overlap |
|---|---|---|---|---|
| 1 | 0.2956 | Inguinal hernia (`C0019294`) | 44 | 5 |
| 2 | 0.2469 | Stable angina (`C0002962`) | 48 | 2 |
| 3 | 0.2149 | Pancreatic neoplasm (`C0346647`) | 44 | 3 |
| 4 | 0.1978 | GERD (`C0017168`) | 111 | 4 |
| 5 | 0.1965 | Boerhaave (`C0014860`) | 56 | 2 |

**Diff vs True:**
- Only in top-1 profile (4):
  - `C0010200` Coughing (P=0.11)
  - `C0085281` Addictive Behavior (P=0.07)
  - `C0817096` Chest (P=0.27)
- Only in true profile (0):

---

