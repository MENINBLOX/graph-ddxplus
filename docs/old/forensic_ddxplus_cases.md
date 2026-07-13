# DDXPlus Forensic Analysis — v56 KG-NB+Cat (v42 KG, lay mode)

- KG: `/mnt/medkg/kg/onlykg_graph_v42_full_universal.pkl`
- Mode: lay (patient_reportable + history + demographic)
- κ=20.0, p_baseline=0.01
- |all_evs|=907, |diseases|=49

## Success cases (rank=1)

### Success #1 — True: GERD (C0017168)

- Rank of true: **1** / 49
- True disease score: -61.71

**Patient evidence (raw input):**
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
- Does the pain radiate to another location? → haut_du_thorax
- How precisely is the pain located? → 3
- How fast did the pain appear? → 2
- Are you significantly overweight compared to people of the same height as you? (yes)
- Do you drink alcohol excessively or do you have an addiction to alcohol? (yes)
- Do you have a hiatal hernia? (yes)
- Have you recently had stools that were black (like coal)? (yes)
- Do you think you are pregnant or are you currently pregnant? (yes)
- Do you have a burning sensation that starts in your stomach then goes up into your throat, and can be associated with a bitter taste in your mouth? (yes)
- Do you have a cough? (yes)
- ... +2 more

**Patient CUIs in profile (count=13):**
- `C0015733` Feces
- `C0085624` Burning sensation
- `C1515974` Anatomic Site
- `C0549206` Patient currently pregnant
- `C0497406` Overweight
- `C0277814` Sitting position
- `C0031354` Pharyngeal structure
- `C0226896` Oral cavity
- `C0085281` Addictive Behavior
- `C1457887` Symptoms
- `C0030193` Pain
- `C0010200` Coughing
- `C0038351` Stomach

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -61.71 | GERD (`C0017168`) ← **TRUE** | 8 | 91 |
| 2 | -62.29 | Inguinal hernia (`C0019294`) | 6 | 40 |
| 3 | -66.09 | Viral pharyngitis (`C0001344`) | 5 | 58 |
| 4 | -68.70 | URTI (`C0041912`) | 5 | 61 |
| 5 | -68.91 | Tuberculosis (`C0041327`) | 6 | 81 |

---

### Success #3 — True: Acute dystonic reactions (C0236832)

- Rank of true: **1** / 49
- True disease score: -34.49

**Patient evidence (raw input):**
- Have you started or taken any antipsychotic medication within the last 7 days? (yes)
- Have you ever felt like you were suffocating for a very short time associated with inability to breathe or speak? (yes)
- Have you been treated in hospital recently for nausea, agitation, intoxication or aggressive behavior and received medication via an intravenous or intramuscular route? (yes)
- Do you have trouble keeping your tongue in your mouth? (yes)
- Do you have a hard time opening/raising one or both eyelids? (yes)
- Do you have annoying muscle spasms in your face, neck or any other part of your body? (yes)
- Have you traveled out of the country in the last 4 weeks? → N

**Patient CUIs in profile (count=6):**
- `C0037763` Spasm
- `C0226896` Oral cavity
- `C0015426` Eyelid structure
- `C0728899` Intoxication
- `C0040408` Tongue
- `C0040615` Antipsychotic Agents

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -34.49 | Acute dystonic reactions (`C0236832`) ← **TRUE** | 4 | 57 |
| 2 | -39.63 | HIV (initial infection) (`C0001175`) | 1 | 48 |
| 3 | -41.04 | Spontaneous rib fracture (`C0478237`) | 0 | 10 |
| 4 | -41.95 | Localized edema (`C0013609`) | 3 | 51 |
| 5 | -43.84 | Anemia (`C0002871`) | 3 | 102 |

---

### Success #5 — True: URTI (C0041912)

- Rank of true: **1** / 49
- True disease score: -51.32

**Patient evidence (raw input):**
- Have you been in contact with a person with similar symptoms in the past 2 weeks? (yes)
- Have you had significantly increased sweating? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → une_lourdeur_ou_serrement
- Do you feel pain somewhere? → dessus_de_tête
- Do you feel pain somewhere? → front
- Do you feel pain somewhere? → joue_D_
- Do you feel pain somewhere? → joue_G_
- Do you feel pain somewhere? → occiput
- How intense is the pain? → 7
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 3
- How fast did the pain appear? → 0
- Do you have a fever (either felt or measured with a thermometer)? (yes)
- Do you have a sore throat? (yes)
- Do you have diffuse (widespread) muscle pain? (yes)
- Do you have nasal congestion or a clear runny nose? (yes)
- Do you have a cough? (yes)
- Have you traveled out of the country in the last 4 weeks? → N

**Patient CUIs in profile (count=13):**
- `C0018670` Head
- `C1515974` Anatomic Site
- `C0007966` Cheek structure
- `C0038990` Sweating
- `C0031350` Pharyngitis
- `C1260880` Rhinorrhea
- `C0015967` Fever
- `C0027424` Nasal congestion (finding)
- `C5236002` Increased (finding)
- `C0030193` Pain
- `C0231528` Myalgia
- `C0010200` Coughing
- `C1457887` Symptoms

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -51.32 | URTI (`C0041912`) ← **TRUE** | 9 | 61 |
| 2 | -55.13 | Chronic rhinosinusitis (`C0037199`) | 8 | 79 |
| 3 | -57.35 | Acute laryngitis (`C0001327`) | 9 | 89 |
| 4 | -57.44 | Viral pharyngitis (`C0001344`) | 7 | 58 |
| 5 | -59.03 | Cluster headache (`C0009088`) | 8 | 65 |

---

### Success #6 — True: URTI (C0041912)

- Rank of true: **1** / 49
- True disease score: -48.04

**Patient evidence (raw input):**
- Do you live with 4 or more people? (yes)
- Do you attend or work in a daycare? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Do you feel pain somewhere? → dessus_de_tête
- Do you feel pain somewhere? → front
- Do you feel pain somewhere? → joue_D_
- Do you feel pain somewhere? → occiput
- Do you feel pain somewhere? → tempe_G_
- How intense is the pain? → 7
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 3
- How fast did the pain appear? → 0
- Do you have a cough that produces colored or more abundant sputum than usual? (yes)
- Do you smoke cigarettes? (yes)
- Do you have a fever (either felt or measured with a thermometer)? (yes)
- Do you have a sore throat? (yes)
- Do you have diffuse (widespread) muscle pain? (yes)
- Do you have a cough? (yes)
- Have you traveled out of the country in the last 4 weeks? → N

**Patient CUIs in profile (count=10):**
- `C0018670` Head
- `C1515974` Anatomic Site
- `C0038056` Sputum
- `C0007966` Cheek structure
- `C0031350` Pharyngitis
- `C0015967` Fever
- `C0030193` Pain
- `C0231528` Myalgia
- `C0010200` Coughing
- `C0230007` Temporal region

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -48.04 | URTI (`C0041912`) ← **TRUE** | 6 | 61 |
| 2 | -50.06 | Ebola (`C0282687`) | 6 | 79 |
| 3 | -50.20 | Viral pharyngitis (`C0001344`) | 5 | 58 |
| 4 | -51.82 | Acute laryngitis (`C0001327`) | 6 | 89 |
| 5 | -52.20 | Acute pulmonary edema (`C0155919`) | 5 | 58 |

---

### Success #7 — True: Inguinal hernia (C0019294)

- Rank of true: **1** / 49
- True disease score: -68.11

**Patient evidence (raw input):**
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
- How fast did the pain appear? → 3
- Do you have any lesions, redness or problems on your skin that you believe are related to the condition you are consulting for? (yes)
- What color is the rash? → pale
- Do your lesions peel off? → N
- Is the rash swollen? → 5
- Where is the affected region located? → fosse_iliaque_G_
- How intense is the pain caused by the rash? → 4
- Is the lesion (or are the lesions) larger than 1cm? → O
- How severe is the itching? → 0
- Were you born prematurely or did you suffer any complication at birth? (yes)
- ... +3 more

**Patient CUIs in profile (count=14):**
- `C1515974` Anatomic Site
- `C0015230` Exanthema
- `C0237849` Peeling of skin
- `C0021853` Intestines
- `C0041834` Erythema
- `C0033774` Pruritus
- `C0019552` Hip structure
- `C0221198` Lesion
- `C0030232` 
- `C5236002` Increased (finding)
- `C0009566` Complication
- `C0030193` Pain
- `C0010200` Coughing
- `C1457887` Symptoms

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -68.11 | Inguinal hernia (`C0019294`) ← **TRUE** | 5 | 40 |
| 2 | -68.61 | Chagas (`C0041234`) | 10 | 122 |
| 3 | -68.92 | Allergic sinusitis (`C0018621`) | 7 | 53 |
| 4 | -69.15 | Acute laryngitis (`C0001327`) | 7 | 89 |
| 5 | -72.19 | Tuberculosis (`C0041327`) | 6 | 81 |

---

### Success #8 — True: Spontaneous pneumothorax (C0032326)

- Rank of true: **1** / 49
- True disease score: -54.82

**Patient evidence (raw input):**
- Do you have chest pain even at rest? (yes)
- Have you ever had a spontaneous pneumothorax? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → déchirante
- Characterize your pain: → violente
- Do you feel pain somewhere? → sein_D_
- How intense is the pain? → 7
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 4
- How fast did the pain appear? → 5
- Are you experiencing shortness of breath or difficulty breathing in a significant way? (yes)
- Do you have a chronic obstructive pulmonary disease (COPD)? (yes)
- Have any of your family members ever had a pneumothorax? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Do you have symptoms that are increased with physical exertion but alleviated with rest? (yes)
- Do you have pain that is increased when you breathe in deeply? (yes)

**Patient CUIs in profile (count=12):**
- `C1515974` Anatomic Site
- `C0013404` Dyspnea
- `C0006141` Breast
- `C0425043` Death of relative
- `C1299586` Has difficulty doing (qualifier value)
- `C0008031` Chest Pain
- `C0024117` Chronic Obstructive Airway Disease
- `C5236002` Increased (finding)
- `C0030193` Pain
- `C0035203` Respiration
- `C1457887` Symptoms
- `C0032326` Pneumothorax

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -54.82 | Spontaneous pneumothorax (`C0032326`) ← **TRUE** | 7 | 80 |
| 2 | -57.75 | Stable angina (`C0002962`) | 6 | 46 |
| 3 | -58.87 | Panic attack (`C0349232`) | 7 | 77 |
| 4 | -58.92 | Epiglottitis (`C0155814`) | 7 | 93 |
| 5 | -59.60 | Atrial fibrillation (`C3264374`) | 6 | 52 |

---

### Success #10 — True: GERD (C0017168)

- Rank of true: **1** / 49
- True disease score: -61.26

**Patient evidence (raw input):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → lancinante_/_choc_électrique
- Characterize your pain: → sensible
- Characterize your pain: → une_brûlure_ou_chaleur
- Characterize your pain: → écoeurante
- Do you feel pain somewhere? → bas_du_thorax
- Do you feel pain somewhere? → hypochondre_D_
- Do you feel pain somewhere? → hypochondre_G_
- Do you feel pain somewhere? → ventre
- Do you feel pain somewhere? → épigastre
- How intense is the pain? → 9
- Does the pain radiate to another location? → haut_du_thorax
- How precisely is the pain located? → 3
- How fast did the pain appear? → 0
- Do you drink alcohol excessively or do you have an addiction to alcohol? (yes)
- Do you smoke cigarettes? (yes)
- Do you have a hiatal hernia? (yes)
- Do you have asthma or have you ever had to use a bronchodilator in the past? (yes)
- Have you recently had stools that were black (like coal)? (yes)
- Do you have a burning sensation that starts in your stomach then goes up into your throat, and can be associated with a bitter taste in your mouth? (yes)
- ... +4 more

**Patient CUIs in profile (count=13):**
- `C0015733` Feces
- `C0085624` Burning sensation
- `C1515974` Anatomic Site
- `C0004096` Asthma
- `C0277814` Sitting position
- `C0224086` Belly of skeletal muscle
- `C0031354` Pharyngeal structure
- `C0226896` Oral cavity
- `C0085281` Addictive Behavior
- `C1457887` Symptoms
- `C0030193` Pain
- `C0010200` Coughing
- `C0038351` Stomach

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -61.26 | GERD (`C0017168`) ← **TRUE** | 8 | 91 |
| 2 | -63.18 | Viral pharyngitis (`C0001344`) | 6 | 58 |
| 3 | -64.75 | Inguinal hernia (`C0019294`) | 5 | 40 |
| 4 | -66.30 | URTI (`C0041912`) | 6 | 61 |
| 5 | -67.06 | Tuberculosis (`C0041327`) | 7 | 81 |

---

### Success #14 — True: Anemia (C0002871)

- Rank of true: **1** / 49
- True disease score: -84.14

**Patient evidence (raw input):**
- Do you have a poor diet? (yes)
- Have you ever had a diagnosis of anemia? (yes)
- Do you have any family members who have been diagnosed with anemia? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → un_tiraillement
- Characterize your pain: → épuisante
- Do you feel pain somewhere? → dessus_de_tête
- Do you feel pain somewhere? → tempe_D_
- Do you feel pain somewhere? → tempe_G_
- How intense is the pain? → 1
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 2
- How fast did the pain appear? → 4
- Are you experiencing shortness of breath or difficulty breathing in a significant way? (yes)
- Do you feel slightly dizzy or lightheaded? (yes)
- Do you feel so tired that you are unable to do your usual activities or are you stuck in your bed all day long? (yes)
- Do you constantly feel fatigued or do you have non-restful sleep? (yes)
- Do you have chronic kidney failure? (yes)
- Are you taking any new oral anticoagulants ((NOACs)? (yes)
- Is your skin much paler than usual? (yes)
- ... +3 more

**Patient CUIs in profile (count=18):**
- `C0018670` Head
- `C1515974` Anatomic Site
- `C0013404` Dyspnea
- `C0022661` Kidney Failure, Chronic
- `C0041667` Underweight
- `C0549206` Patient currently pregnant
- `C0588012` Diet poor
- `C0012833` Dizziness
- `C0425043` Death of relative
- `C0220870` Lightheadedness
- `C0005893` 
- `C1299586` Has difficulty doing (qualifier value)
- `C0015672` Fatigue
- `C0030232` 
- `C0030193` Pain
- `C0002871` Anemia
- `C0035203` Respiration
- `C0230007` Temporal region

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -84.14 | Anemia (`C0002871`) ← **TRUE** | 10 | 102 |
| 2 | -86.11 | SLE (`C0024141`) | 7 | 75 |
| 3 | -86.15 | Panic attack (`C0349232`) | 7 | 77 |
| 4 | -86.15 | Stable angina (`C0002962`) | 6 | 46 |
| 5 | -86.66 | Pericarditis (`C0155679`) | 7 | 85 |

---

### Success #16 — True: URTI (C0041912)

- Rank of true: **1** / 49
- True disease score: -50.08

**Patient evidence (raw input):**
- Have you been in contact with a person with similar symptoms in the past 2 weeks? (yes)
- Do you live with 4 or more people? (yes)
- Do you attend or work in a daycare? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → une_lourdeur_ou_serrement
- Do you feel pain somewhere? → front
- Do you feel pain somewhere? → joue_D_
- Do you feel pain somewhere? → joue_G_
- Do you feel pain somewhere? → tempe_D_
- How intense is the pain? → 8
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 3
- How fast did the pain appear? → 3
- Do you have a cough that produces colored or more abundant sputum than usual? (yes)
- Do you have a sore throat? (yes)
- Do you have diffuse (widespread) muscle pain? (yes)
- Do you have nasal congestion or a clear runny nose? (yes)
- Do you have a cough? (yes)
- Have you traveled out of the country in the last 4 weeks? → N

**Patient CUIs in profile (count=11):**
- `C1515974` Anatomic Site
- `C0038056` Sputum
- `C0007966` Cheek structure
- `C1260880` Rhinorrhea
- `C0031350` Pharyngitis
- `C0027424` Nasal congestion (finding)
- `C0030193` Pain
- `C0231528` Myalgia
- `C0010200` Coughing
- `C1457887` Symptoms
- `C0230007` Temporal region

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -50.08 | URTI (`C0041912`) ← **TRUE** | 7 | 61 |
| 2 | -52.91 | Chronic rhinosinusitis (`C0037199`) | 6 | 79 |
| 3 | -54.34 | Acute laryngitis (`C0001327`) | 7 | 89 |
| 4 | -55.54 | Inguinal hernia (`C0019294`) | 5 | 40 |
| 5 | -55.84 | Viral pharyngitis (`C0001344`) | 5 | 58 |

---

### Success #17 — True: URTI (C0041912)

- Rank of true: **1** / 49
- True disease score: -50.08

**Patient evidence (raw input):**
- Have you been in contact with a person with similar symptoms in the past 2 weeks? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → pénible
- Do you feel pain somewhere? → front
- Do you feel pain somewhere? → joue_D_
- Do you feel pain somewhere? → joue_G_
- Do you feel pain somewhere? → occiput
- Do you feel pain somewhere? → tempe_G_
- How intense is the pain? → 2
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 2
- How fast did the pain appear? → 3
- Do you have a cough that produces colored or more abundant sputum than usual? (yes)
- Do you smoke cigarettes? (yes)
- Do you have a sore throat? (yes)
- Do you have diffuse (widespread) muscle pain? (yes)
- Do you have nasal congestion or a clear runny nose? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Are you exposed to secondhand cigarette smoke on a daily basis? (yes)

**Patient CUIs in profile (count=11):**
- `C1515974` Anatomic Site
- `C0038056` Sputum
- `C0007966` Cheek structure
- `C1260880` Rhinorrhea
- `C0031350` Pharyngitis
- `C0027424` Nasal congestion (finding)
- `C0030193` Pain
- `C0231528` Myalgia
- `C0010200` Coughing
- `C1457887` Symptoms
- `C0230007` Temporal region

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -50.08 | URTI (`C0041912`) ← **TRUE** | 7 | 61 |
| 2 | -52.91 | Chronic rhinosinusitis (`C0037199`) | 6 | 79 |
| 3 | -54.34 | Acute laryngitis (`C0001327`) | 7 | 89 |
| 4 | -55.54 | Inguinal hernia (`C0019294`) | 5 | 40 |
| 5 | -55.84 | Viral pharyngitis (`C0001344`) | 5 | 58 |

---


## Failure cases (rank≥5)

### Failure #2 — True: Bronchitis (C0006277)

- Rank of true: **15** / 49
- True disease score: -64.21

**Patient evidence (raw input):**
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
- Do you have nasal congestion or a clear runny nose? (yes)
- Do you have a cough? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Have you noticed a wheezing sound when you exhale? (yes)

**Patient CUIs in profile (count=13):**
- `C0085624` Burning sensation
- `C0817096` Chest
- `C0013404` Dyspnea
- `C1260880` Rhinorrhea
- `C1515974` Anatomic Site
- `C0043144` Wheezing
- `C0031354` Pharyngeal structure
- `C1299586` Has difficulty doing (qualifier value)
- `C0027424` Nasal congestion (finding)
- `C0024117` Chronic Obstructive Airway Disease
- `C0030193` Pain
- `C0035203` Respiration
- `C0010200` Coughing

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -52.94 | Bronchiolitis (`C0001311`) | 9 | 82 |
| 2 | -55.55 | Acute laryngitis (`C0001327`) | 9 | 89 |
| 3 | -60.05 | Croup (`C0010380`) | 7 | 54 |
| 4 | -60.06 | Bronchiectasis (`C0006267`) | 8 | 107 |
| 5 | -60.83 | Spontaneous pneumothorax (`C0032326`) | 7 | 80 |

**Top-1 vs True profile comparison:**
- Top-1 (Bronchiolitis) has 82 CUIs, score -52.94
- True (Bronchitis) has 110 CUIs, score -64.21
- Patient CUIs in **both** profiles: 6
- Patient CUIs **only in top-1** profile: 3
  - `C0027424` Nasal congestion (finding) (p_top1=0.16)
  - `C1260880` Rhinorrhea (p_top1=0.25)
  - `C1299586` Has difficulty doing (qualifier value) (p_top1=0.14)
- Patient CUIs **only in true** profile: 1
  - `C0030193` Pain (p_true=0.07)

---

### Failure #9 — True: Bronchitis (C0006277)

- Rank of true: **8** / 49
- True disease score: -69.67

**Patient evidence (raw input):**
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
- How precisely is the pain located? → 7
- How fast did the pain appear? → 0
- Do you have a cough that produces colored or more abundant sputum than usual? (yes)
- Do you have a fever (either felt or measured with a thermometer)? (yes)
- Do you have a sore throat? (yes)
- Do you have a chronic obstructive pulmonary disease (COPD)? (yes)
- Do you have nasal congestion or a clear runny nose? (yes)
- Do you have a cough? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Have you noticed a wheezing sound when you exhale? (yes)

**Patient CUIs in profile (count=14):**
- `C0085624` Burning sensation
- `C0817096` Chest
- `C0038056` Sputum
- `C1515974` Anatomic Site
- `C1260880` Rhinorrhea
- `C0031350` Pharyngitis
- `C0015967` Fever
- `C0043144` Wheezing
- `C0031354` Pharyngeal structure
- `C0027424` Nasal congestion (finding)
- `C0024117` Chronic Obstructive Airway Disease
- `C0030193` Pain
- `C0010200` Coughing
- `C0006141` Breast

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -58.81 | URTI (`C0041912`) | 9 | 61 |
| 2 | -59.74 | Acute laryngitis (`C0001327`) | 9 | 89 |
| 3 | -62.55 | Viral pharyngitis (`C0001344`) | 7 | 58 |
| 4 | -63.10 | Bronchiolitis (`C0001311`) | 8 | 82 |
| 5 | -64.57 | Inguinal hernia (`C0019294`) | 6 | 40 |

**Top-1 vs True profile comparison:**
- Top-1 (URTI) has 61 CUIs, score -58.81
- True (Bronchitis) has 110 CUIs, score -69.67
- Patient CUIs in **both** profiles: 5
- Patient CUIs **only in top-1** profile: 4
  - `C0027424` Nasal congestion (finding) (p_top1=0.31)
  - `C0031350` Pharyngitis (p_top1=0.49)
  - `C0031354` Pharyngeal structure (p_top1=0.19)
  - `C1260880` Rhinorrhea (p_top1=0.21)
- Patient CUIs **only in true** profile: 2
  - `C0038056` Sputum (p_true=0.48)
  - `C0817096` Chest (p_true=0.32)

---

### Failure #11 — True: Bronchitis (C0006277)

- Rank of true: **20** / 49
- True disease score: -65.51

**Patient evidence (raw input):**
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
- Do you have a chronic obstructive pulmonary disease (COPD)? (yes)
- Do you have nasal congestion or a clear runny nose? (yes)
- Do you have a cough? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Are your vaccinations up to date? (yes)
- Have you noticed a wheezing sound when you exhale? (yes)
- Are your symptoms more prominent at night? (yes)

**Patient CUIs in profile (count=13):**
- `C0085624` Burning sensation
- `C0013404` Dyspnea
- `C1515974` Anatomic Site
- `C0042196` Vaccination
- `C1260880` Rhinorrhea
- `C0043144` Wheezing
- `C1299586` Has difficulty doing (qualifier value)
- `C0027424` Nasal congestion (finding)
- `C0024117` Chronic Obstructive Airway Disease
- `C0030193` Pain
- `C0035203` Respiration
- `C0010200` Coughing
- `C1457887` Symptoms

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -53.66 | Bronchiolitis (`C0001311`) | 9 | 82 |
| 2 | -59.99 | Acute laryngitis (`C0001327`) | 8 | 89 |
| 3 | -60.05 | Croup (`C0010380`) | 7 | 54 |
| 4 | -60.62 | Anaphylaxis (`C0685898`) | 10 | 149 |
| 5 | -60.86 | URTI (`C0041912`) | 8 | 61 |

**Top-1 vs True profile comparison:**
- Top-1 (Bronchiolitis) has 82 CUIs, score -53.66
- True (Bronchitis) has 110 CUIs, score -65.51
- Patient CUIs in **both** profiles: 6
- Patient CUIs **only in top-1** profile: 3
  - `C0027424` Nasal congestion (finding) (p_top1=0.16)
  - `C1260880` Rhinorrhea (p_top1=0.25)
  - `C1299586` Has difficulty doing (qualifier value) (p_top1=0.14)
- Patient CUIs **only in true** profile: 1
  - `C0030193` Pain (p_true=0.07)

---

### Failure #12 — True: Bronchitis (C0006277)

- Rank of true: **24** / 49
- True disease score: -77.23

**Patient evidence (raw input):**
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
- How precisely is the pain located? → 5
- How fast did the pain appear? → 2
- Are you experiencing shortness of breath or difficulty breathing in a significant way? (yes)
- Do you have a sore throat? (yes)
- Do you have a chronic obstructive pulmonary disease (COPD)? (yes)
- Do you have nasal congestion or a clear runny nose? (yes)
- Do you have a cough? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Are your vaccinations up to date? (yes)
- Have you noticed a wheezing sound when you exhale? (yes)

**Patient CUIs in profile (count=15):**
- `C0085624` Burning sensation
- `C0013404` Dyspnea
- `C1515974` Anatomic Site
- `C1260880` Rhinorrhea
- `C0042196` Vaccination
- `C0031350` Pharyngitis
- `C0043144` Wheezing
- `C0031354` Pharyngeal structure
- `C1299586` Has difficulty doing (qualifier value)
- `C0027424` Nasal congestion (finding)
- `C0024117` Chronic Obstructive Airway Disease
- `C0030193` Pain
- `C0035203` Respiration
- `C0010200` Coughing
- `C0006141` Breast

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -63.70 | Acute laryngitis (`C0001327`) | 9 | 89 |
| 2 | -63.98 | Bronchiolitis (`C0001311`) | 9 | 82 |
| 3 | -65.76 | URTI (`C0041912`) | 9 | 61 |
| 4 | -67.45 | Anaphylaxis (`C0685898`) | 11 | 149 |
| 5 | -68.53 | Viral pharyngitis (`C0001344`) | 7 | 58 |

**Top-1 vs True profile comparison:**
- Top-1 (Acute laryngitis) has 89 CUIs, score -63.70
- True (Bronchitis) has 110 CUIs, score -77.23
- Patient CUIs in **both** profiles: 5
- Patient CUIs **only in top-1** profile: 4
  - `C0027424` Nasal congestion (finding) (p_top1=0.13)
  - `C0031350` Pharyngitis (p_top1=0.39)
  - `C0031354` Pharyngeal structure (p_top1=0.40)
  - `C1260880` Rhinorrhea (p_top1=0.13)
- Patient CUIs **only in true** profile: 1
  - `C0024117` Chronic Obstructive Airway Disease (p_true=0.43)

---

### Failure #15 — True: Larygospasm (C0023066)

- Rank of true: **9** / 49
- True disease score: -23.54

**Patient evidence (raw input):**
- Have you noticed a high pitched sound when breathing in? (yes)
- Have you traveled out of the country in the last 4 weeks? → AmerN
- Are you exposed to secondhand cigarette smoke on a daily basis? (yes)

**Patient CUIs in profile (count=1):**
- `C0035203` Respiration

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -18.06 | Spontaneous rib fracture (`C0478237`) | 0 | 10 |
| 2 | -19.50 | HIV (initial infection) (`C0001175`) | 0 | 48 |
| 3 | -20.21 | Whooping cough (`C0043168`) | 1 | 43 |
| 4 | -21.61 | Acute pulmonary edema (`C0155919`) | 1 | 58 |
| 5 | -21.63 | Pulmonary neoplasm (`C0348343`) | 1 | 48 |

**Top-1 vs True profile comparison:**
- Top-1 (Spontaneous rib fracture) has 10 CUIs, score -18.06
- True (Larygospasm) has 46 CUIs, score -23.54
- Patient CUIs in **both** profiles: 0
- Patient CUIs **only in top-1** profile: 0
- Patient CUIs **only in true** profile: 1
  - `C0035203` Respiration (p_true=0.18)

---

### Failure #23 — True: Influenza (C0021400)

- Rank of true: **20** / 49
- True disease score: -88.99

**Patient evidence (raw input):**
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
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 7
- How fast did the pain appear? → 2
- Do you smoke cigarettes? (yes)
- Do you feel so tired that you are unable to do your usual activities or are you stuck in your bed all day long? (yes)
- Do you have a fever (either felt or measured with a thermometer)? (yes)
- Do you have a sore throat? (yes)
- Do you have any lesions, redness or problems on your skin that you believe are related to the condition you are consulting for? (yes)
- What color is the rash? → rose
- Do your lesions peel off? → N
- ... +12 more

**Patient CUIs in profile (count=18):**
- `C0018670` Head
- `C1515974` Anatomic Site
- `C0015230` Exanthema
- `C0038990` Sweating
- `C0237849` Peeling of skin
- `C0031350` Pharyngitis
- `C0041834` Erythema
- `C0003123` Anorexia
- `C0033774` Pruritus
- `C0015967` Fever
- `C0027530` Neck
- `C0221198` Lesion
- `C0015672` Fatigue
- `C5236002` Increased (finding)
- `C0030193` Pain
- `C0231528` Myalgia
- `C0010200` Coughing
- `C0085393` Immunocompromised Host

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -71.85 | Acute laryngitis (`C0001327`) | 12 | 89 |
| 2 | -76.11 | Ebola (`C0282687`) | 9 | 79 |
| 3 | -78.19 | Sarcoidosis (`C0036202`) | 14 | 144 |
| 4 | -79.61 | Viral pharyngitis (`C0001344`) | 8 | 58 |
| 5 | -79.71 | Acute otitis media (`C0029882`) | 9 | 59 |

**Top-1 vs True profile comparison:**
- Top-1 (Acute laryngitis) has 89 CUIs, score -71.85
- True (Influenza) has 148 CUIs, score -88.99
- Patient CUIs in **both** profiles: 8
- Patient CUIs **only in top-1** profile: 4
  - `C0027530` Neck (p_top1=0.13)
  - `C0033774` Pruritus (p_top1=0.15)
  - `C0041834` Erythema (p_top1=0.17)
  - `C0221198` Lesion (p_top1=0.22)
- Patient CUIs **only in true** profile: 1
  - `C0003123` Anorexia (p_true=0.17)

---

### Failure #25 — True: Influenza (C0021400)

- Rank of true: **12** / 49
- True disease score: -80.49

**Patient evidence (raw input):**
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
- How fast did the pain appear? → 1
- Do you smoke cigarettes? (yes)
- Do you feel so tired that you are unable to do your usual activities or are you stuck in your bed all day long? (yes)
- Do you have a fever (either felt or measured with a thermometer)? (yes)
- Do you have a sore throat? (yes)
- Do you have any lesions, redness or problems on your skin that you believe are related to the condition you are consulting for? (yes)
- What color is the rash? → rose
- Do your lesions peel off? → N
- Is the rash swollen? → 2
- Where is the affected region located? → arrière_du_cou
- ... +9 more

**Patient CUIs in profile (count=17):**
- `C0018670` Head
- `C1515974` Anatomic Site
- `C0015230` Exanthema
- `C1260880` Rhinorrhea
- `C0237849` Peeling of skin
- `C0031350` Pharyngitis
- `C0015967` Fever
- `C0041834` Erythema
- `C0033774` Pruritus
- `C0027530` Neck
- `C0027424` Nasal congestion (finding)
- `C0015672` Fatigue
- `C0221198` Lesion
- `C0030193` Pain
- `C0231528` Myalgia
- `C0010200` Coughing
- `C0085393` Immunocompromised Host

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -61.86 | Acute laryngitis (`C0001327`) | 14 | 89 |
| 2 | -68.51 | Acute rhinosinusitis (`C0149512`) | 12 | 96 |
| 3 | -71.17 | Allergic sinusitis (`C0018621`) | 10 | 53 |
| 4 | -71.24 | Chronic rhinosinusitis (`C0037199`) | 9 | 79 |
| 5 | -72.44 | Viral pharyngitis (`C0001344`) | 9 | 58 |

**Top-1 vs True profile comparison:**
- Top-1 (Acute laryngitis) has 89 CUIs, score -61.86
- True (Influenza) has 148 CUIs, score -80.49
- Patient CUIs in **both** profiles: 10
- Patient CUIs **only in top-1** profile: 4
  - `C0027530` Neck (p_top1=0.13)
  - `C0033774` Pruritus (p_top1=0.15)
  - `C0041834` Erythema (p_top1=0.17)
  - `C0221198` Lesion (p_top1=0.22)
- Patient CUIs **only in true** profile: 0

---

### Failure #26 — True: Acute dystonic reactions (C0236832)

- Rank of true: **8** / 49
- True disease score: -35.71

**Patient evidence (raw input):**
- Do you regularly take stimulant drugs? (yes)
- Have you ever felt like you were suffocating for a very short time associated with inability to breathe or speak? (yes)
- Do you have a hard time opening/raising one or both eyelids? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Do you suddenly have difficulty or an inability to open your mouth or have jaw pain when opening it? (yes)

**Patient CUIs in profile (count=4):**
- `C0226896` Oral cavity
- `C0236000` Jaw pain
- `C0015426` Eyelid structure
- `C1299586` Has difficulty doing (qualifier value)

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -29.94 | HIV (initial infection) (`C0001175`) | 2 | 48 |
| 2 | -31.85 | Spontaneous rib fracture (`C0478237`) | 0 | 10 |
| 3 | -32.67 | Scombroid food poisoning (`C0275143`) | 3 | 81 |
| 4 | -32.87 | Anemia (`C0002871`) | 4 | 102 |
| 5 | -32.87 | Myasthenia gravis (`C0026896`) | 3 | 70 |

**Top-1 vs True profile comparison:**
- Top-1 (HIV (initial infection)) has 48 CUIs, score -29.94
- True (Acute dystonic reactions) has 57 CUIs, score -35.71
- Patient CUIs in **both** profiles: 1
- Patient CUIs **only in top-1** profile: 1
  - `C0236000` Jaw pain (p_top1=0.02)
- Patient CUIs **only in true** profile: 0

---

### Failure #32 — True: Spontaneous rib fracture (C0478237)

- Rank of true: **16** / 49
- True disease score: -59.89

**Patient evidence (raw input):**
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → déchirante
- Characterize your pain: → un_coup_de_couteau
- Characterize your pain: → vive
- Do you feel pain somewhere? → bas_du_thorax
- Do you feel pain somewhere? → côté_du_thorax_D_
- Do you feel pain somewhere? → côté_du_thorax_G_
- Do you feel pain somewhere? → sein_G_
- Do you feel pain somewhere? → thorax_postérieur_G_
- How intense is the pain? → 10
- Does the pain radiate to another location? → thorax_postérieur_D_
- Does the pain radiate to another location? → thorax_postérieur_G_
- How precisely is the pain located? → 9
- How fast did the pain appear? → 8
- Are you experiencing shortness of breath or difficulty breathing in a significant way? (yes)
- Do you drink alcohol excessively or do you have an addiction to alcohol? (yes)
- Are you being treated for osteoporosis? (yes)
- Do you have intense coughing fits? (yes)
- Have you traveled out of the country in the last 4 weeks? → N
- Do you have pain that is increased with movement? (yes)
- ... +1 more

**Patient CUIs in profile (count=11):**
- `C1515974` Anatomic Site
- `C0817096` Chest
- `C0013404` Dyspnea
- `C1299586` Has difficulty doing (qualifier value)
- `C0085281` Addictive Behavior
- `C5236002` Increased (finding)
- `C0030193` Pain
- `C0035203` Respiration
- `C0010200` Coughing
- `C0006141` Breast
- `C0029456` Osteoporosis

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -54.46 | Inguinal hernia (`C0019294`) | 5 | 40 |
| 2 | -54.74 | Pulmonary neoplasm (`C0348343`) | 5 | 48 |
| 3 | -54.97 | Bronchiectasis (`C0006267`) | 8 | 107 |
| 4 | -55.50 | Acute pulmonary edema (`C0155919`) | 5 | 58 |
| 5 | -55.96 | Croup (`C0010380`) | 5 | 54 |

**Top-1 vs True profile comparison:**
- Top-1 (Inguinal hernia) has 40 CUIs, score -54.46
- True (Spontaneous rib fracture) has 10 CUIs, score -59.89
- Patient CUIs in **both** profiles: 1
- Patient CUIs **only in top-1** profile: 4
  - `C0010200` Coughing (p_top1=0.11)
  - `C0085281` Addictive Behavior (p_top1=0.07)
  - `C0817096` Chest (p_top1=0.26)
  - `C1515974` Anatomic Site (p_top1=0.11)
- Patient CUIs **only in true** profile: 1
  - `C0029456` Osteoporosis (p_true=0.06)

---

### Failure #36 — True: Influenza (C0021400)

- Rank of true: **36** / 49
- True disease score: -82.09

**Patient evidence (raw input):**
- Have you had significantly increased sweating? (yes)
- Do you have pain somewhere, related to your reason for consulting? (yes)
- Characterize your pain: → sensible
- Characterize your pain: → une_lourdeur_ou_serrement
- Do you feel pain somewhere? → arrière_du_cou
- Do you feel pain somewhere? → côté_du_cou_G_
- Do you feel pain somewhere? → dessus_de_tête
- Do you feel pain somewhere? → front
- Do you feel pain somewhere? → occiput
- How intense is the pain? → 8
- Does the pain radiate to another location? → nulle_part
- How precisely is the pain located? → 3
- How fast did the pain appear? → 4
- Do you feel so tired that you are unable to do your usual activities or are you stuck in your bed all day long? (yes)
- Do you have a fever (either felt or measured with a thermometer)? (yes)
- What color is the rash? → NA
- Do your lesions peel off? → N
- Is the rash swollen? → 0
- Where is the affected region located? → nulle_part
- How intense is the pain caused by the rash? → 0
- ... +6 more

**Patient CUIs in profile (count=15):**
- `C0018670` Head
- `C1515974` Anatomic Site
- `C0015230` Exanthema
- `C0038990` Sweating
- `C0237849` Peeling of skin
- `C0015967` Fever
- `C0033774` Pruritus
- `C0027530` Neck
- `C0015672` Fatigue
- `C0221198` Lesion
- `C5236002` Increased (finding)
- `C0030193` Pain
- `C0231528` Myalgia
- `C0010200` Coughing
- `C0085393` Immunocompromised Host

**Top-5 predictions:**
| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | -65.25 | Acute laryngitis (`C0001327`) | 10 | 89 |
| 2 | -66.72 | Sarcoidosis (`C0036202`) | 13 | 144 |
| 3 | -69.19 | Ebola (`C0282687`) | 7 | 79 |
| 4 | -72.61 | Chronic rhinosinusitis (`C0037199`) | 6 | 79 |
| 5 | -72.66 | Allergic sinusitis (`C0018621`) | 7 | 53 |

**Top-1 vs True profile comparison:**
- Top-1 (Acute laryngitis) has 89 CUIs, score -65.25
- True (Influenza) has 148 CUIs, score -82.09
- Patient CUIs in **both** profiles: 7
- Patient CUIs **only in top-1** profile: 3
  - `C0027530` Neck (p_top1=0.13)
  - `C0033774` Pruritus (p_top1=0.15)
  - `C0221198` Lesion (p_top1=0.22)
- Patient CUIs **only in true** profile: 0

---


## Aggregate analysis

### Success vs Failure pattern

- Success patients: avg patient_cuis=12.1, avg true_score=-56.40
- Failure patients: avg patient_cuis=12.1, avg true_score=-64.73
- Failure score gap (top1 - true): avg=11.68


## 패턴 분석 — 왜 실패하는가?

### 발견 1. **Bronchitis가 반복 실패** (10건 중 4건)

Failure cases #2, #9, #11, #12 모두 Bronchitis가 true.
- Top-1으로 자주 잘못 예측: Bronchiolitis, Acute laryngitis, Croup, Bronchiectasis

**이유**: 호흡기 disease cluster (cough + dyspnea + wheezing + COPD) 가 너무 공유됨.
Bronchitis profile은 **110 CUIs (크기)** 인 반면, Bronchiolitis는 **82 CUIs (집중)**.

### 발견 2. **Profile size dilution** — 진짜 bottleneck

| Case | True 점수 | True profile size | Top-1 점수 | Top-1 profile size | Patient overlap |
|---|---|---|---|---|---|
| Bronchitis fail | -64.21 | 110 | -52.94 (Bronchiolitis) | 82 | 6 vs 9 |
| Influenza fail #25 | -82.09 | 148 | -65.25 (Laryngitis) | 89 | 7 vs 10 |
| Influenza fail #36 | -82+ | 144 | -69+ | 89 | similar |

**구조적 문제**: Hill function `P(E|D) = w/(w+κ)`에서 **profile 크면 각 edge의 P가 낮아짐** → patient match 시 log P 작아짐 → 점수 손해.

```
Bronchitis (profile=110, large): 각 P(E|D) avg = 0.4 → log(0.4) = -0.92
Bronchiolitis (profile=82,  small): 각 P(E|D) avg = 0.5 → log(0.5) = -0.69
같은 patient overlap(9개)이라도 Bronchiolitis가 +2.1 더 점수 받음
```

### 발견 3. **Success는 specific evidence 보유 시**

Success #3 (Acute dystonic reactions, rank=1):
- "antipsychotic medication", "tongue protrusion", "eyelid problem", "muscle spasm"
- 이 CUIs는 **dystonic reactions에만** 특이적 (다른 disease에 없음)
- Profile 작아도 (57) discriminative power 강함

→ **specificity** 가 size보다 중요.

### 발견 4. **Cluster confusion** — 핵심 의학적 문제

호흡기 cluster (Bronchitis ↔ Bronchiolitis ↔ Laryngitis ↔ Croup ↔ Bronchiectasis) 와
상기도 cluster (URTI ↔ Influenza ↔ Pharyngitis ↔ Allergic sinusitis) 가 자주 confuse.

**의학적 현실**: 이들은 *실제로* 비슷한 증상 → 의사도 differential 어려움.
**KG의 한계**: PubMed IE가 임상 distinguishing feature를 제대로 capture 못 함.

## 해결 방향 — 우선순위

### A. Profile size normalization (즉시 시도 가능, 1시간)
NB scoring을 size-invariant하게:
- 대안 1: `log P(D|patient) = Σ log P(E∈patient|D) / |patient|` (per-evidence avg)
- 대안 2: 코사인 유사도 `cos(patient_vec, profile_vec)`
- 대안 3: Top-K profile compression (각 disease의 top 50 edges만 사용)

### B. Discriminative feature emphasis (중기, IE 재실행 필요)
IE 프롬프트에 "이 disease의 *distinguishing* feature 강조" 추가:
- "What features differentiate this disease from similar conditions?"
- 일반 증상(cough) 보다 specific feature (e.g., bronchitis의 "productive cough with purulent sputum") 우선

### C. Cluster-aware Bayesian (장기, 새 알고리즘)
호흡기 cluster 내에서 별도 pairwise discrimination layer:
- Stage 1: 49 → top-5
- Stage 2: top-5 중 cluster 내부면 specific discrimination
- (이전 v92, v327 패턴 — 단 LLM 사용으로 strict 원칙 위반 가능)

### 추천 즉시 시도

**A의 대안 1 (per-evidence avg)** 가 가장 빠르고 fair한 fix.
- 즉시 구현 가능 (10줄)
- KG 재구축 불필요
- Profile size bias 해소
- 학술적으로 정당 (NB ranking을 size-invariant로 만드는 표준 기법)
