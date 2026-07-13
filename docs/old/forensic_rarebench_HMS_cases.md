# rarebench_HMS Forensic Analysis — v59 cosine (v42 KG, clinical mode)

- KG: /mnt/medkg/kg/onlykg_graph_v42_full_universal.pkl
- Mode: clinical
- |dcs|=78, |all_evs|=2154

## Success cases (rank=1, 10 samples)

### Success #13 — True: Focal segmental glomerulosclerosis 9 (`C4015555`)

- Rank of true: **1** / 78, Score: 0.0512


**Patient evidence (7 CUIs in profile universe):**

- `C0742906` C-reactive protein above reference range
- `C0239981` Hypoalbuminemia
- `C0239937` Microscopic hematuria
- `C0017668` Focal glomerulosclerosis
- `C4022832` Mild proteinuria
- `C0041834` Erythema
- `C0151632` Erythrocyte sedimentation rate raised


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2296 | Focal segmental glomerulosclerosis 2 (`C1858915`) | 1 | 26 |
| 2 | 0.2119 | Focal segmental glomerulosclerosis 3, su (`C1842982`) | 2 | 57 |
| 3 | 0.0980 | Muckle-Wells syndrome/Muckle-Wells syndr (`C0268390`) | 2 | 98 |
| 4 | 0.0593 | Sarcoidosis/Sarcoidosis, susceptibility  (`C0036202`) | 3 | 452 |
| 5 | 0.0535 | Tumor necrosis factor receptor 1 associa (`C1275126`) | 1 | 32 |

---
### Success #21 — True: Familial cold autoinflammatory syndrome 3 (`C3280914`)

- Rank of true: **1** / 78, Score: -1000000000.0000


**Patient evidence (9 CUIs in profile universe):**

- `C0742906` C-reactive protein above reference range
- `C0012833` Dizziness
- `C0239266` Pain of elbow region
- `C0020437` Hypercalcemia
- `C0015967` Fever
- `C0038002` Splenomegaly
- `C0011053` Deafness
- `C0039231` Tachycardia
- `C0151632` Erythrocyte sedimentation rate raised


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1726 | Muckle-Wells syndrome/Muckle-Wells syndr (`C0268390`) | 4 | 98 |
| 2 | 0.1126 | Sarcoidosis/Sarcoidosis, susceptibility  (`C0036202`) | 8 | 452 |
| 3 | 0.1092 | Autosomal dominant otospondylomegaepiphy (`C1861481`) | 1 | 6 |
| 4 | 0.0950 | Systemic-onset juvenile idiopathic arthr (`C0087031`) | 3 | 83 |
| 5 | 0.0792 | 家族性地中海热/Familial mediterranean fever; FM (`C0031069`) | 1 | 35 |

---
### Success #22 — True: Sarcoidosis, susceptibility to, 2 (`C2676468`)

- Rank of true: **1** / 78, Score: 0.0288


**Patient evidence (10 CUIs in profile universe):**

- `C0235896` Pulmonary Infiltrate
- `C0742906` C-reactive protein above reference range
- `C0239266` Pain of elbow region
- `C2240374` Eosinophil count raised (finding)
- `C0019079` Hemoptysis
- `C1262477` Weight Loss
- `C0231807` Dyspnea on exertion
- `C0424551` Impaired exercise tolerance
- `C0235592` Cervical lymphadenopathy
- `C0015672` Fatigue


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1028 | Sarcoidosis/Sarcoidosis, susceptibility  (`C0036202`) | 6 | 452 |
| 2 | 0.0941 | Eosinophilic granulomatosis with polyang (`C0008728`) | 2 | 65 |
| 3 | 0.0796 | Muckle-Wells syndrome/Muckle-Wells syndr (`C0268390`) | 2 | 98 |
| 4 | 0.0789 | Kimura disease (`C0033838`) | 2 | 62 |
| 5 | 0.0661 | Granulomatosis with polyangiitis/Granulo (`C3495801`) | 1 | 20 |

---
### Success #25 — True: Kimura disease (`C0033838`)

- Rank of true: **1** / 78, Score: 0.1801


**Patient evidence (4 CUIs in profile universe):**

- `C0235896` Pulmonary Infiltrate
- `C0236175` Increased circulating IgE concentration
- `C0497156` Lymphadenopathy
- `C2240374` Eosinophil count raised (finding)


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1801 | Kimura disease (`C0033838`) ← **TRUE** | 3 | 62 |
| 2 | 0.1487 | Eosinophilic granulomatosis with polyang (`C0008728`) | 2 | 65 |
| 3 | 0.1045 | Granulomatosis with polyangiitis/Granulo (`C3495801`) | 1 | 20 |
| 4 | 0.0909 | Sarcoidosis, susceptibility to, 2 (`C2676468`) | 2 | 58 |
| 5 | 0.0762 | Felty syndrome/Felty syndrome (`C0015773`) | 1 | 76 |

---
### Success #26 — True: Eosinophilic granulomatosis with polyangiitis (`C0008728`)

- Rank of true: **1** / 78, Score: 0.1487


**Patient evidence (4 CUIs in profile universe):**

- `C0235896` Pulmonary Infiltrate
- `C0013404` Dyspnea
- `C0236175` Increased circulating IgE concentration
- `C2240374` Eosinophil count raised (finding)


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1487 | Eosinophilic granulomatosis with polyang (`C0008728`) ← **TRUE** | 2 | 65 |
| 2 | 0.1096 | Kimura disease (`C0033838`) | 2 | 62 |
| 3 | 0.1045 | Granulomatosis with polyangiitis/Granulo (`C3495801`) | 1 | 20 |
| 4 | 0.0636 | Reactive arthritis (`C0085435`) | 1 | 70 |
| 5 | 0.0455 | Sarcoidosis, susceptibility to, 2 (`C2676468`) | 1 | 58 |

---
### Success #30 — True: SAPHO syndrome (`C0263859`)

- Rank of true: **1** / 78, Score: 0.2210


**Patient evidence (7 CUIs in profile universe):**

- `C0742906` C-reactive protein above reference range
- `C0236000` Jaw pain
- `C0030246` Pustulosis of Palms and Soles
- `C0029443` Osteomyelitis
- `C0020492` Hyperostosis
- `C0151632` Erythrocyte sedimentation rate raised
- `C0750426` Blood leukocyte number above reference range


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2210 | SAPHO syndrome (`C0263859`) ← **TRUE** | 4 | 88 |
| 2 | 0.0980 | Muckle-Wells syndrome/Muckle-Wells syndr (`C0268390`) | 2 | 98 |
| 3 | 0.0670 | Thrombotic thrombocytopenic purpura (`C0034155`) | 2 | 53 |
| 4 | 0.0565 | Familial cold urticaria (`C0343068`) | 3 | 77 |
| 5 | 0.0554 | Sarcoidosis/Sarcoidosis, susceptibility  (`C0036202`) | 3 | 452 |

---
### Success #45 — True: Familial cold autoinflammatory syndrome 3 (`C3280914`)

- Rank of true: **1** / 78, Score: -1000000000.0000


**Patient evidence (5 CUIs in profile universe):**

- `C0042109` Urticaria
- `C0742906` C-reactive protein above reference range
- `C0003864` Arthritis
- `C0015967` Fever
- `C0428974` Supraventricular arrhythmia


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2155 | Muckle-Wells syndrome/Muckle-Wells syndr (`C0268390`) | 4 | 98 |
| 2 | 0.2040 | Familial cold urticaria (`C0343068`) | 4 | 77 |
| 3 | 0.1662 | 家族性地中海热/Familial mediterranean fever; FM (`C0031069`) | 2 | 35 |
| 4 | 0.1482 | CINCA syndrome/Cinca syndrome (`C0409818`) | 2 | 101 |
| 5 | 0.1134 | Reactive arthritis (`C0085435`) | 2 | 70 |

---
### Success #60 — True: Spondyloarthropathy, susceptibility to, 2 (`C1866738`)

- Rank of true: **1** / 78, Score: 0.0000


**Patient evidence (19 CUIs in profile universe):**

- `C0018989` Hemiparesis
- `C0038363` Aphthous Stomatitis
- `C0012569` Diplopia
- `C0011053` Deafness
- `C0424551` Impaired exercise tolerance
- `C0392525` Nephrolithiasis
- `C0040264` Tinnitus
- `C0012833` Dizziness
- `C0149745` Oral Ulcer
- `C0231807` Dyspnea on exertion
- `C0003864` Arthritis
- `C0043352` Xerostomia
- `C5886864` Decreased circulating vitamin D concentration
- `C0026266` Mitral Valve Insufficiency
- `C0015300` Exophthalmos
- ... +4 more


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1439 | Behçet disease/Behcet syndrome (`C0004943`) | 6 | 136 |
| 2 | 0.1292 | Systemic lupus erythematosus (`C0024141`) | 6 | 128 |
| 3 | 0.1157 | Muckle-Wells syndrome/Muckle-Wells syndr (`C0268390`) | 5 | 98 |
| 4 | 0.1157 | Familial cold urticaria (`C0343068`) | 4 | 77 |
| 5 | 0.0851 | Systemic-onset juvenile idiopathic arthr (`C0087031`) | 3 | 83 |

---
### Success #66 — True: Thrombotic thrombocytopenic purpura, hereditary (`C1268935`)

- Rank of true: **1** / 78, Score: 0.0891


**Patient evidence (5 CUIs in profile universe):**

- `C0012833` Dizziness
- `C0040997` Trigeminal Neuralgia
- `C0424551` Impaired exercise tolerance
- `C0039070` Syncope
- `C0040034` Thrombocytopenia


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1130 | Thrombotic thrombocytopenic purpura (`C0034155`) | 1 | 53 |
| 2 | 0.0891 | Thrombotic thrombocytopenic purpura, her (`C1268935`) ← **TRUE** | 2 | 83 |
| 3 | 0.0759 | Scleroderma (`C0011644`) | 3 | 359 |
| 4 | 0.0587 | Cryoglobulinemic vasculitis/Cryoglobulin (`C0340992`) | 1 | 63 |
| 5 | 0.0543 | Sarcoidosis/Sarcoidosis, susceptibility  (`C0036202`) | 3 | 452 |

---
### Success #74 — True: Familial Mediterranean fever, AD (`C1851347`)

- Rank of true: **1** / 78, Score: 0.1725


**Patient evidence (12 CUIs in profile universe):**

- `C0742906` C-reactive protein above reference range
- `C0008031` Chest Pain
- `C0020676` Hypothyroidism
- `C0013404` Dyspnea
- `C0015967` Fever
- `C0032231` Pleurisy
- `C0027497` Nausea
- `C0038002` Splenomegaly
- `C0853986` Lymphocyte count decreased
- `C0015672` Fatigue
- `C0232462` Decrease in appetite
- `C0085593` Chills


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1725 | Familial Mediterranean fever, AD (`C1851347`) ← **TRUE** | 3 | 91 |
| 2 | 0.1658 | 家族性地中海热/Familial mediterranean fever; FM (`C0031069`) | 2 | 35 |
| 3 | 0.1287 | Reactive arthritis (`C0085435`) | 4 | 70 |
| 4 | 0.1172 | Systemic-onset juvenile idiopathic arthr (`C0087031`) | 4 | 83 |
| 5 | 0.1148 | Systemic lupus erythematosus (`C0024141`) | 4 | 128 |

---

## Failure cases (rank≥5, 10 samples)

### Failure #1 — True: Spondyloarthropathy, susceptibility to, 1 (`C1862852`)

- Rank of true: **51** / 78, Score: 0.0000


**Patient evidence (4 CUIs in profile universe):**

- `C0037011` Shoulder Pain
- `C0239266` Pain of elbow region
- `C0750426` Blood leukocyte number above reference range
- `C0836924` Thrombocytosis


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.0927 | Immunoglobulin A vasculitis (`C0034152`) | 2 | 73 |
| 2 | 0.0846 | Polymyositis (`C0085655`) | 1 | 66 |
| 3 | 0.0721 | 系统性硬化症/Systemic sclerosi; SSc/Systemic s (`C0036421`) | 1 | 85 |
| 4 | 0.0523 | Thrombotic thrombocytopenic purpura (`C0034155`) | 1 | 53 |
| 5 | 0.0364 | Familial cold urticaria (`C0343068`) | 1 | 77 |

**Top-1 vs True comparison:**

- Profile size: top-1=73, true=44
- Patient CUIs in **both**: 0
- **Only top-1**: 2
  - `C0750426` Blood leukocyte number above reference range (P=0.06)
  - `C0836924` Thrombocytosis (P=0.08)
- **Only true**: 0

---
### Failure #3 — True: Behçet disease/Behcet syndrome (`C0004943`)

- Rank of true: **6** / 78, Score: 0.0934


**Patient evidence (29 CUIs in profile universe):**

- `C0008031` Chest Pain
- `C0038363` Aphthous Stomatitis
- `C0012569` Diplopia
- `C0027497` Nausea
- `C0011053` Deafness
- `C0332563` Papule
- `C0018681` Headache
- `C0039070` Syncope
- `C0392525` Nephrolithiasis
- `C0040264` Tinnitus
- `C0012833` Dizziness
- `C2240374` Eosinophil count raised (finding)
- `C0241005` Creatine phosphokinase serum increased
- `C0149745` Oral Ulcer
- `C0231807` Dyspnea on exertion
- ... +14 more


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1265 | Systemic lupus erythematosus (`C0024141`) | 8 | 128 |
| 2 | 0.1095 | Thrombotic thrombocytopenic purpura, her (`C1268935`) | 6 | 83 |
| 3 | 0.1025 | Immunoglobulin A vasculitis (`C0034152`) | 4 | 73 |
| 4 | 0.0955 | CINCA syndrome/Cinca syndrome (`C0409818`) | 6 | 101 |
| 5 | 0.0952 | Sarcoidosis/Sarcoidosis, susceptibility  (`C0036202`) | 16 | 452 |

**Top-1 vs True comparison:**

- Profile size: top-1=128, true=136
- Patient CUIs in **both**: 3
- **Only top-1**: 5
  - `C0008031` Chest Pain (P=0.10)
  - `C0012833` Dizziness (P=0.17)
  - `C0015672` Fatigue (P=0.31)
  - `C0018681` Headache (P=0.15)
  - `C0231528` Myalgia (P=0.11)
- **Only true**: 2
  - `C0027497` Nausea (P=0.06)
  - `C0038363` Aphthous Stomatitis (P=0.20)

---
### Failure #5 — True: Vitiligo-Associated multiple autoimmune disease susceptibility 1 (`C1847835`)

- Rank of true: **8** / 78, Score: 0.0000


**Patient evidence (13 CUIs in profile universe):**

- `C0031154` Peritonitis
- `C0030312` Pancytopenia
- `C0015967` Fever
- `C0033687` Proteinuria
- `C0011991` Diarrhea
- `C0241005` Creatine phosphokinase serum increased
- `C0746674` Generalized muscle weakness
- `C0038002` Splenomegaly
- `C0003862` Arthralgia
- `C0231528` Myalgia
- `C0000737` Abdominal Pain
- `C0424551` Impaired exercise tolerance
- `C0015672` Fatigue


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1569 | Familial Mediterranean fever, AD (`C1851347`) | 5 | 91 |
| 2 | 0.1499 | Tumor necrosis factor receptor 1 associa (`C1275126`) | 3 | 32 |
| 3 | 0.1475 | 家族性地中海热/Familial mediterranean fever; FM (`C0031069`) | 2 | 35 |
| 4 | 0.1376 | Immunoglobulin A vasculitis (`C0034152`) | 4 | 73 |
| 5 | 0.1357 | Systemic-onset juvenile idiopathic arthr (`C0087031`) | 4 | 83 |

**Top-1 vs True comparison:**

- Profile size: top-1=91, true=11
- Patient CUIs in **both**: 0
- **Only top-1**: 5
  - `C0000737` Abdominal Pain (P=0.08)
  - `C0003862` Arthralgia (P=0.10)
  - `C0015967` Fever (P=0.15)
  - `C0031154` Peritonitis (P=0.12)
  - `C0033687` Proteinuria (P=0.02)
- **Only true**: 0

---
### Failure #6 — True: Renal cell carcinoma, nonpapillary (`C0007134`)

- Rank of true: **10** / 78, Score: 0.0000


**Patient evidence (9 CUIs in profile universe):**

- `C0007134` Renal Cell Carcinoma
- `C0003864` Arthritis
- `C0149871` Deep Vein Thrombosis
- `C0017658` Glomerulonephritis
- `C0239937` Microscopic hematuria
- `C0033687` Proteinuria
- `C0035078` Kidney Failure
- `C0853986` Lymphocyte count decreased
- `C0042164` Uveitis


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1690 | Reactive arthritis (`C0085435`) | 3 | 70 |
| 2 | 0.1674 | Focal segmental glomerulosclerosis 9 (`C4015555`) | 2 | 41 |
| 3 | 0.1633 | Focal segmental glomerulosclerosis 6 (`C3279905`) | 2 | 52 |
| 4 | 0.1485 | Immunoglobulin A vasculitis (`C0034152`) | 3 | 73 |
| 5 | 0.1117 | Focal segmental glomerulosclerosis 3, su (`C1842982`) | 1 | 57 |

**Top-1 vs True comparison:**

- Profile size: top-1=70, true=20
- Patient CUIs in **both**: 0
- **Only top-1**: 3
  - `C0003864` Arthritis (P=0.14)
  - `C0017658` Glomerulonephritis (P=0.15)
  - `C0042164` Uveitis (P=0.13)
- **Only true**: 0

---
### Failure #7 — True: Primary Sjögren syndrome/Sjogren syndrome (`C0086981`)

- Rank of true: **6** / 78, Score: 0.0854


**Patient evidence (13 CUIs in profile universe):**

- `C0006267` Bronchiectasis
- `C0742906` C-reactive protein above reference range
- `C0012833` Dizziness
- `C0030252` Palpitations
- `C0017168` Gastroesophageal reflux disease
- `C0027497` Nausea
- `C0241005` Creatine phosphokinase serum increased
- `C0000737` Abdominal Pain
- `C0043352` Xerostomia
- `C0004604` Back Pain
- `C0015672` Fatigue
- `C0151632` Erythrocyte sedimentation rate raised
- `C0011168` Deglutition Disorders


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1071 | Cryoglobulinemic vasculitis/Cryoglobulin (`C1852456`) | 2 | 24 |
| 2 | 0.0963 | Sarcoidosis/Sarcoidosis, susceptibility  (`C0036202`) | 6 | 452 |
| 3 | 0.0912 | Muckle-Wells syndrome/Muckle-Wells syndr (`C0268390`) | 3 | 98 |
| 4 | 0.0909 | Diffuse cutaneous systemic sclerosis (`C1258104`) | 3 | 84 |
| 5 | 0.0866 | Thrombotic thrombocytopenic purpura, her (`C1268935`) | 3 | 83 |

**Top-1 vs True comparison:**

- Profile size: top-1=24, true=97
- Patient CUIs in **both**: 0
- **Only top-1**: 2
  - `C0017168` Gastroesophageal reflux disease (P=0.06)
  - `C0027497` Nausea (P=0.06)
- **Only true**: 3
  - `C0006267` Bronchiectasis (P=0.02)
  - `C0015672` Fatigue (P=0.08)
  - `C0043352` Xerostomia (P=0.18)

---
### Failure #8 — True: Giant cell arteritis/Temporal arteritis (`C0039483`)

- Rank of true: **39** / 78, Score: 0.0000


**Patient evidence (12 CUIs in profile universe):**

- `C0085593` Chills
- `C0241137` Pallor of skin
- `C0030193` Pain
- `C0018681` Headache
- `C0027497` Nausea
- `C1262477` Weight Loss
- `C0424551` Impaired exercise tolerance
- `C0015672` Fatigue
- `C0042963` Vomiting
- `C0039070` Syncope
- `C0151632` Erythrocyte sedimentation rate raised
- `C0750426` Blood leukocyte number above reference range


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1686 | Familial cold urticaria (`C0343068`) | 5 | 77 |
| 2 | 0.1501 | Thrombotic thrombocytopenic purpura (`C0034155`) | 4 | 53 |
| 3 | 0.1213 | Systemic-onset juvenile idiopathic arthr (`C0087031`) | 4 | 83 |
| 4 | 0.1141 | Reactive arthritis (`C0085435`) | 3 | 70 |
| 5 | 0.1124 | Systemic lupus erythematosus (`C0024141`) | 3 | 128 |

**Top-1 vs True comparison:**

- Profile size: top-1=77, true=10
- Patient CUIs in **both**: 0
- **Only top-1**: 5
  - `C0018681` Headache (P=0.08)
  - `C0027497` Nausea (P=0.08)
  - `C0030193` Pain (P=0.15)
  - `C0085593` Chills (P=0.09)
  - `C0750426` Blood leukocyte number above reference range (P=0.06)
- **Only true**: 0

---
### Failure #10 — True: Takayasu arteritis/Takayasu arteritis (`C0039263`)

- Rank of true: **9** / 78, Score: 0.0598


**Patient evidence (11 CUIs in profile universe):**

- `C4022792` Reduced left ventricular ejection fraction
- `C0013404` Dyspnea
- `C0015967` Fever
- `C0035078` Kidney Failure
- `C0149721` Left Ventricular Hypertrophy
- `C0033774` Pruritus
- `C0010200` Coughing
- `C0027059` Myocarditis
- `C0031039` Pericardial effusion
- `C0015672` Fatigue
- `C0007193` Cardiomyopathy, Dilated


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1395 | Sarcoidosis/Sarcoidosis, susceptibility  (`C0036202`) | 7 | 452 |
| 2 | 0.1165 | Familial Mediterranean fever, AD (`C1851347`) | 3 | 91 |
| 3 | 0.0919 | Scleroderma (`C0011644`) | 6 | 359 |
| 4 | 0.0898 | Tumor necrosis factor receptor 1 associa (`C1275126`) | 2 | 32 |
| 5 | 0.0794 | Systemic lupus erythematosus (`C0024141`) | 2 | 128 |

**Top-1 vs True comparison:**

- Profile size: top-1=452, true=27
- Patient CUIs in **both**: 1
- **Only top-1**: 6
  - `C0007193` Cardiomyopathy, Dilated (P=0.19)
  - `C0010200` Coughing (P=0.38)
  - `C0013404` Dyspnea (P=0.34)
  - `C0015672` Fatigue (P=0.27)
  - `C0027059` Myocarditis (P=0.25)
- **Only true**: 0

---
### Failure #11 — True: Granulomatosis with polyangiitis/Granulomatosis with polyangiitis (`C3495801`)

- Rank of true: **59** / 78, Score: 0.0000


**Patient evidence (9 CUIs in profile universe):**

- `C0742906` C-reactive protein above reference range
- `C0008031` Chest Pain
- `C0015967` Fever
- `C0019079` Hemoptysis
- `C0004238` Atrial Fibrillation
- `C0014583` Episcleritis
- `C0010200` Coughing
- `C0039231` Tachycardia
- `C0750426` Blood leukocyte number above reference range


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1455 | Sarcoidosis/Sarcoidosis, susceptibility  (`C0036202`) | 8 | 452 |
| 2 | 0.1188 | Tumor necrosis factor receptor 1 associa (`C1275126`) | 2 | 32 |
| 3 | 0.1170 | Familial Mediterranean fever, AD (`C1851347`) | 2 | 91 |
| 4 | 0.0827 | Muckle-Wells syndrome/Muckle-Wells syndr (`C0268390`) | 2 | 98 |
| 5 | 0.0792 | 家族性地中海热/Familial mediterranean fever; FM (`C0031069`) | 1 | 35 |

**Top-1 vs True comparison:**

- Profile size: top-1=452, true=20
- Patient CUIs in **both**: 0
- **Only top-1**: 8
  - `C0004238` Atrial Fibrillation (P=0.11)
  - `C0008031` Chest Pain (P=0.31)
  - `C0010200` Coughing (P=0.38)
  - `C0014583` Episcleritis (P=0.14)
  - `C0015967` Fever (P=0.20)
- **Only true**: 0

---
### Failure #12 — True: Reactive arthritis (`C0085435`)

- Rank of true: **5** / 78, Score: 0.0448


**Patient evidence (7 CUIs in profile universe):**

- `C0236175` Increased circulating IgE concentration
- `C0037763` Spasm
- `C0238656` Ankle pain
- `C0003862` Arthralgia
- `C0231528` Myalgia
- `C0043352` Xerostomia
- `C0149931` Migraine Disorders


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1386 | Thrombotic thrombocytopenic purpura, her (`C1268935`) | 4 | 83 |
| 2 | 0.1378 | Polyarteritis nodosa (`C0031036`) | 4 | 74 |
| 3 | 0.1273 | Immunoglobulin A vasculitis (`C0034152`) | 2 | 73 |
| 4 | 0.1145 | Pediatric systemic lupus erythematosus (`C1274834`) | 3 | 72 |
| 5 | 0.1086 | Systemic-onset juvenile idiopathic arthr (`C0087031`) | 2 | 83 |

**Top-1 vs True comparison:**

- Profile size: top-1=83, true=70
- Patient CUIs in **both**: 1
- **Only top-1**: 3
  - `C0037763` Spasm (P=0.05)
  - `C0149931` Migraine Disorders (P=0.05)
  - `C0231528` Myalgia (P=0.10)
- **Only true**: 0

---
### Failure #15 — True: Granulomatosis with polyangiitis/Granulomatosis with polyangiitis (`C3495801`)

- Rank of true: **60** / 78, Score: 0.0000


**Patient evidence (11 CUIs in profile universe):**

- `C0239266` Pain of elbow region
- `C0028961` Oliguria
- `C0017658` Glomerulonephritis
- `C0015967` Fever
- `C0019079` Hemoptysis
- `C0035078` Kidney Failure
- `C0151701` Pulmonary hemorrhage
- `C0037011` Shoulder Pain
- `C0010200` Coughing
- `C0424551` Impaired exercise tolerance
- `C0009763` Conjunctivitis


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1510 | Immunoglobulin A vasculitis (`C0034152`) | 3 | 73 |
| 2 | 0.1304 | Reactive arthritis (`C0085435`) | 3 | 70 |
| 3 | 0.0928 | Familial Mediterranean fever, AD (`C1851347`) | 2 | 91 |
| 4 | 0.0906 | Pediatric systemic lupus erythematosus (`C1274834`) | 2 | 72 |
| 5 | 0.0887 | 系统性硬化症/Systemic sclerosi; SSc/Systemic s (`C0036421`) | 2 | 85 |

**Top-1 vs True comparison:**

- Profile size: top-1=73, true=20
- Patient CUIs in **both**: 0
- **Only top-1**: 3
  - `C0015967` Fever (P=0.07)
  - `C0017658` Glomerulonephritis (P=0.15)
  - `C0151701` Pulmonary hemorrhage (P=0.15)
- **Only true**: 0

---

## Aggregate

- Success: avg pcuis=8.2, true_score=-199999999.9109
- Failure: avg pcuis=11.8, true_score=0.0283
- Failure score gap (top1 - true): 0.1112
