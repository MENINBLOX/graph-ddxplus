# symcat Forensic Analysis — v59 cosine (v42 KG, lay mode)

- KG: /mnt/medkg/kg/onlykg_graph_v42_full_universal.pkl
- Mode: lay
- |dcs|=49, |all_evs|=884

## Success cases (rank=1, 10 samples)

### Success #73 — True: Abscess of nose (`C0264263`)

- Rank of true: **1** / 49, Score: 0.3648


**Patient evidence (4 CUIs in profile universe):**

- `C0013404` Dyspnea
- `C0010200` Coughing
- `C1260880` Rhinorrhea
- `C0027424` Nasal congestion (finding)


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.3648 | Abscess of nose (`C0264263`) ← **TRUE** | 2 | 36 |
| 2 | 0.3433 | Acute bronchitis (`C0149514`) | 4 | 60 |
| 3 | 0.3146 | Acute sinusitis (`C0149512`) | 3 | 96 |
| 4 | 0.3060 | Acute bronchiolitis (`C0001311`) | 4 | 82 |
| 5 | 0.1762 | Acariasis (`C0026229`) | 1 | 44 |

---
### Success #74 — True: Abscess of nose (`C0264263`)

- Rank of true: **1** / 49, Score: 0.5245


**Patient evidence (6 CUIs in profile universe):**

- `C0013456` Earache
- `C0013404` Dyspnea
- `C1260880` Rhinorrhea
- `C0010200` Coughing
- `C0027424` Nasal congestion (finding)
- `C0018681` Headache


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.5245 | Abscess of nose (`C0264263`) ← **TRUE** | 4 | 36 |
| 2 | 0.3479 | Acute sinusitis (`C0149512`) | 5 | 96 |
| 3 | 0.3091 | Acute bronchitis (`C0149514`) | 5 | 60 |
| 4 | 0.2499 | Acute bronchiolitis (`C0001311`) | 4 | 82 |
| 5 | 0.2433 | Acariasis (`C0026229`) | 2 | 44 |

---
### Success #75 — True: Abscess of nose (`C0264263`)

- Rank of true: **1** / 49, Score: 0.6263


**Patient evidence (4 CUIs in profile universe):**

- `C1260880` Rhinorrhea
- `C0015967` Fever
- `C0018681` Headache
- `C0027424` Nasal congestion (finding)


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.6263 | Abscess of nose (`C0264263`) ← **TRUE** | 4 | 36 |
| 2 | 0.3441 | Acute sinusitis (`C0149512`) | 4 | 96 |
| 3 | 0.2980 | Acariasis (`C0026229`) | 2 | 44 |
| 4 | 0.2716 | Adrenal cancer (`C0750887`) | 2 | 17 |
| 5 | 0.2363 | Acute bronchitis (`C0149514`) | 4 | 60 |

---
### Success #76 — True: Abscess of nose (`C0264263`)

- Rank of true: **1** / 49, Score: 0.3221


**Patient evidence (3 CUIs in profile universe):**

- `C0015967` Fever
- `C0013456` Earache
- `C0042963` Vomiting


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.3221 | Abscess of nose (`C0264263`) ← **TRUE** | 2 | 36 |
| 2 | 0.1833 | Alcohol withdrawal (`C0236663`) | 2 | 73 |
| 3 | 0.1677 | Adrenal cancer (`C0750887`) | 1 | 17 |
| 4 | 0.1424 | Acute fatty liver of pregnancy (AFLP) (`C1455728`) | 2 | 32 |
| 5 | 0.1286 | Acute sinusitis (`C0149512`) | 2 | 96 |

---
### Success #78 — True: Abscess of nose (`C0264263`)

- Rank of true: **1** / 49, Score: 0.4040


**Patient evidence (6 CUIs in profile universe):**

- `C0013404` Dyspnea
- `C0018681` Headache
- `C1260880` Rhinorrhea
- `C0010200` Coughing
- `C0027424` Nasal congestion (finding)
- `C0042963` Vomiting


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.4040 | Abscess of nose (`C0264263`) ← **TRUE** | 3 | 36 |
| 2 | 0.3091 | Acute bronchitis (`C0149514`) | 5 | 60 |
| 3 | 0.3071 | Acute sinusitis (`C0149512`) | 4 | 96 |
| 4 | 0.2499 | Acute bronchiolitis (`C0001311`) | 4 | 82 |
| 5 | 0.2433 | Acariasis (`C0026229`) | 2 | 44 |

---
### Success #79 — True: Abscess of nose (`C0264263`)

- Rank of true: **1** / 49, Score: 0.3214


**Patient evidence (2 CUIs in profile universe):**

- `C0010200` Coughing
- `C0027424` Nasal congestion (finding)


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.3214 | Abscess of nose (`C0264263`) ← **TRUE** | 1 | 36 |
| 2 | 0.3098 | Acute bronchitis (`C0149514`) | 2 | 60 |
| 3 | 0.2910 | Acute sinusitis (`C0149512`) | 2 | 96 |
| 4 | 0.2492 | Acariasis (`C0026229`) | 1 | 44 |
| 5 | 0.1689 | Acute bronchiolitis (`C0001311`) | 2 | 82 |

---
### Success #80 — True: Abscess of nose (`C0264263`)

- Rank of true: **1** / 49, Score: 0.4528


**Patient evidence (5 CUIs in profile universe):**

- `C0013456` Earache
- `C0015967` Fever
- `C0010200` Coughing
- `C0027424` Nasal congestion (finding)
- `C0242429` Sore Throat


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.4528 | Abscess of nose (`C0264263`) ← **TRUE** | 3 | 36 |
| 2 | 0.3267 | Acute bronchitis (`C0149514`) | 4 | 60 |
| 3 | 0.2836 | Acute sinusitis (`C0149512`) | 4 | 96 |
| 4 | 0.1929 | Acute bronchiolitis (`C0001311`) | 3 | 82 |
| 5 | 0.1576 | Acariasis (`C0026229`) | 1 | 44 |

---
### Success #81 — True: Abscess of nose (`C0264263`)

- Rank of true: **1** / 49, Score: 0.4546


**Patient evidence (1 CUIs in profile universe):**

- `C0027424` Nasal congestion (finding)


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.4546 | Abscess of nose (`C0264263`) ← **TRUE** | 1 | 36 |
| 2 | 0.3524 | Acariasis (`C0026229`) | 1 | 44 |
| 3 | 0.2247 | Acute sinusitis (`C0149512`) | 1 | 96 |
| 4 | 0.1104 | Acute bronchitis (`C0149514`) | 1 | 60 |
| 5 | 0.0802 | Acute bronchiolitis (`C0001311`) | 1 | 82 |

---
### Success #82 — True: Abscess of nose (`C0264263`)

- Rank of true: **1** / 49, Score: 0.4052


**Patient evidence (6 CUIs in profile universe):**

- `C0013404` Dyspnea
- `C0015967` Fever
- `C1260880` Rhinorrhea
- `C0010200` Coughing
- `C0027424` Nasal congestion (finding)
- `C0242429` Sore Throat


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.4052 | Abscess of nose (`C0264263`) ← **TRUE** | 3 | 36 |
| 2 | 0.3998 | Acute bronchitis (`C0149514`) | 6 | 60 |
| 3 | 0.3284 | Acute bronchiolitis (`C0001311`) | 5 | 82 |
| 4 | 0.3070 | Acute sinusitis (`C0149512`) | 4 | 96 |
| 5 | 0.1454 | Anal fistula (`C0149889`) | 3 | 67 |

---
### Success #83 — True: Abscess of nose (`C0264263`)

- Rank of true: **1** / 49, Score: 0.4438


**Patient evidence (5 CUIs in profile universe):**

- `C0015967` Fever
- `C1260880` Rhinorrhea
- `C0010200` Coughing
- `C0027424` Nasal congestion (finding)
- `C0242429` Sore Throat


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.4438 | Abscess of nose (`C0264263`) ← **TRUE** | 3 | 36 |
| 2 | 0.3719 | Acute bronchitis (`C0149514`) | 5 | 60 |
| 3 | 0.3363 | Acute sinusitis (`C0149512`) | 4 | 96 |
| 4 | 0.2501 | Acute bronchiolitis (`C0001311`) | 4 | 82 |
| 5 | 0.1593 | Anal fistula (`C0149889`) | 3 | 67 |

---

## Failure cases (rank≥5, 10 samples)

### Failure #1 — True: Abdominal aortic aneurysm (`C0162871`)

- Rank of true: **36** / 49, Score: 0.0000


**Patient evidence (1 CUIs in profile universe):**

- `C0013404` Dyspnea


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.3245 | Acute bronchospasm (`C0741804`) | 1 | 23 |
| 2 | 0.2839 | Air embolism (`C0013926`) | 1 | 36 |
| 3 | 0.2453 | Acute bronchiolitis (`C0001311`) | 1 | 82 |
| 4 | 0.2306 | Allergy (`C0002111`) | 1 | 34 |
| 5 | 0.1927 | Amyloidosis (`C0002726`) | 1 | 110 |

**Top-1 vs True comparison:**

- Profile size: top-1=23, true=26
- Patient CUIs in **both**: 0
- **Only top-1**: 1
  - `C0013404` Dyspnea (P=0.10)
- **Only true**: 0

---
### Failure #2 — True: Abdominal aortic aneurysm (`C0162871`)

- Rank of true: **37** / 49, Score: 0.0000


**Patient evidence (2 CUIs in profile universe):**

- `C0013404` Dyspnea
- `C0038999` Swelling


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2295 | Acute bronchospasm (`C0741804`) | 1 | 23 |
| 2 | 0.2093 | Abscess of nose (`C0264263`) | 1 | 36 |
| 3 | 0.2008 | Air embolism (`C0013926`) | 1 | 36 |
| 4 | 0.1735 | Acute bronchiolitis (`C0001311`) | 1 | 82 |
| 5 | 0.1631 | Allergy (`C0002111`) | 1 | 34 |

**Top-1 vs True comparison:**

- Profile size: top-1=23, true=26
- Patient CUIs in **both**: 0
- **Only top-1**: 1
  - `C0013404` Dyspnea (P=0.10)
- **Only true**: 0

---
### Failure #3 — True: Abdominal aortic aneurysm (`C0162871`)

- Rank of true: **36** / 49, Score: 0.0000


**Patient evidence (1 CUIs in profile universe):**

- `C0000731` Swollen abdomen (finding)


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1757 | Acute fatty liver of pregnancy (AFLP) (`C1455728`) | 1 | 32 |
| 2 | 0.0592 | Amyloidosis (`C0002726`) | 1 | 110 |
| 3 | 0.0000 | Acanthosis nigricans (`C0000889`) | 0 | 28 |
| 4 | 0.0000 | Acne (`C0001144`) | 0 | 46 |
| 5 | 0.0000 | Acute bronchiolitis (`C0001311`) | 0 | 82 |

**Top-1 vs True comparison:**

- Profile size: top-1=32, true=26
- Patient CUIs in **both**: 0
- **Only top-1**: 1
  - `C0000731` Swollen abdomen (finding) (P=0.10)
- **Only true**: 0

---
### Failure #4 — True: Abdominal aortic aneurysm (`C0162871`)

- Rank of true: **36** / 49, Score: 0.0000


**Patient evidence (1 CUIs in profile universe):**

- `C0038999` Swelling


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2961 | Abscess of nose (`C0264263`) | 1 | 36 |
| 2 | 0.1029 | Ankylosing spondylitis (`C0038013`) | 1 | 76 |
| 3 | 0.0990 | Abscess of the pharynx (`C0155843`) | 1 | 51 |
| 4 | 0.0892 | Anal fistula (`C0149889`) | 1 | 67 |
| 5 | 0.0867 | Aphthous ulcer (`C0038363`) | 1 | 28 |

**Top-1 vs True comparison:**

- Profile size: top-1=36, true=26
- Patient CUIs in **both**: 0
- **Only top-1**: 1
  - `C0038999` Swelling (P=0.12)
- **Only true**: 0

---
### Failure #5 — True: Abdominal aortic aneurysm (`C0162871`)

- Rank of true: **38** / 49, Score: 0.0000


**Patient evidence (2 CUIs in profile universe):**

- `C0013404` Dyspnea
- `C0000737` Abdominal Pain


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2295 | Acute bronchospasm (`C0741804`) | 1 | 23 |
| 2 | 0.2008 | Air embolism (`C0013926`) | 1 | 36 |
| 3 | 0.1735 | Acute bronchiolitis (`C0001311`) | 1 | 82 |
| 4 | 0.1718 | Acute pancreatitis (`C0001339`) | 1 | 29 |
| 5 | 0.1631 | Allergy (`C0002111`) | 1 | 34 |

**Top-1 vs True comparison:**

- Profile size: top-1=23, true=26
- Patient CUIs in **both**: 0
- **Only top-1**: 1
  - `C0013404` Dyspnea (P=0.10)
- **Only true**: 0

---
### Failure #6 — True: Abdominal aortic aneurysm (`C0162871`)

- Rank of true: **37** / 49, Score: 0.0000


**Patient evidence (1 CUIs in profile universe):**

- `C0000737` Abdominal Pain


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2429 | Acute pancreatitis (`C0001339`) | 1 | 29 |
| 2 | 0.1996 | Abdominal hernia (`C0178282`) | 1 | 39 |
| 3 | 0.1583 | Acute fatty liver of pregnancy (AFLP) (`C1455728`) | 1 | 32 |
| 4 | 0.1068 | Anal fistula (`C0149889`) | 1 | 67 |
| 5 | 0.0661 | Ankylosing spondylitis (`C0038013`) | 1 | 76 |

**Top-1 vs True comparison:**

- Profile size: top-1=29, true=26
- Patient CUIs in **both**: 0
- **Only top-1**: 1
  - `C0000737` Abdominal Pain (P=0.09)
- **Only true**: 0

---
### Failure #7 — True: Abdominal aortic aneurysm (`C0162871`)

- Rank of true: **36** / 49, Score: 0.0000


**Patient evidence (1 CUIs in profile universe):**

- `C0013404` Dyspnea


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.3245 | Acute bronchospasm (`C0741804`) | 1 | 23 |
| 2 | 0.2839 | Air embolism (`C0013926`) | 1 | 36 |
| 3 | 0.2453 | Acute bronchiolitis (`C0001311`) | 1 | 82 |
| 4 | 0.2306 | Allergy (`C0002111`) | 1 | 34 |
| 5 | 0.1927 | Amyloidosis (`C0002726`) | 1 | 110 |

**Top-1 vs True comparison:**

- Profile size: top-1=23, true=26
- Patient CUIs in **both**: 0
- **Only top-1**: 1
  - `C0013404` Dyspnea (P=0.10)
- **Only true**: 0

---
### Failure #8 — True: Abdominal aortic aneurysm (`C0162871`)

- Rank of true: **40** / 49, Score: 0.0000


**Patient evidence (3 CUIs in profile universe):**

- `C0013404` Dyspnea
- `C0000737` Abdominal Pain
- `C0030252` Palpitations


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2374 | Adrenal adenoma (`C0206667`) | 1 | 27 |
| 2 | 0.2351 | Air embolism (`C0013926`) | 2 | 36 |
| 3 | 0.2008 | Adrenal cancer (`C0750887`) | 1 | 17 |
| 4 | 0.1873 | Acute bronchospasm (`C0741804`) | 1 | 23 |
| 5 | 0.1872 | Amyloidosis (`C0002726`) | 2 | 110 |

**Top-1 vs True comparison:**

- Profile size: top-1=27, true=26
- Patient CUIs in **both**: 0
- **Only top-1**: 1
  - `C0030252` Palpitations (P=0.12)
- **Only true**: 0

---
### Failure #9 — True: Abdominal aortic aneurysm (`C0162871`)

- Rank of true: **37** / 49, Score: 0.0000


**Patient evidence (2 CUIs in profile universe):**

- `C0241137` Pallor of skin
- `C0000731` Swollen abdomen (finding)


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2797 | Actinic keratosis (`C0022602`) | 1 | 18 |
| 2 | 0.2048 | Anemia due to malignancy (`C0002871`) | 1 | 102 |
| 3 | 0.1488 | Adrenal cancer (`C0750887`) | 1 | 17 |
| 4 | 0.1335 | Aplastic anemia (`C0002874`) | 1 | 55 |
| 5 | 0.1242 | Acute fatty liver of pregnancy (AFLP) (`C1455728`) | 1 | 32 |

**Top-1 vs True comparison:**

- Profile size: top-1=18, true=26
- Patient CUIs in **both**: 0
- **Only top-1**: 1
  - `C0241137` Pallor of skin (P=0.14)
- **Only true**: 0

---
### Failure #10 — True: Abdominal aortic aneurysm (`C0162871`)

- Rank of true: **36** / 49, Score: 0.0000


**Patient evidence (1 CUIs in profile universe):**

- `C0013404` Dyspnea


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.3245 | Acute bronchospasm (`C0741804`) | 1 | 23 |
| 2 | 0.2839 | Air embolism (`C0013926`) | 1 | 36 |
| 3 | 0.2453 | Acute bronchiolitis (`C0001311`) | 1 | 82 |
| 4 | 0.2306 | Allergy (`C0002111`) | 1 | 34 |
| 5 | 0.1927 | Amyloidosis (`C0002726`) | 1 | 110 |

**Top-1 vs True comparison:**

- Profile size: top-1=23, true=26
- Patient CUIs in **both**: 0
- **Only top-1**: 1
  - `C0013404` Dyspnea (P=0.10)
- **Only true**: 0

---

## Aggregate

- Success: avg pcuis=4.2, true_score=0.4320
- Failure: avg pcuis=1.5, true_score=0.0000
- Failure score gap (top1 - true): 0.2664
