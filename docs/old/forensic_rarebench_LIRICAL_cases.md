# rarebench_LIRICAL Forensic Analysis — v59 cosine (v42 KG, clinical mode)

- KG: /mnt/medkg/kg/onlykg_graph_v42_full_universal.pkl
- Mode: clinical
- |dcs|=272, |all_evs|=4289

## Success cases (rank=1, 10 samples)

### Success #1 — True: Muckle-Wells syndrome/Muckle-Wells syndrome (`C0268390`)

- Rank of true: **1** / 272, Score: 0.2490


**Patient evidence (7 CUIs in profile universe):**

- `C0042109` Urticaria
- `C0742906` C-reactive protein above reference range
- `C0151683` Neutrophilia (finding)
- `C0030353` Papilledema
- `C0018681` Headache
- `C0009763` Conjunctivitis
- `C0151632` Erythrocyte sedimentation rate raised


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2490 | Muckle-Wells syndrome/Muckle-Wells syndr (`C0268390`) ← **TRUE** | 5 | 98 |
| 2 | 0.0624 | Tuberous sclerosis-2 (`C1860707`) | 1 | 50 |
| 3 | 0.0603 | Muenke syndrome/Muenke syndrome (`C1864436`) | 1 | 50 |
| 4 | 0.0420 | 尼曼匹克病C型/Niemann-Pick disease type C; NPD (`C0220756`) | 1 | 104 |
| 5 | 0.0391 | Ichthyosis-hypotrichosis syndrome/Ichthy (`C4510566`) | 1 | 60 |

---
### Success #2 — True: 戊二酸血症 I 型/Glutaric acidemia type I; GA-I/Glutaryl-CoA dehydrogenase deficiency/Glutaric acidemia I (`C0268595`)

- Rank of true: **1** / 272, Score: 0.1906


**Patient evidence (3 CUIs in profile universe):**

- `C2243051` Large head (disorder)
- `C0013421` Dystonia
- `C0018946` Hematoma, Subdural


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1906 | 戊二酸血症 I 型/Glutaric acidemia type I; GA-I (`C0268595`) ← **TRUE** | 3 | 140 |
| 2 | 0.1460 | 3-Methylglutaconic aciduria with deafnes (`C4040739`) | 1 | 61 |
| 3 | 0.1167 | Parkinsonian-pyramidal syndrome/Parkinso (`C1850100`) | 1 | 28 |
| 4 | 0.1002 | Pantothenate kinase-associated neurodege (`C0018523`) | 1 | 112 |
| 5 | 0.0946 | GM1-gangliosidosis, type III (`C0268273`) | 1 | 56 |

---
### Success #4 — True: Galactosialidosis/Galactosialidosis (`C0268233`)

- Rank of true: **1** / 272, Score: 0.2034


**Patient evidence (9 CUIs in profile universe):**

- `C0521525` Short neck
- `C0002985` Angiokeratoma
- `C0003507` Aortic Valve Stenosis
- `C0007758` Cerebellar Ataxia
- `C0019214` Hepatosplenomegaly
- `C0018777` Conductive hearing loss
- `C2216370` Cherry red spot of the macula
- `C0349588` Short stature
- `C1858085` Malar hypoplasia


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2034 | Galactosialidosis/Galactosialidosis (`C0268233`) ← **TRUE** | 4 | 83 |
| 2 | 0.0805 | 尼曼匹克病C型/Niemann-Pick disease type C; NPD (`C0220756`) | 2 | 104 |
| 3 | 0.0784 | Desmosterolosis/DESMOSTEROLOSIS (`C1865596`) | 2 | 119 |
| 4 | 0.0758 | Muckle-Wells syndrome/Muckle-Wells syndr (`C0268390`) | 2 | 98 |
| 5 | 0.0622 | Aarskog-Scott syndrome/Aarskog-Scott syn (`C0175701`) | 2 | 88 |

---
### Success #7 — True: Hajdu-Cheney syndrome/Hajdu-Cheney syndrome (`C0917715`)

- Rank of true: **1** / 272, Score: 0.2133


**Patient evidence (5 CUIs in profile universe):**

- `C0029456` Osteoporosis
- `C0917990` Acro-Osteolysis
- `C0011053` Deafness
- `C0025990` Micrognathism
- `C0349588` Short stature


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2133 | Hajdu-Cheney syndrome/Hajdu-Cheney syndr (`C0917715`) ← **TRUE** | 3 | 105 |
| 2 | 0.1162 | Werner syndrome/Werner syndrome (`C0043119`) | 2 | 91 |
| 3 | 0.0932 | Tietz syndrome/Tietz albinism-deafness s (`C0391816`) | 1 | 60 |
| 4 | 0.0838 | Aarskog-Scott syndrome/Aarskog-Scott syn (`C0175701`) | 2 | 88 |
| 5 | 0.0826 | Microcephaly 3, primary, autosomal reces (`C1858108`) | 1 | 25 |

---
### Success #10 — True: Congenital disorder of glycosylation, type IIP (`C4225190`)

- Rank of true: **1** / 272, Score: 0.2708


**Patient evidence (3 CUIs in profile universe):**

- `C0020443` Hypercholesterolemia
- `C2711227` Steatohepatitis
- `C1314665` Serum alkaline phosphatase raised


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2708 | Congenital disorder of glycosylation, ty (`C4225190`) ← **TRUE** | 2 | 22 |
| 2 | 0.0963 | Glycogen storage disease due to liver gl (`C0017925`) | 1 | 63 |
| 3 | 0.0776 | Ataxia-oculomotor apraxia type 1/Ataxia, (`C1859598`) | 1 | 94 |
| 4 | 0.0353 | Camurati-Engelmann disease/Camurati-Enge (`C0011989`) | 1 | 109 |
| 5 | 0.0000 | Apert syndrome/Apert syndrome (`C0001193`) | 0 | 83 |

---
### Success #19 — True: Camurati-Engelmann disease/Camurati-Engelmann disease (`C0011989`)

- Rank of true: **1** / 272, Score: 0.1802


**Patient evidence (9 CUIs in profile universe):**

- `C0015300` Exophthalmos
- `C0005745` Blepharoptosis
- `C1837260` Prominent forehead
- `C0003862` Arthralgia
- `C0151825` Bone pain
- `C0020492` Hyperostosis
- `C0231712` Waddling gait
- `C0338656` Impaired cognition
- `C1314665` Serum alkaline phosphatase raised


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1802 | Camurati-Engelmann disease/Camurati-Enge (`C0011989`) ← **TRUE** | 5 | 109 |
| 2 | 0.0877 | Van den Ende-Gupta syndrome (`C1833136`) | 1 | 22 |
| 3 | 0.0701 | Inclusion body myopathy with Paget disea (`C1833662`) | 1 | 29 |
| 4 | 0.0591 | Mucolipidosis type IV/Mucolipidosis IV (`C0238286`) | 2 | 131 |
| 5 | 0.0523 | Joubert syndrome 30 (`C4539937`) | 1 | 52 |

---
### Success #25 — True: Ichthyosis-hypotrichosis syndrome/Ichthyosis, congenital, autosomal recessive 11 (`C4510566`)

- Rank of true: **1** / 272, Score: 0.1195


**Patient evidence (3 CUIs in profile universe):**

- `C1862863` Sparse body hair
- `C0085636` Photophobia
- `C0005741` Blepharitis


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1195 | Ichthyosis-hypotrichosis syndrome/Ichthy (`C4510566`) ← **TRUE** | 2 | 60 |
| 2 | 0.0711 | Cornelia de Lange syndrome/Cornelia de L (`C0270972`) | 1 | 105 |
| 3 | 0.0544 | Oculocutaneous albinism type 2/Albinism, (`C0268495`) | 1 | 78 |
| 4 | 0.0514 | Mucolipidosis type IV/Mucolipidosis IV (`C0238286`) | 1 | 131 |
| 5 | 0.0120 | X-linked hypohidrotic ectodermal dysplas (`C0162359`) | 1 | 80 |

---
### Success #32 — True: Papillon-Lefèvre syndrome/Papillon-Lefevre syndrome (`C0030360`)

- Rank of true: **1** / 272, Score: 0.3343


**Patient evidence (4 CUIs in profile universe):**

- `C4025886` Severe periodontitis
- `C0399385` Early tooth exfoliation
- `C0022596` Palmoplantar Keratosis
- `C0016436` Folliculitis


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.3343 | Papillon-Lefèvre syndrome/Papillon-Lefev (`C0030360`) ← **TRUE** | 3 | 49 |
| 2 | 0.0888 | Epidermolytic palmoplantar keratoderma/P (`C1721006`) | 1 | 38 |
| 3 | 0.0823 | Poikiloderma with neutropenia (`C1858723`) | 1 | 60 |
| 4 | 0.0493 | Smith-Magenis syndrome/Smith-Magenis syn (`C0795864`) | 1 | 158 |
| 5 | 0.0000 | Apert syndrome/Apert syndrome (`C0001193`) | 0 | 83 |

---
### Success #42 — True: Charcot-Marie-Tooth disease, demyelinating, type 1C (`C0270913`)

- Rank of true: **1** / 272, Score: 0.2525


**Patient evidence (8 CUIs in profile universe):**

- `C0750937` Ataxia, Appendicular
- `C5551413` Reduced sensation of skin
- `C0751837` Gait Ataxia
- `C0039273` Talipes cavus
- `C1112256` Sensorimotor neuropathy
- `C1857640` Decreased nerve conduction velocity
- `C0030554` Paresthesia
- `C0338656` Impaired cognition


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2525 | Charcot-Marie-Tooth disease, demyelinati (`C0270913`) ← **TRUE** | 4 | 68 |
| 2 | 0.1777 | Ataxia-oculomotor apraxia type 1/Ataxia, (`C1859598`) | 4 | 94 |
| 3 | 0.1039 | Krabbe disease (`C0023521`) | 2 | 61 |
| 4 | 0.0744 | Inclusion body myopathy with Paget disea (`C1833662`) | 1 | 29 |
| 5 | 0.0722 | Autosomal dominant Charcot-Marie-Tooth d (`C1836485`) | 1 | 55 |

---
### Success #54 — True: Muenke syndrome/Muenke syndrome (`C1864436`)

- Rank of true: **1** / 272, Score: 0.2022


**Patient evidence (5 CUIs in profile universe):**

- `C0025362` Mental Retardation
- `C0557874` Global developmental delay
- `C4021164` Bicoronal synostosis
- `C0011053` Deafness
- `C0014544` Epilepsy


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2022 | Muenke syndrome/Muenke syndrome (`C1864436`) ← **TRUE** | 3 | 50 |
| 2 | 0.1176 | Autosomal recessive spastic paraplegia t (`C3888209`) | 3 | 94 |
| 3 | 0.1161 | Bilateral frontoparietal polymicrogyria/ (`C1847352`) | 2 | 51 |
| 4 | 0.1137 | Galactosialidosis/Galactosialidosis (`C0268233`) | 3 | 83 |
| 5 | 0.1069 | Vici syndrome/Vici syndrome (`C1855772`) | 2 | 67 |

---

## Failure cases (rank≥5, 10 samples)

### Failure #3 — True: Autosomal dominant hyper-IgE syndrome/Hyper-IgE recurrent infection syndrome (`C2936739`)

- Rank of true: **200** / 272, Score: -1000000000.0000


**Patient evidence (3 CUIs in profile universe):**

- `C1837260` Prominent forehead
- `C0013595` Eczema
- `C1845847` Coarse facial features


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1135 | Schinzel-Giedion syndrome/Schinzel-Giedi (`C0265227`) | 2 | 159 |
| 2 | 0.0968 | Galactosialidosis/Galactosialidosis (`C0268233`) | 1 | 83 |
| 3 | 0.0815 | Dyggve-Melchior-Clausen disease/Dyggve-M (`C0265286`) | 1 | 85 |
| 4 | 0.0660 | Craniofrontonasal dysplasia/Craniofronto (`C0220767`) | 1 | 93 |
| 5 | 0.0573 | Larsen syndrome/Larsen syndrome (`C0175778`) | 1 | 141 |

**Top-1 vs True comparison:**

- Profile size: top-1=159, true=0
- Patient CUIs in **both**: 0
- **Only top-1**: 2
  - `C1837260` Prominent forehead (P=0.15)
  - `C1845847` Coarse facial features (P=0.15)
- **Only true**: 0

---
### Failure #5 — True: DYRK1A-related intellectual disability syndrome/Mental retardation, autosomal dominant 7 (`C5568143`)

- Rank of true: **267** / 272, Score: -1000000000.0000


**Patient evidence (11 CUIs in profile universe):**

- `C0009952` Febrile Convulsions
- `C0014306` Enophthalmos
- `C0023012` Language Delay
- `C1836047` Long face
- `C4551563` Microcephaly (physical finding)
- `C2051831` Pectus excavatum
- `C1844813` Widely spaced teeth
- `C0232466` Feeding difficulties
- `C0494475` Tonic-Clonic Seizures
- `C0007758` Cerebellar Ataxia
- `C1854113` Prominent nasal bridge


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.0888 | Cohen syndrome/Cohen syndrome (`C0265223`) | 3 | 115 |
| 2 | 0.0677 | Pierpont syndrome/Pierpont syndrome (`C1865644`) | 1 | 40 |
| 3 | 0.0635 | Wiedemann-Steiner syndrome/Wiedemann-Ste (`C1854630`) | 2 | 90 |
| 4 | 0.0563 | X-linked hypohidrotic ectodermal dysplas (`C0162359`) | 1 | 80 |
| 5 | 0.0491 | Ataxia-oculomotor apraxia type 1/Ataxia, (`C1859598`) | 1 | 94 |

**Top-1 vs True comparison:**

- Profile size: top-1=115, true=0
- Patient CUIs in **both**: 0
- **Only top-1**: 3
  - `C0232466` Feeding difficulties (P=0.13)
  - `C1836047` Long face (P=0.15)
  - `C4551563` Microcephaly (physical finding) (P=0.09)
- **Only true**: 0

---
### Failure #6 — True: Mental retardation, autosomal dominant 42 (`C4310774`)

- Rank of true: **21** / 272, Score: 0.0460


**Patient evidence (6 CUIs in profile universe):**

- `C0232466` Feeding difficulties
- `C0557874` Global developmental delay
- `C0085583` Choreoathetosis
- `C1839546` Microretrognathia
- `C0026827` Muscle hypotonia
- `C0038271` Stereotyped Behavior


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1684 | Musculocontractural Ehlers-Danlos syndro (`C1866294`) | 1 | 10 |
| 2 | 0.1469 | Oculocerebrorenal syndrome of Lowe/Lowe  (`C0028860`) | 3 | 73 |
| 3 | 0.1037 | CTCF-related neurodevelopmental disorder (`C4750955`) | 1 | 34 |
| 4 | 0.1010 | Parkinsonian-pyramidal syndrome/Parkinso (`C1850100`) | 1 | 28 |
| 5 | 0.0987 | Van den Ende-Gupta syndrome (`C1833136`) | 1 | 22 |

**Top-1 vs True comparison:**

- Profile size: top-1=10, true=54
- Patient CUIs in **both**: 1
- **Only top-1**: 0
- **Only true**: 0

---
### Failure #8 — True: Oliver-Mcfarlane syndrome (`C1848745`)

- Rank of true: **10** / 272, Score: 0.0577


**Patient evidence (13 CUIs in profile universe):**

- `C0854699` Trichomegaly
- `C0028738` Nystagmus
- `C0751837` Gait Ataxia
- `C0020676` Hypothyroidism
- `C0039273` Talipes cavus
- `C0231686` Gait, Unsteady
- `C0700078` Decreased tendon reflex
- `C0349588` Short stature
- `C0271183` Severe myopia
- `C0013362` Dysarthria
- `C0013274` Patent ductus arteriosus
- `C4048273` Chorioretinal atrophy
- `C0270921` Axonal neuropathy


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1494 | Cone rod dystrophy/Cone-Rod dystrophy 2 (`C4085590`) | 2 | 28 |
| 2 | 0.1203 | Autosomal dominant Charcot-Marie-Tooth d (`C1836485`) | 3 | 55 |
| 3 | 0.1073 | Autoimmune polyendocrinopathy type 1/Aut (`C0085859`) | 1 | 12 |
| 4 | 0.0694 | Choreoacanthocytosis/CHOREOACANTHOCYTOSI (`C0393576`) | 3 | 104 |
| 5 | 0.0683 | Ataxia-oculomotor apraxia type 1/Ataxia, (`C1859598`) | 2 | 94 |

**Top-1 vs True comparison:**

- Profile size: top-1=28, true=74
- Patient CUIs in **both**: 0
- **Only top-1**: 2
  - `C0028738` Nystagmus (P=0.03)
  - `C0271183` Severe myopia (P=0.03)
- **Only true**: 1
  - `C0854699` Trichomegaly (P=0.20)

---
### Failure #9 — True: Immunoskeletal dysplasia with neurodevelopmental abnormalities (`C4479452`)

- Rank of true: **248** / 272, Score: -1000000000.0000


**Patient evidence (15 CUIs in profile universe):**

- `C0010278` Craniosynostosis
- `C1184923` Lordosis deformity of lumbar spine
- `C1860834` Infantile muscular hypotonia
- `C1839323` Small chin
- `C2051831` Pectus excavatum
- `C0857379` Abnormal pinna morphology
- `C0813230` Serum triglycerides increased
- `C0410528` Skeletal dysplasia
- `C0557874` Global developmental delay
- `C0005741` Blepharitis
- `C0239399` Short extremities
- `C0423109` Upward slant of palpebral fissure
- `C4551649` Congenital Dysplasia Of The Hip
- `C0085110` Severe Combined Immunodeficiency
- `C0015310` Exotropia


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1008 | Hypochondroplasia/Hypochondroplasia (`C0410529`) | 2 | 63 |
| 2 | 0.0780 | CODAS syndrome/CODAS syndrome (`C1838180`) | 2 | 81 |
| 3 | 0.0730 | Craniofrontonasal dysplasia/Craniofronto (`C0220767`) | 3 | 93 |
| 4 | 0.0688 | Acromicric dysplasia/Acromicric dysplasi (`C0265287`) | 3 | 121 |
| 5 | 0.0606 | Desmosterolosis/DESMOSTEROLOSIS (`C1865596`) | 2 | 119 |

**Top-1 vs True comparison:**

- Profile size: top-1=63, true=0
- Patient CUIs in **both**: 0
- **Only top-1**: 2
  - `C0239399` Short extremities (P=0.13)
  - `C1184923` Lordosis deformity of lumbar spine (P=0.21)
- **Only true**: 0

---
### Failure #11 — True: Arthrogryposis, distal, with impaired proprioception and touch (`C4310692`)

- Rank of true: **244** / 272, Score: -1000000000.0000


**Patient evidence (5 CUIs in profile universe):**

- `C0036439` Scoliosis, unspecified
- `C0151786` Muscle Weakness
- `C0232466` Feeding difficulties
- `C4551649` Congenital Dysplasia Of The Hip
- `C0240635` Byzanthine arch palate


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1612 | Musculocontractural Ehlers-Danlos syndro (`C1866294`) | 1 | 10 |
| 2 | 0.1004 | Pierpont syndrome/Pierpont syndrome (`C1865644`) | 1 | 40 |
| 3 | 0.0989 | NKX6-2-related autosomal recessive hypom (`C4479653`) | 1 | 19 |
| 4 | 0.0981 | Vici syndrome/Vici syndrome (`C1855772`) | 1 | 67 |
| 5 | 0.0943 | Cohen syndrome/Cohen syndrome (`C0265223`) | 2 | 115 |

**Top-1 vs True comparison:**

- Profile size: top-1=10, true=0
- Patient CUIs in **both**: 0
- **Only top-1**: 1
  - `C0151786` Muscle Weakness (P=0.09)
- **Only true**: 0

---
### Failure #12 — True: Ehlers-Danlos syndrome, classic type, 2 (`C0268336`)

- Rank of true: **58** / 272, Score: 0.0000


**Patient evidence (4 CUIs in profile universe):**

- `C0151786` Muscle Weakness
- `C1837658` Gross motor development delay
- `C0241074` Hyperextensible skin
- `C0575158` Kyphoscoliosis deformity of spine


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1802 | Musculocontractural Ehlers-Danlos syndro (`C1866294`) | 1 | 10 |
| 2 | 0.1071 | Congenital contractural arachnodactyly/C (`C0220668`) | 1 | 98 |
| 3 | 0.0940 | Autosomal dominant centronuclear myopath (`C1834558`) | 1 | 63 |
| 4 | 0.0890 | AGel amyloidosis/Amyloidosis, Finnish ty (`C1622345`) | 1 | 88 |
| 5 | 0.0869 | Jervell and Lange-Nielsen syndrome/Jerve (`C0022387`) | 1 | 48 |

**Top-1 vs True comparison:**

- Profile size: top-1=10, true=12
- Patient CUIs in **both**: 0
- **Only top-1**: 1
  - `C0151786` Muscle Weakness (P=0.09)
- **Only true**: 0

---
### Failure #13 — True: Spondyloepimetaphyseal dysplasia with joint laxity/Spondyloepimetaphyseal dysplasia with joint laxity, type 1, with or without fractures (`C4017377`)

- Rank of true: **235** / 272, Score: -1000000000.0000


**Patient evidence (12 CUIs in profile universe):**

- `C0019554` Hip Dislocation
- `C0015300` Exophthalmos
- `C0010495` Cutis Laxa
- `C0086437` Joint laxity
- `C1844704` Platyspondyly
- `C1837260` Prominent forehead
- `C0025990` Micrognathism
- `C0026827` Muscle hypotonia
- `C1849955` Limited elbow movement
- `C2981150` Uranostaphyloschisis
- `C1850135` Flared metaphysis
- `C0392476` Epiphyseal dysplasia


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1190 | Musculocontractural Ehlers-Danlos syndro (`C1866294`) | 1 | 10 |
| 2 | 0.1063 | CODAS syndrome/CODAS syndrome (`C1838180`) | 3 | 81 |
| 3 | 0.1053 | Pseudoachondroplasia/Pseudoachondroplasi (`C0410538`) | 3 | 97 |
| 4 | 0.0856 | Craniofrontonasal dysplasia/Craniofronto (`C0220767`) | 4 | 93 |
| 5 | 0.0761 | Dyggve-Melchior-Clausen disease/Dyggve-M (`C0265286`) | 2 | 85 |

**Top-1 vs True comparison:**

- Profile size: top-1=10, true=0
- Patient CUIs in **both**: 0
- **Only top-1**: 1
  - `C0026827` Muscle hypotonia (P=0.11)
- **Only true**: 0

---
### Failure #14 — True: CTCF-related neurodevelopmental disorder/Intellectual developmental disorder, autosomal dominant 21 (`C4750955`)

- Rank of true: **9** / 272, Score: 0.0454


**Patient evidence (20 CUIs in profile universe):**

- `C4551563` Microcephaly (physical finding)
- `C1868571` High arched eyebrow
- `C0426414` Small nose
- `C0017168` Gastroesophageal reflux disease
- `C0431447` Synophrys
- `C0557874` Global developmental delay
- `C0015934` Fetal Growth Retardation
- `C3806482` Recurrent respiratory infections
- `C0854699` Trichomegaly
- `C0266544` Microcornea
- `C1840077` Anteverted nostril
- `C0014306` Enophthalmos
- `C0014877` Esotropia
- `C0426429` Broad nasal tip
- `C1853242` Midface retrusion
- ... +5 more


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.2271 | Cornelia de Lange syndrome/Cornelia de L (`C0270972`) | 9 | 105 |
| 2 | 0.1234 | Musculocontractural Ehlers-Danlos syndro (`C1866294`) | 1 | 10 |
| 3 | 0.0912 | Craniofrontonasal dysplasia/Craniofronto (`C0220767`) | 4 | 93 |
| 4 | 0.0700 | Leprechaunism/Donohue syndrome (`C0265344`) | 2 | 78 |
| 5 | 0.0689 | Myhre syndrome/Myhre syndrome (`C0796081`) | 3 | 139 |

**Top-1 vs True comparison:**

- Profile size: top-1=105, true=34
- Patient CUIs in **both**: 1
- **Only top-1**: 8
  - `C0017168` Gastroesophageal reflux disease (P=0.21)
  - `C0266544` Microcornea (P=0.14)
  - `C0431447` Synophrys (P=0.16)
  - `C0557874` Global developmental delay (P=0.09)
  - `C0854699` Trichomegaly (P=0.10)
- **Only true**: 0

---
### Failure #15 — True: Intellectual disability-coarse face-macrocephaly-cerebellar hypotrophy syndrome/Spinocerebellar ataxia, autosomal recessive 20 (`C5190595`)

- Rank of true: **173** / 272, Score: 0.0000


**Patient evidence (10 CUIs in profile universe):**

- `C0025362` Mental Retardation
- `C0028738` Nystagmus
- `C1845847` Coarse facial features
- `C1836542` Depressed nasal bridge
- `C0262404` Cerebellar degeneration
- `C0678230` Congenital Epicanthus
- `C0557874` Global developmental delay
- `C0560046` Unable to walk (finding)
- `C0020534` Orbital separation excessive
- `C0014544` Epilepsy


**Top-5 predictions (cosine score):**

| Rank | Score | Disease (CUI) | Patient CUI ∩ Profile | Profile size |
|---|---|---|---|---|
| 1 | 0.1334 | Galactosialidosis/Galactosialidosis (`C0268233`) | 4 | 83 |
| 2 | 0.1101 | Vici syndrome/Vici syndrome (`C1855772`) | 3 | 67 |
| 3 | 0.1019 | Schinzel-Giedion syndrome/Schinzel-Giedi (`C0265227`) | 5 | 159 |
| 4 | 0.0974 | Cornelia de Lange syndrome/Cornelia de L (`C0270972`) | 3 | 105 |
| 5 | 0.0887 | Wiedemann-Steiner syndrome/Wiedemann-Ste (`C1854630`) | 2 | 90 |

**Top-1 vs True comparison:**

- Profile size: top-1=83, true=14
- Patient CUIs in **both**: 0
- **Only top-1**: 4
  - `C0014544` Epilepsy (P=0.11)
  - `C0025362` Mental Retardation (P=0.09)
  - `C0557874` Global developmental delay (P=0.05)
  - `C1845847` Coarse facial features (P=0.16)
- **Only true**: 0

---

## Aggregate

- Success: avg pcuis=5.6, true_score=0.2216
- Failure: avg pcuis=9.9, true_score=-499999999.9851
- Failure score gap (top1 - true): 500000000.1293
