## 증상 속성 (Symptom Attributes)

각 증상(finding)에는 임상 병력청취의 표준 축에 해당하는 속성을 부여하여 disease–phenotype edge에 qualified statement로 부착한다. 속성 집합은 증상 특성화의 교과서 표준인 Bates의 seven attributes[1]를 기준으로 하며, 속성을 별도 노드로 승격하지 않고 phenotype에 부착된 수식자로 두는 방식은 GA4GH Phenopacket 스키마[2]와 일치한다. IE 단계에서 Table 1의 6개 후보 속성을 추출한다. Bates의 일곱 축 중 동반증상(associated)은 속성이 아니라 KG에서 함께 나타나는 finding으로 표현한다. 다만 (i) 모든 finding이 6개를 모두 갖지는 않으며 — 속성은 원문이 명시할 때만 부여되어 대체로 sparse하다 — (ii) 이 6개는 후보이며, KG scoring에 실제로 사용할 속성은 평가 단계에서 결정한다. 각 속성 값은 표준 개념으로 정규화하며[3], 다인자 속성은 개념들의 집합으로 둔다.

**Table 1.** Candidate symptom attributes, value types, and normalization targets.

| Attribute | Bates axis | Value type | Normalization |
|---|---|---|---|
| Location | Location | concept (multi-valued) | UMLS CUI (anatomical site; SNOMED CT Body structure) |
| Character | Quality | free text | not normalized (no controlled standard) |
| Severity | Quantity | ordinal | controlled vocabulary: {mild, moderate, severe} |
| Timing | Timing | ordinal + text | onset controlled: {sudden, gradual}; duration: free text |
| Aggravating | Aggravating / relieving | concept (multi-valued) | UMLS CUI; edge qualifier: HPO *Aggravated by* |
| Relieving | Aggravating / relieving | concept (multi-valued) | UMLS CUI; edge qualifier: HPO *Ameliorated by* |

*Notes.* Controlled ordinals correspond to HPO axes: *Severity* (HP:0012824), *Temporal pattern* (HP:0011008); edge qualifiers use HPO *Aggravated by* (HP:0025285) and *Ameliorated by* (HP:0025254) [4]. Concept values are linked to UMLS [5]; anatomical sites additionally carry SNOMED CT Body structure codes (123037004) [6]. Radiation is not a separate attribute; it is represented within Location as a direction-preserving relation (see below).

- **location**은 증상이 나타나는 해부학적 부위로, UMLS CUI로 정규화하며 UMLS 내 SNOMED CT Body structure 하위계층으로도 식별된다[5][6]. 방사(radiation)는 별도 축이 아니라 위치의 한 형태로 포함하되, 방향은 진단적으로 변별적이므로 "어디에서 어디로"를 보존한다.
- **character**는 증상의 성상(예: sharp, dull, burning)이다. 성상은 포괄적인 통제 표준 어휘가 없어 대부분 free-text로 둔다. 어떤 성상어가 표준에 대응하는지는 추출할 성상 어휘(finding 범위)를 고정한 뒤 실측해야 정해지므로, 본 스키마에서는 개별 매핑을 규정하지 않는다[4]. 다만 "productive cough(가래 기침, HP:0031245)"[4]처럼 성상이 그 자체로 독립된 phenotype을 이루는 경우에는 속성이 아니라 별도 finding으로 표현한다.
- **severity**는 증상의 정도로, {mild, moderate, severe} 통제어휘로 두며 이는 HPO Severity(HP:0012824)의 서수 term에 대응한다[4].
- **timing**은 onset(발현 속도)과 duration(지속 기간)을 포함한다. onset은 {sudden, gradual} 통제어휘로 두어 HPO Temporal pattern(HP:0011008; Acute/Insidious onset)에 대응시키며, 발병 나이를 다루는 HPO Onset(HP:0003674)과 구별한다[4]. duration은 표준 통제어휘가 없어 문자열로 보존한다.
- **aggravating / relieving**은 증상을 악화·유발하거나 완화하는 인자다. 각 인자는 UMLS CUI로 정규화하고, 관계는 HPO Aggravated by(HP:0025285)·Ameliorated by(HP:0025254)로 표현한다[4].

---

### References

[1] Bickley LS, Szilagyi PG. Bates' Pocket Guide to Physical Examination and History Taking. 7th ed. Philadelphia: Wolters Kluwer Health/Lippincott Williams & Wilkins; 2013. Chapter 3, Interviewing and the Health History. "The Seven Attributes of a Symptom", p. 38. ISBN 978-1-4511-7322-2.
-> 인용 이유: 속성 집합을 임의로 고른 게 아니라 증상 특성화의 교과서 표준(seven attributes)에 앵커한다는 근거. 같은 p.38에 OLD CARTS·OPQRST mnemonic이 함께 수록되어 character·aggravating/relieving·radiation 항목의 근거도 동일 페이지에서 확인됨.

[2] Jacobsen JOB, Baudis M, Baynam GS, et al.; GA4GH Phenopacket Modeling Consortium; Haendel MA, Robinson PN. The GA4GH Phenopacket schema defines a computable representation of clinical data. Nat Biotechnol. 2022 Jun;40(6):817-820. doi: 10.1038/s41587-022-01357-4. PMID: 35705716; PMCID: PMC9363006.
-> 인용 이유: 속성을 별도 노드가 아니라 phenotype에 부착된 수식자(modifiers·severity·onset 필드)로 두는 표현 방식의 국제 표준 선례.

[3] Noémie Elhadad, Sameer Pradhan, Sharon Gorman, Suresh Manandhar, Wendy Chapman, and Guergana Savova. 2015. SemEval-2015 Task 14: Analysis of Clinical Text. In Proceedings of the 9th International Workshop on Semantic Evaluation (SemEval 2015), pages 303–310, Denver, Colorado. Association for Computational Linguistics.
-> 인용 이유: 임상 텍스트의 속성 슬롯(body location→UMLS CUI, severity, course)을 표준에 정규화하는 방식의 선행 연구 근거.

[4] Gargano MA, Matentzoglu N, Coleman B, et al. The Human Phenotype Ontology in 2024: phenotypes around the world. Nucleic Acids Res. 2024 Jan 5;52(D1):D1333-D1346. doi: 10.1093/nar/gkad1005. PMID: 37953324; PMCID: PMC10767975.
-> 인용 이유: severity·timing·character·aggravating·relieving를 HPO 표준 term으로 정규화하는 근거이자 term ID 출처(Severity HP:0012824, Temporal pattern HP:0011008, Aggravated by HP:0025285, Ameliorated by HP:0025254, Pain characteristic HP:0025280, Productive cough HP:0031245; EBI OLS4로 라벨 확인 2026-06-30).

[5] Bodenreider O. The Unified Medical Language System (UMLS): integrating biomedical terminology. Nucleic Acids Res. 2004 Jan 1;32(Database issue):D267-70. doi: 10.1093/nar/gkh061. PMID: 14681409; PMCID: PMC308795.
-> 인용 이유: location을 표준 용어체계인 UMLS Anatomy로 정규화하는 근거.

[6] SNOMED International. *SNOMED CT Editorial Guide — Body Structure.* (Body structure 123037004; Finding site 363698007.) https://docs.snomed.org/ (accessed 2026-06-30).
-> 인용 이유: location을 SNOMED Body structure(Finding site 123037004)로 정규화하는 근거.
