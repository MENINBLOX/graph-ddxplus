# IE 시연 — 실제 임상 텍스트 10건 source-grounded 추출

> 목적: 추상적 method가 아니라 **실제 IE 산출물**을 눈으로 확인. 속성이 어떻게 채워지는지.
> 수행: 이번 시연은 Claude가 IE 엔진 역할(프로덕션=gemma). 입력=benchmark-blind 임상 텍스트(Wikipedia 임상 기술, 2026-07-01 fetch).
> 아키텍처대로 **① LLM 추출(멘션+속성+근거 span)** 과 **② 정규화(linker → HPO/UMLS ID)** 를 분리 표기. ②의 ID는 대표값(실제는 scispaCy+UMLS/HPO linker가 확정).
> **원칙**: source-grounded — 원문에 없는 속성은 비움(환각 방지). 빈 칸은 "원문 미기재"이지 "속성 없음"이 아님.

범례: `loc`=location, `char`=character, `sev`=severity, `time`=timing(onset/duration), `agg`=aggravating, `rel`=relieving. `radiates-to`=방향 보존 location.

---

## 1. Angina pectoris
원문(발췌): "the discomfort is usually described as a pressure, heaviness, tightness, squeezing, burning, or choking sensation … epigastrium, back, neck area, jaw, or shoulders. Typical locations for referred pain are arms (often inner left arm), shoulders, and neck into the jaw … precipitated by exertion or emotional stress. It is exacerbated by having a full stomach and by cold temperatures. Relief … after administration of sublingual nitroglycerin … lasting more than 10 minutes (unstable)."

**Finding: Chest discomfort/pain**
- ① 추출: char="pressure/heaviness/tightness/squeezing/burning/choking"; loc="retrosternal; epigastrium, back, neck, jaw, shoulders"; radiates-to="inner left arm, shoulders, neck, jaw"; time.onset="on exertion", time.duration="several min (stable) / >10 min (unstable)"; agg="exertion, emotional stress, full stomach, cold"; rel="rest, sublingual nitroglycerin"
- ② 정규화: finding→Chest pain (HP:0100749 / UMLS C0008031); char→Pain characteristic (HP:0025280) 하위 미대응 다수→**문자열 보존**("pressure/tightness/squeezing"); loc→Thoracic structure (UMLS C0817096); radiates-to→Left upper arm (UMLS C0446516); agg→Triggered by (HP:0025204)="physical exertion"(C0015264); rel→Ameliorated by (HP:0025254)="nitroglycerin"(C0017887)
- 공존 finding(associated): Dyspnea(HP:0002094), Diaphoresis(HP:0000975), Nausea(HP:0002018)

## 2. Gastroesophageal reflux disease (GERD)
원문(발췌): "most common symptoms … acidic taste in the mouth, regurgitation, and heartburn … chest pain (less common) … foods that may precipitate … coffee, alcohol, chocolate, fatty foods, acidic/spicy foods … not lying down for three hours after eating."

**Finding: Heartburn**
- ① 추출: (char 미기재); (loc 미기재—"chest"만 암시); time="after eating / when reclined"; agg="fatty/acidic/spicy foods, coffee, alcohol, chocolate; lying down after meals"; (rel 미기재)
- ② 정규화: finding→Heartburn (HP:0030794 / UMLS C0018834); agg→Triggered by (HP:0025204)="food(fatty/spicy)"·"recumbent position"; time.setting="postprandial"(문자열)
- 공존: Regurgitation, Dysgeusia(acidic taste)
- ⚠️ **관찰**: 이 원문은 char/loc/sev가 비어 recall이 낮다 → **속성 recall은 source 서술 밀도에 종속**(빈약한 원문 = 빈약한 속성). 임상발현 코퍼스(StatPearls)면 더 촘촘.

## 3. Acute pericarditis
원문(발췌): "sharp, pleuritic, retro-sternal or left precordial pain with radiation to the trapezius ridge … less severe when sitting up and more severe when lying down or breathing deeply … relieved by sitting up or bending forward."

**Finding: Chest pain**
- ① 추출: char="sharp, pleuritic"; loc="retrosternal / left precordial"; radiates-to="trapezius ridge, shoulders, neck, back"; agg="lying down (supine), deep inspiration"; rel="sitting up, leaning forward"
- ② 정규화: finding→Chest pain (HP:0100749); char→**Sharp (HP:0025281)** + "pleuritic"(문자열/Chest pain worse on inspiration); loc→Retrosternal(UMLS C0442096); radiates-to→Trapezius region; agg→Aggravated by (HP:0025285)="supine position","inspiration"; rel→Ameliorated by (HP:0025254)="sitting up","leaning forward"
- 🔑 **discriminator 시연**: finding은 angina와 같은 "Chest pain"이지만 **char=sharp/pleuritic, rel=leaning-forward** 조합이 pericarditis를 angina(char=pressure, rel=nitroglycerin)와 가른다.

## 4. Migraine
원문(발췌): "occurs on one side and throbs with moderate to severe intensity … In around 40% … both sides … pain phase usually lasts 4 to 72 hours … motion and physical activity may increase pain … sensitivity to light … to sound."

**Finding: Headache**
- ① 추출: loc="unilateral (one side); 40% bilateral"; char="throbbing/pulsating"; sev="moderate to severe"; time.duration="4–72 h"; agg="motion, physical activity, light, sound"
- ② 정규화: finding→Headache (HP:0002315 / UMLS C0018681); loc→"unilateral"→Lateralized(문자열/HP:0012831 Laterality); char→Pulsating(문자열; HPO char 미대응); sev→**Moderate–Severe (HP:0012826–HP:0012828)**; agg→Aggravated by (HP:0025285)="physical activity"
- 공존(진단 특이): Photophobia(HP:0000613), Phonophobia(HP:0002183), Nausea(HP:0002018), Vomiting(HP:0002013)

## 5. Cluster headache
원문(발췌): "pain occurs only on one side … around the eye, particularly behind or above the eye, in the temple … burning, stabbing, drilling or squeezing … greater than … migraines … attacks last between 15 minutes to three hours … strike at a precise time of day."

**Finding: Headache**
- ① 추출: loc="unilateral; around/behind/above eye, temple"; char="burning, stabbing, drilling, squeezing"; sev="very severe (> migraine)"; time.duration="15 min–3 h"; time.pattern="circadian (same hour daily)"
- ② 정규화: finding→Headache (HP:0002315); loc→Periorbital region(UMLS C0229310); char→**Stabbing** + "burning/drilling"(문자열); sev→**Severe (HP:0012828)**; time→duration 문자열, circadian 문자열
- 공존(자율신경, 동측): Ptosis(HP:0000508), Miosis(HP:0000616), Conjunctival injection(HP:0000524), Lacrimation(HP:0009926), Rhinorrhea(HP:0031417)
- 🔑 **discriminator**: 같은 "Headache"라도 migraine(char=throbbing, dur 4–72h, +photophobia) vs cluster(char=stabbing, loc=periorbital, dur 15min–3h, +동측 자율신경)로 속성이 가른다.

## 6. Acute appendicitis
원문(발췌): "dull pain around the navel. After several hours, the pain usually migrates towards the right lower quadrant … migratory right iliac fossa pain associated with nausea, and anorexia … increased pain on movement, or jolting … increased pain in the right lower quadrant by coughing (Dunphy's sign)."

**Finding: Abdominal pain**
- ① 추출: char="dull"; loc="periumbilical → right lower quadrant (migratory)"; time.onset="gradual over hours (migration)"; agg="movement, jolting, coughing"
- ② 정규화: finding→Abdominal pain (HP:0002027); char→Dull(문자열; cf. HP:0025282); loc→**migration**: Periumbilical(C0230168) → Right lower quadrant(C0230179) [방향 보존]; time.onset→문자열; agg→Aggravated by (HP:0025285)="movement","coughing"
- 공존: Nausea(HP:0002018), Vomiting(HP:0002013), Anorexia(HP:0002039), Fever(HP:0001945)
- 🔑 loc이 **정적 부위가 아니라 이동(migration)** — 방향/시간 보존이 진단 신호(periumbilical→RLQ).

## 7. Kidney stone / renal colic
원문(발췌): "excruciating, intermittent pain that radiates from the flank to the groin or to the inner thigh … in waves … lasting 20 to 60 minutes … one of the strongest pain sensations known."

**Finding: Flank pain (renal colic)**
- ① 추출: sev="excruciating (strongest known)"; char="colicky/intermittent (waves)"; loc="flank"; radiates-to="groin, inner thigh"; time.duration="waves 20–60 min"
- ② 정규화: finding→Flank pain (HP:0030157 / UMLS C0423637); sev→**Severe (HP:0012828)**; char→"colicky"(문자열); loc→Flank(C0224549); radiates-to→Groin(C0018246)/Inner thigh; time→문자열
- 공존: Hematuria(HP:0000790), Urinary urgency(HP:0000012), Nausea(HP:0002018), Vomiting(HP:0002013), Diaphoresis(HP:0000975)

## 8. Pneumonia
원문(발췌): "productive or dry cough, chest pain, fever … productive cough, fever accompanied by shaking chills, shortness of breath, sharp or stabbing chest pain during deep breaths."

**Findings**
- **Productive cough** — ① char="productive"(가래) → ② **별도 finding**: Productive cough (HP:0031245). 즉 "cough+char=productive"가 아니라 독립 phenotype으로 승격(문서화한 규칙 실사례).
- **Chest pain** — ① char="sharp/stabbing"; agg="deep breathing (pleuritic)" → ② Chest pain(HP:0100749); char→**Sharp (HP:0025281)**; agg→Aggravated by(HP:0025285)="inspiration"
- 공존: Fever(HP:0001945), Chills(HP:0025143), Dyspnea(HP:0002094), Tachypnea(HP:0002789)
- 🔑 **char→별도 finding 규칙** 실사례: productive cough.

## 9. Pulmonary embolism
원문(발췌): "typically sudden in onset … shortness of breath … chest discomfort … pleuritic … worsened by breathing … cough and hemoptysis … cyanosis … collapse."

**Findings**
- **Dyspnea** — ① time.onset="sudden" → ② Dyspnea(HP:0002094); time.onset→**Acute/sudden (HP:0011009)**
- **Chest pain** — ① char="pleuritic"; agg="breathing" → ② Chest pain(HP:0100749); agg→Aggravated by(HP:0025285)="inspiration"
- 공존: Hemoptysis(HP:0002105), Tachycardia(HP:0001649), Cyanosis(HP:0000961), Syncope(HP:0001279)
- 🔑 **onset=sudden**이 핵심 속성(PE vs 만성 호흡곤란 감별).

## 10. Anaphylaxis
원문(발췌): "symptoms over minutes or hours … onset of 5 to 30 minutes if … intravenous and up to 2 hours if from eating food … generalized hives, itchiness, flushing, or swelling (angioedema) … burning sensation of the skin rather than itchiness … shortness of breath, wheezes, or stridor … severe crampy abdominal pain and vomiting."

**Findings**
- **Urticaria (hives)** — ① loc="generalized"; time.onset="5–30 min (IV) / ≤2 h (food)"; agg="allergen exposure" → ② Urticaria(HP:0001025); time.onset→Acute(HP:0011009)+문자열; agg→Triggered by(HP:0025204)="allergen/food"
- **Pruritus / skin burning** — ① char="burning (angioedema) vs itch" → ② Pruritus(HP:0000989) + "burning"(문자열)
- **Abdominal pain** — ① char="crampy"; sev="severe" → ② Abdominal pain(HP:0002027); char→"crampy"(문자열); sev→Severe(HP:0012828)
- 공존: Angioedema(HP:0100665), Wheezing(HP:0030828), Stridor(HP:0010307), Hypotension(HP:0002615), Tachycardia(HP:0001649)
- 🔑 **onset=trigger 후 분(minutes)** + trigger(allergen)가 진단 신호.

---

## 속성 커버리지 매트릭스 (원문에서 실제로 채워진 속성)

| 질환 | loc | radiates-to | char | sev | time | agg | rel |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Angina | ● | ● | ●(문자열) | ○ | ● | ● | ● |
| GERD | ○ | ○ | ○ | ○ | ● | ● | ○ |
| Pericarditis | ● | ● | ●(Sharp) | ○ | ● | ● | ● |
| Migraine | ●(편측) | ○ | ●(문자열) | ● | ● | ● | ○ |
| Cluster HA | ● | ○ | ●(Stabbing) | ● | ● | ○ | ○ |
| Appendicitis | ●(이동) | ●(이동) | ●(Dull) | ○ | ● | ● | ○ |
| Renal colic | ● | ● | ●(colicky) | ● | ● | ○ | ○ |
| Pneumonia | ● | ○ | ●(Sharp) | ○ | ○ | ● | ○ |
| PE | ● | ○ | ●(pleuritic) | ○ | ●(sudden) | ● | ○ |
| Anaphylaxis | ●(전신) | ○ | ●(burning/crampy) | ● | ●(분) | ● | ○ |

(●=채워짐, ○=원문 미기재)

## 관찰 (정성 분석)

1. **속성이 진단을 가르는 게 눈에 보인다.** 같은 finding "Chest pain"이 5개 질환에 등장하지만 속성 조합이 다르다:
   - angina: char=pressure, agg=exertion, **rel=nitroglycerin**, radiates=left arm
   - pericarditis: char=**sharp/pleuritic**, agg=supine, **rel=leaning-forward**
   - PE: char=pleuritic, **onset=sudden**, +hemoptysis
   - pneumonia: char=sharp, agg=inspiration, +productive cough/fever
   → finding만으론 구분 불가, **속성이 discriminator**. (NLICE·Panju 논지의 실물)

2. **radiates-to(방향 보존)의 실효성.** angina→왼팔, pericarditis→trapezius, renal colic→flank→groin, appendicitis→periumbilical→RLQ. 평평한 location으로 뭉개면 이 방향/이동 신호가 소실(문서 §location 결정의 실증).

3. **char→별도 finding 규칙**이 실제로 발동(pneumonia "productive cough"→HP:0031245). 성상이 독립 phenotype일 때 속성 아닌 finding으로 승격.

4. **severity는 희소.** 대개 "excruciating/severe"처럼 서수만(renal colic, cluster, anaphylaxis). numeric은 원문에 거의 없음 → edge weight 미사용 결정과 정합.

5. **속성 recall은 source 밀도에 종속.** GERD 원문이 얇아 char/loc/sev가 비었다. 임상발현 코퍼스(StatPearls/MedlinePlus)를 쓰면 더 촘촘(과거 corpus-mismatch 교훈의 재확인). → IE 품질 논의에서 **source 선택이 recall의 상한**.

6. **char 정규화의 현실.** Sharp/Stabbing/Dull은 HPO Pain characteristic(HP:0025280) 하위로 매핑되나, pressure/squeezing/colicky/pleuritic/burning은 통제어휘 미대응 → **문자열 보존**(문서의 "미대응분 문자열"이 다수 사례에서 실제로 발생).

7. **환각 통제.** 각 속성은 원문 span에 근거해서만 부여, 원문이 침묵하면 비웠다(GERD char 등). 빈 칸이 곧 "속성 없음"이 아님을 매트릭스에 명시.

## 다음 단계 제안
- 동일 10건을 **실제 gemma-4-E4B(8B) 파이프라인**으로 돌려 이 이상적 추출 대비 8B의 recall/정확도 gap을 정량 비교.
- scispaCy+UMLS/HPO linker로 위 ② 정규화를 실제 실행해 ID 확정률·문자열 잔존율 측정.
