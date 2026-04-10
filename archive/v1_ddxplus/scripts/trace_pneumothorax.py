import sys
sys.path.insert(0, '.')
from collections import deque
from src.data_loader import DDXPlusLoader
from src.umls_kg import UMLSKG

loader = DDXPlusLoader()
patients = loader.load_patients(split="test")

# Patient 7
patient = patients[7]
gt_eng = loader.fr_to_eng.get(patient.pathology, patient.pathology)
print(f"Patient: {patient.age}yo {patient.sex}, GT: {gt_eng}")
print(f"Initial evidence: {patient.initial_evidence}")

# Get initial evidence name
initial_cui = loader.get_symptom_cui(patient.initial_evidence)
initial_name = None
for code, info in loader.symptom_mapping.items():
    if info.get("cui") == initial_cui:
        initial_name = info.get("name", code)
        break
print(f"Initial CUI: {initial_cui}, Name: {initial_name}")

# Patient positive CUIs
patient_positive_cuis = set()
for ev in patient.evidences:
    code = ev.split("_@_")[0] if "_@_" in ev else ev
    cui = loader.get_symptom_cui(code)
    if cui:
        patient_positive_cuis.add(cui)
print(f"Positive CUIs: {len(patient_positive_cuis)}")

# Run simulation with optimal config
kg = UMLSKG()
kg.reset_state()
kg.state.add_confirmed(initial_cui)

rank_history = deque(maxlen=10)

print(f"\n| IL | Question (2nd hop) | Response | Top-1 Diagnosis | Score |")
print(f"|:--:|-------------------|:--------:|----------------|:-----:|")
print(f"| 0 | — (Initial: {initial_name}) | + | — | — |")

for step in range(50):
    # Get candidates (deny5, noante)
    if not (kg.state.confirmed_cuis - {initial_cui}):
        query = """
        MATCH (s:Symptom {cui: $initial_cui})-[:INDICATES]->(d:Disease)
        MATCH (d)<-[:INDICATES]-(related:Symptom)
        WHERE related.cui <> $initial_cui
          AND NOT related.cui IN $asked_cuis
        WITH related, count(DISTINCT d) AS disease_coverage
        RETURN related.cui AS cui, related.name AS name, disease_coverage
        ORDER BY disease_coverage DESC
        LIMIT 10
        """
        with kg.driver.session() as session:
            result = session.run(query, initial_cui=initial_cui, asked_cuis=list(kg.state.asked_cuis))
            candidates = [{"cui": r["cui"], "name": r["name"]} for r in result]
    else:
        query = """
        MATCH (confirmed:Symptom)-[:INDICATES]->(d:Disease)
        WHERE confirmed.cui IN $confirmed_cuis
        WITH DISTINCT d
        OPTIONAL MATCH (d)<-[:INDICATES]-(denied:Symptom)
        WHERE denied.cui IN $denied_cuis
        WITH d, count(DISTINCT denied) AS denied_count
        WHERE denied_count < 5
        WITH collect(DISTINCT d) AS valid_diseases
        WHERE size(valid_diseases) > 0
        UNWIND valid_diseases AS d
        MATCH (d)<-[:INDICATES]-(next:Symptom)
        WHERE NOT next.cui IN $confirmed_cuis
          AND NOT next.cui IN $denied_cuis
          AND NOT next.cui IN $asked_cuis
        WITH next, d
        MATCH (d)<-[:INDICATES]-(conf:Symptom)
        WHERE conf.cui IN $confirmed_cuis
        WITH next, count(DISTINCT d) AS coverage, count(DISTINCT conf) AS cooccur_count
        RETURN next.cui AS cui, next.name AS name, coverage AS disease_coverage
        ORDER BY toFloat(cooccur_count) * coverage DESC
        LIMIT 10
        """
        with kg.driver.session() as session:
            result = session.run(query,
                confirmed_cuis=list(kg.state.confirmed_cuis),
                denied_cuis=list(kg.state.denied_cuis),
                asked_cuis=list(kg.state.asked_cuis))
            candidates = [{"cui": r["cui"], "name": r["name"]} for r in result]

    if not candidates:
        break

    sel = candidates[0]
    hit = sel["cui"] in patient_positive_cuis

    if hit:
        kg.state.add_confirmed(sel["cui"])
    else:
        kg.state.add_denied(sel["cui"])

    diag = kg.get_diagnosis_candidates(top_k=3)
    top1 = diag[0] if diag else None

    marker = "**+**" if hit else "-"
    name_fmt = f"**{sel['name']}**" if hit else sel['name']
    top1_name = top1.name if top1 else "—"
    top1_score = f"{top1.score:.3f}" if top1 else "—"

    kg_dist = [(c.cui, c.score) for c in diag] if diag else []
    current_ranks = tuple(cui for cui, _ in kg_dist[:3])
    rank_history.append(current_ranks)

    il = step + 1
    print(f"| {il} | {name_fmt} | {marker} | {top1_name} | {top1_score} |")

    if len(rank_history) >= 5:
        recent = list(rank_history)[-5:]
        if all(r == recent[0] for r in recent):
            # Print top-3
            top3_names = [c.name for c in diag[:3]]
            print(f"| | **Top-3 Stable → Stop** | | **{top1_name} (정답)** | **{top1_score}** |")
            print(f"\nTop-3: {', '.join(top3_names)}")
            break

kg.close()
