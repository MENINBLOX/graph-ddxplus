"""2-stage CoT IE — Stage 2: classify each extracted evidence as symptom/sign.

Stage 1 (reused): clean concise findings already extracted per disease
(e.g. v103c_attr_ddx49_per_disease: "groin", "pain", "bulge", "tenderness").

Stage 2 (this script): for each unique finding of a disease, the LLM reasons
(CoT, reasoning field filled BEFORE category → chain-of-thought) and assigns a
BINARY label symptom|sign. If sign, it records which test/exam is needed and why
(stored in the improved pydantic schema).

Academically clean: classification is principle-based (definitional: a symptom is
patient-experienced/self-reportable; a sign needs a clinician's exam or a
lab/imaging test). Few-shot exemplars are GENERIC clinical concepts unrelated to
any benchmark disease (CLAUDE.md 원칙 #5), not benchmark cases.

guided_json (StructuredOutputsParams) enforces the schema, so output is valid.
Output keeps the SAME aggregated schema (+category) so v103_build_kg_cui can build.
"""
import os, sys, json, argparse, glob
from pathlib import Path

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "category": {"type": "string", "enum": ["symptom", "sign"]},
                    "required_test": {"type": "string"},
                    "test_reason": {"type": "string"},
                },
                "required": ["name", "reasoning", "category"],
            },
        }
    },
    "required": ["items"],
}

PROMPT = '''You classify each clinical finding as a SYMPTOM or a SIGN. Reason first, then label.

DEFINITIONS (principle-based):
- symptom = a SUBJECTIVE experience the patient feels and can self-report WITHOUT a clinician (pain, cough, a lump/swelling the patient notices, itching, fatigue, dizziness, nausea). Anatomical LOCATION terms the patient can point to (groin, flank, chest) count as symptom-side context.
- sign = something requiring a CLINICIAN: an objective physical-exam finding (heart murmur, stridor on auscultation, hepatomegaly on palpation) OR a laboratory/imaging/procedure result (elevated white cell count, consolidation on chest X-ray) OR a prior condition/risk known only from records.
If a finding is a SIGN, also state which test or examination is needed to obtain it and why.

EXAMPLES (generic, for calibration only):
- "chest pain" → reasoning: patient feels and reports it directly; category: symptom
- "shortness of breath" → reasoning: subjective sensation the patient reports; category: symptom
- "groin" → reasoning: body location the patient can indicate; category: symptom
- "heart murmur" → reasoning: only detectable by a clinician on auscultation; category: sign; required_test: "cardiac auscultation / echocardiography"; test_reason: "to detect and characterize abnormal heart sounds"
- "elevated white blood cell count" → reasoning: a laboratory measurement; category: sign; required_test: "complete blood count"; test_reason: "to quantify leukocytes"
- "hepatomegaly" → reasoning: found on physical exam or imaging, not self-reported; category: sign; required_test: "abdominal palpation / ultrasound"; test_reason: "to assess liver size"

Now classify EVERY finding below for the disease "{disease}". Output JSON {{"items":[...]}} with one entry per finding.

FINDINGS:
{findings}'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="pilot/data/cache/v103c_attr_ddx49_per_disease")
    ap.add_argument("--out_dir", default="pilot/data/cache/v103e_classified_per_disease")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{args.in_dir}/*.json"))
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0.0, max_tokens=4096,
                              structured_outputs=StructuredOutputsParams(json=SCHEMA))

    # build one prompt per disease (list of finding names)
    tasks = []
    for f in files:
        o = json.load(open(f))
        names = list(o["aggregated"].keys())
        if not names: continue
        flist = "\n".join(f"- {n}" for n in names)
        tasks.append((o, PROMPT.format(disease=o["disease"], findings=flist)))

    convs = [[{"role": "user", "content": p}] for _, p in tasks]
    outs = llm.chat(convs, sampling, use_tqdm=True)

    for (o, _), out in zip(tasks, outs):
        try:
            res = json.loads(out.outputs[0].text)
            cls = {it["name"].strip().lower(): it for it in res.get("items", []) if it.get("name")}
        except Exception:
            cls = {}
        new_agg = {}
        for name, ent in o["aggregated"].items():
            c = cls.get(name.strip().lower(), {})
            cat = c.get("category", "symptom")
            ent = dict(ent)
            ent["category"] = cat
            ent["class_reasoning"] = c.get("reasoning", "")
            ent["required_test"] = c.get("required_test", "")
            ent["test_reason"] = c.get("test_reason", "")
            new_agg[name] = ent
        oo = dict(o); oo["aggregated"] = new_agg
        json.dump(oo, open(f"{args.out_dir}/{o['cui']}.json", "w"))
        from collections import Counter
        cc = Counter(v["category"] for v in new_agg.values())
        print(f"  {o['disease']}: {dict(cc)}", flush=True)


if __name__ == "__main__":
    main()
