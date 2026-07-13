#!/usr/bin/env python3
"""v102 — Structured IE with Pydantic schema + vLLM guided_json.

LLM 출력이 schema에 100% conform하도록 token-level constrain (xgrammar/outlines backend).

학술적 근거:
- HPO modifiers (HP:0012824 Severity, HP:0011008 Onset, HP:0040279 Frequency, HP:0012834 Laterality)
- SNOMED CT (Location 363698007, Episodicity 246456000)
- HL7 FHIR Observation (duration, certainty)
- Phenopackets v2 (excluded field for negative findings)
"""
from __future__ import annotations
import os, sys, json, argparse, re, time
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")


# ============================================================================
# Pydantic schema — academically-grounded ontology mapping
# ============================================================================

# Anatomical location enum (UMLS A1.2 Anatomy aligned)
LocationEnum = Literal[
    "head","face","eye","eyelid","ear","nose","mouth","lip","tongue","throat","larynx",
    "neck","chest","abdomen","epigastric","back","pelvis","groin",
    "arm","shoulder","elbow","wrist","hand","finger",
    "leg","thigh","hip","knee","ankle","foot","toe",
    "skin","joint","bone","muscle","lung","heart","kidney","liver","brain",
    "generalized","systemic"
]

# HP:0012824 Severity modifier
SeverityEnum = Literal["mild","moderate","severe","profound","critical","variable"]

# HP:0012834 Laterality
LateralityEnum = Literal["left","right","bilateral","unilateral","central"]

# HP:0011009/HP:0011010 Onset pace
OnsetPaceEnum = Literal["sudden","rapid","gradual","insidious","variable"]

# HP:0011008 Age of onset
OnsetAgeEnum = Literal["congenital","neonatal","infantile","childhood","juvenile","adult","late","any"]

# FHIR Observation effectivePeriod
DurationEnum = Literal["seconds","minutes","hours","days","weeks","months","years","variable"]

# SNOMED 246456000 Episodicity
EpisodicityEnum = Literal["constant","intermittent","recurrent","episodic","paroxysmal","single"]

# Color qualifier
ColorEnum = Literal["pink","red","purple","blue","yellow","white","brown","cyanotic","pallor"]


class PhenotypeAttributes(BaseModel):
    """Phenotype attributes — qualitative only.

    NOTE: frequency_in_disease 제거 (2026-05-28).
    이유: LLM이 자료 없이 임의 숫자 생성 (hallucination). 실제 prevalence는
    별도 statistical pipeline (multi-source IE counting or PubMed co-occurrence)에서 계산.

    Qualitative attributes은 LLM이 medical knowledge로 reliable하게 추출 가능
    (HPO + SNOMED standardized vocab).
    """
    location: Optional[List[LocationEnum]] = Field(None, description="HP:0410014 anatomical location")
    severity: Optional[SeverityEnum] = Field(None, description="HP:0012824")
    onset_pace: Optional[OnsetPaceEnum] = Field(None, description="HP:0011009/HP:0011010")
    character: Optional[List[str]] = Field(None, description="phenotype-specific quality (sharp/dull/burning for pain)")


class Phenotype(BaseModel):
    name: str = Field(..., description="phenotype/symptom name (English, UMLS-aligned)")
    attributes: PhenotypeAttributes


class Disease(BaseModel):
    disease: str = Field(..., description="disease name (English)")
    phenotypes: List[Phenotype] = Field(..., min_length=3, max_length=8)


class IEOutput(BaseModel):
    diseases: List[Disease] = Field(..., min_length=1, max_length=5)


# ============================================================================
# Prompt
# ============================================================================

PROMPT_TPL = """\
List medical diseases commonly presenting with "{phen}".

For each disease, list 5-12 typical phenotypes. Each phenotype has optional attributes:
- location: list of body parts (controlled vocab)
- laterality: left/right/bilateral/unilateral/central
- severity: mild/moderate/severe/profound/critical
- onset_pace: sudden/rapid/gradual/insidious
- onset_age: congenital/neonatal/infantile/childhood/juvenile/adult/late/any
- duration: seconds/minutes/hours/days/weeks/months/years/variable
- episodicity: constant/intermittent/recurrent/episodic/paroxysmal/single
- character: list of free-text quality descriptors (e.g., pain: sharp/dull/burning)
- color: pink/red/purple/blue/yellow/white/brown/cyanotic/pallor
- frequency_in_disease: 0.0-1.0 (prevalence in this disease)
- trigger: list of triggering factors
- excluded: true if explicitly absent

OMIT attribute keys if unknown. Use ONLY the controlled enum values listed above.

Output JSON matching the schema exactly.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phen", default="Anaphylaxis")
    ap.add_argument("--out", default="pilot/data/cache/v102_structured.json")
    ap.add_argument("--validate_only", action="store_true",
                    help="Skip vLLM, just print schema")
    args = ap.parse_args()

    # Generate JSON schema
    schema = IEOutput.model_json_schema()
    print(f"=== JSON Schema (Pydantic → JSON Schema) ===", flush=True)
    print(json.dumps(schema, indent=2)[:2000], flush=True)
    print(f"... (truncated)\n", flush=True)

    if args.validate_only:
        print("Schema validation mode — exiting", flush=True)
        return

    prompt = PROMPT_TPL.format(phen=args.phen)
    print(f"=== Running v102 structured IE: query='{args.phen}' ===", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=8192, gpu_memory_utilization=0.85,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})

    sampling = SamplingParams(
        temperature=0.4,
        max_tokens=6144,
        top_p=0.95,
        structured_outputs=StructuredOutputsParams(json=schema)
    )

    t0 = time.time()
    out = llm.chat([[{"role":"user", "content": prompt}]], sampling)[0]
    text = out.outputs[0].text
    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.1f}s, {len(text)} chars", flush=True)

    raw_path = args.out.replace(".json", "_raw.txt")
    with open(raw_path, "w") as f: f.write(text)

    # Validate via Pydantic
    try:
        parsed = json.loads(text)
        validated = IEOutput(**parsed)
        print(f"\n✓ Pydantic validation PASSED — {len(validated.diseases)} diseases", flush=True)
        for d in validated.diseases:
            n_phen = len(d.phenotypes)
            print(f"  - {d.disease}: {n_phen} phenotypes", flush=True)
            for p in d.phenotypes[:3]:
                attrs_used = {k: v for k, v in p.attributes.model_dump().items() if v is not None and v is not False}
                print(f"      [{p.name}] attrs: {attrs_used}", flush=True)
    except Exception as e:
        print(f"\n❌ Validation FAILED: {e}", flush=True)
        print(f"   Raw output[:500]: {text[:500]}", flush=True)
        return

    with open(args.out, "w") as f:
        json.dump(validated.model_dump(), f, indent=2)
    print(f"\nSaved validated output → {args.out}", flush=True)


if __name__ == "__main__":
    main()
