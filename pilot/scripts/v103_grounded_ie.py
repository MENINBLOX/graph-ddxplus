#!/usr/bin/env python3
"""v103 — Source-grounded structured IE with evidence-span citation.

핵심 변경 vs v102:
1. Input: PubMed abstract 또는 clinical text (source-grounded)
2. 각 attribute에 evidence_span (text quote) 필수
3. LLM이 자기 지식으로 채움 금지 — text에 없으면 omit
4. Validation: citation 없는 attribute 자동 제거

학술적 정당성:
- LLM hallucination 영역 제거 (qualitative descriptive도 self-knowledge 차단)
- 모든 추출 정보가 reproducible (source text trace 가능)
- HPO/SNOMED standardized vocab + Phenopackets v2 evidence pattern
"""
from __future__ import annotations
import os, sys, json, argparse, re, time
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")


# Same controlled vocabularies as v102
LocationEnum = Literal[
    "head","face","eye","eyelid","ear","nose","mouth","lip","tongue","throat","larynx",
    "neck","chest","abdomen","epigastric","back","pelvis","groin","arm","shoulder",
    "elbow","wrist","hand","finger","leg","thigh","hip","knee","ankle","foot","toe",
    "skin","joint","bone","muscle","lung","heart","kidney","liver","brain",
    "generalized","systemic"
]
SeverityEnum = Literal["mild","moderate","severe","profound","critical","variable"]
OnsetPaceEnum = Literal["sudden","rapid","gradual","insidious","variable"]


class LocationAttribute(BaseModel):
    values: List[LocationEnum] = Field(..., min_length=1)
    evidence_span: str = Field(..., min_length=5, max_length=200)


class SeverityAttribute(BaseModel):
    """Severity with enum-strict value (HP:0012824)."""
    value: SeverityEnum
    evidence_span: str = Field(..., min_length=5, max_length=200)


class OnsetPaceAttribute(BaseModel):
    """Onset pace with enum-strict value (HP:0011009/HP:0011010)."""
    value: OnsetPaceEnum
    evidence_span: str = Field(..., min_length=5, max_length=200)


class CharacterAttribute(BaseModel):
    values: List[str] = Field(..., min_length=1, max_length=5)
    evidence_span: str = Field(..., min_length=5, max_length=200)


class PhenotypeGrounded(BaseModel):
    """Phenotype with evidence-grounded attributes.

    All attributes are Optional. If attribute is mentioned in source text,
    extract value + evidence_span. Otherwise OMIT entirely (do not hallucinate)."""
    name: str
    location: Optional[LocationAttribute] = None
    severity: Optional[SeverityAttribute] = None
    onset_pace: Optional[OnsetPaceAttribute] = None
    character: Optional[CharacterAttribute] = None


class IEOutputGrounded(BaseModel):
    """List of phenotypes extracted from one source document."""
    phenotypes: List[PhenotypeGrounded] = Field(..., min_length=0, max_length=15)


# ============================================================================
# Prompt — explicitly text-grounded
# ============================================================================

PROMPT_GROUNDED = """\
You are extracting medical phenotypes from a source document.

SOURCE TEXT:
\"\"\"
{source_text}
\"\"\"

DISEASE: {disease}

EXTRACTION RULES (strict, must follow):
1. Extract ONLY phenotypes (symptoms/signs/findings) EXPLICITLY mentioned in the source text above.
2. For each phenotype, extract attributes ONLY IF they are STATED in the text.
3. Do NOT use your general medical knowledge. If the text does not specify an attribute, OMIT that attribute entirely.
4. Each attribute value must be supported by a verbatim text quote (evidence_span) — minimum 5 characters.
5. If text says "pain" without specifying location/severity/onset, output ONLY {{"name": "pain"}} — NO attribute fields.
6. CRITICAL: Match attribute SEMANTICS, not just text presence:
   - "rapidly" / "rapid" / "sudden" → onset_pace (NOT severity)
   - "severe" / "mild" / "moderate" → severity
   - "abdomen" / "chest" → location
   - "cramping" / "burning" / "sharp" → character
   Mis-categorization is a critical error.
7. If a phrase doesn't fit any controlled vocabulary, OMIT it. Do not force-fit.
8. ONSET_PACE rule (strict): only extract if the text explicitly states the
   TIMING/SPEED of THIS phenotype's appearance. Valid examples:
   - "rapid-onset urticaria" → urticaria has onset_pace=rapid
   - "sudden chest pain" → pain has onset_pace=sudden
   - "develops over hours" → onset_pace=gradual
   Invalid examples (DO NOT extract):
   - "indicates upper airway obstruction" — describes mechanism, not onset
   - "reflects bronchospasm" — describes mechanism, not onset
   - General disease description ("the disease is acute") — applies to disease, not this phenotype
   If onset is not explicitly stated for THIS specific phenotype, OMIT the onset_pace field.
9. Apply same rule to severity: only extract if text uses severity words
   ("mild","moderate","severe","profound","critical") DIRECTLY modifying THIS phenotype.

CONTROLLED VOCABULARIES (use EXACT values, no variants):
- location: ["head","face","eye","eyelid","ear","nose","mouth","lip","tongue","throat","larynx","neck","chest","abdomen","epigastric","back","pelvis","groin","arm","shoulder","elbow","wrist","hand","finger","leg","thigh","hip","knee","ankle","foot","toe","skin","joint","bone","muscle","lung","heart","kidney","liver","brain","generalized","systemic"]
- severity: ["mild","moderate","severe","profound","critical","variable"]
- onset_pace: ["sudden","rapid","gradual","insidious","variable"]

If text says "mild to moderate", choose ONE value (e.g., "moderate") and quote the supporting text.

Output strict JSON. OMIT attribute keys not stated in source text.
"""


# ============================================================================
# Demonstration with synthetic PubMed-like abstract
# ============================================================================

DEMO_ABSTRACT = """\
Anaphylaxis is a severe, potentially fatal systemic allergic reaction with sudden onset.
Patients typically present with rapid-onset urticaria (hives) on the skin, sometimes
accompanied by severe pruritus (itching). Angioedema involving the face, lips, and throat
is a hallmark, often progressing within minutes. Respiratory involvement is critical:
stridor indicates upper airway obstruction (larynx), while wheezing reflects bronchospasm
in the lung. Gastrointestinal symptoms include abdominal cramping and nausea, usually mild
to moderate. Hypotension can develop rapidly, manifesting as dizziness and pallor.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pilot/data/cache/v103_grounded.json")
    ap.add_argument("--disease", default="Anaphylaxis")
    ap.add_argument("--source_file", help="Optional: path to source text file (uses DEMO_ABSTRACT if omitted)")
    args = ap.parse_args()

    source_text = DEMO_ABSTRACT
    if args.source_file:
        source_text = open(args.source_file).read()

    print(f"=== v103 grounded IE: disease={args.disease} ===", flush=True)
    print(f"Source text ({len(source_text)} chars):\n{source_text[:300]}...\n", flush=True)

    schema = IEOutputGrounded.model_json_schema()
    prompt = PROMPT_GROUNDED.format(source_text=source_text, disease=args.disease)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=8192, gpu_memory_utilization=0.85,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(
        temperature=0.2,  # low for grounded extraction
        max_tokens=4096,
        top_p=0.9,
        structured_outputs=StructuredOutputsParams(json=schema)
    )

    t0 = time.time()
    out = llm.chat([[{"role":"user", "content": prompt}]], sampling)[0]
    text = out.outputs[0].text
    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.1f}s, {len(text)} chars", flush=True)

    raw_path = args.out.replace(".json", "_raw.txt")
    with open(raw_path, "w") as f: f.write(text)

    # Validate
    ONSET_KEYWORDS = {"sudden", "rapid", "rapidly", "gradual", "gradually",
                       "insidious", "abrupt", "immediate", "minutes", "hours",
                       "instant", "acute", "chronic", "over time", "progressive"}
    SEVERITY_KEYWORDS = {"mild", "moderate", "severe", "profound", "critical",
                          "intense", "extreme", "slight", "marked", "significant"}

    def has_onset_keyword(text):
        t = text.lower()
        return any(kw in t for kw in ONSET_KEYWORDS)

    def has_severity_keyword(text):
        t = text.lower()
        return any(kw in t for kw in SEVERITY_KEYWORDS)

    try:
        parsed = json.loads(text)
        validated = IEOutputGrounded(**parsed)
        print(f"\n✓ Pydantic validation PASSED — {len(validated.phenotypes)} phenotypes (pre-filter)", flush=True)

        # Post-validation: enforce keyword in evidence_span
        n_dropped = 0
        for p in validated.phenotypes:
            if p.onset_pace and not has_onset_keyword(p.onset_pace.evidence_span):
                print(f"  [{p.name}] DROPPING onset_pace='{p.onset_pace.value}' — no onset keyword in '{p.onset_pace.evidence_span[:50]}'", flush=True)
                p.onset_pace = None
                n_dropped += 1
            if p.severity and not has_severity_keyword(p.severity.evidence_span):
                print(f"  [{p.name}] DROPPING severity='{p.severity.value}' — no severity keyword in '{p.severity.evidence_span[:50]}'", flush=True)
                p.severity = None
                n_dropped += 1
        print(f"\n  Post-validation dropped {n_dropped} unsupported attributes\n", flush=True)

        # Citation-aware analysis
        for p in validated.phenotypes:
            attrs_with_evidence = []
            if p.location: attrs_with_evidence.append(f"location={p.location.values} ← \"{p.location.evidence_span[:60]}\"")
            if p.severity: attrs_with_evidence.append(f"severity={p.severity.value} ← \"{p.severity.evidence_span[:60]}\"")
            if p.onset_pace: attrs_with_evidence.append(f"onset={p.onset_pace.value} ← \"{p.onset_pace.evidence_span[:60]}\"")
            if p.character: attrs_with_evidence.append(f"character={p.character.values} ← \"{p.character.evidence_span[:60]}\"")
            n_attrs = len(attrs_with_evidence)
            print(f"  [{p.name}] ({n_attrs} attrs with citation)", flush=True)
            for a in attrs_with_evidence:
                # Validate evidence_span actually in source
                span = re.search(r'"([^"]*)"$', a)
                if span:
                    quote = span.group(1)
                    in_source = quote.lower() in source_text.lower()
                    mark = "✓" if in_source else "❌ NOT IN SOURCE"
                    print(f"    {a} [{mark}]", flush=True)
                else:
                    print(f"    {a}", flush=True)
    except Exception as e:
        print(f"\n❌ Validation FAILED: {e}", flush=True)
        print(f"   Raw[:500]: {text[:500]}", flush=True)
        return

    with open(args.out, "w") as f:
        json.dump(validated.model_dump(), f, indent=2)
    print(f"\nSaved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
