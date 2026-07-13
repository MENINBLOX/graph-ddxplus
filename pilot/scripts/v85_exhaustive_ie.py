#!/usr/bin/env python3
"""v85 — Exhaustive LLM IE (no phenotype count limit).

Cycle 7 fix: CLAUDE.md 원칙 6 (phenotype 수 제한 금지) 적용.
v80~v84는 "15-30 findings" 같은 제한으로 SymCat coverage 30%에 그침.

v85 prompt:
- NO phenotype count limit
- "Exhaustive list of ALL symptoms/signs/findings"
- Multi-dimension directive: lay terms, anatomical, onset, severity, demographic, symptom variants
- Benchmark-blind few-shot (generic medical disease only)
- 입력: disease name only (벤치마크 question/symptom list 미포함)
"""
from __future__ import annotations
import os, sys, json, argparse, re, time
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")


FEWSHOT = """\
## Examples (generic medical diseases for prompt calibration — NOT from any evaluation benchmark)

### Disease: Iron deficiency anemia
- Fatigue — 70
- Tiredness — 65
- Weakness — 60
- Pallor of skin — 60
- Pallor of conjunctiva — 60
- Pale lips — 50
- Pale palms — 40
- Heavy menstrual bleeding (women of reproductive age) — 45
- Heavy periods — 45
- Menorrhagia — 35
- Pica (craving ice or non-food) — 15
- Craving ice — 12
- Eating ice — 10
- Koilonychia (spoon-shaped nails) — 10
- Brittle nails — 25
- Spoon-shaped nails — 8
- Glossitis (smooth shiny tongue) — 15
- Sore tongue — 18
- Beefy red tongue — 8
- Angular cheilitis — 8
- Cracked corners of mouth — 12
- Restless legs syndrome — 15
- Leg restlessness at night — 15
- Exertional dyspnea — 50
- Shortness of breath on exertion — 50
- Shortness of breath with walking — 35
- Hair loss — 25
- Brittle hair — 20
- Dry skin — 30
- Itchy skin — 18
- Headache — 30
- Dizziness — 25
- Lightheadedness — 20
- Cold hands — 28
- Cold feet — 26
- Tinnitus (ringing in ears) — 10
- Decreased appetite — 18
- Difficulty concentrating — 30
- Brain fog — 20
- Chest pain (with severe anemia) — 12
- Palpitations — 22
- Rapid heartbeat — 20

### Disease: Acute appendicitis
- Periumbilical pain shifting to right lower quadrant — 75
- Right lower quadrant pain — 80
- Lower abdominal pain — 70
- Sharp abdominal pain — 55
- McBurney point tenderness — 70
- Rebound tenderness — 60
- Guarding on palpation — 50
- Rovsing sign positive — 30
- Psoas sign positive — 25
- Obturator sign positive — 15
- Anorexia (loss of appetite) — 70
- Loss of appetite — 70
- Decreased appetite — 65
- Nausea — 65
- Vomiting — 55
- Low-grade fever — 50
- Fever — 45
- Constipation — 25
- Diarrhea — 20
- Bloating — 18
- Abdominal distension — 15
- Pain worsened with movement — 50
- Pain worsened with coughing — 45
- Limping or hunched posture — 30
- Inability to lie flat — 25
- Tachycardia — 35
- Rapid heartbeat — 30
- Sweating — 20

### Disease: Hypothyroidism
- Fatigue — 75
- Tiredness — 70
- Lethargy — 50
- Cold intolerance — 60
- Feeling cold often — 55
- Cold hands — 35
- Cold feet — 35
- Weight gain — 50
- Unexplained weight gain — 45
- Constipation — 50
- Dry skin — 55
- Coarse skin — 30
- Hair loss — 40
- Hair thinning — 35
- Brittle hair — 30
- Brittle nails — 25
- Hoarse voice — 25
- Voice deepening — 18
- Bradycardia (slow heart rate) — 30
- Slow pulse — 28
- Periorbital edema — 25
- Puffy face — 30
- Macroglossia (enlarged tongue) — 10
- Goiter (enlarged thyroid) — 15
- Neck swelling — 18
- Depression — 30
- Sadness — 28
- Brain fog — 35
- Memory problems — 28
- Difficulty concentrating — 35
- Muscle cramps — 25
- Muscle stiffness — 22
- Joint stiffness — 25
- Joint pain — 22
- Menstrual irregularity (women) — 25
- Heavy periods — 18
- Carpal tunnel syndrome — 15
- Numbness in hands — 18
- Reduced libido — 25
- Delayed deep tendon reflex relaxation — 35
"""


PROMPT_TEMPLATE = """\
# Task
You are a medical knowledge base. For the disease "{disease}", list ALL
symptoms, signs, and clinical features that a patient may report or that
would be noted on examination. Be EXHAUSTIVE — include every relevant
finding regardless of frequency.

For each, estimate the percentage (1-99) of typical patients exhibiting it.

CRITICAL RULES:
- EXHAUSTIVE: Do NOT limit the number. List as many findings as you can
  recall, including rare and unusual presentations.
- MULTIPLE FORMS for the same concept (LAY + CLINICAL + SYNONYMS):
  - "Shortness of breath" + "Dyspnea" + "Difficulty breathing"
  - "Pale skin" + "Pallor"
  - "Belly pain" + "Abdominal pain"
- Include ANATOMICAL specificity:
  - "Right lower quadrant pain" (not just "abdominal pain")
  - "Periorbital edema" (not just "edema")
- Include ONSET/SEVERITY/DURATION qualifiers:
  - "Sudden onset chest pain"
  - "Severe headache"
  - "Chronic cough"
- Include DEMOGRAPHIC associations:
  - "More common in women over 50"
  - "History of smoking"
- Include EXAM findings (signs visible to clinician):
  - "Murphy sign positive", "Rebound tenderness", "Murmur"
- DO NOT include:
  - Diagnostic tests (CBC, X-ray, MRI, biopsy)
  - Treatments or medications
  - Disease names that are NOT phenotypes of this disease

Output rules:
- One line per finding: `- <finding> — <percentage>`
- Vary percentages across the FULL range (1-99)

{fewshot}

### Disease: {disease}
List exhaustively below.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=-1, help="-1 = all")
    ap.add_argument("--include_rarebench", action="store_true",
                    help="Include RareBench (11K disease, slow)")
    args = ap.parse_args()

    diseases = []
    seen = set()
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    for dn in icd:
        if dn not in seen:
            diseases.append((dn, "ddxplus")); seen.add(dn)
    parsed = json.load(open("data/symcat/symcat_parsed_full.json"))
    for dn in parsed["disease_symptom_pairs"]:
        if dn not in seen:
            diseases.append((dn, "symcat")); seen.add(dn)

    # Add RareBench diseases (CLAUDE.md 원칙 2: 모든 benchmark cover) — optional
    if args.include_rarebench:
        try:
            rb_dis = json.load(open("data/rarebench/disease_umls_mapping.json"))["mapping"]
            for did, info in rb_dis.items():
                dn = info.get("umls_name") or did
                if dn and dn not in seen:
                    diseases.append((dn, "rarebench")); seen.add(dn)
        except Exception as e:
            print(f"  RareBench mapping load failed (skip): {e}", flush=True)

    print(f"Disease pool (full): {len(diseases)} "
          f"(DDXPlus: {sum(1 for _, s in diseases if s == 'ddxplus')}, "
          f"SymCat: {sum(1 for _, s in diseases if s == 'symcat')}, "
          f"RareBench: {sum(1 for _, s in diseases if s == 'rarebench')})",
          flush=True)
    end = args.end_idx if args.end_idx >= 0 else len(diseases)
    diseases = diseases[args.start_idx:end]
    print(f"This shard: [{args.start_idx}:{end}] = {len(diseases)} diseases", flush=True)

    prompts = []; meta = []
    for dn, src in diseases:
        p = PROMPT_TEMPLATE.format(disease=dn, fewshot=FEWSHOT)
        for s in range(args.n_samples):
            prompts.append(p)
            meta.append({"disease": dn, "source": src, "sample": s})
    print(f"Total prompts: {len(prompts)}", flush=True)

    print("Loading vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=6144, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=args.temperature,
                              max_tokens=args.max_tokens, top_p=0.95)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    disease_data = defaultdict(lambda: defaultdict(list))
    line_re = re.compile(r"-\s*([^—\-]+?)\s*[—\-]\s*(\d+)")
    t0 = time.time()

    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i+args.batch]
        meta_chunk = meta[i:i+args.batch]
        convs = [[{"role": "user", "content": p}] for p in chunk]
        outs = llm.chat(convs, sampling)
        for m, o in zip(meta_chunk, outs):
            try: text = o.outputs[0].text
            except: text = ""
            for line in text.split("\n"):
                line = line.strip()
                mat = line_re.match(line)
                if not mat: continue
                phen = mat.group(1).strip().rstrip(".,;:()").strip()
                if not phen or len(phen) < 3 or len(phen) > 150: continue
                low = phen.lower()
                if any(skip in low for skip in [
                    " test", "x-ray", "ultrasound", "scan ", "biopsy",
                    "diagnostic ", "treatment", "medication", "antibiotic",
                    "therapy", "vaccine", " mri", " ct ", "panel", "level of",
                    "blood test"
                ]): continue
                pct = int(mat.group(2))
                if not (1 <= pct <= 99): continue
                disease_data[m["disease"]][low].append(pct)
        if (i // args.batch) % 10 == 0:
            print(f"  {i+len(chunk)}/{len(prompts)} ({time.time()-t0:.0f}s)", flush=True)

    n_records = 0
    with open(args.out, "w") as f:
        for dn, src in diseases:
            phens = disease_data.get(dn, {})
            agg = {phen: {"prob": sum(pcts)/len(pcts)/100.0, "n_seen": len(pcts)}
                   for phen, pcts in phens.items()}
            rec = {"disease": dn, "source": src, "n_samples": args.n_samples,
                   "phenotypes": agg}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_records += len(agg)
    print(f"\nSaved {len(diseases)} disease records → {args.out}", flush=True)
    print(f"Total phenotype entries: {n_records}, avg/disease: {n_records/max(len(diseases),1):.1f}",
          flush=True)


if __name__ == "__main__":
    main()
