"""Aggregate the IE model panel into one adjudicated table: gemma-4-E4B (V4 baseline) vs
the panel models, on location + severity, raw + source-adjudicated F1. Stdlib only."""
import json, os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v147_adjudicate import adjudicate
from score_maccrobat_attrs import parse_gold

ANN_DIR = "pilot/cache/maccrobat/brat"
# label -> pred path (gemma-E4B via V4 is the fixed-architecture baseline)
MODELS = {
    "gemma4-E4B": "pilot/cache/maccrobat/v147_V4_pred.json",
    "qwen25-7b":  "pilot/cache/maccrobat/v148_qwen25-7b_pred.json",
    "gemma4-12b": "pilot/cache/maccrobat/v148_gemma4-12b_pred.json",
    "qwen25-14b": "pilot/cache/maccrobat/v148_qwen25-14b_pred.json",
    "medgemma-4b": "pilot/cache/maccrobat/v148_medgemma-4b_pred.json",
    "biomistral-7b":  "pilot/cache/maccrobat/v148_biomistral-7b_pred.json",
    "openbio-8b":     "pilot/cache/maccrobat/v148_openbio-8b_pred.json",
    "med42-8b":       "pilot/cache/maccrobat/v148_med42-8b_pred.json",
    "phi35-mini":     "pilot/cache/maccrobat/v148_phi35-mini_pred.json",
}

def main():
    have = {k: json.load(open(v)) for k, v in MODELS.items() if os.path.exists(v)}
    for attr in ("location", "severity"):
        gold = parse_gold(ANN_DIR, "severity" if attr == "severity" else "biological_structure")
        ng = sum(len(v) for v in gold.values())
        print(f"\n### {attr.upper()}  (V4 prompt fixed; gold pairs={ng})")
        print(f"{'model':12} {'rawP':>5} {'rawR':>5} {'rawF1':>6} | {'adjP':>5} {'adjR':>5} {'adjF1':>6} | {'bktB':>5}")
        rows = []
        for k, preds in have.items():
            (Pb, Rb, Fb), (Pa, Ra, Fa), B = adjudicate(preds, gold, attr)
            rows.append((k, Pb, Rb, Fb, Pa, Ra, Fa, B))
        for k, Pb, Rb, Fb, Pa, Ra, Fa, B in sorted(rows, key=lambda r: -r[6]):
            print(f"{k:12} {Pb:5.2f} {Rb:5.2f} {Fb:6.2f} | {Pa:5.2f} {Ra:5.2f} {Fa:6.2f} | {B:5}")

if __name__ == "__main__":
    main()
