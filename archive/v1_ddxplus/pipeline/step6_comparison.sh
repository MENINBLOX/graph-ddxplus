#!/bin/bash
# Step 6: 선행연구 비교 및 확인 실험
#
# (a) MEDDxAgent 동일 조건 비교 (100건, fixed IL=5,10,15)
# (b) Complete Profile (134,529건, 상한선 측정)
# (c) 확인 실험 (ANOVA 고정 요인의 GTPA@1 검증)
#
# 사용법:
#   bash pipeline/step6_comparison.sh

set -e
cd "$(dirname "$0")/.."

PORTS="7687,7688,7689,7690,7691,7692,7693,7694"
WORKERS=8

echo "============================================"
echo "Step 6: Comparison Experiments"
echo "============================================"

# (a) MEDDxAgent 비교
echo ""
echo "--- (a) MEDDxAgent Fixed IL Comparison ---"
if [ -f "results/fixed_il_comparison.json" ]; then
    echo "  [skip] 이미 완료"
else
    uv run python scripts/experiment_fixed_il_comparison.py \
        --workers $WORKERS --ports "$PORTS"
fi

# (b) Complete Profile
echo ""
echo "--- (b) Complete Profile Benchmark ---"
if [ -f "results/complete_profile_134529.json" ]; then
    echo "  [skip] 이미 완료"
else
    uv run python scripts/experiment_complete_profile.py \
        --workers $WORKERS --ports "$PORTS"
fi

# (c) 확인 실험
echo ""
echo "--- (c) Confirmatory Experiments ---"
if [ -f "results/confirmatory_134529.json" ]; then
    echo "  [skip] 이미 완료"
else
    uv run python scripts/experiment_confirmatory.py \
        --workers $WORKERS --ports "$PORTS"
fi

echo ""
echo "============================================"
echo "완료!"
echo "============================================"

# 결과 요약
echo ""
echo "=== 결과 요약 ==="

if [ -f "results/fixed_il_comparison.json" ]; then
    echo ""
    echo "(a) MEDDxAgent 비교:"
    python3 -c "
import json
with open('results/fixed_il_comparison.json') as f:
    d = json.load(f)
for il_key, result in d.items():
    if isinstance(result, dict) and 'gtpa_1' in result:
        print(f'  IL={il_key}: GTPA@1={result[\"gtpa_1\"]*100:.0f}%')
" 2>/dev/null || true
fi

if [ -f "results/complete_profile_134529.json" ]; then
    echo ""
    echo "(b) Complete Profile:"
    python3 -c "
import json
with open('results/complete_profile_134529.json') as f:
    d = json.load(f)
print(f'  GTPA@1={d[\"gtpa_1\"]*100:.2f}%, GTPA@10={d[\"gtpa_10\"]*100:.2f}%')
" 2>/dev/null || true
fi

if [ -f "results/confirmatory_134529.json" ]; then
    echo ""
    echo "(c) 확인 실험:"
    python3 -c "
import json
with open('results/confirmatory_134529.json') as f:
    d = json.load(f)
for v, r in d.items():
    if isinstance(r, dict) and 'gtpa_1' in r:
        print(f'  {v}: GTPA@1={r[\"gtpa_1\"]*100:.2f}%')
" 2>/dev/null || true
fi
