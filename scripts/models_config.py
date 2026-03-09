"""모델 설정 파일.

소형 오픈소스 LLM + KG 통합 효과 검증.

실험 설계:
- Category 1: LLM only (baseline)
- Category 2: LLM + KG (본 연구 제안 방법)

모델 선정 기준 (2026.02.26):
- artificialanalysis.ai/models/open-source/tiny 상위 5개
- artificialanalysis.ai/models/open-source/small 상위 5개
- RTX 4090 GPU에서 구동 가능
- VL 모델 제외 (Qwen3-VL-8B-Thinking 예외: Thinking 성능 검증용)
- Llama-3.1-8B-Instruct: 선행 연구 MEDDxAgent 직접 비교용
"""

# =============================================================================
# 소형 오픈소스 LLM (Tiny, 1.2B~8B)
# =============================================================================

TINY_LLM_MODELS = [
    # 1. Qwen3 4B Thinking (Thinking)
    "Qwen/Qwen3-4B-Thinking-2507",
    # 2. Qwen3 4B Instruct (Non-Thinking)
    "Qwen/Qwen3-4B-Instruct-2507",
    # 3. EXAONE 4.0 1.2B (Thinking)
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    # 4. Qwen3 1.7B (Thinking)
    "Qwen/Qwen3-1.7B",
    # 5. Ministral 3B
    "mistralai/Ministral-3-3B-Instruct-2512",
    # 6. Llama-3.1-8B-Instruct (MEDDxAgent 비교용)
    "meta-llama/Llama-3.1-8B-Instruct",
]

# =============================================================================
# 중형 오픈소스 LLM (Small, 8B~24B)
# =============================================================================

SMALL_LLM_MODELS = [
    # 7. GPT-OSS 20B (Thinking)
    "openai/gpt-oss-20b",
    # 8. Nemotron Nano 9B v2 (Thinking + Non-Thinking 둘 다 테스트)
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    # 9. DeepSeek R1 Qwen3 8B (Thinking)
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    # 10. LFM2 8B (Non-Thinking)
    "LiquidAI/LFM2-8B-A1B",
    # 11. Ministral 14B (Non-Thinking)
    "mistralai/Ministral-3-14B-Instruct-2512",
]

# =============================================================================
# Thinking 모델 목록 (응답에서 <think> 블록 제거 필요)
# =============================================================================

THINKING_MODELS = [
    "Qwen/Qwen3-4B-Thinking-2507",
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "Qwen/Qwen3-1.7B",
    "openai/gpt-oss-20b",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
]

# =============================================================================
# 전체 모델 목록
# =============================================================================

ALL_MODELS = TINY_LLM_MODELS + SMALL_LLM_MODELS

# =============================================================================
# 상용 LLM (사용 안 함)
# =============================================================================
# 상용 LLM 제외 근거:
# - 재현성: API 버전 변경/중단으로 실험 재현 불가
# - 프라이버시: 환자 데이터 외부 전송 → HIPAA 등 규정 위반 가능
# - 비용: 대규모 실험 시 API 비용 부담

LARGE_LLM_MODELS = []

# =============================================================================
# 모델 정보
# =============================================================================

MODEL_INFO = """
모델 선정 기준 (2026.02.26):
- artificialanalysis.ai/models/open-source/tiny 상위 5개
- artificialanalysis.ai/models/open-source/small 상위 5개
- RTX 4090 GPU에서 구동 가능
- VL 모델 제외

## Tiny 모델 (6개, 1.2B~8B)

| 모델 | 파라미터 | 유형 |
|------|----------|------|
| Qwen3-4B-Thinking-2507 | 4B | Thinking |
| Qwen3-4B-Instruct-2507 | 4B | Instruct |
| EXAONE-4.0-1.2B | 1.2B | Thinking |
| Qwen3-1.7B | 1.7B | Thinking |
| Ministral-3-3B | 3B | Instruct |
| Llama-3.1-8B-Instruct | 8B | Instruct (MEDDxAgent 비교) |

## Small 모델 (5개, 8B~20B)

| 모델 | 파라미터 | 유형 |
|------|----------|------|
| gpt-oss-20b | 20B | Thinking |
| Nemotron-Nano-9B-v2 | 9B | Thinking + Non-Thinking |
| DeepSeek-R1-0528-Qwen3-8B | 8B | Thinking |
| LFM2-8B-A1B | 8B | Non-Thinking |
| Ministral-3-14B | 14B | Non-Thinking |

총 11개 모델
"""
