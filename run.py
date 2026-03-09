#!/usr/bin/env python3
"""Graph-DDXPlus: LLM + UMLS KG 의료 진단 벤치마크.

Usage:
    uv run python run.py                    # 기본 (100 샘플)
    uv run python run.py -n 1000            # 1000 샘플
    uv run python run.py -m gpt-oss:20b     # 다른 모델
    uv run python run.py --full             # 전체 테스트셋 (134,529)
"""

import argparse
import time
from pathlib import Path

from src.data_loader import DDXPlusLoader
from src.evaluator import BenchmarkResult
from src.llm_agent import AgentConfig, LLMDiagnosisAgent
from src.patient_simulator import PatientSimulator, ResponseType
from src.umls_kg import UMLSKG


def run_benchmark(
    model: str = "qwen3:4b-instruct-2507-fp16",
    n_samples: int | None = 100,
    verbose: bool = True,
) -> BenchmarkResult:
    """벤치마크 실행.

    Args:
        model: Ollama 모델명
        n_samples: 샘플 수 (None이면 전체)
        verbose: 진행 상황 출력

    Returns:
        BenchmarkResult
    """
    loader = DDXPlusLoader()
    kg = UMLSKG()
    agent = LLMDiagnosisAgent(AgentConfig(model=model, temperature=0.1))

    patients = loader.load_patients(split="test", n_samples=n_samples)
    total = len(patients)

    if verbose:
        print(f"{'='*60}")
        print(f"Model: {model}")
        print(f"Samples: {total:,}")
        print(f"{'='*60}")

    results: list[dict] = []
    start = time.time()

    for idx, patient in enumerate(patients):
        sim = PatientSimulator(patient, loader)
        kg.reset_state()
        agent.reset()

        # 초기 증상
        init_cui = sim.get_initial_evidence_cui()
        if not init_cui:
            continue

        init_name = loader.symptom_mapping.get(patient.initial_evidence, {}).get("name", "")
        kg.state.add_confirmed(init_cui)
        conf_names, den_names = [init_name], []

        # 진단 루프
        il = 0
        while il < 25:
            should_stop, _ = kg.should_stop()
            if should_stop:
                break

            cands = kg.get_candidate_symptoms(init_cui, limit=10)
            if not cands:
                break

            diag = kg.get_diagnosis_candidates(top_k=5)
            sel = agent.select_next_symptom(init_name, conf_names, den_names, cands, diag)
            if not sel:
                break

            sym = next((c for c in cands if c.cui == sel), None)
            if not sym:
                break

            resp = sim.ask(sel)
            il += 1

            if resp.response_type == ResponseType.VALID_YES:
                kg.state.add_confirmed(sel)
                conf_names.append(sym.name)
            else:
                kg.state.add_denied(sel)
                den_names.append(sym.name)

        # 최종 진단
        diag = kg.get_diagnosis_candidates(top_k=5)
        pred = agent.get_final_diagnosis(init_name, conf_names, den_names, diag) if diag else None
        correct = pred == sim.get_ground_truth_cui()

        results.append({"il": il, "correct": correct})

        # 진행 상황 출력
        if verbose and (idx + 1) % 100 == 0:
            elapsed = time.time() - start
            acc = sum(r["correct"] for r in results) / len(results)
            avg_il = sum(r["il"] for r in results) / len(results)
            rate = (idx + 1) / elapsed
            remaining = (total - idx - 1) / rate if rate > 0 else 0
            print(
                f"[{idx+1:,}/{total:,}] GTPA@1: {acc:.1%}, IL: {avg_il:.1f} | "
                f"{elapsed/60:.1f}min, ~{remaining/60:.1f}min left"
            )

    kg.close()
    elapsed = time.time() - start

    # 결과 계산
    n = len(results)
    acc = sum(r["correct"] for r in results) / n if n > 0 else 0
    avg_il = sum(r["il"] for r in results) / n if n > 0 else 0

    result = BenchmarkResult(
        n_samples=n,
        gtpa_at_1=acc,
        ddr=0.0,  # DD 메트릭은 별도 계산 필요
        ddp=0.0,
        ddf1=0.0,
        avg_il=avg_il,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS - {model}")
        print(f"{'='*60}")
        print(f"Samples: {n:,}")
        print(f"GTPA@1: {acc:.2%}")
        print(f"Avg IL: {avg_il:.2f}")
        print(f"Time: {elapsed/60:.1f} min")
        print(f"{'='*60}")
        print(f"\nvs AARLC: GTPA@1 {acc:.2%} vs 75.39% ({(acc-0.7539)*100:+.2f}%)")
        print(f"vs AARLC: IL {avg_il:.2f} vs 25.75 ({avg_il-25.75:+.2f})")

    return result


def main() -> None:
    """메인 함수."""
    parser = argparse.ArgumentParser(description="Graph-DDXPlus Benchmark")
    parser.add_argument("-m", "--model", default="qwen3:4b-instruct-2507-fp16", help="Ollama model")
    parser.add_argument("-n", "--n-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--full", action="store_true", help="Run on full test set")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    n_samples = None if args.full else args.n_samples
    run_benchmark(model=args.model, n_samples=n_samples, verbose=not args.quiet)


if __name__ == "__main__":
    main()
