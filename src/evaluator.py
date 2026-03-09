"""평가 메트릭.

DDXPlus 평가 지표: GTPA@1, DDR, DDF1, IL
"""

from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """단일 환자 평가 결과."""

    patient_id: int
    ground_truth: str  # 정답 질환 CUI
    predicted: str | None  # 예측 질환 CUI
    predicted_dd: list[tuple[str, float]]  # 예측 감별진단 [(CUI, prob), ...]
    ground_truth_dd: list[str]  # 정답 감별진단 CUI 목록
    interaction_length: int
    correct: bool  # Top-1 정확


@dataclass
class BenchmarkResult:
    """벤치마크 결과."""

    n_samples: int
    gtpa_at_1: float  # Ground Truth Probability Assignment @1
    ddr: float  # Differential Diagnosis Recall
    ddp: float  # Differential Diagnosis Precision
    ddf1: float  # Differential Diagnosis F1
    avg_il: float  # Average Interaction Length

    def __str__(self) -> str:
        return (
            f"Benchmark Results (n={self.n_samples}):\n"
            f"  GTPA@1: {self.gtpa_at_1:.2%}\n"
            f"  DDR:    {self.ddr:.2%}\n"
            f"  DDP:    {self.ddp:.2%}\n"
            f"  DDF1:   {self.ddf1:.2%}\n"
            f"  Avg IL: {self.avg_il:.2f}"
        )


class Evaluator:
    """평가기."""

    def __init__(self) -> None:
        self.results: list[EvaluationResult] = []

    def add_result(self, result: EvaluationResult) -> None:
        """평가 결과 추가."""
        self.results.append(result)

    def clear(self) -> None:
        """결과 초기화."""
        self.results.clear()

    def compute_gtpa_at_1(self) -> float:
        """GTPA@1 계산.

        Top-1 정확도: 예측 질환이 정답과 일치하는 비율.
        """
        if not self.results:
            return 0.0
        correct = sum(1 for r in self.results if r.correct)
        return correct / len(self.results)

    def compute_dd_metrics(self) -> tuple[float, float, float]:
        """DDR, DDP, DDF1 계산.

        DDR = |예측DD ∩ 정답DD| / |정답DD|
        DDP = |예측DD ∩ 정답DD| / |예측DD|
        DDF1 = 2 * DDR * DDP / (DDR + DDP)
        """
        if not self.results:
            return 0.0, 0.0, 0.0

        total_recall_num = 0
        total_recall_den = 0
        total_precision_num = 0
        total_precision_den = 0

        for r in self.results:
            gt_set = set(r.ground_truth_dd)
            pred_set = {cui for cui, _ in r.predicted_dd}

            intersection = len(gt_set & pred_set)

            total_recall_num += intersection
            total_recall_den += len(gt_set) if gt_set else 1
            total_precision_num += intersection
            total_precision_den += len(pred_set) if pred_set else 1

        ddr = total_recall_num / total_recall_den if total_recall_den > 0 else 0.0
        ddp = total_precision_num / total_precision_den if total_precision_den > 0 else 0.0

        if ddr + ddp > 0:
            ddf1 = 2 * ddr * ddp / (ddr + ddp)
        else:
            ddf1 = 0.0

        return ddr, ddp, ddf1

    def compute_avg_il(self) -> float:
        """평균 IL 계산."""
        if not self.results:
            return 0.0
        return sum(r.interaction_length for r in self.results) / len(self.results)

    def compute_all(self) -> BenchmarkResult:
        """모든 메트릭 계산."""
        gtpa_at_1 = self.compute_gtpa_at_1()
        ddr, ddp, ddf1 = self.compute_dd_metrics()
        avg_il = self.compute_avg_il()

        return BenchmarkResult(
            n_samples=len(self.results),
            gtpa_at_1=gtpa_at_1,
            ddr=ddr,
            ddp=ddp,
            ddf1=ddf1,
            avg_il=avg_il,
        )
