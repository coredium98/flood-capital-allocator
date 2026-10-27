"""Pure cost-benefit simulation and threshold optimization logic."""

from __future__ import annotations

from typing import Sequence, TypedDict


class ThresholdPoint(TypedDict):
    threshold: float
    precision: float
    recall: float
    areas_flagged: int
    true_positives: int
    intervention_cost: float
    prevented_damage: float
    net_benefit: float
    roi: float


DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))


def simulate_thresholds(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    cost_per_area: float,
    damage_per_area: float,
    effectiveness_rate: float,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[ThresholdPoint]:
    """Evaluate classification and financial metrics across probability cutoffs."""
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have identical lengths.")

    total_actual_floods = sum(1 for y in y_true if int(y) == 1)
    results: list[ThresholdPoint] = []

    for raw_t in thresholds:
        threshold = round(float(raw_t), 2)
        preds = [1 if float(p) >= threshold else 0 for p in y_proba]

        areas_flagged = sum(preds)
        true_positives = sum(1 for yp, yt in zip(preds, y_true) if yp == 1 and int(yt) == 1)

        precision = (true_positives / areas_flagged) if areas_flagged > 0 else 0.0
        recall = (true_positives / total_actual_floods) if total_actual_floods > 0 else 0.0

        intervention_cost = areas_flagged * float(cost_per_area)
        prevented_damage = true_positives * float(damage_per_area) * float(effectiveness_rate)
        net_benefit = prevented_damage - intervention_cost
        roi = (net_benefit / intervention_cost * 100.0) if intervention_cost > 0 else 0.0

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "areas_flagged": int(areas_flagged),
                "true_positives": int(true_positives),
                "intervention_cost": float(intervention_cost),
                "prevented_damage": float(prevented_damage),
                "net_benefit": float(net_benefit),
                "roi": float(roi),
            }
        )

    return results


def find_optimal_threshold(results: Sequence[ThresholdPoint]) -> ThresholdPoint:
    """Return the threshold scenario that maximizes expected net economic benefit."""
    if not results:
        raise ValueError("results cannot be empty.")
    return max(results, key=lambda item: item["net_benefit"])
