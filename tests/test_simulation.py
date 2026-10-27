"""Unit tests for the flood-risk cost-benefit simulation and threshold optimizer."""

from __future__ import annotations

import unittest
from pathlib import Path

from flood_risk.model_loader import resolve_model_path
from flood_risk.simulation import find_optimal_threshold, simulate_thresholds


class TestFloodRiskSimulation(unittest.TestCase):
    def test_simulate_thresholds_math_and_roi(self) -> None:
        y_true = [1, 1, 0, 0, 1]
        y_proba = [0.90, 0.65, 0.60, 0.20, 0.40]

        results = simulate_thresholds(
            y_true=y_true,
            y_proba=y_proba,
            cost_per_area=1.0,
            damage_per_area=5.0,
            effectiveness_rate=0.80,
            thresholds=[0.30, 0.50, 0.80],
        )

        self.assertEqual(len(results), 3)

        # At threshold 0.50: predictions are [1, 1, 1, 0, 0] -> 3 flagged, 2 true positives
        t_50 = results[1]
        self.assertEqual(t_50["threshold"], 0.50)
        self.assertEqual(t_50["areas_flagged"], 3)
        self.assertEqual(t_50["true_positives"], 2)
        self.assertAlmostEqual(t_50["precision"], 2 / 3)
        self.assertAlmostEqual(t_50["recall"], 2 / 3)
        self.assertAlmostEqual(t_50["intervention_cost"], 3.0)
        self.assertAlmostEqual(t_50["prevented_damage"], 8.0)
        self.assertAlmostEqual(t_50["net_benefit"], 5.0)
        self.assertAlmostEqual(t_50["roi"], (5.0 / 3.0) * 100.0)

        optimal = find_optimal_threshold(results)
        self.assertEqual(optimal["threshold"], 0.30)
        self.assertAlmostEqual(optimal["net_benefit"], 8.0)

    def test_zero_interventions_handles_division_by_zero(self) -> None:
        results = simulate_thresholds(
            y_true=[1, 0],
            y_proba=[0.10, 0.05],
            cost_per_area=2.0,
            damage_per_area=10.0,
            effectiveness_rate=0.70,
            thresholds=[0.95],
        )
        self.assertEqual(results[0]["areas_flagged"], 0)
        self.assertEqual(results[0]["precision"], 0.0)
        self.assertEqual(results[0]["roi"], 0.0)
        self.assertEqual(results[0]["net_benefit"], 0.0)

    def test_mismatched_lengths_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            simulate_thresholds([1, 0], [0.5], 1.0, 5.0, 0.7)

    def test_model_artifact_exists(self) -> None:
        model_path = resolve_model_path()
        self.assertIsNotNone(model_path)
        self.assertTrue(Path(model_path).exists())


if __name__ == "__main__":
    unittest.main()
