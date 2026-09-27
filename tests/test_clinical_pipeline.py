"""Synthetic checks for the shared two-mask measurement pipeline."""

import unittest

import numpy as np

from clinical.aggregation import aggregate_video_ratio
from clinical.grading import ratio_to_grade
from clinical.measurements import compute_ratio, measure_frame
from evaluation.ratio import evaluate_ratios
from evaluation.segmentation import evaluate_regions


class ClinicalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.adenoid = np.zeros((10, 100), dtype=np.uint8)
        self.airway = np.zeros((10, 100), dtype=np.uint8)
        self.adenoid.flat[:300] = 1
        self.airway.flat[300:] = 1

    def test_fraction_of_total_ratio(self):
        self.assertAlmostEqual(compute_ratio(self.adenoid, self.airway), 0.30)

    def test_common_prediction_and_aggregation(self):
        prediction = measure_frame(42, self.adenoid, self.airway)
        self.assertEqual(prediction["adenoid_area"], 300)
        self.assertEqual(prediction["airway_area"], 700)
        self.assertTrue(prediction["valid_frame"])
        self.assertAlmostEqual(prediction["ratio"], 0.30)
        self.assertAlmostEqual(aggregate_video_ratio([prediction]), 0.30)

    def test_ratio_evaluation_and_grading(self):
        metrics = evaluate_ratios([0.2, 0.4], [0.3, 0.5])
        self.assertEqual(metrics["count"], 2)
        self.assertAlmostEqual(metrics["mae"], 0.1)
        self.assertAlmostEqual(metrics["rmse"], 0.1)
        self.assertEqual(ratio_to_grade(0.30, [0.25, 0.5]), 1)

    def test_named_region_segmentation_metrics(self):
        metrics = evaluate_regions(
            {"optic_cup": self.adenoid, "optic_disc": self.airway},
            {"optic_cup": self.adenoid, "optic_disc": self.airway},
        )
        self.assertEqual(metrics["optic_cup_dice"], 1.0)
        self.assertEqual(metrics["optic_disc_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
