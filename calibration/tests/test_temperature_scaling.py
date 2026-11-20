"""Numerical regression checks for the calibration protocol."""

import unittest

import numpy as np

from calibration.temperature_scaling import (
    _nll_and_derivative,
    evaluate_logits,
    fit_temperature,
    probabilities,
)


class TemperatureScalingTests(unittest.TestCase):
    def test_softmax_matches_known_probabilities(self):
        expected = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        np.testing.assert_allclose(probabilities(np.log(expected)), expected, rtol=1e-14)

    def test_binary_and_multiclass_brier_use_class_sum(self):
        binary = evaluate_logits(np.log([[0.8, 0.2]]), [0])
        self.assertAlmostEqual(binary["brier"], 0.08)
        multiclass = evaluate_logits(np.log([[0.7, 0.2, 0.1]]), [1])
        self.assertAlmostEqual(multiclass["brier"], 1.14)
        self.assertAlmostEqual(multiclass["nll"], -np.log(0.2))

    def test_padding_has_zero_probability_and_no_metric_effect(self):
        logits = np.array([[2.0, 1.0], [0.2, 1.4]])
        padded = np.pad(logits, ((0, 0), (0, 2)), constant_values=-np.inf)
        np.testing.assert_array_equal(probabilities(padded)[:, 2:], 0.0)
        left = evaluate_logits(logits, [0, 0])
        right = evaluate_logits(padded, [0, 0])
        self.assertEqual(left, right)
        self.assertAlmostEqual(
            fit_temperature(logits, [0, 0])["temperature"],
            fit_temperature(padded, [0, 0])["temperature"],
        )

    def test_ece_right_closed_boundary_and_confidence_one(self):
        uniform = evaluate_logits(np.zeros((1, 3)), [0], n_bins=15)
        self.assertEqual(uniform["bins"][4]["count"], 1)
        self.assertAlmostEqual(uniform["ece"], 2.0 / 3.0)
        self.assertIsNone(uniform["bins"][0]["accuracy"])
        certain = evaluate_logits([[0.0, -1000.0]], [1], n_bins=15)
        self.assertEqual(certain["bins"][-1]["count"], 1)
        self.assertEqual(certain["mean_confidence"], 1.0)
        self.assertEqual(certain["ece"], 1.0)
        self.assertEqual(certain["nll"], 1000.0)

    def test_temperature_fit_matches_analytic_binary_optimum(self):
        # All examples predict class zero with logit margin 4. Three of four
        # predictions are correct, so the NLL optimum has p(class zero)=0.75.
        logits = np.tile([4.0, 0.0], (4, 1))
        labels = np.array([0, 0, 0, 1])
        fitted = fit_temperature(logits, labels)
        expected_temperature = 4.0 / np.log(3.0)
        self.assertAlmostEqual(fitted["temperature"], expected_temperature, places=8)
        self.assertLess(fitted["nll_after"], fitted["nll_before"])
        self.assertTrue(fitted["converged"])
        self.assertIsNone(fitted["boundary"])
        self.assertGreater(fitted["temperature"], 0.0)
        np.testing.assert_array_equal(
            probabilities(logits, fitted["temperature"]).argmax(axis=1), logits.argmax(axis=1)
        )

    def test_convex_derivative_matches_finite_difference(self):
        logits = np.array([[2.0, 0.0, -np.inf], [1.0, 0.5, 2.0], [-1.0, 0.0, 0.5]])
        labels = np.array([0, 1, 2])
        beta, step = 0.73, 1e-5
        _, derivative = _nll_and_derivative(logits, labels, beta)
        higher, _ = _nll_and_derivative(logits, labels, beta + step)
        lower, _ = _nll_and_derivative(logits, labels, beta - step)
        self.assertAlmostEqual(derivative, (higher - lower) / (2 * step), places=9)

    def test_logits_can_be_shifted_per_example(self):
        logits = np.array([[4.0, 1.0, 0.0], [0.5, 1.5, 0.0], [2.0, 0.0, 1.0]])
        shifted = logits + np.array([[10000.0], [-10000.0], [123.0]])
        np.testing.assert_allclose(probabilities(logits), probabilities(shifted), atol=1e-14)
        fitted = fit_temperature(logits, [0, 0, 2])
        shifted_fit = fit_temperature(shifted, [0, 0, 2])
        self.assertAlmostEqual(fitted["temperature"], shifted_fit["temperature"], places=10)

    def test_flat_logits_and_boundary_solutions(self):
        flat = fit_temperature(np.zeros((3, 2)), [0, 1, 0])
        self.assertEqual(flat["temperature"], 1.0)
        self.assertEqual(flat["nll_before"], flat["nll_after"])
        sharpened = fit_temperature([[2.0, 0.0], [0.0, 2.0]], [0, 1])
        self.assertEqual(sharpened["boundary"], "upper")
        self.assertAlmostEqual(sharpened["temperature"], 1e-4)
        flattened = fit_temperature([[2.0, 0.0], [2.0, 0.0]], [1, 1])
        self.assertEqual(flattened["boundary"], "lower")
        self.assertAlmostEqual(flattened["temperature"], 1e4)

    def test_invalid_inputs_are_rejected(self):
        invalid_logits = [[], [[1.0]], [[np.nan, 0.0]], [[np.inf, 0.0]], [[0.0, -np.inf]]]
        for logits in invalid_logits:
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                probabilities(logits)
        for labels in ([2], [-1], [0.5], [0, 1]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate_logits([[0.0, 1.0]], labels)
        with self.assertRaises(ValueError):
            evaluate_logits([[0.0, 1.0, -np.inf]], [2])
        for temperature in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                probabilities([[0.0, 1.0]], temperature)
        with self.assertRaises(ValueError):
            evaluate_logits([[0.0, 1.0]], [0], n_bins=0)


if __name__ == "__main__":
    unittest.main()
