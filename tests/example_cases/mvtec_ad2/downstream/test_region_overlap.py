"""Partial PRO AUC checks for region weighting, ties and streaming evaluation."""

import unittest
import numpy as np

from examples.mvtec_ad2.downstream.evaluation import RegionOverlapMetric


class RegionOverlapTests(unittest.TestCase):
    def metric(self, bucket, truth, bins=10):
        metric = RegionOverlapMetric(bins)
        metric.update(np.asarray(bucket), np.asarray(truth, dtype=bool))
        negative = np.bincount(np.asarray(bucket)[~np.asarray(truth, dtype=bool)], minlength=bins)
        return metric, negative

    def test_perfect_reversed_and_tied_predictions(self):
        truth = [[[1, 0], [0, 0]]]
        for bucket, expected in (([[[9, 0], [0, 0]]], 1.),
                                 ([[[0, 9], [9, 9]]], 0.),
                                 ([[[5, 5], [5, 5]]], .025)):
            metric, negative = self.metric(bucket, truth)
            self.assertAlmostEqual(metric.compute(negative), expected)

    def test_regions_have_equal_weight_and_healthy_images_count(self):
        truth = np.zeros((2, 4, 6), dtype=bool)
        truth[0, 0, 0] = True
        truth[0, 2:4, 4:6] = True
        bucket = np.full(truth.shape, 5)
        bucket[truth] = 0
        bucket[0, 0, 0] = 9
        metric, negative = self.metric(bucket, truth)
        self.assertEqual(metric.regions, 2)
        self.assertAlmostEqual(metric.compute(negative), .5)
        streamed = RegionOverlapMetric(10)
        for i in range(2):
            streamed.update(bucket[i:i+1], truth[i:i+1])
        self.assertEqual(metric.compute(negative), streamed.compute(negative))

    def test_endpoint_interpolation_and_diagonal_connectivity(self):
        metric, negative = self.metric([[[9, 9, 0]]], [[[1, 0, 0]]])
        self.assertAlmostEqual(metric.compute(negative), .05)
        metric, negative = self.metric([[[9, 0], [0, 9]]], [[[1, 0], [0, 1]]])
        self.assertEqual(metric.regions, 1)
        self.assertAlmostEqual(metric.compute(negative), 1.)

    def test_undefined_without_regions_or_background(self):
        for truth in (np.zeros((1, 2, 2)), np.ones((1, 2, 2))):
            metric, negative = self.metric(np.zeros((1, 2, 2), dtype=int), truth)
            self.assertIsNone(metric.compute(negative))
