from hybrid_sample_generator.datasets.record_datasets import (
    HybridSampleDataset,
    OriginalSampleDataset,
    RealAnomalyDataset,
    SyntheticAnomalyDataset,
)
from hybrid_sample_generator.datasets.study_datasets import StudyDatasets
from tests.datasets.support import DatasetStudyTestCase


class StudyDatasetsTests(DatasetStudyTestCase):
    def test_factories_create_fresh_typed_dataset_views(self):
        datasets = StudyDatasets(self.repository, self.store)

        first_originals = datasets.original_samples(return_artifacts=("record",))
        second_originals = datasets.original_samples(return_artifacts=("record",))

        self.assertIsInstance(first_originals, OriginalSampleDataset)
        self.assertIsNot(first_originals, second_originals)
        self.assertIsInstance(datasets.real_anomalies(), RealAnomalyDataset)
        self.assertIsInstance(
            datasets.synthetic_anomalies(), SyntheticAnomalyDataset
        )
        self.assertIsInstance(datasets.hybrid_samples(), HybridSampleDataset)


if __name__ == "__main__":
    import unittest

    unittest.main()
