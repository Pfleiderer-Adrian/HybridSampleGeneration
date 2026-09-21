# Repository-backed datasets

`StudyDatasets` creates short-lived `OriginalSampleDataset`,
`RealAnomalyDataset`, `SyntheticAnomalyDataset` and `HybridSampleDataset` views
over repository records. Original views can filter `has_anomaly` and
`is_annotated`. They do not scan folders or align files by basename. Dataset
objects are not persistent state of `HybridDataGenerator`; callers choose
explicitly whether a view should load arrays into RAM.
