# Hybrid Sample Generation
![Python](https://img.shields.io/badge/Python-14354C?style=flat&logo=python&logoColor=green) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Tests](https://github.com/Pfleiderer-Adrian/HybridSampleGeneration/actions/workflows/tests.yml/badge.svg)](https://github.com/Pfleiderer-Adrian/HybridSampleGeneration/actions/workflows/tests.yml) [![DOI:AMLDS63918.2025.11159383](http://img.shields.io/badge/DOI-AMLDS63918.2025.11159383-B31B1B.svg)](https://doi.org/10.1109/AMLDS63918.2025.11159383)

This project extracts real anomalies from labelled 2D images or 3D volumes,
trains a generative model, creates multiple synthetic variants and places them
into original control samples. It is based on the IEEE paper
[AMLDS63918.2025.11159383](https://doi.org/10.1109/AMLDS63918.2025.11159383).

![High-level overview of hybrid sample generation](assets/high_level.png)

## Start here

- [Install the project](getting-started/installation.md) and [run a study](getting-started/quick-start.md).
- Learn about [input data](guides/input-data.md), [pipeline stages](guides/pipeline.md), and [resuming a study](guides/continuing-studies.md).
- Browse the [configuration reference](configuration/index.md) or use the search box to find a parameter.
- Read about [study storage](concepts/study-storage.md), [evaluation](guides/evaluation.md), and [visualization](guides/visualization.md).
- See [citation and license](about.md).
