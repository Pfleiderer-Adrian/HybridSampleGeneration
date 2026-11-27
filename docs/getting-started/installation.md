# Installation

## Requirements

- Python 3.10, 3.11 or 3.12
- PyTorch 2.2 or newer. A CUDA GPU is recommended for training the generator;
  all other stages run on the CPU.
- Tk for the [study browser](../guides/visualization.md). It is part of most
  Python installations; on Debian or Ubuntu install `python3-tk` if
  `import tkinter` fails.

## Install the package

Install PyTorch first if you need a specific CUDA build; follow the
[PyTorch instructions](https://pytorch.org/get-started/locally/). Otherwise
pip selects a default build as a dependency.

=== "From PyPI"

    ```bash
    python -m pip install hybrid-sample-generator
    ```

=== "From a repository checkout"

    ```bash
    git clone https://github.com/Pfleiderer-Adrian/HybridSampleGeneration.git
    cd HybridSampleGeneration
    python -m pip install -e .
    ```

The package installs its runtime dependencies: PyTorch, Optuna, NumPy, SciPy,
pandas, scikit-image, Matplotlib and tqdm. Check the installation with:

```bash
python -c "import hybrid_sample_generator; print(hybrid_sample_generator.__version__)"
```

Then continue with the [quick start](quick-start.md).

## Optional dependencies

| Extra | Needed for | Installs |
| --- | --- | --- |
| `examples` | the image and NIfTI example loaders in the repository | nibabel, Pillow |
| `dinov3` | the DINOv3 downstream models of the [MVTec AD 2 benchmark](../mvtec-ad2/index.md) | transformers, Pillow |

```bash
python -m pip install -e ".[examples,dinov3]"
```

The examples themselves are part of the repository, not of the PyPI package,
so they need a repository checkout.

## Image harmonization

The optional PCTNet and LBM harmonizers of the
[classical fusion backend](../guides/fusion.md#classical-fusion) use `libcom`
and require a CUDA GPU. Install the pinned dependency only when image
harmonization is needed:

=== "From a repository checkout"

    ```bash
    python -m pip install -r requirements-harmonization.txt
    ```

=== "From PyPI"

    ```bash
    python -m pip install "libcom @ git+https://github.com/bcmi/libcom.git@7a9a73ab7be2ebe4fdd92718aa778cba7bfcd92c"
    ```

`libcom` downloads the selected pretrained model files on first use. The
revision is pinned because the current PyPI release of `libcom` does not
contain the LBM backend. It is installed separately because PyPI rejects
direct URL dependencies in package metadata, including optional extras.
