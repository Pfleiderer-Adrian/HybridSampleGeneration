# Installation

The pipeline uses PyTorch, Optuna, NumPy, SciPy, pandas, scikit-image and
Matplotlib. Additional model and file-format dependencies are listed in
`requirements.txt`.

Install PyTorch in the variant appropriate for the local CPU/CUDA environment,
then install the project in editable mode:

```bash
python -m pip install -e .
```

Exact PyTorch and CUDA versions depend on the target system. A GPU is useful for
training but the orchestration and repository layers do not require one.
