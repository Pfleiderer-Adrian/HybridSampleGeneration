# Experimental prototypes

Unsupported diffusion and learned residual-alpha fusion prototypes are isolated
under `experiments/`. They are excluded from the stable package API and
registries and require the optional dependencies in
`experiments/requirements.txt`. Their APIs, configuration formats and
checkpoints may change without notice; see [`experiments/README.md`](https://github.com/Pfleiderer-Adrian/HybridSampleGeneration/blob/main/experiments/README.md) for their
current status.

Install and test them separately only when working on the prototypes:

```bash
python -m pip install -e ".[experiments]"
python -m unittest discover -s experiments/tests -v
```
