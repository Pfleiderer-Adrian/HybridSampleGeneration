# Experimental prototypes

This directory contains unsupported research prototypes that are intentionally
excluded from the stable `hybrid_sample_generator` package API and registries.

The code may be incomplete, incompatible with some environments, or changed
without notice. Do not rely on its configuration or checkpoint formats.

## Latent Diffusion LoRA

Location: `generation/latent_diffusion_lora/`

Status:

- prototype for 2D generation;
- end-to-end training and generation are not yet verified;
- checkpoint compatibility and hardware requirements are not guaranteed;
- requires the optional dependencies in `experiments/requirements.txt`.

## Learned residual-alpha fusion

Location: `fusion/learned_residual_alpha/`

Status:

- prototype for learned 2D and 3D fusion;
- isolated helper, checkpoint, training, and fusion tests are retained;
- training quality and production stability are not yet validated;
- not available through the stable fusion registry.

## Installation and tests

Install the package with the experimental dependency group:

```bash
python -m pip install -e ".[experiments]"
```

Run the isolated prototype tests explicitly:

```bash
python -m unittest discover -s experiments/tests -v
```

A prototype should return to the stable package only after reproducible
end-to-end training and inference, checkpoint reload tests, documented hardware
requirements, and a supported example are available.
