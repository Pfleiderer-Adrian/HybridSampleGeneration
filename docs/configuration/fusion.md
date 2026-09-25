# Fusion

`config.fusion.backend` holds the selected backend name. Call
`config.fusion.set_backend(name)` to select a backend and reset its parameters
to that backend's defaults. Available backends are
[`classical`](reference/fusion-classical.md) for alpha blending and
[`poisson`](reference/fusion-poisson.md) for gradient-domain blending.

```python
config.fusion.set_backend("classical")
config.fusion.parameters.max_alpha = 0.9
config.validate()
```

```python
config.fusion.set_backend("poisson")
config.fusion.parameters.guidance_mode = "mixed"
config.validate()
```

The backend reference page lists every available parameter.
