# Fusion

`config.fusion.backend` holds the selected backend name. Call `config.fusion.set_backend(name)` to select a backend and reset its parameters to that backend's defaults. Currently, [`classical`](reference/fusion-classical.md) is registered.

```python
config.fusion.set_backend("classical")
config.fusion.parameters.max_alpha = 0.9
config.validate()
```

The backend reference page lists every available parameter.
