# Contributing

Bug reports and suggestions are welcome as
[GitHub issues](https://github.com/Pfleiderer-Adrian/HybridSampleGeneration/issues).
For a bug, include the configuration (`configuration.json` of the study), the
failing pipeline stage and the complete error message.

## Development setup

```bash
git clone https://github.com/Pfleiderer-Adrian/HybridSampleGeneration.git
cd HybridSampleGeneration
python -m pip install -e ".[examples]"
```

## Before opening a pull request

1. Run the [tests](testing.md).
2. If configuration parameters changed, regenerate the
   [configuration reference](documentation.md#configuration-reference).
3. If behavior changed, update the affected guide and build the
   [documentation](documentation.md) in strict mode.

Pull requests are checked by the `Tests` and `Documentation` workflows.

## Where things belong

The [project structure](project-structure.md) lists the packages. New
generator models are registered in `hybrid_sample_generator/generation/registry.py`,
new fusion backends in `hybrid_sample_generator/fusion/registry.py`; both
appear in the configuration reference once the generator script knows their
parameter explanations.

Contributions are licensed under the project's
[GPL-3.0 license](../about.md#license).
