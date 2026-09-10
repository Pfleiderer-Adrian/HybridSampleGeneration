# Package structure

The production implementation lives below `hybrid_sample_generator/`. Its
top-level packages follow feature and dependency boundaries:

```text
configuration  domain  persistence  imaging
pipeline       extraction  matching
generation     fusion      evaluation
datasets       visualization
```

`pipeline` orchestrates work. It may call feature packages, but reusable image
operations do not depend on the pipeline. `examples/` may import the production
package; production code must not import examples.

The documented public API is available directly from the package:

```python
from hybrid_sample_generator import (
    Configuration,
    HybridDataGenerator,
    InputSample,
    evaluate_study,
    load_config_file,
)
```

Import production APIs exclusively from `hybrid_sample_generator`; no legacy
package roots are retained.

Tests mirror feature boundaries below `tests/`. MVTec AD 2 is an example
integration below `examples/mvtec_ad2/`, not a dependency of the core package.
