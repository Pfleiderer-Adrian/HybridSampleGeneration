# Package versions and releases

Package versions are derived from Git tags by `setuptools-scm`; there is no
version field to update in `pyproject.toml`. Use PEP 440-compatible release tags
such as `v1.0.1`. A build from that exact tag has version `1.0.1`; commits after
the tag receive a development version and must not be uploaded as that release.

Before creating a release, commit all changes, run the tests and create the next
tag through a GitHub Release. PyPI versions are immutable, so every release needs
a new tag/version. The core installation contains only the library runtime
dependencies. Dependencies used by the repository examples can be installed with:

```bash
python -m pip install -e ".[examples]"
```
