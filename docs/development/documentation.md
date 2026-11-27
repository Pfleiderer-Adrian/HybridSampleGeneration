# Documentation

The documentation is a [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
site built from the Markdown files in `docs/`. Navigation, theme and Markdown
extensions are configured in `mkdocs.yml`.

## Preview locally

```bash
python -m pip install "mkdocs-material>=9.7,<10"
mkdocs serve
```

`mkdocs serve` rebuilds the site on every change and serves it at
`http://127.0.0.1:8000`. Before opening a pull request, run the same build as
the CI:

```bash
mkdocs build --strict
```

Strict mode fails on broken internal links and on pages that are missing from
the navigation. Add every new page to `nav` in `mkdocs.yml`.

## Configuration reference

The pages in `docs/configuration/reference/` are generated; do not edit them by
hand. Parameter names, types and defaults come from the configuration classes,
the explanations from `DESCRIPTIONS` in `scripts/generate_config_reference.py`.
After changing a parameter or an explanation, regenerate and verify the pages:

```bash
python scripts/generate_config_reference.py
python scripts/generate_config_reference.py --check
```

The check reports fields without an explanation and outdated reference pages.
The test workflow runs it on every pull request.

## Writing conventions

- Describe the current behavior. Changes between versions belong in the
  release notes, not in the guides.
- Use the terms of the [glossary](../concepts/glossary.md).
- Check code examples against the code; the
  [quick start](../getting-started/quick-start.md) script is meant to run
  unchanged.
- Guides explain behavior and link to the reference for complete parameter
  lists instead of repeating them.

## Publishing

The `Documentation` workflow builds the site for every pull request that
touches `docs/`, `mkdocs.yml` or `README.md`, and checks the links in the
Markdown files. On `main` it also deploys the site to GitHub Pages.
