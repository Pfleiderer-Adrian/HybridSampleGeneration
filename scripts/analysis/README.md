# Research analysis scripts

The scripts previously mixed into `notes/` and `reports/` are maintained here:

| Source | Location | Default data/output location |
|---|---|---|
| 36 audit scripts | `audits/<audit-name>/` | `/mnt/results/mvtec2/audits/<audit-name>/` |
| 5 report generators | `reports/<report-name>/` | `/mnt/results/mvtec2/reports/<report-name>/` |
| Manuscript and bibliography sources | `../../docs/paper/` | Generated assets remain in external reports |

`paths.py` defines the output locations. Set `HSG_RESULTS_ROOT` to use another
base for audit/report outputs. Original experiment datasets, checkpoints and
environment paths are separate and retain their recorded locations.

Run scripts from the project root, for example:

```bash
.venv/bin/python scripts/analysis/reports/paper_hybrid_generator_2026-11-05/build_assets.py
.venv/bin/python scripts/analysis/audits/ad_visibility_audit_2026-11-02/controlled_can_test.py --help
```

Some scripts perform GPU generation or update diagnostic experiment data; they
retain their original behavior and dependencies. Moving a script does not make
it a read-only operation. The inventory generator still requires its original
`/tmp/study_*` input lists. Use the existing AD environment for AD-dependent
scripts and the existing project environment for ordinary analyses.

Direct script execution appends the checkout to `sys.path`. It deliberately
keeps explicitly supplied frozen `PYTHONPATH` entries first, which is required by
the training-replay script's runtime provenance checks.

Historical local aliases under `notes/` and `reports/` point to the new locations.
These aliases are ignored by Git and serve older links and archived commands.
New code should reference this source tree and the external result directories.
The originals of relocated scripts/text and the verified transfer manifest are
stored under `/mnt/results/mvtec2/cleanup/`; no frozen training snapshot or
checkpoint has been edited.
