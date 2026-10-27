# Analysis and figure code map

The reusable acquisition and analysis package is in `src/chronic_pixel_pipeline/`. The catalog in this directory groups every existing root Python workflow by purpose and records the script entry point.

Run `python -m workflows.run --list` to browse or `python -m workflows.run NAME --help` for an individual workflow. Arguments after NAME are passed through unchanged. Working directory and DATA_ROOT behavior are inherited from the original script.

Existing root scripts remain available for compatibility with running jobs, imports and the four locally modified scripts that predate this organization. They have not been moved or overwritten. The catalog provides a stable, grouped interface without duplicating their implementations. Future refactoring can move reusable logic into the package with dedicated regression validation.

| Group | Responsibility |
|---|---|
| acquisition | Survey, wireless and combined sorting |
| quality | Feature precomputation, labeling, classifier training and application |
| tracking | UnitMatch tracking and compatibility patch |
| analysis | Neural/IMU analysis and decoding |
| figures | Drift, unit, pair and tracking diagnostics |
| publishing | General analysis-figure mirroring |

Generated outputs stay outside Git.
