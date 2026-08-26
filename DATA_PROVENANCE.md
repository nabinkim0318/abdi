# Data provenance

The public portfolio repository is **synthetic-only by default**.

The committed demo dataset is fully synthetic and generated locally with a
fixed random seed. No row represents a real person, patient, or clinical
record, and no patient-level records from TCGA, GDC, CPTAC, CMI, or other
external clinical datasets are redistributed in the current repository tree.

Earlier development versions used external clinical datasets locally; the
public portfolio demo is now fully synthetic. That historical note is not a
license determination about those external sources.

## Authoritative files

| File | Role |
| --- | --- |
| `bias_audit_tool/sample_data/demo.csv` | Authoritative synthetic demo dataset |
| `bias_audit_tool/sample_data/demo_benchmark.json` | Synthetic demonstration benchmark (proportions) |
| `scripts/generate_demo_data.py` | Deterministic generator (`SEED = 42`, `N_ROWS = 400`) |

Regenerate both artifacts with:

```bash
python scripts/generate_demo_data.py
```

The generator makes no network calls and does not read external datasets.
Re-running it with the committed script must reproduce the committed CSV
and JSON byte-for-byte.

## What the demo is for

The dataset exists solely to demonstrate application behavior:
preprocessing recommendations, binary classification, held-out evaluation
plots, and group-wise fairness diagnostics.

Synthetic group distributions and outcome relationships are intentionally
constructed to exercise those code paths. They are software demonstration
fixtures, **not** estimates of real-world demographic or clinical
relationships, population prevalence, epidemiology, or regulatory
thresholds.

User-uploaded datasets are supplied by the user and may contain sensitive
information. The bundled repository demo itself is synthetic. This document
does not claim that the application is privacy-safe, HIPAA compliant, or
unable to expose sensitive data in user uploads.

## Do not commit raw or external data

Do not add real patient-level records, downloaded clinical extracts, or
local working copies under:

- `data/raw/`
- `sample_data/raw/`
- `bias_audit_tool/sample_data/raw/`

`.gitignore` ignores those directories and ignores CSV files except the
synthetic demo path above.

## Data dictionary (`demo.csv`)

| Column | Type | Meaning | Generation | Role | Missingness |
| --- | --- | --- | --- | --- | --- |
| `feature_a` | numeric (float) | Unnamed synthetic predictor | `Normal(0, 1)` from `numpy.random.default_rng(42)` | Ordinary feature | None |
| `feature_b` | numeric (float) | Unnamed synthetic predictor | `Normal(0.2, 1.2)`, then ~8% of values set to missing | Ordinary feature | Intentional ~8% missing to exercise imputation |
| `feature_c` | categorical | Unnamed synthetic predictor (`low` / `medium` / `high`) | Independent draws with fixed probabilities | Ordinary feature | None |
| `age_band` | categorical | Synthetic banding label (`band_1` / `band_2` / `band_3`), not chronological age | Independent draws with fixed probabilities | Ordinary feature (name heuristic may also list it as a candidate) | None |
| `demo_group` | categorical | Synthetic grouping label (`Group A` / `Group B`) | Bernoulli draw with P(Group A) = 0.42 | Candidate sensitive attribute. After the app's exploratory one-hot step, the live selector shows the merged column `demo_group_mapped`. | None |
| `outcome` | binary integer (`0` / `1`) | Synthetic classification target | Bernoulli draw from a logistic of `feature_a`, `feature_b`, `feature_c`, plus a **demo-only** intercept shift by `demo_group` | Target | None |

There are no unique personal identifiers, names, dates of birth, addresses,
case IDs, or medical-record IDs.

## Synthetic demonstration benchmark

`demo_benchmark.json` is:

```json
{
  "Group A": 0.5,
  "Group B": 0.5
}
```

Values are **proportions** (they sum to 1.0), matching the representation-
analysis UI. This is a synthetic demonstration benchmark, not a claim about
population prevalence.

## Notebooks

This repository currently has **no committed notebooks**. Notebook output
stripping (for example `nbstripout`) is not configured because there are
no `.ipynb` files to guard.

## Git history

The current repository tree uses only the bundled synthetic demo.

The canonical repository branch history was rewritten to remove previously
committed clinical sample files and exploratory notebooks. Those historical
artifacts are no longer reachable from branches or tags maintained in the
canonical repository.

Git hosting providers, forks, clones, caches, or pull-request refs may
retain unreachable historical objects outside the branch/tag history
controlled by this repository. This document does not claim universal
deletion, third-party clone erasure, guaranteed backend garbage
collection, HIPAA compliance, or privacy certification.

A private local recovery ref (`refs/backup/pre-history-sanitization`)
intentionally retains the unsanitized history for emergency recovery and
must never be pushed.
