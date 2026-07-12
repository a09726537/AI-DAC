# Dataset Availability

This repository does not redistribute large third-party benchmark datasets.

## Included directly

The controlled SQL laboratory dataset partitions are included because they are part of the reproducible AI-DAC evaluation artifact:

- `data/processed/controlled_sql_train.csv`
- `data/processed/controlled_sql_validation.csv`
- `data/processed/controlled_sql_test.csv`

## Documented through manifests

Large public benchmark datasets are documented through manifests in:

- `reproducibility/manifests/`

These manifests record dataset names, file availability, file sizes, hashes where applicable, and evaluation role.

## Public benchmark datasets

The following datasets were used for benchmark comparability or robustness evidence:

- UNSW-NB15
- NSL-KDD
- CSE-CIC-IDS2018
- TON_IoT
- BoT-IoT
- LogHub, provenance only / inactive in final benchmark table

These datasets are not redistributed in this repository because of size, licensing, and repository-health constraints. Users should obtain them from their official public sources and place them in the expected local external dataset directory before reproducing benchmark-related checks.



## Interpretation boundary



The controlled SQL laboratory dataset provides the main RDBMS-specific evidence. Public benchmark datasets support comparability and robustness checks, but they are not treated as substitutes for relational database audit logs.

