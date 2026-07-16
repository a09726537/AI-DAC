# AI-DAC — Reproducibility Package

This repository contains the reproducibility package associated with the doctoral dissertation:

**Triple-Loop Learning for Lifecycle-Aware Database Cybersecurity: A Recursive Learning Framework**  
**William Kandolo**  
University of Vienna, Doctoral Dissertation, 2026

Repository: <https://github.com/a09726537/AI-DAC>

---

## Purpose

This repository supports the independent inspection, reproduction, and audit of the controlled experimental results reported in the dissertation.

The reproducibility package includes:

- dataset-provenance manifests;
- controlled SQL training, validation, and test partitions;
- evaluation and audit scripts;
- generated experimental results;
- metric registers;
- reproducibility reports;
- integrity-verification files; and
- documentation of the controlled laboratory configuration.

The objective is to make the reported AI-DAC results transparent, traceable, repeatable, and auditable within the boundaries of the documented experimental environment.

The repository does **not** claim universal or production-scale generalization across all relational database systems, organizations, infrastructures, or threat environments.

---

## Data Provenance and Legal Scope

The current AI-DAC evaluation does **not** use confidential, proprietary, or operational security data obtained from companies, government institutions, public authorities, or other external organizations.

The experiments are based exclusively on the following categories of data.

### 1. Controlled laboratory data

SQL events and database-security telemetry generated on controlled laboratory servers specifically configured for the AI-DAC experiments.

These events were produced within an isolated and documented experimental environment and were not collected from a live company, government, customer, or University production system.

### 2. Public cybersecurity benchmark datasets

Publicly available cybersecurity datasets used for external benchmarking and comparative validation, including:

- UNSW-NB15;
- NSL-KDD;
- CSE-CIC-IDS2018;
- BoT-IoT;
- TON-IoT; and
- LogHub.

These datasets are used as supplementary benchmark evidence. They do not replace the primary controlled SQL evaluation and should not be interpreted as direct equivalents of relational database telemetry.

Users of this repository remain responsible for observing the original licences, citation requirements, and terms of use associated with each public dataset.

### 3. Controlled synthetic attack scenarios

Reproducible SQL misuse, privilege-abuse, anomalous-access, and adversarial scenarios executed within the isolated laboratory environment.

The synthetic scenarios were created for defensive cybersecurity research, evaluation, and reproducibility testing.

---

## Data Exclusions

The reproducibility package does not contain:

- confidential company data;
- internal government data;
- production database logs;
- operational customer records;
- personal customer information;
- confidential University systems data;
- third-party organizational security records; or
- data collected through privileged access to an external organization.

The University of Vienna affiliation identifies the academic context of the dissertation. It does not indicate that the University supplied the experimental data.

Accordingly, the controlled laboratory experiments do not depend on authorization letters from companies, government institutions, or public authorities.

---

## Dataset-Provenance Manifests

The `reproducibility/manifests` directory contains a dedicated provenance manifest for each controlled or public dataset used in the evaluation.

The available manifests include:

| Manifest | Dataset or source |
|---|---|
| `controlled_sql_lab_manifest.json` | Controlled SQL laboratory dataset |
| `UNSW_NB15_manifest.json` | UNSW-NB15 |
| `NSL_KDD_manifest.json` | NSL-KDD |
| `CSE_CIC_IDS2018_manifest.json` | CSE-CIC-IDS2018 |
| `BoT_IoT_manifest.json` | BoT-IoT |
| `TON_IoT_manifest.json` | TON-IoT |
| `LogHub_manifest.json` | LogHub |
| `dataset_manifest_summary.csv` | Consolidated dataset manifest summary |

These manifests support the traceability of the experimental evidence by documenting the origin, scope, classification, and intended role of each dataset.

The manifests also distinguish the primary controlled SQL dataset from the public datasets used for supplementary external validation.

---

## Thesis Context

AI-DAC stands for **Artificial Intelligence–Driven Anomaly Detection and Control**.

AI-DAC is the operational research artifact used to implement and evaluate the proposed **Lifecycle-Aware Triple-Loop Learning Framework** for adaptive relational database cybersecurity.

The architecture integrates:

- anomaly detection;
- adaptive security response;
- meta-learning and concept-drift recovery;
- SHAP-based explainability;
- retrieval-augmented generation for contextual explanations;
- governance-aware decision filtering;
- human oversight;
- audit-trail generation; and
- reproducibility controls.

The principal contribution is not any individual model or component in isolation. It is the integration of detection, adaptation, reflection, explanation, governance, and evidence preservation into a lifecycle-aware and reproducible decision-support architecture for relational database cybersecurity.

---

## Experimental Scope

The primary AI-DAC evaluation is based on a controlled SQL dataset generated within the documented laboratory environment.

The controlled dataset contains:

- **47,832 SQL events**;
- normal and anomalous database activities;
- separate training, validation, and test partitions; and
- controlled attack and misuse scenarios.

Public cybersecurity datasets are used as external benchmarks to evaluate whether selected detection components remain effective outside the primary controlled SQL dataset.

The reported findings are bounded by:

- the documented laboratory infrastructure;
- the selected datasets;
- the implemented feature-engineering procedures;
- the configured model parameters;
- the selected decision thresholds;
- the defined attack scenarios; and
- the evaluation protocols included in the reproducibility package.

---

## Final Reproducibility Audit

The final reproducibility audit covered **45 unique reported metrics**:

- **39 metrics** reproduced with an exact `OK` status;
- **6 metrics** reproduced with a `Rounded OK` status; and
- **0 referenced output files** were missing.

An `OK` result indicates that the reproduced value matched the registered value exactly.

A `Rounded OK` result indicates that the reproduced value matched the reported value after applying the documented numerical-rounding convention.

---

## Main Reproduced Claims

| Reproduced claim | Value |
|---|---:|
| Controlled SQL dataset size | 47,832 events |
| Controlled SQL test-partition size | 7,174 events |
| Controlled SQL test-set accuracy | 0.98 |
| Precision | 0.95 |
| Recall | 0.95 |
| F1-score | 0.95 |
| ROC-AUC | 0.97 |
| Response-risk reduction | 27.8% |
| Governance audit-trail completeness | 98.3% |
| SHAP–RAG overall usefulness score | 4.21 |
| Drift-recovery reduction | 83% |
| Full AI-DAC F1-score | 0.950 |
| No-lifecycle ablation F1-score | 0.921 |
| Transformer baseline F1-score | 0.900 |
| UNSW-NB15 F1-score / ROC-AUC | 0.941 / 0.965 |
| NSL-KDD F1-score / ROC-AUC | 0.957 / 0.973 |

The complete metric definitions, source files, rounding conventions, and reproduction statuses are provided in the final audit report and machine-readable audit summaries.

---

## Repository Files

| File or directory | Purpose |
|---|---|
| `aidac_reproducibility_package.tar.gz` | Complete compressed reproducibility archive |
| `aidac_reproducibility_package.sha256` | SHA-256 checksum used to verify archive integrity |
| `final_reproducibility_audit_report.txt` | Human-readable final audit report |
| `final_reproducibility_audit_summary.csv` | Metric-level audit results in CSV format |
| `final_reproducibility_audit_summary.json` | Machine-readable audit results in JSON format |
| `reproducibility/manifests/` | Dataset-provenance manifests and consolidated summary |
| `CITATION.cff` | Citation metadata for the reproducibility package |
| `README.md` | Repository scope, documentation, and usage instructions |

---

## Verify Package Integrity

After cloning or downloading the repository, verify the integrity of the reproducibility archive by running:

```bash
sha256sum -c aidac_reproducibility_package.sha256
