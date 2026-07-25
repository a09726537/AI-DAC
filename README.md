# AI-DAC — Preparatory Reproducibility and Feasibility Package

This repository contains the preparatory reproducibility package associated with the doctoral research project:

**Triple-Loop Learning for Lifecycle-Aware Database Cybersecurity: A Recursive Learning Framework**  
**William Kandolo**  
University of Vienna, Doctoral Research Project, 2026

Repository: <https://github.com/a09726537/AI-DAC>

---

## Repository Status

**PREPARATORY / PRE-CONFIRMATORY**

The material currently available in this repository documents controlled laboratory development, technical feasibility testing, pipeline verification, preliminary benchmarking, and protocol-readiness activities.

The numerical values, audit records, comparison tables, and generated outputs currently included in the repository are preparatory results. They must not be interpreted as final confirmatory dissertation evidence and must not be used to accept or reject the dissertation hypotheses.

The confirmatory evaluation will be conducted only after the formal protocol freeze. A separate immutable thesis-evaluation release will then identify the exact software commit, dataset manifests, preprocessing rules, parameters, decision thresholds, execution commands, raw outputs, statistical procedures, and audit records used for the confirmatory evaluation.

The active `main` branch remains a development and preparatory research branch.

---

## Purpose

This repository supports the independent inspection, reconstruction, and audit of the preparatory AI-DAC experimental pipeline.

The current package includes:

- dataset-provenance manifests;
- controlled SQL training, validation, and test partitions;
- evaluation and audit scripts;
- preparatory experimental outputs;
- preliminary metric registers;
- reproducibility reports;
- integrity-verification files; and
- documentation of the controlled laboratory configuration.

Its purpose is to make the preparatory AI-DAC implementation and evaluation process transparent, traceable, repeatable, and auditable within the boundaries of the documented experimental environment.

The repository currently supports:

- verification of the controlled laboratory pipeline;
- reconstruction of preparatory outputs;
- testing of preprocessing and feature-engineering procedures;
- validation of metric-calculation scripts;
- baseline integration;
- technical feasibility assessment;
- governance and audit-pipeline testing; and
- preparation of the frozen confirmatory protocol.

The repository does **not** currently claim:

- completion of the confirmatory dissertation evaluation;
- final acceptance or rejection of the dissertation hypotheses;
- universal generalization across all relational database systems;
- production-scale validation;
- operational certification;
- unrestricted autonomous enforcement capability; or
- generalization across all organizations, infrastructures, or threat environments.

---

## Data Provenance and Legal Scope

The current AI-DAC preparatory evaluation does **not** use confidential, proprietary, or operational security data obtained from companies, government institutions, public authorities, hospitals, banks, police organizations, or other external organizations.

The preparatory experiments are based exclusively on the following categories of data.

### 1. Controlled laboratory data

SQL events and database-security telemetry were generated on controlled laboratory servers configured specifically for the AI-DAC research project.

These events were produced within an isolated and documented experimental environment. They were not collected from a live company, government, customer, hospital, bank, police, or University production system.

The controlled laboratory dataset is the primary evidence source for the relational database-security evaluation.

### 2. Public cybersecurity benchmark datasets

Publicly available cybersecurity datasets are used for supplementary benchmarking and comparative validation.

The principal public benchmarks currently used in the preparatory evaluation include:

- UNSW-NB15;
- NSL-KDD;
- CSE-CIC-IDS2018; and
- LogHub.

BoT-IoT and TON-IoT are treated as **conditional supplementary robustness benchmarks**. They will be included in the confirmatory evaluation only if the protocol-freeze gate is passed, including verification of:

- dataset availability;
- licence and citation requirements;
- provenance documentation;
- preprocessing reproducibility;
- feature compatibility;
- computational feasibility; and
- inclusion in the frozen experimental plan.

Public benchmark datasets provide supplementary external evidence. They do not replace the primary controlled SQL evaluation and must not be interpreted as direct equivalents of relational database telemetry.

Users of this repository remain responsible for observing the original licences, citation requirements, access conditions, and terms of use associated with each public dataset.

### 3. Controlled synthetic attack scenarios

The repository documents reproducible SQL misuse, privilege-abuse, anomalous-access, and adversarial scenarios executed within the isolated laboratory environment.

These scenarios were created exclusively for defensive cybersecurity research, feasibility assessment, evaluation, and reproducibility testing.

They do not involve unauthorized access to external systems or operational organizational infrastructure.

---

## Data Exclusions

The preparatory reproducibility package does not contain:

- confidential company data;
- internal government data;
- police operational data;
- hospital or medical records;
- banking or financial customer records;
- production database logs from external organizations;
- personal customer information;
- confidential University production-system data;
- third-party organizational security records;
- data collected through privileged access to an external organization; or
- data obtained from Belgian Police, banks, or hospitals.

The University of Vienna affiliation identifies the academic context of the doctoral research. It does not indicate that the University supplied the experimental datasets or granted access to production systems.

The controlled laboratory experiments therefore do not depend on authorization letters from companies, government institutions, police organizations, banks, hospitals, or other public authorities.

---

## Dataset-Provenance Manifests

The `reproducibility/manifests/` directory contains provenance manifests for the controlled and public datasets used or considered in the preparatory evaluation.

The available manifests include:

| Manifest | Dataset or source | Current role |
|---|---|---|
| `controlled_sql_lab_manifest.json` | Controlled SQL laboratory dataset | Primary controlled evaluation |
| `UNSW_NB15_manifest.json` | UNSW-NB15 | Supplementary external benchmark |
| `NSL_KDD_manifest.json` | NSL-KDD | Supplementary external benchmark |
| `CSE_CIC_IDS2018_manifest.json` | CSE-CIC-IDS2018 | Supplementary external benchmark |
| `LogHub_manifest.json` | LogHub | Supplementary log-analysis benchmark |
| `BoT_IoT_manifest.json` | BoT-IoT | Conditional supplementary benchmark |
| `TON_IoT_manifest.json` | TON-IoT | Conditional supplementary benchmark |
| `dataset_manifest_summary.csv` | Consolidated dataset overview | Cross-dataset provenance summary |

These manifests support evidence traceability by documenting, where applicable:

- dataset origin;
- access location;
- licence or terms of use;
- checksum;
- acquisition date;
- preprocessing status;
- feature scope;
- data classification;
- intended evaluation role;
- inclusion status; and
- known limitations.

The manifests distinguish:

- the primary controlled SQL dataset;
- fixed supplementary public benchmarks;
- conditional supplementary benchmarks; and
- datasets considered but not included in the frozen confirmatory protocol.

The existence of a manifest does not by itself mean that a dataset is included in the final confirmatory evaluation. Final inclusion will be determined and recorded during the formal protocol freeze.

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
- audit-trail generation;
- evidence preservation; and
- reproducibility controls.

The principal scientific contribution is not any individual model or component in isolation.

The proposed contribution is the integration of detection, adaptation, reflective learning, explanation, governance, human oversight, and evidence preservation into a lifecycle-aware and reproducible decision-support architecture for relational database cybersecurity.

The current repository documents the preparatory implementation and feasibility-testing stage of that architecture.

---

## Experimental Scope

The primary preparatory AI-DAC evaluation is based on a controlled SQL dataset generated within the documented laboratory environment.

The controlled dataset contains:

- **47,832 SQL events**;
- normal and anomalous database activities;
- separate training, validation, and test partitions; and
- controlled attack, misuse, and privilege-abuse scenarios.

The exact integer counts associated with each frozen partition will be taken from the corresponding dataset manifest during the protocol freeze.

Any percentages displayed in preparatory documentation are descriptive rounded values. Confirmatory analyses will use the exact integer counts recorded in the frozen manifest.

Public cybersecurity datasets are used as supplementary benchmarks to assess whether selected detection components remain effective outside the primary controlled SQL dataset.

The findings currently documented in the repository are bounded by:

- the documented laboratory infrastructure;
- the selected preparatory datasets;
- the implemented feature-engineering procedures;
- the current model configurations;
- the current reference parameters;
- the selected decision thresholds;
- the defined attack scenarios;
- the available computational resources; and
- the preparatory evaluation procedures included in the repository.

These boundaries must be considered when interpreting any visible metric or output.

---

## Preparatory Metrics

The numerical values currently visible in this repository originate from controlled development, feasibility, pipeline-verification, and preparatory reproducibility executions.

They demonstrate technical feasibility and internal pipeline consistency only.

They are not final confirmatory dissertation results.

Examples of preparatory metrics currently represented in the repository include:

| Preparatory metric | Observed value |
|---|---:|
| Controlled SQL dataset size | 47,832 events |
| Controlled SQL test-partition size | 7,174 events |
| Preliminary test-set accuracy | 0.98 |
| Preliminary precision | 0.95 |
| Preliminary recall | 0.95 |
| Preliminary F1-score | 0.95 |
| Preliminary ROC-AUC | 0.97 |
| Simulated response-risk reduction | 27.8% |
| Preparatory governance audit-trail completeness | 98.3% |
| Preparatory SHAP–RAG usefulness score | 4.21 |
| Preliminary drift-recovery reduction | 83% |
| Preparatory full AI-DAC F1-score | 0.950 |
| Preparatory no-lifecycle ablation F1-score | 0.921 |
| Preparatory transformer baseline F1-score | 0.900 |
| Preparatory UNSW-NB15 F1-score / ROC-AUC | 0.941 / 0.965 |
| Preparatory NSL-KDD F1-score / ROC-AUC | 0.957 / 0.973 |

These values must be interpreted as **pre-confirmatory diagnostics**.

They may change after:

- formal protocol freeze;
- dataset-manifest verification;
- correction of implementation defects;
- final preprocessing specification;
- execution of the frozen confirmatory protocol;
- application of predefined exclusion rules;
- statistical analysis; and
- independent reproducibility review.

No preparatory metric may be used retroactively as confirmatory evidence unless it is regenerated under the formally frozen protocol and linked to the immutable thesis-evaluation release.

---

## Preparatory Reproducibility Audit

The current reproducibility audit is a **preparatory audit**.

It evaluates whether the development-stage metrics and outputs can be reconstructed from the currently documented:

- scripts;
- configurations;
- manifests;
- metric registers;
- intermediate outputs;
- result files; and
- audit procedures.

The preparatory audit currently covers **45 unique registered metrics**:

- **39 metrics** reconstructed with an `OK` status;
- **6 metrics** reconstructed with a `Rounded OK` status; and
- **0 referenced preparatory output files** recorded as missing at the time of the audit.

Within this preparatory context:

- `OK` means that a reconstructed development-stage value matched the registered preparatory value exactly;
- `Rounded OK` means that a reconstructed value matched the registered preparatory value after applying the documented numerical-rounding convention.

These labels indicate internal reproducibility within the preparatory package only.

They do **not** indicate:

- confirmation of a dissertation hypothesis;
- completion of the final dissertation evaluation;
- external validation;
- production readiness; or
- final scientific acceptance of the reported values.

The final confirmatory audit will be conducted separately against the immutable thesis-evaluation release.

---

## Protocol Freeze

Before the confirmatory evaluation begins, the following elements will be formally frozen and documented:

- exact software commit;
- repository tag or release identifier;
- dataset manifests;
- dataset checksums;
- dataset inclusion and exclusion decisions;
- preprocessing rules;
- feature definitions;
- training, validation, and test partitions;
- reference model parameters;
- anomaly thresholds;
- explanation thresholds;
- reward-function coefficients;
- governance constraints;
- primary and secondary outcomes;
- experimental units;
- baseline configurations;
- ablation configurations;
- exclusion criteria;
- multiplicity families;
- statistical decision rules;
- execution commands;
- software environment;
- hardware environment;
- random-seed policy;
- raw-output locations;
- audit-report locations; and
- integrity-verification procedures.

After the protocol freeze, these elements may not be modified on the basis of observed confirmatory results.

Any necessary deviation will be documented explicitly, justified, timestamped, and separated from the predefined confirmatory analysis.

---

## Thesis-Evaluation Release

The repository state currently available on the `main` branch is not the immutable thesis-evaluation release.

After formal protocol freeze, a dedicated release will identify the exact software and evidence state used for the confirmatory dissertation evaluation.

The future thesis-evaluation release will include or reference:

- an immutable commit hash;
- a signed or annotated version tag;
- dataset manifests and checksums;
- frozen configuration files;
- execution commands;
- dependency specifications;
- container or environment definitions;
- raw confirmatory outputs;
- processed result tables;
- statistical-analysis outputs;
- audit records;
- integrity hashes; and
- a release-specific reproducibility report.

A suitable current preparatory version identifier is:

```text
v0.9.0-preconfirmatory
