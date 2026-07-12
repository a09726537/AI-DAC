# AI-DAC — Reproducibility Package

This repository contains the reproducibility package for the doctoral thesis:

**Lifecycle-Aware Database Cybersecurity: A Triple-Loop Learning Approach**  
**William Kandolo**, University of Vienna, Doctoral Thesis, 2026.

Repository: <https://github.com/a09726537/AI-DAC>

---

## Purpose

This repository supports the reproduction and audit of the controlled laboratory results reported in the dissertation.

The package contains dataset manifests, evaluation scripts, generated result files, audit records, metric registers, and the controlled SQL train, validation, and test partitions used to support the AI-DAC evaluation.

The repository is intended to make the thesis results inspectable, repeatable, and bounded to the documented laboratory configuration. It does **not** claim production-scale generalization across all enterprise database environments.

---

## Thesis Context

AI-DAC stands for **Artificial Intelligence–Driven Anomaly Detection and Control**.

The system operationalizes a **Lifecycle-Aware Triple-Loop Learning Framework** for adaptive database cybersecurity. It combines:

- anomaly detection,
- adaptive response,
- meta-learning and drift recovery,
- SHAP-based explainability,
- RAG-supported explanation,
- governance filtering,
- auditability,
- and reproducibility controls.

The core contribution is not any single component in isolation, but the integration of these components into a lifecycle-aware, governed, and reproducible decision-support architecture for relational database cybersecurity.

---

## Final Reproducibility Audit

The final reproducibility audit covered **45 unique metrics**:

- **39 metrics** reproduced with exact `OK` status
- **6 metrics** reproduced with `Rounded OK` status
- **0 missing referenced output files**

---

## Main Reproduced Claims

| Claim | Reproduced value |
|---|---:|
| Controlled SQL dataset size | 47,832 events |
| Controlled SQL test size | 7,174 events |
| Ordinary test-set accuracy | 0.98 |
| Precision | 0.95 |
| Recall | 0.95 |
| F1-score | 0.95 |
| ROC-AUC | 0.97 |
| Response-risk reduction | 27.8% |
| Governance audit completeness | 98.3% |
| SHAP--RAG overall usefulness | 4.21 |
| Drift recovery reduction | 83% |
| Full AI-DAC F1 | 0.950 |
| No-lifecycle F1 | 0.921 |
| Transformer baseline F1 | 0.900 |
| UNSW-NB15 F1 / ROC-AUC | 0.941 / 0.965 |
| NSL-KDD F1 / ROC-AUC | 0.957 / 0.973 |

---

## Repository Files

| File | Purpose |
|---|---|
| `aidac_reproducibility_package.tar.gz` | Full reproducibility archive |
| `aidac_reproducibility_package.sha256` | SHA-256 checksum for integrity verification |
| `final_reproducibility_audit_report.txt` | Human-readable final audit report |
| `final_reproducibility_audit_summary.csv` | Metric-level audit summary |
| `final_reproducibility_audit_summary.json` | Machine-readable audit summary |
| `CITATION.cff` | Citation metadata for the reproducibility package |
| `README.md` | Repository documentation |

---

## Verify Package Integrity

Run:

```bash
sha256sum -c aidac_reproducibility_package.sha256
