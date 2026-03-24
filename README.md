# Patient-Reported Outcomes in Young IBD Patients in a LMIC

Analysis code and manuscript for the study: **"Patient-Reported Outcomes, Disability, and Substance Use Risk in Young Patients With Inflammatory Bowel Disease in a Low- and Middle-Income Country"**

Submitted to **JCC Plus** (Journal of Crohn's and Colitis Plus) — Special Issue: *The Global Burden of Inflammatory Bowel Disease*

## Study Overview

Cross-sectional study of **148 young patients with IBD** (105 CD, 38 UC, 5 IBD-U) at Hospital das Clínicas, Faculty of Medicine, University of São Paulo (HC-FMUSP), Brazil — a tertiary public hospital within the Brazilian Unified Health System (SUS).

Four validated PRO instruments were administered:
- **CRAFFT 2.1** — substance use screening (all patients)
- **IBD-Disk** — disability index (all patients)
- **IBDQ** — quality of life, 32-item (patients ≥ 18 years)
- **IMPACT-III** — quality of life, paediatric (patients < 18 years)

## Key Findings

- **UC patients report significantly worse QoL than CD** across all IBDQ domains (p = 0.01–0.04)
- **Female patients have greater disability** than males (IBD-Disk: 44 vs 31, p = 0.035; emotions p < 0.001)
- **Extraintestinal burden** (joint pain + fatigue + sleep) strongly associated with worse QoL (all domains p < 0.001)
- **22.2% screen positive for substance use risk** (CRAFFT ≥ 2), correlated with older age at diagnosis (p = 0.013)
- **IBD-Disk and IBDQ strongly correlated** (Spearman ρ = −0.71, p < 0.001)

## Repository Structure

```
├── calculos/                  # Reproducible analysis scripts
│   ├── 01_table1_demographics.py
│   ├── 02_table1_pro_scores.py
│   ├── 03_table1_statistics.py
│   ├── 04_correlations.py
│   ├── 05_crafft_positive_vs_negative.py
│   ├── 06_regression_crafft.py
│   ├── 07_figures.py
│   ├── 08_extraintestinal_vs_qol.py
│   ├── 09_sex_differences.py
│   └── 10_age_correlations.py
├── clean_*.py                 # Data cleaning/processing scripts
├── manuscrito/
│   ├── manuscrito_jcc_plus.docx
│   └── figures/
│       ├── figure1_ibdq_by_diagnosis.png
│       └── figure2_ibddisk_vs_ibdq_correlation.png
├── CLAUDE.md                  # Project context and JCC Plus editorial standards
└── README.md
```

## Running the Analysis

Each script in `calculos/` reads from `data_clean/` (not included — see Data Availability below) and prints results to the terminal:

```bash
python3 calculos/01_table1_demographics.py
python3 calculos/02_table1_pro_scores.py
python3 calculos/03_table1_statistics.py
# ... etc
```

### Requirements

```
Python >= 3.9
pandas >= 2.2
scipy >= 1.14
numpy
matplotlib
statsmodels >= 0.14
openpyxl
```

## Data Availability

The clinical datasets contain protected health information (patient names and hospital registration numbers) and are **not included in this repository** in compliance with Brazilian data protection legislation (Lei Geral de Proteção de Dados — LGPD).

De-identified data may be made available upon reasonable request to the corresponding author, subject to ethical review and institutional approval.

**Ethics approval:** CAAE 82679518.2.0000 (Ethics Committee, HC-FMUSP)

## Authors

- **Jane Oba, MD** — Paediatric Gastroenterology, HC-FMUSP, São Paulo, Brazil
- [Additional authors to be added]

## License

Analysis code is available under the MIT License. The manuscript and figures are copyright of the authors.
