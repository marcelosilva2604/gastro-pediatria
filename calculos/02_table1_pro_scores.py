#!/usr/bin/env python3
"""
02_table1_pro_scores.py
Gera escores PRO por diagnóstico para Table 1.
Lê: data_clean/crafft_clean.csv, ibddisk_clean.csv, ibdq_clean.csv, impact_clean.csv
Uso: python3 calculos/02_table1_pro_scores.py
"""

import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DC = os.path.join(BASE, 'data_clean')

crafft = pd.read_csv(os.path.join(DC, 'crafft_clean.csv'))
ibddisk = pd.read_csv(os.path.join(DC, 'ibddisk_clean.csv'))
ibdq = pd.read_csv(os.path.join(DC, 'ibdq_clean.csv'))
impact = pd.read_csv(os.path.join(DC, 'impact_clean.csv'))

def report_score(df, score_col, label, diags=['All', 'CD', 'UC', 'IBD-U']):
    print(f"\n--- {label} ---")
    for d in diags:
        sub = df if d == 'All' else df[df['DIAGNOSTICO'] == d]
        n = len(sub)
        if n == 0:
            print(f"  {d}: n=0")
            continue
        s = sub[score_col].dropna()
        print(f"  {d} (n={n}): median {s.median():.0f} (IQR {s.quantile(0.25):.0f}–{s.quantile(0.75):.0f}), range {s.min():.0f}–{s.max():.0f}")

print("=" * 70)
print("TABLE 1 — PRO SCORES BY DIAGNOSIS")
print("=" * 70)

# CRAFFT
crafft_d = crafft[crafft['DIAGNOSTICO'].notna()].drop_duplicates('PACIENTE', keep='first')
print(f"\n{'='*40}")
print(f"CRAFFT (1a avaliação por paciente)")
print(f"{'='*40}")
print(f"Total pacientes: {len(crafft_d)}")

report_score(crafft_d, 'Total Score', 'CRAFFT Total Score')

print(f"\n  CRAFFT Positive Screen (>=2) por diagnóstico:")
for d in ['All', 'CD', 'UC', 'IBD-U']:
    sub = crafft_d if d == 'All' else crafft_d[crafft_d['DIAGNOSTICO'] == d]
    n = len(sub)
    pos = (sub['Risk Interpretation'] == 'Positive screen (≥2)').sum()
    neg = (sub['Risk Interpretation'] == 'Negative screen (<2)').sum()
    print(f"    {d} (n={n}): Pos={pos} ({pos/n*100:.1f}%), Neg={neg} ({neg/n*100:.1f}%)")

# IBD-Disk
ibddisk_d = ibddisk[ibddisk['DIAGNOSTICO'].notna()].drop_duplicates('PACIENTE', keep='first')
print(f"\n{'='*40}")
print(f"IBD-DISK (1a avaliação por paciente)")
print(f"{'='*40}")
print(f"Total pacientes: {len(ibddisk_d)}")

report_score(ibddisk_d, 'Total_IBD_Disk_Score', 'IBD-Disk Total Score')

# IBDQ full (32 items, >=18y)
ibdq_full = ibdq[(ibdq['DIAGNOSTICO'].notna()) & (ibdq['Item 32'].notna())]
ibdq_ff = ibdq_full.drop_duplicates('PACIENTE', keep='first')
print(f"\n{'='*40}")
print(f"IBDQ FULL 32 ITEMS (>=18 anos, 1a avaliação)")
print(f"{'='*40}")
print(f"Total pacientes: {len(ibdq_ff)}")

report_score(ibdq_ff, 'Total', 'IBDQ Total')

for dom, col in [('Bowel Symptoms', 'Sintomas Intestinais'),
                  ('Systemic Symptoms', 'Sintomas Sistêmicos'),
                  ('Emotional Well-being', 'Bem-Estar Emocional'),
                  ('Social Function', 'Interação Social')]:
    report_score(ibdq_ff, col, f'IBDQ — {dom}')

# IMPACT-III (<18y)
impact_v = impact[(impact['DIAGNOSTICO'].notna()) & (impact['Total'] > 0)]
impact_f = impact_v.drop_duplicates('PACIENTE', keep='first')
print(f"\n{'='*40}")
print(f"IMPACT-III (<18 anos, 1a avaliação, Total > 0)")
print(f"{'='*40}")
print(f"Total pacientes: {len(impact_f)}")

report_score(impact_f, 'Total', 'IMPACT-III Total', ['All', 'CD', 'UC'])

for dom in ['Domínio: Sintomas', 'Domínio: Emocional', 'Domínio: Social',
            'Domínio: Bem-estar', 'Domínio: Tratamento']:
    report_score(impact_f, dom, f'IMPACT-III — {dom}', ['All', 'CD', 'UC'])

print(f"\n{'=' * 70}")
print("FIM — 02_table1_pro_scores.py")
