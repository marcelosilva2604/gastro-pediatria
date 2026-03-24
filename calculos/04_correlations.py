#!/usr/bin/env python3
"""
04_correlations.py
Correlações de Spearman entre instrumentos PRO.
Uso: python3 calculos/04_correlations.py
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DC = os.path.join(BASE, 'data_clean')

crafft = pd.read_csv(os.path.join(DC, 'crafft_clean.csv'))
ibddisk = pd.read_csv(os.path.join(DC, 'ibddisk_clean.csv'))
ibdq = pd.read_csv(os.path.join(DC, 'ibdq_clean.csv'))

# First evaluation per patient, with diagnosis
crafft_f = crafft[crafft['DIAGNOSTICO'].notna()].drop_duplicates('PACIENTE', keep='first')[['PACIENTE', 'Total Score', 'Risk Interpretation', 'DIAGNOSTICO']].rename(columns={'Total Score': 'CRAFFT_Score'})
ibddisk_f = ibddisk[ibddisk['DIAGNOSTICO'].notna()].drop_duplicates('PACIENTE', keep='first')[['PACIENTE', 'Total_IBD_Disk_Score']].rename(columns={'Total_IBD_Disk_Score': 'IBD_Disk_Score'})
ibdq_full = ibdq[(ibdq['DIAGNOSTICO'].notna()) & (ibdq['Item 32'].notna())].drop_duplicates('PACIENTE', keep='first')[['PACIENTE', 'Total', 'Sintomas Intestinais', 'Sintomas Sistêmicos', 'Bem-Estar Emocional', 'Interação Social']].rename(columns={'Total': 'IBDQ_Total'})

print("=" * 70)
print("CORRELAÇÕES DE SPEARMAN ENTRE INSTRUMENTOS PRO")
print("=" * 70)

# Merge CRAFFT + IBD-Disk
merged_cd = crafft_f.merge(ibddisk_f, on='PACIENTE', how='inner')
print(f"\nPacientes com CRAFFT + IBD-Disk: {len(merged_cd)}")

rho, p = spearmanr(merged_cd['CRAFFT_Score'], merged_cd['IBD_Disk_Score'])
print(f"  CRAFFT vs IBD-Disk: rho = {rho:.3f}, p = {p:.4f}")

# Merge CRAFFT + IBDQ
merged_cq = crafft_f.merge(ibdq_full, on='PACIENTE', how='inner')
print(f"\nPacientes com CRAFFT + IBDQ full: {len(merged_cq)}")

rho, p = spearmanr(merged_cq['CRAFFT_Score'], merged_cq['IBDQ_Total'])
print(f"  CRAFFT vs IBDQ Total: rho = {rho:.3f}, p = {p:.4f}")

for dom, col in [('Bowel', 'Sintomas Intestinais'), ('Systemic', 'Sintomas Sistêmicos'),
                  ('Emotional', 'Bem-Estar Emocional'), ('Social', 'Interação Social')]:
    rho, p = spearmanr(merged_cq['CRAFFT_Score'], merged_cq[col])
    print(f"  CRAFFT vs IBDQ {dom}: rho = {rho:.3f}, p = {p:.4f}")

# Merge IBD-Disk + IBDQ
merged_dq = ibddisk_f.merge(ibdq_full, on='PACIENTE', how='inner')
print(f"\nPacientes com IBD-Disk + IBDQ full: {len(merged_dq)}")

rho, p = spearmanr(merged_dq['IBD_Disk_Score'], merged_dq['IBDQ_Total'])
print(f"  IBD-Disk vs IBDQ Total: rho = {rho:.3f}, p = {p:.4f}")

for dom, col in [('Bowel', 'Sintomas Intestinais'), ('Systemic', 'Sintomas Sistêmicos'),
                  ('Emotional', 'Bem-Estar Emocional'), ('Social', 'Interação Social')]:
    rho, p = spearmanr(merged_dq['IBD_Disk_Score'], merged_dq[col])
    print(f"  IBD-Disk vs IBDQ {dom}: rho = {rho:.3f}, p = {p:.4f}")

# All 3 instruments
merged_all = crafft_f.merge(ibddisk_f, on='PACIENTE').merge(ibdq_full, on='PACIENTE')
print(f"\nPacientes com CRAFFT + IBD-Disk + IBDQ full: {len(merged_all)}")

print(f"\nMatriz de correlação completa:")
cols = ['CRAFFT_Score', 'IBD_Disk_Score', 'IBDQ_Total']
for i, c1 in enumerate(cols):
    for c2 in cols[i+1:]:
        rho, p = spearmanr(merged_all[c1], merged_all[c2])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {c1} vs {c2}: rho = {rho:.3f}, p = {p:.4f} {sig}")

print(f"\n{'=' * 70}")
print("FIM — 04_correlations.py")
