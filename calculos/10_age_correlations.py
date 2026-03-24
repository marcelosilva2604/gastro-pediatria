#!/usr/bin/env python3
"""
10_age_correlations.py
Correlações da idade ao diagnóstico com todos os PROs.
Uso: python3 calculos/10_age_correlations.py
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
import warnings
import os

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DC = os.path.join(BASE, 'data_clean')

pts = pd.read_csv(os.path.join(DC, 'todospacientes_clean.csv'))
inc = pts[pts['STATUS'] == 'INCLUÍDO']
age_map = inc.set_index('NOME')['IDADE_DIAGNOSTICO'].to_dict()

crafft = pd.read_csv(os.path.join(DC, 'crafft_clean.csv')).drop_duplicates('PACIENTE', keep='first')
ibddisk = pd.read_csv(os.path.join(DC, 'ibddisk_clean.csv')).drop_duplicates('PACIENTE', keep='first')
ibdq = pd.read_csv(os.path.join(DC, 'ibdq_clean.csv'))
ibdq_full = ibdq[ibdq['Item 32'].notna()].drop_duplicates('PACIENTE', keep='first').copy()

for df in [crafft, ibddisk, ibdq_full]:
    df['IDADE'] = df['PACIENTE'].map(age_map)

def fp(p):
    if p < 0.001: return '< 0.001'
    return f'{p:.4f}'

print("=" * 70)
print("CORRELAÇÕES — IDADE AO DIAGNÓSTICO vs PROs")
print("=" * 70)

# =============================================
# 1. IDADE vs CRAFFT
# =============================================
print("\n" + "=" * 50)
print("1. IDADE vs CRAFFT")
print("=" * 50)

sub = crafft[['IDADE', 'Total Score']].dropna()
rho, p = spearmanr(sub['IDADE'], sub['Total Score'])
sig = ' ***' if p < 0.05 else ''
print(f"  Idade vs CRAFFT score: rho = {rho:.3f}, p = {fp(p)}{sig} (n={len(sub)})")

# Paris <17 vs >=17 CRAFFT
young = crafft[crafft['IDADE'] < 17]['Total Score']
older = crafft[crafft['IDADE'] >= 17]['Total Score']
u, p = mannwhitneyu(young, older)
sig = ' ***' if p < 0.05 else ''
print(f"  CRAFFT <17y vs >=17y: {young.median():.0f} vs {older.median():.0f}, p = {fp(p)}{sig} (n={len(young)} vs {len(older)})")

# =============================================
# 2. IDADE vs IBD-DISK
# =============================================
print("\n" + "=" * 50)
print("2. IDADE vs IBD-DISK")
print("=" * 50)

sub = ibddisk[['IDADE', 'Total_IBD_Disk_Score']].dropna()
rho, p = spearmanr(sub['IDADE'], sub['Total_IBD_Disk_Score'])
sig = ' ***' if p < 0.05 else ''
print(f"  Idade vs IBD-Disk total: rho = {rho:.3f}, p = {fp(p)}{sig} (n={len(sub)})")

disk_domains = {
    'Item1_Abdominal_Pain': 'Abdominal Pain',
    'Item6_Energy': 'Energy',
    'Item5_Sleep': 'Sleep',
    'Item7_Emotions': 'Emotions',
    'Item8_Body_Image': 'Body Image',
    'Item9_Sexual_Function': 'Sexual Function',
    'Item10_Joint_Pain': 'Joint Pain',
}

print("\n  By domain:")
for col, label in disk_domains.items():
    if col in ibddisk.columns:
        sub = ibddisk[['IDADE', col]].dropna()
        rho, p = spearmanr(sub['IDADE'], sub[col])
        sig = ' ***' if p < 0.05 else ''
        if p < 0.05:
            print(f"    Idade vs {label}: rho = {rho:.3f}, p = {fp(p)}{sig} (n={len(sub)})")

# =============================================
# 3. IDADE vs IBDQ
# =============================================
print("\n" + "=" * 50)
print("3. IDADE vs IBDQ")
print("=" * 50)

for label, col in [('Total', 'Total'), ('Bowel', 'Sintomas Intestinais'),
                    ('Systemic', 'Sintomas Sistêmicos'), ('Emotional', 'Bem-Estar Emocional'),
                    ('Social', 'Interação Social')]:
    sub = ibdq_full[['IDADE', col]].dropna()
    rho, p = spearmanr(sub['IDADE'], sub[col])
    sig = ' ***' if p < 0.05 else ''
    print(f"  Idade vs IBDQ {label}: rho = {rho:.3f}, p = {fp(p)}{sig} (n={len(sub)})")

# =============================================
# RESUMO
# =============================================
print(f"\n{'=' * 70}")
print("RESUMO — ACHADOS SIGNIFICATIVOS COM IDADE (p < 0.05)")
print("=" * 70)
print("""
IDADE AO DIAGNÓSTICO:
  - Correlaciona com CRAFFT score: rho = 0.207, p = 0.013
    (diagnóstico mais tardio → mais risco de substâncias)
  - CRAFFT <17y vs >=17y: score diferente, p = 0.027
  - Correlaciona com IBD-Disk Emotions: rho = 0.225, p = 0.010
    (diagnóstico mais tardio → mais impacto emocional)
  - Correlaciona com IBD-Disk Sexual Function: rho = 0.208, p = 0.018
    (diagnóstico mais tardio → mais impacto sexual)
""")
print("FIM — 10_age_correlations.py")
