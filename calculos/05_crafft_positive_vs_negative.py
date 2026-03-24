#!/usr/bin/env python3
"""
05_crafft_positive_vs_negative.py
Compara QoL e disability entre CRAFFT positivo vs negativo (Table 2).
Uso: python3 calculos/05_crafft_positive_vs_negative.py
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DC = os.path.join(BASE, 'data_clean')

crafft = pd.read_csv(os.path.join(DC, 'crafft_clean.csv'))
ibddisk = pd.read_csv(os.path.join(DC, 'ibddisk_clean.csv'))
ibdq = pd.read_csv(os.path.join(DC, 'ibdq_clean.csv'))

np.random.seed(42)

def format_p(p):
    if p < 0.001: return '< 0.001'
    elif p < 0.01: return f'{p:.3f}'
    else: return f'{p:.2f}'

def wilson_ci(k, n):
    from scipy.stats import norm
    z = norm.ppf(0.975)
    p_hat = k / n
    denom = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denom
    spread = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denom
    return max(0, center - spread), min(1, center + spread)

# First eval per patient
crafft_f = crafft[crafft['DIAGNOSTICO'].notna()].drop_duplicates('PACIENTE', keep='first')
ibddisk_f = ibddisk[ibddisk['DIAGNOSTICO'].notna()].drop_duplicates('PACIENTE', keep='first')[['PACIENTE', 'Total_IBD_Disk_Score']]
ibdq_full = ibdq[(ibdq['DIAGNOSTICO'].notna()) & (ibdq['Item 32'].notna())].drop_duplicates('PACIENTE', keep='first')

# Add CRAFFT status
crafft_f = crafft_f[['PACIENTE', 'Total Score', 'Risk Interpretation', 'DIAGNOSTICO', 'IDADE_DIAGNOSTICO']].copy()
crafft_f['CRAFFT_Status'] = crafft_f['Risk Interpretation'].apply(lambda x: 'Positive' if 'Positive' in str(x) else 'Negative')

print("=" * 70)
print("TABLE 2 — PRO SCORES BY CRAFFT STATUS")
print("=" * 70)

pos = crafft_f[crafft_f['CRAFFT_Status'] == 'Positive']
neg = crafft_f[crafft_f['CRAFFT_Status'] == 'Negative']
print(f"\nCRAFFT Positive (>=2): n = {len(pos)}")
print(f"CRAFFT Negative (<2):  n = {len(neg)}")

# Demographics by CRAFFT status
print(f"\n--- DEMOGRAPHICS BY CRAFFT STATUS ---")
for label, grp in [('Positive', pos), ('Negative', neg)]:
    print(f"\n  {label} (n={len(grp)}):")
    # Diagnosis
    for d in ['CD', 'UC', 'IBD-U']:
        c = (grp['DIAGNOSTICO'] == d).sum()
        print(f"    {d}: {c} ({c/len(grp)*100:.1f}%)")
    # Age at diagnosis
    a = grp['IDADE_DIAGNOSTICO'].dropna()
    print(f"    Age at diagnosis: median {a.median():.0f} (IQR {a.quantile(0.25):.0f}–{a.quantile(0.75):.0f})")

# Chi-squared for diagnosis
ct = pd.crosstab(crafft_f['CRAFFT_Status'], crafft_f['DIAGNOSTICO'])
print(f"\nDiagnosis by CRAFFT status:")
print(ct)
chi2, p, _, _ = chi2_contingency(ct)
print(f"Chi-squared: p = {format_p(p)}")

# Age comparison
u, p_age = mannwhitneyu(pos['IDADE_DIAGNOSTICO'].dropna(), neg['IDADE_DIAGNOSTICO'].dropna())
print(f"Age at diagnosis: Mann-Whitney p = {format_p(p_age)}")

# IBD-Disk by CRAFFT status
print(f"\n--- IBD-DISK BY CRAFFT STATUS ---")
merged = crafft_f[['PACIENTE', 'CRAFFT_Status']].merge(ibddisk_f, on='PACIENTE')
pos_disk = merged[merged['CRAFFT_Status'] == 'Positive']['Total_IBD_Disk_Score']
neg_disk = merged[merged['CRAFFT_Status'] == 'Negative']['Total_IBD_Disk_Score']

print(f"  Positive (n={len(pos_disk)}): median {pos_disk.median():.0f} (IQR {pos_disk.quantile(0.25):.0f}–{pos_disk.quantile(0.75):.0f})")
print(f"  Negative (n={len(neg_disk)}): median {neg_disk.median():.0f} (IQR {neg_disk.quantile(0.25):.0f}–{neg_disk.quantile(0.75):.0f})")
u, p = mannwhitneyu(pos_disk, neg_disk, alternative='two-sided')
print(f"  Mann-Whitney: U={u:.0f}, p = {format_p(p)}")

# IBDQ by CRAFFT status
print(f"\n--- IBDQ BY CRAFFT STATUS ---")
ibdq_cols = ['PACIENTE', 'Total', 'Sintomas Intestinais', 'Sintomas Sistêmicos', 'Bem-Estar Emocional', 'Interação Social']
merged_q = crafft_f[['PACIENTE', 'CRAFFT_Status']].merge(ibdq_full[ibdq_cols], on='PACIENTE')
pos_q = merged_q[merged_q['CRAFFT_Status'] == 'Positive']
neg_q = merged_q[merged_q['CRAFFT_Status'] == 'Negative']

print(f"  Positive com IBDQ full: n = {len(pos_q)}")
print(f"  Negative com IBDQ full: n = {len(neg_q)}")

for label, col in [('Total', 'Total'), ('Bowel', 'Sintomas Intestinais'),
                    ('Systemic', 'Sintomas Sistêmicos'), ('Emotional', 'Bem-Estar Emocional'),
                    ('Social', 'Interação Social')]:
    p_vals = pos_q[col].dropna()
    n_vals = neg_q[col].dropna()
    if len(p_vals) >= 2 and len(n_vals) >= 2:
        u, p = mannwhitneyu(p_vals, n_vals, alternative='two-sided')
        print(f"\n  IBDQ {label}:")
        print(f"    Positive: median {p_vals.median():.0f} (IQR {p_vals.quantile(0.25):.0f}–{p_vals.quantile(0.75):.0f})")
        print(f"    Negative: median {n_vals.median():.0f} (IQR {n_vals.quantile(0.25):.0f}–{n_vals.quantile(0.75):.0f})")
        print(f"    Mann-Whitney: U={u:.0f}, p = {format_p(p)}")
    else:
        print(f"\n  IBDQ {label}: n insuficiente para comparação")

# Summary table for manuscript
print(f"\n{'=' * 70}")
print("RESUMO TABLE 2 (para manuscrito)")
print(f"{'=' * 70}")
print(f"{'Variable':<30} {'CRAFFT Neg':<20} {'CRAFFT Pos':<20} {'p-value':<10}")
print(f"{'-'*80}")
print(f"{'n':<30} {len(neg):<20} {len(pos):<20}")
print(f"{'IBD-Disk, median (IQR)':<30} {f'{neg_disk.median():.0f} ({neg_disk.quantile(0.25):.0f}–{neg_disk.quantile(0.75):.0f})':<20} {f'{pos_disk.median():.0f} ({pos_disk.quantile(0.25):.0f}–{pos_disk.quantile(0.75):.0f})':<20}", end='')
u, p = mannwhitneyu(pos_disk, neg_disk)
print(f" {format_p(p)}")

for label, col in [('IBDQ Total', 'Total'), ('  Bowel', 'Sintomas Intestinais'),
                    ('  Systemic', 'Sintomas Sistêmicos'), ('  Emotional', 'Bem-Estar Emocional'),
                    ('  Social', 'Interação Social')]:
    pv = pos_q[col].dropna()
    nv = neg_q[col].dropna()
    if len(pv) >= 2 and len(nv) >= 2:
        u, p = mannwhitneyu(pv, nv)
        print(f"{label + ', median (IQR)':<30} {f'{nv.median():.0f} ({nv.quantile(0.25):.0f}–{nv.quantile(0.75):.0f})':<20} {f'{pv.median():.0f} ({pv.quantile(0.25):.0f}–{pv.quantile(0.75):.0f})':<20} {format_p(p)}")

print(f"\n{'=' * 70}")
print("FIM — 05_crafft_positive_vs_negative.py")
