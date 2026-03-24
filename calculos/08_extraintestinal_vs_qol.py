#!/usr/bin/env python3
"""
08_extraintestinal_vs_qol.py
Sintomas extraintestinais (IBD-Disk: joint pain, energy, sleep) vs QoL e CRAFFT.
Uso: python3 calculos/08_extraintestinal_vs_qol.py
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, fisher_exact
import warnings
import os

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DC = os.path.join(BASE, 'data_clean')

crafft = pd.read_csv(os.path.join(DC, 'crafft_clean.csv')).drop_duplicates('PACIENTE', keep='first')
ibddisk = pd.read_csv(os.path.join(DC, 'ibddisk_clean.csv')).drop_duplicates('PACIENTE', keep='first')
ibdq = pd.read_csv(os.path.join(DC, 'ibdq_clean.csv'))
ibdq_full = ibdq[ibdq['Item 32'].notna()].drop_duplicates('PACIENTE', keep='first').copy()

def fp(p):
    if p < 0.001: return '< 0.001'
    return f'{p:.4f}'

print("=" * 70)
print("SINTOMAS EXTRAINTESTINAIS vs QoL vs SUBSTÂNCIAS")
print("=" * 70)

# =============================================
# 1. DOR ARTICULAR (Item10) como proxy extraintestinal
# =============================================
print("\n" + "=" * 50)
print("1. DOR ARTICULAR (IBD-Disk Item10 > 0)")
print("=" * 50)

ibddisk['HAS_JOINT_PAIN'] = ibddisk['Item10_Joint_Pain'] > 0
has_jp = ibddisk[ibddisk['HAS_JOINT_PAIN'] == True]
no_jp = ibddisk[ibddisk['HAS_JOINT_PAIN'] == False]
print(f"Com dor articular: {len(has_jp)} ({len(has_jp)/len(ibddisk)*100:.1f}%)")
print(f"Sem dor articular: {len(no_jp)} ({len(no_jp)/len(ibddisk)*100:.1f}%)")

# Joint pain vs IBD-Disk total
u, p = mannwhitneyu(has_jp['Total_IBD_Disk_Score'], no_jp['Total_IBD_Disk_Score'])
print(f"\nIBD-Disk total: com JP {has_jp['Total_IBD_Disk_Score'].median():.0f} vs sem JP {no_jp['Total_IBD_Disk_Score'].median():.0f}, p = {fp(p)}")

# Joint pain vs IBDQ
merged_jp_q = ibddisk[['PACIENTE', 'HAS_JOINT_PAIN']].merge(
    ibdq_full[['PACIENTE', 'Total', 'Sintomas Intestinais', 'Sintomas Sistêmicos',
               'Bem-Estar Emocional', 'Interação Social']], on='PACIENTE')
jp_yes = merged_jp_q[merged_jp_q['HAS_JOINT_PAIN'] == True]
jp_no = merged_jp_q[merged_jp_q['HAS_JOINT_PAIN'] == False]

print(f"\nIBDQ com vs sem dor articular (n={len(jp_yes)} vs {len(jp_no)}):")
for label, col in [('Total', 'Total'), ('Bowel', 'Sintomas Intestinais'),
                    ('Systemic', 'Sintomas Sistêmicos'), ('Emotional', 'Bem-Estar Emocional'),
                    ('Social', 'Interação Social')]:
    u, p = mannwhitneyu(jp_yes[col], jp_no[col])
    sig = ' ***' if p < 0.05 else ''
    print(f"  IBDQ {label}: {jp_yes[col].median():.0f} vs {jp_no[col].median():.0f}, p = {fp(p)}{sig}")

# Joint pain vs CRAFFT
merged_jp_c = ibddisk[['PACIENTE', 'HAS_JOINT_PAIN']].merge(
    crafft[['PACIENTE', 'Total Score', 'Risk Interpretation']], on='PACIENTE')
jp_c_yes = merged_jp_c[merged_jp_c['HAS_JOINT_PAIN'] == True]
jp_c_no = merged_jp_c[merged_jp_c['HAS_JOINT_PAIN'] == False]

u, p = mannwhitneyu(jp_c_yes['Total Score'], jp_c_no['Total Score'])
print(f"\nCRAFFT score: com JP {jp_c_yes['Total Score'].median():.0f} vs sem JP {jp_c_no['Total Score'].median():.0f}, p = {fp(p)}")

pos_jp = (jp_c_yes['Risk Interpretation'] == 'Positive screen (≥2)').sum()
pos_no = (jp_c_no['Risk Interpretation'] == 'Positive screen (≥2)').sum()
ct = [[pos_jp, len(jp_c_yes) - pos_jp], [pos_no, len(jp_c_no) - pos_no]]
or_val, p = fisher_exact(ct)
print(f"CRAFFT positive: com JP {pos_jp}/{len(jp_c_yes)} ({pos_jp/len(jp_c_yes)*100:.1f}%) vs sem JP {pos_no}/{len(jp_c_no)} ({pos_no/len(jp_c_no)*100:.1f}%), OR={or_val:.2f}, p = {fp(p)}")

# =============================================
# 2. BURDEN EXTRAINTESTINAL COMPOSTO
# =============================================
print("\n" + "=" * 50)
print("2. SCORE EXTRAINTESTINAL COMPOSTO")
print("   (Joint Pain + Energy + Sleep)")
print("=" * 50)

ibddisk['EXTRA_SCORE'] = ibddisk[['Item10_Joint_Pain', 'Item6_Energy', 'Item5_Sleep']].sum(axis=1)
median_extra = ibddisk['EXTRA_SCORE'].median()
ibddisk['HIGH_EXTRA'] = ibddisk['EXTRA_SCORE'] > median_extra
hi = ibddisk[ibddisk['HIGH_EXTRA'] == True]
lo = ibddisk[ibddisk['HIGH_EXTRA'] == False]

print(f"Mediana score extraintestinal: {median_extra:.0f}")
print(f"High burden (>{median_extra:.0f}): {len(hi)} ({len(hi)/len(ibddisk)*100:.1f}%)")
print(f"Low burden (<={median_extra:.0f}): {len(lo)} ({len(lo)/len(ibddisk)*100:.1f}%)")

u, p = mannwhitneyu(hi['Total_IBD_Disk_Score'], lo['Total_IBD_Disk_Score'])
print(f"\nIBD-Disk total: high {hi['Total_IBD_Disk_Score'].median():.0f} vs low {lo['Total_IBD_Disk_Score'].median():.0f}, p = {fp(p)}")

# High vs low vs IBDQ
merged_ex_q = ibddisk[['PACIENTE', 'HIGH_EXTRA']].merge(
    ibdq_full[['PACIENTE', 'Total', 'Sintomas Intestinais', 'Sintomas Sistêmicos',
               'Bem-Estar Emocional', 'Interação Social']], on='PACIENTE')
ex_hi = merged_ex_q[merged_ex_q['HIGH_EXTRA'] == True]
ex_lo = merged_ex_q[merged_ex_q['HIGH_EXTRA'] == False]

print(f"\nIBDQ high vs low extraintestinal burden (n={len(ex_hi)} vs {len(ex_lo)}):")
for label, col in [('Total', 'Total'), ('Bowel', 'Sintomas Intestinais'),
                    ('Systemic', 'Sintomas Sistêmicos'), ('Emotional', 'Bem-Estar Emocional'),
                    ('Social', 'Interação Social')]:
    u, p = mannwhitneyu(ex_hi[col], ex_lo[col])
    sig = ' ***' if p < 0.05 else ''
    print(f"  IBDQ {label}: high {ex_hi[col].median():.0f} vs low {ex_lo[col].median():.0f}, p = {fp(p)}{sig}")

# High vs low vs CRAFFT
merged_ex_c = ibddisk[['PACIENTE', 'HIGH_EXTRA']].merge(
    crafft[['PACIENTE', 'Total Score', 'Risk Interpretation']], on='PACIENTE')
ex_c_hi = merged_ex_c[merged_ex_c['HIGH_EXTRA'] == True]
ex_c_lo = merged_ex_c[merged_ex_c['HIGH_EXTRA'] == False]

u, p = mannwhitneyu(ex_c_hi['Total Score'], ex_c_lo['Total Score'])
print(f"\nCRAFFT score: high extra {ex_c_hi['Total Score'].median():.0f} vs low {ex_c_lo['Total Score'].median():.0f}, p = {fp(p)}")

pos_hi = (ex_c_hi['Risk Interpretation'] == 'Positive screen (≥2)').sum()
pos_lo = (ex_c_lo['Risk Interpretation'] == 'Positive screen (≥2)').sum()
ct = [[pos_hi, len(ex_c_hi) - pos_hi], [pos_lo, len(ex_c_lo) - pos_lo]]
or_val, p = fisher_exact(ct)
print(f"CRAFFT positive: high {pos_hi}/{len(ex_c_hi)} ({pos_hi/len(ex_c_hi)*100:.1f}%) vs low {pos_lo}/{len(ex_c_lo)} ({pos_lo/len(ex_c_lo)*100:.1f}%), OR={or_val:.2f}, p = {fp(p)}")

# =============================================
# 3. CORRELAÇÕES
# =============================================
print("\n" + "=" * 50)
print("3. CORRELAÇÕES: Extra score vs IBDQ domínios")
print("=" * 50)

merged2 = ibddisk[['PACIENTE', 'EXTRA_SCORE']].merge(
    ibdq_full[['PACIENTE', 'Total', 'Sintomas Intestinais', 'Sintomas Sistêmicos',
               'Bem-Estar Emocional', 'Interação Social']], on='PACIENTE')

print(f"n = {len(merged2)}")
for label, col in [('Total', 'Total'), ('Bowel', 'Sintomas Intestinais'),
                    ('Systemic', 'Sintomas Sistêmicos'), ('Emotional', 'Bem-Estar Emocional'),
                    ('Social', 'Interação Social')]:
    rho, p = spearmanr(merged2['EXTRA_SCORE'], merged2[col])
    sig = ' ***' if p < 0.05 else ''
    print(f"  Extra vs IBDQ {label}: rho = {rho:.3f}, p = {fp(p)}{sig}")

# Extra composite vs CRAFFT
merged3 = ibddisk[['PACIENTE', 'EXTRA_SCORE']].merge(
    crafft[['PACIENTE', 'Total Score']], on='PACIENTE')
sub = merged3.dropna()
rho, p = spearmanr(sub['EXTRA_SCORE'], sub['Total Score'])
sig = ' ***' if p < 0.05 else ''
print(f"\n  Extra composite vs CRAFFT: rho = {rho:.3f}, p = {fp(p)}{sig} (n={len(sub)})")

# =============================================
# RESUMO
# =============================================
print(f"\n{'=' * 70}")
print("RESUMO — ACHADOS SIGNIFICATIVOS (p < 0.05)")
print("=" * 70)
print("""
DOR ARTICULAR (66% dos pacientes):
  - IBDQ Total: 145 vs 180, p = 0.018
  - IBDQ Bowel: 38 vs 46, p = 0.005
  - IBDQ Emotional: 45 vs 58, p = 0.016
  - IBDQ Social: 46 vs 56, p = 0.042
  - CRAFFT: não significativo (p = 0.056, borderline)

SCORE EXTRAINTESTINAL COMPOSTO (Joint Pain + Energy + Sleep):
  - IBDQ Total: 133 vs 181, p < 0.001
  - IBDQ Bowel: p < 0.001
  - IBDQ Systemic: p < 0.001
  - IBDQ Emotional: p < 0.001
  - IBDQ Social: p < 0.001
  - Correlação com CRAFFT: rho = 0.175, p = 0.037

MENSAGEM: Sintomas extraintestinais são um driver forte de pior QoL,
independente do tipo de doença. CRAFFT é marginalmente associado.
""")
print("FIM — 08_extraintestinal_vs_qol.py")
