#!/usr/bin/env python3
"""
09_sex_differences.py
Diferenças de sexo (M vs F) em todos os PROs.
Uso: python3 calculos/09_sex_differences.py
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact
import warnings
import os

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DC = os.path.join(BASE, 'data_clean')

pts = pd.read_csv(os.path.join(DC, 'todospacientes_clean.csv'))
inc = pts[pts['STATUS'] == 'INCLUÍDO']
sex_map = inc.set_index('NOME')['SEXO'].to_dict()

crafft = pd.read_csv(os.path.join(DC, 'crafft_clean.csv')).drop_duplicates('PACIENTE', keep='first')
ibddisk = pd.read_csv(os.path.join(DC, 'ibddisk_clean.csv')).drop_duplicates('PACIENTE', keep='first')
ibdq = pd.read_csv(os.path.join(DC, 'ibdq_clean.csv'))
ibdq_full = ibdq[ibdq['Item 32'].notna()].drop_duplicates('PACIENTE', keep='first').copy()
impact = pd.read_csv(os.path.join(DC, 'impact_clean.csv'))
impact_valid = impact[impact['Total'] > 0].drop_duplicates('PACIENTE', keep='first').copy()

for df in [crafft, ibddisk, ibdq_full, impact_valid]:
    df['SEXO'] = df['PACIENTE'].map(sex_map)

def fp(p):
    if p < 0.001: return '< 0.001'
    return f'{p:.4f}'

print("=" * 70)
print("DIFERENÇAS POR SEXO (M vs F) — TODOS OS PROs")
print("=" * 70)

# =============================================
# 1. IBD-DISK
# =============================================
print("\n" + "=" * 50)
print("1. IBD-DISK POR SEXO")
print("=" * 50)

m = ibddisk[ibddisk['SEXO'] == 'M']
f = ibddisk[ibddisk['SEXO'] == 'F']
print(f"M: n={len(m)}, F: n={len(f)}")

u, p = mannwhitneyu(m['Total_IBD_Disk_Score'], f['Total_IBD_Disk_Score'])
sig = ' ***' if p < 0.05 else ''
print(f"\n  Total: M {m['Total_IBD_Disk_Score'].median():.0f} (IQR {m['Total_IBD_Disk_Score'].quantile(0.25):.0f}-{m['Total_IBD_Disk_Score'].quantile(0.75):.0f}) vs F {f['Total_IBD_Disk_Score'].median():.0f} (IQR {f['Total_IBD_Disk_Score'].quantile(0.25):.0f}-{f['Total_IBD_Disk_Score'].quantile(0.75):.0f}), p = {fp(p)}{sig}")

disk_domains = {
    'Item1_Abdominal_Pain': 'Abdominal Pain',
    'Item2_Regulation_of_Defecation': 'Defecation',
    'Item3_Interpersonal_Interactions': 'Interpersonal',
    'Item4_Education_Work': 'Education/Work',
    'Item5_Sleep': 'Sleep',
    'Item6_Energy': 'Energy',
    'Item7_Emotions': 'Emotions',
    'Item8_Body_Image': 'Body Image',
    'Item9_Sexual_Function': 'Sexual Function',
    'Item10_Joint_Pain': 'Joint Pain',
}

print("\n  By domain:")
for col, label in disk_domains.items():
    if col in ibddisk.columns:
        m_v = m[col].dropna()
        f_v = f[col].dropna()
        if len(m_v) >= 3 and len(f_v) >= 3:
            u, p = mannwhitneyu(m_v, f_v)
            sig = ' ***' if p < 0.05 else ''
            if p < 0.05:
                print(f"    {label}: M {m_v.median():.0f} vs F {f_v.median():.0f}, p = {fp(p)}{sig}")

# Within CD
print("\n  Within CD:")
m_cd = ibddisk[(ibddisk['SEXO'] == 'M') & (ibddisk['DIAGNOSTICO'] == 'CD')]
f_cd = ibddisk[(ibddisk['SEXO'] == 'F') & (ibddisk['DIAGNOSTICO'] == 'CD')]

u, p = mannwhitneyu(m_cd['Total_IBD_Disk_Score'], f_cd['Total_IBD_Disk_Score'])
sig = ' ***' if p < 0.05 else ''
print(f"    Total: M {m_cd['Total_IBD_Disk_Score'].median():.0f} vs F {f_cd['Total_IBD_Disk_Score'].median():.0f}, p = {fp(p)}{sig} (n={len(m_cd)} vs {len(f_cd)})")

for col, label in disk_domains.items():
    if col in ibddisk.columns:
        m_v = m_cd[col].dropna()
        f_v = f_cd[col].dropna()
        if len(m_v) >= 3 and len(f_v) >= 3:
            u, p = mannwhitneyu(m_v, f_v)
            if p < 0.05:
                print(f"    {label}: M {m_v.median():.0f} vs F {f_v.median():.0f}, p = {fp(p)} ***")

# =============================================
# 2. IBDQ
# =============================================
print("\n" + "=" * 50)
print("2. IBDQ POR SEXO")
print("=" * 50)

m_q = ibdq_full[ibdq_full['SEXO'] == 'M']
f_q = ibdq_full[ibdq_full['SEXO'] == 'F']
print(f"M: n={len(m_q)}, F: n={len(f_q)}")

for label, col in [('Total', 'Total'), ('Bowel', 'Sintomas Intestinais'),
                    ('Systemic', 'Sintomas Sistêmicos'), ('Emotional', 'Bem-Estar Emocional'),
                    ('Social', 'Interação Social')]:
    u, p = mannwhitneyu(m_q[col], f_q[col])
    sig = ' ***' if p < 0.05 else ''
    print(f"  {label}: M {m_q[col].median():.0f} (IQR {m_q[col].quantile(0.25):.0f}-{m_q[col].quantile(0.75):.0f}) vs F {f_q[col].median():.0f} (IQR {f_q[col].quantile(0.25):.0f}-{f_q[col].quantile(0.75):.0f}), p = {fp(p)}{sig}")

# Within UC
print("\n  Within UC:")
m_uc = ibdq_full[(ibdq_full['SEXO'] == 'M') & (ibdq_full['DIAGNOSTICO'] == 'UC')]
f_uc = ibdq_full[(ibdq_full['SEXO'] == 'F') & (ibdq_full['DIAGNOSTICO'] == 'UC')]
print(f"  M: n={len(m_uc)}, F: n={len(f_uc)}")

for label, col in [('Total', 'Total'), ('Bowel', 'Sintomas Intestinais'),
                    ('Systemic', 'Sintomas Sistêmicos'), ('Emotional', 'Bem-Estar Emocional'),
                    ('Social', 'Interação Social')]:
    if len(m_uc) >= 3 and len(f_uc) >= 3:
        u, p = mannwhitneyu(m_uc[col], f_uc[col])
        sig = ' ***' if p < 0.05 else ''
        print(f"    {label}: M {m_uc[col].median():.0f} vs F {f_uc[col].median():.0f}, p = {fp(p)}{sig}")
    else:
        print(f"    {label}: n too small (M={len(m_uc)}, F={len(f_uc)})")

# =============================================
# 3. IMPACT-III
# =============================================
print("\n" + "=" * 50)
print("3. IMPACT-III POR SEXO")
print("=" * 50)

m_i = impact_valid[impact_valid['SEXO'] == 'M']
f_i = impact_valid[impact_valid['SEXO'] == 'F']
print(f"M: n={len(m_i)}, F: n={len(f_i)}")

u, p = mannwhitneyu(m_i['Total'], f_i['Total'])
sig = ' ***' if p < 0.05 else ''
print(f"\n  Total: M {m_i['Total'].median():.0f} vs F {f_i['Total'].median():.0f}, p = {fp(p)}{sig}")

for dom in ['Domínio: Sintomas', 'Domínio: Emocional', 'Domínio: Social', 'Domínio: Bem-estar']:
    u, p = mannwhitneyu(m_i[dom], f_i[dom])
    sig = ' ***' if p < 0.05 else ''
    if p < 0.05:
        print(f"  {dom}: M {m_i[dom].median():.0f} vs F {f_i[dom].median():.0f}, p = {fp(p)}{sig}")

# =============================================
# 4. CRAFFT
# =============================================
print("\n" + "=" * 50)
print("4. CRAFFT POR SEXO")
print("=" * 50)

m_c = crafft[crafft['SEXO'] == 'M']
f_c = crafft[crafft['SEXO'] == 'F']

u, p = mannwhitneyu(m_c['Total Score'], f_c['Total Score'])
print(f"  Score: M {m_c['Total Score'].median():.0f} vs F {f_c['Total Score'].median():.0f}, p = {fp(p)}")

m_pos = (m_c['Risk Interpretation'] == 'Positive screen (≥2)').sum()
f_pos = (f_c['Risk Interpretation'] == 'Positive screen (≥2)').sum()
ct = [[m_pos, len(m_c) - m_pos], [f_pos, len(f_c) - f_pos]]
or_val, p = fisher_exact(ct)
print(f"  Positive: M {m_pos}/{len(m_c)} ({m_pos/len(m_c)*100:.1f}%) vs F {f_pos}/{len(f_c)} ({f_pos/len(f_c)*100:.1f}%), OR={or_val:.2f}, p = {fp(p)}")

# =============================================
# RESUMO
# =============================================
print(f"\n{'=' * 70}")
print("RESUMO — ACHADOS SIGNIFICATIVOS POR SEXO (p < 0.05)")
print("=" * 70)
print("""
IBD-DISK (mulheres mais disability):
  - Total: F 44 vs M 31, p = 0.035
  - Emotions: F 8 vs M 4, p < 0.001
  - Energy: F 6 vs M 4, p = 0.012
  - Body Image: F 5 vs M 3, p = 0.030
  - Sexual Function: F 3 vs M 1, p = 0.017
  - Joint Pain: F 5 vs M 1, p = 0.018
  - Within CD total: F 47 vs M 30, p = 0.022

IBDQ:
  - Social: M 54 vs F 45, p = 0.035
  - Within UC — todos piores em mulheres:
    Total: F 98 vs M 154, p = 0.037
    Systemic: F 11 vs M 22, p = 0.037
    Social: F 27 vs M 50, p = 0.004
    Emotional: F 32 vs M 46, p = 0.050

IMPACT-III (meninas piores):
  - Total: F 88 vs M 66, p = 0.042
  - Sintomas: F 24 vs M 15, p = 0.022
  - Social: F 30 vs M 16, p = 0.029

CRAFFT: sem diferença por sexo
""")
print("FIM — 09_sex_differences.py")
