#!/usr/bin/env python3
"""
03_table1_statistics.py
Testes estatísticos e 95% CIs para Table 1.
Uso: python3 calculos/03_table1_statistics.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DC = os.path.join(BASE, 'data_clean')

pts = pd.read_csv(os.path.join(DC, 'todospacientes_clean.csv'))
inc = pts[pts['STATUS'] == 'INCLUÍDO']
crafft = pd.read_csv(os.path.join(DC, 'crafft_clean.csv'))
ibddisk = pd.read_csv(os.path.join(DC, 'ibddisk_clean.csv'))
ibdq = pd.read_csv(os.path.join(DC, 'ibdq_clean.csv'))
impact = pd.read_csv(os.path.join(DC, 'impact_clean.csv'))

def format_p(p):
    if p < 0.001: return '< 0.001'
    elif p < 0.01: return f'{p:.3f}'
    else: return f'{p:.2f}'

def bootstrap_ci_median(data, n_boot=10000, ci=95):
    data = data.dropna().values
    if len(data) < 3:
        return None, None
    medians = [np.median(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    return np.percentile(medians, (100-ci)/2), np.percentile(medians, 100-(100-ci)/2)

def wilson_ci(k, n, ci=0.95):
    if n == 0: return 0, 0
    z = stats.norm.ppf(1 - (1-ci)/2)
    p_hat = k / n
    denom = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denom
    spread = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denom
    return max(0, center - spread), min(1, center + spread)

print("=" * 70)
print("TABLE 1 — STATISTICAL TESTS & 95% CIs")
print("=" * 70)

# =============================================
# 1. SEX
# =============================================
print("\n--- SEXO (CD vs UC vs IBD-U) ---")
sex_known = inc[inc['SEXO'].notna()]
ct = pd.crosstab(sex_known['DIAGNOSTICO'], sex_known['SEXO'])
print(ct)
chi2, p_sex, dof, expected = chi2_contingency(ct)
print(f"Chi-squared: chi2={chi2:.2f}, df={dof}, p = {format_p(p_sex)}")
print(f"Expected counts:\n{pd.DataFrame(expected, index=ct.index, columns=ct.columns).round(1)}")
if (expected < 5).any():
    print("AVISO: expected < 5 em algumas celas — Fisher's exact recomendado")

print("\n95% CI (Wilson) para proporção de masculino:")
for d in ['CD', 'UC', 'IBD-U']:
    sub = sex_known[sex_known['DIAGNOSTICO'] == d]
    m = (sub['SEXO'] == 'M').sum()
    lo, hi = wilson_ci(m, len(sub))
    print(f"  {d}: {m}/{len(sub)} = {m/len(sub)*100:.1f}% (95% CI: {lo*100:.1f}–{hi*100:.1f}%)")

# =============================================
# 2. AGE AT DIAGNOSIS
# =============================================
print("\n--- IDADE AO DIAGNÓSTICO ---")
cd_age = inc[inc['DIAGNOSTICO'] == 'CD']['IDADE_DIAGNOSTICO'].dropna()
uc_age = inc[inc['DIAGNOSTICO'] == 'UC']['IDADE_DIAGNOSTICO'].dropna()
ibdu_age = inc[inc['DIAGNOSTICO'] == 'IBD-U']['IDADE_DIAGNOSTICO'].dropna()

h, p_kw = kruskal(cd_age, uc_age, ibdu_age)
print(f"Kruskal-Wallis (3 grupos): H={h:.2f}, p = {format_p(p_kw)}")

u, p_cu = mannwhitneyu(cd_age, uc_age, alternative='two-sided')
print(f"Mann-Whitney CD vs UC: U={u:.0f}, p = {format_p(p_cu)}")

print("\n95% CI (bootstrap) para mediana:")
for label, data in [('All', inc['IDADE_DIAGNOSTICO'].dropna()), ('CD', cd_age), ('UC', uc_age), ('IBD-U', ibdu_age)]:
    lo, hi = bootstrap_ci_median(data)
    if lo is not None:
        print(f"  {label}: median {data.median():.0f}, 95% CI [{lo:.0f}–{hi:.0f}]")

# =============================================
# 3. PARIS
# =============================================
print("\n--- CLASSIFICAÇÃO DE PARIS ---")
inc2 = inc.copy()
inc2['PARIS'] = inc2['IDADE_DIAGNOSTICO'].apply(lambda x: 'A1a' if x < 10 else ('A1b' if x < 17 else 'A2'))
ct_p = pd.crosstab(inc2['DIAGNOSTICO'], inc2['PARIS'])
print(ct_p)
chi2_p, p_paris, dof_p, exp_p = chi2_contingency(ct_p)
print(f"Chi-squared: chi2={chi2_p:.2f}, df={dof_p}, p = {format_p(p_paris)}")
if (exp_p < 5).any():
    print("AVISO: expected < 5 — interpretar com cautela")

# =============================================
# 4. CRAFFT
# =============================================
print("\n--- CRAFFT ---")
crafft_d = crafft[crafft['DIAGNOSTICO'].notna()].drop_duplicates('PACIENTE', keep='first')
cd_c = crafft_d[crafft_d['DIAGNOSTICO'] == 'CD']['Total Score']
uc_c = crafft_d[crafft_d['DIAGNOSTICO'] == 'UC']['Total Score']
ibdu_c = crafft_d[crafft_d['DIAGNOSTICO'] == 'IBD-U']['Total Score']

h_c, p_c = kruskal(cd_c, uc_c, ibdu_c)
print(f"Kruskal-Wallis score: H={h_c:.2f}, p = {format_p(p_c)}")

ct_cr = pd.crosstab(crafft_d['DIAGNOSTICO'], crafft_d['Risk Interpretation'])
chi2_cr, p_cr, _, _ = chi2_contingency(ct_cr)
print(f"Chi-squared positive screen: chi2={chi2_cr:.2f}, p = {format_p(p_cr)}")

print("\n95% CI (Wilson) para positive screen:")
for d in ['CD', 'UC', 'IBD-U']:
    sub = crafft_d[crafft_d['DIAGNOSTICO'] == d]
    pos = (sub['Risk Interpretation'] == 'Positive screen (≥2)').sum()
    lo, hi = wilson_ci(pos, len(sub))
    print(f"  {d}: {pos}/{len(sub)} = {pos/len(sub)*100:.1f}% (95% CI: {lo*100:.1f}–{hi*100:.1f}%)")

# =============================================
# 5. IBD-DISK
# =============================================
print("\n--- IBD-DISK ---")
ibddisk_d = ibddisk[ibddisk['DIAGNOSTICO'].notna()].drop_duplicates('PACIENTE', keep='first')
cd_id = ibddisk_d[ibddisk_d['DIAGNOSTICO'] == 'CD']['Total_IBD_Disk_Score']
uc_id = ibddisk_d[ibddisk_d['DIAGNOSTICO'] == 'UC']['Total_IBD_Disk_Score']
ibdu_id = ibddisk_d[ibddisk_d['DIAGNOSTICO'] == 'IBD-U']['Total_IBD_Disk_Score']

h_id, p_id = kruskal(cd_id, uc_id, ibdu_id)
print(f"Kruskal-Wallis: H={h_id:.2f}, p = {format_p(p_id)}")
u_id, p_id2 = mannwhitneyu(cd_id, uc_id, alternative='two-sided')
print(f"Mann-Whitney CD vs UC: U={u_id:.0f}, p = {format_p(p_id2)}")

print("\n95% CI (bootstrap) para mediana:")
for label, data in [('All', ibddisk_d['Total_IBD_Disk_Score']), ('CD', cd_id), ('UC', uc_id)]:
    lo, hi = bootstrap_ci_median(data)
    print(f"  {label}: median {data.median():.0f}, 95% CI [{lo:.0f}–{hi:.0f}]")

# =============================================
# 6. IBDQ
# =============================================
print("\n--- IBDQ (CD vs UC) ---")
ibdq_full = ibdq[(ibdq['DIAGNOSTICO'].notna()) & (ibdq['Item 32'].notna())]
ibdq_ff = ibdq_full.drop_duplicates('PACIENTE', keep='first')
cd_q = ibdq_ff[ibdq_ff['DIAGNOSTICO'] == 'CD']
uc_q = ibdq_ff[ibdq_ff['DIAGNOSTICO'] == 'UC']

for label, col in [('Total', 'Total'), ('Bowel', 'Sintomas Intestinais'),
                    ('Systemic', 'Sintomas Sistêmicos'), ('Emotional', 'Bem-Estar Emocional'),
                    ('Social', 'Interação Social')]:
    u, p = mannwhitneyu(cd_q[col], uc_q[col], alternative='two-sided')
    print(f"  {label}: U={u:.0f}, p = {format_p(p)}")

print("\n95% CI (bootstrap) para mediana IBDQ Total:")
for label, data in [('All', ibdq_ff['Total']), ('CD', cd_q['Total']), ('UC', uc_q['Total'])]:
    lo, hi = bootstrap_ci_median(data)
    print(f"  {label}: median {data.median():.0f}, 95% CI [{lo:.0f}–{hi:.0f}]")

# =============================================
# 7. IMPACT-III
# =============================================
print("\n--- IMPACT-III (CD vs UC) ---")
imp_v = impact[(impact['DIAGNOSTICO'].notna()) & (impact['Total'] > 0)]
imp_f = imp_v.drop_duplicates('PACIENTE', keep='first')
cd_i = imp_f[imp_f['DIAGNOSTICO'] == 'CD']['Total']
uc_i = imp_f[imp_f['DIAGNOSTICO'] == 'UC']['Total']

if len(uc_i) >= 2:
    u_i, p_i = mannwhitneyu(cd_i, uc_i, alternative='two-sided')
    print(f"  Total: U={u_i:.0f}, p = {format_p(p_i)}")
else:
    print("  UC n muito pequeno para comparação")

# =============================================
# RESUMO
# =============================================
print(f"\n{'=' * 70}")
print("RESUMO DOS P-VALUES PARA TABLE 1")
print(f"{'=' * 70}")
print(f"  Sexo (chi-squared):           p = {format_p(p_sex)}")
print(f"  Idade diagnóstico (KW):       p = {format_p(p_kw)}")
print(f"    CD vs UC (MW):              p = {format_p(p_cu)}")
print(f"  Paris (chi-squared):          p = {format_p(p_paris)}  ***")
print(f"  CRAFFT score (KW):            p = {format_p(p_c)}")
print(f"  CRAFFT positive (chi-sq):     p = {format_p(p_cr)}")
print(f"  IBD-Disk (KW):                p = {format_p(p_id)}")
print(f"    CD vs UC (MW):              p = {format_p(p_id2)}")

for label, col in [('IBDQ Total', 'Total'), ('IBDQ Bowel', 'Sintomas Intestinais'),
                    ('IBDQ Systemic', 'Sintomas Sistêmicos'), ('IBDQ Emotional', 'Bem-Estar Emocional'),
                    ('IBDQ Social', 'Interação Social')]:
    u, p = mannwhitneyu(cd_q[col], uc_q[col], alternative='two-sided')
    sig = '  ***' if p < 0.05 else '  *' if p < 0.1 else ''
    print(f"  {label} CD vs UC (MW):     p = {format_p(p)}{sig}")

if len(uc_i) >= 2:
    print(f"  IMPACT-III CD vs UC (MW):     p = {format_p(p_i)}")

print("\n*** p < 0.05  |  * p < 0.10")
print(f"\n{'=' * 70}")
print("FIM — 03_table1_statistics.py")
