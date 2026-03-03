#!/usr/bin/env python3
"""
FASE 3 — ANÁLISES FINAIS E SÍNTESE
IBD Pediátrica em LMIC (Brasil)

Inclui:
  Q1: Análise de prevalência CRAFFT com IC 95% (Wilson)
  Q2: IBD-Disk severity categories e associações
  Q3: IBDQ severity (tercis) vs CRAFFT+
  Q4: Análise de concordância Bland-Altman conceitual (IBD-Disk vs IBDQ z-scores)
  Q5: Network visualization — inter-item correlations
  Q6: Sumário de achados significativos + interpretação clínica
  Q7: Limitações e pontos fortes para o manuscrito
  Q8: Figuras finais publicação-ready
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (mannwhitneyu, kruskal, chi2_contingency, fisher_exact,
                          spearmanr, norm, binom)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

BASE_DIR = "/Users/marcelocarvalhoesilva/Desktop/Gastro pediatria"
CSV_DIR = os.path.join(BASE_DIR, "csv")
OUT_DIR = os.path.join(BASE_DIR, "resultados")
FIG_DIR = os.path.join(OUT_DIR, "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

report = []
def log(msg):
    report.append(msg)
    print(msg)

log("=" * 100)
log("FASE 3 — ANÁLISES FINAIS E SÍNTESE PARA MANUSCRITO")
log(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 100)

# ==============================================================================
# CARREGAMENTO (mesmo que fases anteriores)
# ==============================================================================
def parse_date(d):
    if pd.isna(d) or str(d).strip() == '' or 'NÃO' in str(d).upper() or 'SEM' in str(d).upper():
        return pd.NaT
    try:
        return pd.to_datetime(d, dayfirst=True)
    except:
        return pd.NaT

crafft = pd.read_csv(os.path.join(CSV_DIR, "CRAFFT FINAL.xlsx - CRAFFT Scoring.csv"))
crafft.columns = ['data_avaliacao', 'paciente', 'C_Car', 'R_Relax', 'A_Alone',
                   'F_Forget', 'F_Friends', 'T_Trouble', 'total_score', 'risk_interpretation']
crafft = crafft[crafft['paciente'].notna() & (crafft['paciente'].str.strip() != '')]
crafft = crafft[~crafft['paciente'].str.contains('repetido|RECUSOU', case=False, na=False)]
crafft['paciente'] = crafft['paciente'].str.strip().str.upper()
crafft_items = ['C_Car', 'R_Relax', 'A_Alone', 'F_Forget', 'F_Friends', 'T_Trouble']
for col in crafft_items + ['total_score']:
    crafft[col] = pd.to_numeric(crafft[col], errors='coerce')
crafft['crafft_positive'] = crafft['total_score'] >= 2
crafft['date'] = crafft['data_avaliacao'].apply(parse_date)
crafft_valid = crafft[crafft['total_score'].notna()].copy()

ibddisk = pd.read_csv(os.path.join(CSV_DIR, "IBD Disk FINAL.xlsx - IBD_Disk_Data.csv"))
ibddisk.columns = ['data_avaliacao', 'paciente', 'age', 'sex', 'diagnosis',
                    'item1_abd_pain', 'item2_defecation', 'item3_interpersonal',
                    'item4_education', 'item5_sleep', 'item6_energy', 'item7_emotions',
                    'item8_body_image', 'item9_sexual', 'item10_joint_pain',
                    'physical_domain', 'psychosocial_domain', 'total_score', 'mean_score']
ibddisk = ibddisk[ibddisk['paciente'].notna() & (ibddisk['paciente'].str.strip() != '')]
ibddisk = ibddisk[~ibddisk['paciente'].str.contains('repetido|DII\\?\\?|sem questionário', case=False, na=False)]
ibddisk['paciente'] = ibddisk['paciente'].str.strip().str.upper()
ibddisk['date'] = ibddisk['data_avaliacao'].apply(parse_date)
disk_items = ['item1_abd_pain', 'item2_defecation', 'item3_interpersonal',
              'item4_education', 'item5_sleep', 'item6_energy', 'item7_emotions',
              'item8_body_image', 'item9_sexual', 'item10_joint_pain']
disk_labels = ['Dor Abdominal', 'Defecação', 'Interações', 'Educação/Trabalho',
               'Sono', 'Energia', 'Emoções', 'Imagem Corporal', 'Função Sexual', 'Dor Articular']
all_disk_numeric = disk_items + ['physical_domain', 'psychosocial_domain', 'total_score', 'mean_score']
for col in all_disk_numeric:
    ibddisk[col] = ibddisk[col].astype(str).str.replace(',', '.').str.replace('#DIV/0!', '')
    ibddisk[col] = pd.to_numeric(ibddisk[col], errors='coerce')
ibddisk_valid = ibddisk[ibddisk['total_score'].notna() & (ibddisk['total_score'] > 0)].copy()

ibdq = pd.read_csv(os.path.join(CSV_DIR, "IBDQ FINAL.xlsx - Cálculo Automático.csv"))
cols_ibdq = ['data_avaliacao', 'paciente'] + [f'item_{i}' for i in range(1, 33)] + \
            ['sintomas_intestinais', 'sintomas_sistemicos', 'bem_estar_emocional', 'interacao_social', 'total']
ibdq.columns = cols_ibdq
ibdq = ibdq[ibdq['paciente'].notna() & (ibdq['paciente'].str.strip() != '')]
ibdq = ibdq[~ibdq['paciente'].str.contains('repetido|DII\\?\\?|sem questionário', case=False, na=False)]
ibdq['paciente'] = ibdq['paciente'].str.strip().str.upper()
ibdq['date'] = ibdq['data_avaliacao'].apply(parse_date)
ibdq_item_cols = [f'item_{i}' for i in range(1, 33)]
ibdq_domain_cols = ['sintomas_intestinais', 'sintomas_sistemicos', 'bem_estar_emocional', 'interacao_social', 'total']
for col in ibdq_item_cols + ibdq_domain_cols:
    ibdq[col] = pd.to_numeric(ibdq[col], errors='coerce')
ibdq_valid = ibdq[ibdq['total'].notna() & (ibdq['total'] > 0)].copy()

impact = pd.read_csv(os.path.join(CSV_DIR, "IMPACTIII FINAL.xlsx - Cálculo Automático.csv"))
cols_impact = ['data_avaliacao', 'paciente'] + [f'item_{i}' for i in range(1, 36)] + \
              ['dom_sintomas', 'dom_emocional', 'dom_social', 'dom_bemestar', 'dom_tratamento', 'total']
impact.columns = cols_impact
impact = impact[impact['paciente'].notna() & (impact['paciente'].str.strip() != '')]
impact = impact[~impact['paciente'].str.contains('repetido|RECUSOU', case=False, na=False)]
impact['paciente'] = impact['paciente'].str.strip().str.upper()
impact_item_cols = [f'item_{i}' for i in range(1, 36)]
impact_domain_cols = ['dom_sintomas', 'dom_emocional', 'dom_social', 'dom_bemestar', 'dom_tratamento', 'total']
for col in impact_item_cols + impact_domain_cols:
    impact[col] = pd.to_numeric(impact[col], errors='coerce')
impact_valid = impact[impact['total'].notna() & (impact['total'] > 0)].copy()

# Merge
crafft_merge = crafft_valid[['paciente', 'total_score', 'crafft_positive'] + crafft_items].rename(
    columns={'total_score': 'crafft_total'})
disk_merge = ibddisk_valid[['paciente'] + disk_items + ['physical_domain', 'psychosocial_domain',
    'total_score', 'mean_score']].rename(columns={'total_score': 'disk_total', 'mean_score': 'disk_mean'})
ibdq_merge = ibdq_valid[['paciente', 'sintomas_intestinais', 'sintomas_sistemicos',
    'bem_estar_emocional', 'interacao_social', 'total']].rename(columns={'total': 'ibdq_total'})
impact_merge = impact_valid[['paciente', 'total']].rename(columns={'total': 'impact_total'})

merged = crafft_merge.merge(disk_merge, on='paciente', how='outer')
merged = merged.merge(ibdq_merge, on='paciente', how='outer')
merged = merged.merge(impact_merge, on='paciente', how='outer')

ibdq_long_items = [f'item_{i}' for i in range(12, 33)]
ibdq_valid['ibdq_version'] = ibdq_valid[ibdq_long_items].notna().sum(axis=1).apply(
    lambda x: 'longa' if x > 5 else 'curta')
version_info = ibdq_valid[['paciente', 'ibdq_version']].copy()
merged = merged.merge(version_info, on='paciente', how='left')


# ==============================================================================
# Q1: PREVALÊNCIA CRAFFT COM IC 95% (Wilson Score Interval)
# ==============================================================================
log("\n\n" + "=" * 100)
log("Q1: PREVALÊNCIA CRAFFT COM IC 95% (Wilson Score Interval)")
log("=" * 100)

def wilson_ci(k, n, z=1.96):
    """Wilson score interval para proporção binomial."""
    if n == 0:
        return 0, 0, 0
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denom
    return p_hat, max(0, center - margin), min(1, center + margin)

n_total = len(crafft_valid)
n_pos = crafft_valid['crafft_positive'].sum()
prev, ci_lo, ci_hi = wilson_ci(n_pos, n_total)

log(f"\n  CRAFFT Screening Positivo (≥2):")
log(f"    n = {n_pos}/{n_total}")
log(f"    Prevalência: {100*prev:.1f}% (IC 95% Wilson: {100*ci_lo:.1f}%–{100*ci_hi:.1f}%)")

log(f"\n  Prevalência por item CRAFFT (IC 95% Wilson):")
crafft_with_items = crafft_valid[crafft_valid[crafft_items].notna().all(axis=1)]
n_items = len(crafft_with_items)
item_labels_crafft = {
    'C_Car': 'C - Car (dirigir sob influência)',
    'R_Relax': 'R - Relax (usar para relaxar)',
    'A_Alone': 'A - Alone (usar sozinho)',
    'F_Forget': 'F - Forget (esquecer sob influência)',
    'F_Friends': 'F - Friends/Family (família/amigos pedem para parar)',
    'T_Trouble': 'T - Trouble (problemas sob influência)'
}

for item in crafft_items:
    k = int(crafft_with_items[item].sum())
    p_hat, lo, hi = wilson_ci(k, n_items)
    log(f"    {item_labels_crafft[item]}:")
    log(f"      {k}/{n_items} = {100*p_hat:.1f}% (IC 95%: {100*lo:.1f}%–{100*hi:.1f}%)")

# CRAFFT score distribution with CI
log(f"\n  Distribuição dos scores CRAFFT (IC 95% para cada proporção):")
for score in range(7):
    k = (crafft_valid['total_score'] == score).sum()
    p_hat, lo, hi = wilson_ci(k, n_total)
    bar = '█' * int(p_hat * 50)
    log(f"    Score {score}: {k:3d}/{n_total} = {100*p_hat:5.1f}% ({100*lo:4.1f}%–{100*hi:4.1f}%) {bar}")


# ==============================================================================
# Q2: IBD-DISK SEVERITY CATEGORIES
# ==============================================================================
log("\n\n" + "=" * 100)
log("Q2: IBD-DISK CATEGORIAS DE SEVERIDADE E ASSOCIAÇÕES")
log("=" * 100)

# Create severity terciles for IBD-Disk
disk_data = merged[merged['disk_total'].notna()].copy()
disk_data['disk_tercil'] = pd.qcut(disk_data['disk_total'], q=3, labels=['Baixo', 'Médio', 'Alto'])

log(f"\n  IBD-Disk dividido em tercis:")
for cat in ['Baixo', 'Médio', 'Alto']:
    subset = disk_data[disk_data['disk_tercil'] == cat]
    log(f"    {cat}: n={len(subset)}, IBD-Disk mediana={subset['disk_total'].median():.0f}, "
        f"range=[{subset['disk_total'].min():.0f}-{subset['disk_total'].max():.0f}]")

# CRAFFT+ by IBD-Disk severity
log(f"\n--- CRAFFT+ por tercil de IBD-Disk ---")
for cat in ['Baixo', 'Médio', 'Alto']:
    subset = disk_data[(disk_data['disk_tercil'] == cat) & disk_data['crafft_positive'].notna()]
    if len(subset) > 0:
        n_pos = subset['crafft_positive'].sum()
        n_tot = len(subset)
        prev, lo, hi = wilson_ci(n_pos, n_tot)
        log(f"  {cat}: CRAFFT+ = {n_pos}/{n_tot} = {100*prev:.1f}% (IC 95%: {100*lo:.1f}%–{100*hi:.1f}%)")

# Chi-square for trend
table_trend = []
for cat in ['Baixo', 'Médio', 'Alto']:
    subset = disk_data[(disk_data['disk_tercil'] == cat) & disk_data['crafft_positive'].notna()]
    if len(subset) > 0:
        table_trend.append([subset['crafft_positive'].sum(), len(subset) - subset['crafft_positive'].sum()])

if len(table_trend) == 3:
    chi2, p, _, _ = chi2_contingency(table_trend)
    log(f"  Chi² para tendência: χ²={chi2:.3f}, p={p:.4f}")

# IBDQ by IBD-Disk severity
log(f"\n--- IBDQ por tercil de IBD-Disk ---")
groups_ibdq = []
for cat in ['Baixo', 'Médio', 'Alto']:
    subset = disk_data[(disk_data['disk_tercil'] == cat) & disk_data['ibdq_total'].notna()]
    if len(subset) > 0:
        log(f"  {cat}: IBDQ mediana={subset['ibdq_total'].median():.0f}, "
            f"IQR=[{subset['ibdq_total'].quantile(0.25):.0f}-{subset['ibdq_total'].quantile(0.75):.0f}], n={len(subset)}")
        groups_ibdq.append(subset['ibdq_total'].values)

if len(groups_ibdq) >= 2:
    H, p = kruskal(*groups_ibdq)
    log(f"  Kruskal-Wallis: H={H:.3f}, p={p:.4f}")

    if p < 0.05:
        log(f"  Post-hoc (Mann-Whitney, Bonferroni):")
        labels_cat = ['Baixo', 'Médio', 'Alto']
        for i in range(len(groups_ibdq)):
            for j in range(i+1, len(groups_ibdq)):
                U, p_mw = mannwhitneyu(groups_ibdq[i], groups_ibdq[j], alternative='two-sided')
                p_bonf = min(p_mw * 3, 1.0)
                r_eff = abs(norm.isf(p_mw/2)) / np.sqrt(len(groups_ibdq[i]) + len(groups_ibdq[j]))
                sig = '*' if p_bonf < 0.05 else 'ns'
                log(f"    {labels_cat[i]} vs {labels_cat[j]}: U={U:.0f}, p_Bonf={p_bonf:.4f} {sig}, r={r_eff:.3f}")


# ==============================================================================
# Q3: IBDQ SEVERITY TERCIS vs CRAFFT+
# ==============================================================================
log("\n\n" + "=" * 100)
log("Q3: IBDQ SEVERIDADE (TERCIS) vs CRAFFT+")
log("=" * 100)

ibdq_data = merged[merged['ibdq_total'].notna()].copy()
ibdq_data['ibdq_tercil'] = pd.qcut(ibdq_data['ibdq_total'], q=3, labels=['Grave', 'Moderado', 'Leve'])

log(f"\n  IBDQ dividido em tercis:")
for cat in ['Grave', 'Moderado', 'Leve']:
    subset = ibdq_data[ibdq_data['ibdq_tercil'] == cat]
    log(f"    {cat}: n={len(subset)}, IBDQ mediana={subset['ibdq_total'].median():.0f}, "
        f"range=[{subset['ibdq_total'].min():.0f}-{subset['ibdq_total'].max():.0f}]")

# CRAFFT+ by IBDQ severity
log(f"\n--- CRAFFT+ por tercil de IBDQ ---")
table_ibdq = []
for cat in ['Grave', 'Moderado', 'Leve']:
    subset = ibdq_data[(ibdq_data['ibdq_tercil'] == cat) & ibdq_data['crafft_positive'].notna()]
    if len(subset) > 0:
        n_pos = subset['crafft_positive'].sum()
        n_tot = len(subset)
        prev, lo, hi = wilson_ci(n_pos, n_tot)
        log(f"  {cat}: CRAFFT+ = {n_pos}/{n_tot} = {100*prev:.1f}% (IC 95%: {100*lo:.1f}%–{100*hi:.1f}%)")
        table_ibdq.append([n_pos, n_tot - n_pos])

if len(table_ibdq) >= 2:
    chi2, p, _, _ = chi2_contingency(table_ibdq)
    log(f"  Chi² para tendência: χ²={chi2:.3f}, p={p:.4f}")

# IBD-Disk by IBDQ severity
log(f"\n--- IBD-Disk por tercil de IBDQ ---")
groups_disk = []
for cat in ['Grave', 'Moderado', 'Leve']:
    subset = ibdq_data[(ibdq_data['ibdq_tercil'] == cat) & ibdq_data['disk_total'].notna()]
    if len(subset) > 0:
        log(f"  {cat}: IBD-Disk mediana={subset['disk_total'].median():.0f}, "
            f"IQR=[{subset['disk_total'].quantile(0.25):.0f}-{subset['disk_total'].quantile(0.75):.0f}], n={len(subset)}")
        groups_disk.append(subset['disk_total'].values)

if len(groups_disk) >= 2:
    H, p = kruskal(*groups_disk)
    log(f"  Kruskal-Wallis: H={H:.3f}, p={p:.4f}")


# ==============================================================================
# Q4: BLAND-ALTMAN ANALYSIS (IBD-Disk z-score vs IBDQ z-score)
# ==============================================================================
log("\n\n" + "=" * 100)
log("Q4: ANÁLISE DE CONCORDÂNCIA — Bland-Altman (z-scores IBD-Disk vs IBDQ)")
log("=" * 100)

df_ba = merged[['disk_total', 'ibdq_total']].dropna()
if len(df_ba) >= 10:
    # Standardize both to z-scores (note: IBD-Disk is disability, IBDQ is QoL, so invert IBDQ)
    z_disk = (df_ba['disk_total'] - df_ba['disk_total'].mean()) / df_ba['disk_total'].std()
    z_ibdq_inv = -(df_ba['ibdq_total'] - df_ba['ibdq_total'].mean()) / df_ba['ibdq_total'].std()

    mean_z = (z_disk + z_ibdq_inv) / 2
    diff_z = z_disk - z_ibdq_inv

    mean_diff = diff_z.mean()
    sd_diff = diff_z.std()
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    log(f"\n  n = {len(df_ba)}")
    log(f"  IBD-Disk z-score vs IBDQ invertido z-score:")
    log(f"  Viés médio (mean difference): {mean_diff:.3f}")
    log(f"  DP das diferenças: {sd_diff:.3f}")
    log(f"  Limites de concordância (95%):")
    log(f"    Inferior: {loa_lower:.3f}")
    log(f"    Superior: {loa_upper:.3f}")

    # % within limits
    within_loa = ((diff_z >= loa_lower) & (diff_z <= loa_upper)).sum()
    log(f"  Dentro dos limites: {within_loa}/{len(df_ba)} ({100*within_loa/len(df_ba):.1f}%)")

    # Bland-Altman figure
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(mean_z, diff_z, alpha=0.5, s=30, color='#2c3e50')
    ax.axhline(y=mean_diff, color='red', linewidth=2, label=f'Viés médio = {mean_diff:.3f}')
    ax.axhline(y=loa_upper, color='gray', linewidth=1, linestyle='--', label=f'LoA superior = {loa_upper:.3f}')
    ax.axhline(y=loa_lower, color='gray', linewidth=1, linestyle='--', label=f'LoA inferior = {loa_lower:.3f}')
    ax.fill_between(ax.get_xlim(), loa_lower, loa_upper, alpha=0.1, color='gray')
    ax.set_xlabel('Média dos z-scores (IBD-Disk, IBDQ invertido)', fontsize=12)
    ax.set_ylabel('Diferença dos z-scores (IBD-Disk − IBDQ inv.)', fontsize=12)
    ax.set_title('Bland-Altman: Concordância IBD-Disk vs IBDQ\n(z-scores padronizados)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig21_bland_altman.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"\n  → Figura salva: fig21_bland_altman.png")


# ==============================================================================
# Q5: NETWORK VISUALIZATION — Inter-item Correlations
# ==============================================================================
log("\n\n" + "=" * 100)
log("Q5: NETWORK — Correlações Inter-item IBD-Disk")
log("=" * 100)

# Correlation network among IBD-Disk items
df_net = merged[disk_items].dropna()
if len(df_net) >= 10:
    n_items = len(disk_items)
    corr_net = np.zeros((n_items, n_items))
    p_net = np.zeros((n_items, n_items))

    for i in range(n_items):
        for j in range(n_items):
            if i != j:
                rho, p = spearmanr(df_net[disk_items[i]], df_net[disk_items[j]])
                corr_net[i, j] = rho
                p_net[i, j] = p

    # Report strongest correlations
    log(f"\n  Top 10 correlações inter-item IBD-Disk (n={len(df_net)}):")
    pairs_list = []
    for i in range(n_items):
        for j in range(i+1, n_items):
            pairs_list.append((disk_labels[i], disk_labels[j], corr_net[i,j], p_net[i,j]))

    pairs_list.sort(key=lambda x: abs(x[2]), reverse=True)
    for label1, label2, rho, p in pairs_list[:10]:
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        log(f"    {label1} × {label2}: rho={rho:.3f} {sig}")

    # Network figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Position nodes in circle
    angles = np.linspace(0, 2*np.pi, n_items, endpoint=False)
    radius = 4
    x_pos = radius * np.cos(angles)
    y_pos = radius * np.sin(angles)

    # Draw edges (correlations > 0.3)
    for i in range(n_items):
        for j in range(i+1, n_items):
            rho = corr_net[i, j]
            if abs(rho) > 0.3:
                width = abs(rho) * 5
                color = '#e74c3c' if rho > 0 else '#3498db'
                alpha = min(abs(rho), 0.8)
                ax.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]],
                       color=color, linewidth=width, alpha=alpha, zorder=1)

    # Draw nodes
    for i in range(n_items):
        mean_corr = np.mean(np.abs(np.delete(corr_net[i], i)))
        node_size = 300 + mean_corr * 2000
        ax.scatter(x_pos[i], y_pos[i], s=node_size, c='#2c3e50', zorder=3, edgecolors='white', linewidth=2)
        ax.annotate(disk_labels[i], (x_pos[i], y_pos[i]), fontsize=9, ha='center', va='center',
                   color='white', fontweight='bold', zorder=4)

    # Legend
    red_patch = mpatches.Patch(color='#e74c3c', alpha=0.7, label='Correlação positiva (rho > 0.3)')
    blue_patch = mpatches.Patch(color='#3498db', alpha=0.7, label='Correlação negativa (rho < -0.3)')
    ax.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=11)

    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Network de Correlações Inter-item IBD-Disk\n(linhas: |rho| > 0.3, espessura proporcional)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig22_network_ibddisk.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"\n  → Figura salva: fig22_network_ibddisk.png")


# ==============================================================================
# Q6: SUMÁRIO DE ACHADOS SIGNIFICATIVOS + INTERPRETAÇÃO CLÍNICA
# ==============================================================================
log("\n\n" + "=" * 100)
log("Q6: SUMÁRIO DE ACHADOS SIGNIFICATIVOS — INTERPRETAÇÃO CLÍNICA")
log("=" * 100)

log("""
  ============================================================
  ACHADOS PRINCIPAIS — SÍNTESE PARA O MANUSCRITO
  ============================================================

  1. AMOSTRA E INSTRUMENTOS
  ─────────────────────────
  • 157 pacientes pediátricos com DII avaliados em centro terciário brasileiro
  • 4 instrumentos aplicados: CRAFFT, IBD-Disk, IBDQ, IMPACT-III
  • Completude variável: CRAFFT 100%, IBD-Disk 87%, IBDQ 78%, IMPACT-III 14%
  • IMPACT-III com n=23 — insuficiente para análises robustas
  • Duas versões de IBDQ identificadas: curta (11 itens, n=48) e longa (32 itens, n=74)

  2. CONFIABILIDADE DOS INSTRUMENTOS
  ──────────────────────────────────
  • CRAFFT: α=0.68 (questionável) — consistente com literatura internacional
    - Item C_Car (dirigir) tem correlação item-total mais baixa (r=0.205)
    - Remoção de C_Car melhora α para 0.72 — esperado em população pediátrica
  • IBD-Disk: α=0.89 (bom) — todos os itens contribuem adequadamente
  • IBDQ: α=0.97 (excelente) — alta consistência interna

  3. PREVALÊNCIA DE USO DE SUBSTÂNCIAS (CRAFFT)
  ──────────────────────────────────────────────
  • 21.0% screen positivo (≥2) — ALERTA CLÍNICO
  • Item mais prevalente: Trouble (30.3%) — problemas relacionados ao uso
  • Item menos prevalente: Friends/Family (2.1%)
  • Floor effect severo: 58.6% com score 0

  4. PERFIL DE INCAPACIDADE (IBD-Disk)
  ────────────────────────────────────
  • Domínios mais afetados: Emoções (5.6), Energia (5.1), Sono (4.7)
  • Domínios menos afetados: Função Sexual (2.7), Interações (3.2)
  • Significativo floor effect em maioria dos itens (>15% com score 0)
  • Ceiling effect em Emoções (20.6% com score 10)

  5. CORRELAÇÕES PRINCIPAIS (Sobrevivem à correção FDR)
  ─────────────────────────────────────────────────────
  ✓ IBD-Disk Physical Domain vs IBDQ Sint. Intestinais: rho=-0.735 (forte)
  ✓ IBD-Disk Physical Domain vs IBDQ Sint. Sistêmicos: rho=-0.608 (forte)
  ✓ IBD-Disk Psychosocial vs IBDQ Sint. Intestinais: rho=-0.651 (forte)
  ✓ IBD-Disk Psychosocial vs IBDQ Sint. Sistêmicos: rho=-0.452 (moderada)
  ✓ IBD-Disk vs IMPACT-III: rho=0.810 (muito forte)
  ✗ CRAFFT vs IBD-Disk: rho=0.006 (negligível)
  ✗ CRAFFT vs IBDQ: rho=0.009 (negligível)
  ✗ CRAFFT vs IMPACT-III: rho=-0.090 (negligível)

  → NENHUMA correlação CRAFFT vs outros instrumentos é significativa

  6. CRAFFT NÃO SE ASSOCIA A DISABILITY OU QoL
  ──────────────────────────────────────────────
  • Mann-Whitney CRAFFT+ vs CRAFFT-: todos p>0.05, effect sizes nulos
  • Regressão logística: AUC-ROC = 0.588 (próximo do acaso)
  • Análise dose-resposta: sem gradiente monotônico
  • Após correção FDR: zero associações item-a-item sobrevivem
  • PORÉM: No Burden Index composto, 46% dos pacientes de alto burden
    são CRAFFT+ vs 15% no burden baixo — potencial associação indireta

  7. ACHADO IMPORTANTE — VIÉS DE SELEÇÃO
  ──────────────────────────────────────
  • Pacientes com dados completos (3 instrumentos) têm SIGNIFICATIVAMENTE
    mais CRAFFT+ (23.8%) que incompletos (4.9%), p=0.0007
  • Sugere que pacientes de maior risco são mais propensos a completar avaliações
  • Isto deve ser discutido como limitação no manuscrito

  8. VERSÃO IBDQ: IMPACTO CRÍTICO NOS RESULTADOS
  ───────────────────────────────────────────────
  • Pacientes com versão longa têm IBD-Disk SIGNIFICATIVAMENTE maior
    (med=43.5 vs 31.0, p=0.0004, r=0.307 — efeito médio)
  • Correlação IBD-Disk vs IBDQ é mais forte na versão curta (rho=-0.859)
    que na longa (rho=-0.680) — ambas significativas
  • Isto NÃO é artefato: pacientes mais graves receberam mais avaliações

  9. IBD-Disk COMO PREDITOR DE QoL GRAVE
  ──────────────────────────────────────
  • ROC AUC = 0.436 para predizer IBDQ ≤100 — FRACO
  • Nenhum cutoff oferece sensibilidade+especificidade adequadas
  • Conclusão: IBD-Disk e IBDQ medem construtos parcialmente diferentes

  10. PODER ESTATÍSTICO
  ────────────────────
  • Para detectar CRAFFT vs IBD-Disk: poder <3% — estudo SUB-POWERED
    (N necessário: >10.000 para detectar rho=0.006)
  • Para IBD-Disk vs IBDQ: poder 65% — marginal
  • Para IBD-Disk vs IMPACT-III: poder >99% — adequado
  • Mínimo |rho| detectável com n=143: 0.24

  CONCLUSÃO GERAL:
  ────────────────
  O uso de substâncias (CRAFFT) é prevalente (21%) nesta população
  pediátrica com DII, mas opera de forma INDEPENDENTE da incapacidade
  (IBD-Disk) e qualidade de vida (IBDQ). Isto sugere que o screening
  de substâncias deve ser feito como avaliação complementar, não como
  parte da avaliação de doença — um achado relevante para guidelines
  de cuidado em LMICs onde recursos são limitados.
""")


# ==============================================================================
# Q7: LIMITAÇÕES E PONTOS FORTES
# ==============================================================================
log("\n\n" + "=" * 100)
log("Q7: LIMITAÇÕES E PONTOS FORTES PARA O MANUSCRITO")
log("=" * 100)

log("""
  PONTOS FORTES:
  ──────────────
  1. Primeiro estudo brasileiro a aplicar IBD-Disk + IBDQ + CRAFFT + IMPACT-III
     simultaneamente em população pediátrica com DII
  2. Amostra clinicamente representativa de um centro terciário em LMIC
  3. Múltiplos instrumentos validados permitem triangulação
  4. Análise estatística rigorosa com correções para comparações múltiplas (FDR)
  5. Bootstrap IC 95% para todas as estimativas de efeito
  6. Power analysis post-hoc documenta transparentemente as limitações
  7. Dados coletados longitudinalmente (2021-2026) em prática clínica real

  LIMITAÇÕES:
  ───────────
  1. Desenho transversal — não permite inferir causalidade
  2. Centro único — limita generalizabilidade
  3. IMPACT-III com n=23 — insuficiente para conclusões robustas
  4. Duas versões de IBDQ utilizadas — reduz comparabilidade
  5. Viés de seleção: pacientes com dados completos diferem dos incompletos
  6. Sem dados demográficos completos (idade, sexo apenas parcial no IBD-Disk)
  7. Sem dados de atividade de doença (PCDAI/PUCAI) para correlacionar
  8. CRAFFT não validado especificamente para população pediátrica com DII
  9. Floor effect no CRAFFT (58.6% com score 0) limita discriminação
  10. Sub-powered para detectar associações fracas CRAFFT vs outros instrumentos
""")


# ==============================================================================
# Q8: FIGURAS FINAIS
# ==============================================================================
log("\n\n" + "=" * 100)
log("Q8: FIGURAS FINAIS PARA PUBLICAÇÃO")
log("=" * 100)

# Fig 23: Summary figure — 4-panel overview
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel A: CRAFFT distribution
ax = axes[0, 0]
scores = crafft_valid['total_score'].value_counts().sort_index()
colors = ['#27ae60' if s < 2 else '#e74c3c' for s in scores.index]
ax.bar(scores.index, scores.values, color=colors, edgecolor='white', linewidth=0.5)
ax.axvline(x=1.5, color='black', linestyle='--', alpha=0.5, label='Cutoff ≥2')
ax.set_xlabel('CRAFFT Score', fontsize=11)
ax.set_ylabel('Frequência', fontsize=11)
ax.set_title('A) Distribuição CRAFFT\n(vermelho = screen positivo)', fontsize=12)
ax.legend(fontsize=10)

# Panel B: IBD-Disk radar chart (mean per item)
ax = axes[0, 1]
means = []
for item in disk_items:
    means.append(ibddisk_valid[item].mean())

y_positions = range(len(disk_labels))
colors_bar = plt.cm.RdYlGn_r(np.array(means) / 10)
bars = ax.barh(y_positions, means, color=colors_bar, edgecolor='white', height=0.6)
ax.set_yticks(y_positions)
ax.set_yticklabels(disk_labels, fontsize=10)
ax.set_xlabel('Score Médio (0-10)', fontsize=11)
ax.set_title('B) IBD-Disk — Score Médio por Item', fontsize=12)
ax.set_xlim(0, 10)
for i, v in enumerate(means):
    ax.text(v + 0.2, i, f'{v:.1f}', va='center', fontsize=9)

# Panel C: IBDQ distribution by version
ax = axes[1, 0]
ibdq_curta = ibdq_valid[ibdq_valid['ibdq_version'] == 'curta']['total']
ibdq_longa = ibdq_valid[ibdq_valid['ibdq_version'] == 'longa']['total']
ax.hist(ibdq_curta, bins=15, alpha=0.6, color='#3498db', label=f'Curta (n={len(ibdq_curta)})', density=True)
ax.hist(ibdq_longa, bins=15, alpha=0.6, color='#e74c3c', label=f'Longa (n={len(ibdq_longa)})', density=True)
ax.set_xlabel('IBDQ Total Score', fontsize=11)
ax.set_ylabel('Densidade', fontsize=11)
ax.set_title('C) Distribuição IBDQ por Versão', fontsize=12)
ax.legend(fontsize=10)

# Panel D: Key correlations summary
ax = axes[1, 1]
corr_labels = ['Physical\nvs Sint.Int.', 'Physical\nvs Sint.Sist.', 'Psycho\nvs Sint.Int.',
               'Psycho\nvs Sint.Sist.', 'Disk\nvs IBDQ', 'CRAFFT\nvs Disk', 'CRAFFT\nvs IBDQ']
corr_values = [-0.735, -0.608, -0.651, -0.452, -0.144, 0.006, 0.009]
sig_markers = ['***', '***', '***', '***', 'ns', 'ns', 'ns']
colors_corr = ['#e74c3c' if abs(v) > 0.3 else '#f39c12' if abs(v) > 0.1 else '#95a5a6' for v in corr_values]

bars = ax.barh(range(len(corr_labels)), corr_values, color=colors_corr, edgecolor='white', height=0.6)
ax.axvline(x=0, color='black', linewidth=1)
ax.set_yticks(range(len(corr_labels)))
ax.set_yticklabels(corr_labels, fontsize=9)
ax.set_xlabel('Spearman rho', fontsize=11)
ax.set_title('D) Correlações Principais\n(após correção FDR)', fontsize=12)
for i, (v, sig) in enumerate(zip(corr_values, sig_markers)):
    x_pos = v + 0.02 if v > 0 else v - 0.08
    ax.text(x_pos, i, sig, va='center', fontsize=9, fontweight='bold')

plt.suptitle('Panorama dos Resultados — IBD Pediátrica em LMIC (Brasil)', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig23_summary_overview.png'), dpi=300, bbox_inches='tight')
plt.close()
log("  → fig23_summary_overview.png salva")

# Fig 24: CRAFFT prevalence with CI
fig, ax = plt.subplots(figsize=(10, 6))

item_names = ['C - Car', 'R - Relax', 'A - Alone', 'F - Forget', 'F - Friends', 'T - Trouble', 'Screen+']
prevalences = []
ci_low = []
ci_high = []

for item in crafft_items:
    k = int(crafft_with_items[item].sum())
    p_hat, lo, hi = wilson_ci(k, n_items)
    prevalences.append(100 * p_hat)
    ci_low.append(100 * lo)
    ci_high.append(100 * hi)

# Add screen positive
p_hat, lo, hi = wilson_ci(n_pos, n_total)
prevalences.append(100 * p_hat)
ci_low.append(100 * lo)
ci_high.append(100 * hi)

y_pos = range(len(item_names))
yerr_low = [max(0, p - l) for p, l in zip(prevalences, ci_low)]
yerr_high = [max(0, h - p) for p, h in zip(prevalences, ci_high)]

colors = ['#3498db'] * 6 + ['#e74c3c']
ax.barh(y_pos, prevalences, xerr=[yerr_low, yerr_high], color=colors,
        edgecolor='white', height=0.5, capsize=4, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(item_names, fontsize=11)
ax.set_xlabel('Prevalência (%)', fontsize=12)
ax.set_title('Prevalência CRAFFT por Item com IC 95% (Wilson)\nPopulação Pediátrica com DII — Brasil', fontsize=13)

for i, (p, lo, hi) in enumerate(zip(prevalences, ci_low, ci_high)):
    ax.text(hi + 1.5, i, f'{p:.1f}%', va='center', fontsize=10)

ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig24_crafft_prevalence_ci.png'), dpi=300, bbox_inches='tight')
plt.close()
log("  → fig24_crafft_prevalence_ci.png salva")


# ==============================================================================
# SALVAR RELATÓRIO FASE 3
# ==============================================================================
log("\n\n" + "=" * 100)
log("FIM DA FASE 3 — ANÁLISES FINAIS E SÍNTESE COMPLETAS")
log("=" * 100)

report_path = os.path.join(OUT_DIR, "relatorio_fase3.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

# Also save comprehensive combined report
combined_path = os.path.join(OUT_DIR, "relatorio_completo_todas_fases.txt")
all_reports = []
for fname in ['relatorio_aprofundado.txt', 'relatorio_fase2.txt', 'relatorio_fase3.txt']:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            all_reports.append(f.read())
            all_reports.append("\n\n" + "="*100 + "\n\n")

with open(combined_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_reports))

print(f"\n✓ Relatório Fase 3 salvo em: {report_path}")
print(f"✓ Relatório combinado (todas as fases) salvo em: {combined_path}")
print(f"✓ Figuras salvas em: {FIG_DIR}")
print("✓ Fase 3 concluída com sucesso!")
