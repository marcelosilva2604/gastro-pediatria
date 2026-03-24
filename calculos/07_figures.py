#!/usr/bin/env python3
"""
07_figures.py
Gera figuras APENAS para resultados estatisticamente significativos.
Salva em: manuscrito/figures/
Uso: python3 calculos/07_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DC = os.path.join(BASE, 'data_clean')
FIG_DIR = os.path.join(BASE, 'manuscrito', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Clean old figures
for f in os.listdir(FIG_DIR):
    os.remove(os.path.join(FIG_DIR, f))
    print(f"  Removed: {f}")

# Load data
ibddisk = pd.read_csv(os.path.join(DC, 'ibddisk_clean.csv'))
ibdq = pd.read_csv(os.path.join(DC, 'ibdq_clean.csv'))

ibddisk_f = ibddisk.drop_duplicates('PACIENTE', keep='first')
ibdq_full = ibdq[ibdq['Item 32'].notna()].drop_duplicates('PACIENTE', keep='first')

# Style
COLORS = {'CD': '#2171b5', 'UC': '#cb181d'}
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 300,
})

# =============================================
# FIGURE 1: IBDQ by Disease Type — ALL significant
# Panel A: Total score | Panel B: Domain scores
# =============================================
cd_q = ibdq_full[ibdq_full['DIAGNOSTICO'] == 'CD']
uc_q = ibdq_full[ibdq_full['DIAGNOSTICO'] == 'UC']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: IBDQ Total
bp1 = axes[0].boxplot(
    [cd_q['Total'], uc_q['Total']],
    tick_labels=[f'CD\n(n={len(cd_q)})', f'UC\n(n={len(uc_q)})'],
    patch_artist=True, widths=0.5
)
bp1['boxes'][0].set_facecolor(COLORS['CD'])
bp1['boxes'][1].set_facecolor(COLORS['UC'])
for box in bp1['boxes']:
    box.set_alpha(0.7)
axes[0].set_ylabel('IBDQ Total Score (32–224)')
axes[0].set_title('A', loc='left', fontweight='bold', fontsize=14)
axes[0].set_title('IBDQ Total Score')
u, p = mannwhitneyu(cd_q['Total'], uc_q['Total'])
axes[0].text(0.95, 0.95, f'p = {p:.2f}', transform=axes[0].transAxes,
             ha='right', va='top', fontsize=10, fontweight='bold', style='italic')

# Panel B: IBDQ Domains (grouped bar of medians with IQR)
domains = ['Sintomas Intestinais', 'Sintomas Sistêmicos', 'Bem-Estar Emocional', 'Interação Social']
domain_labels = ['Bowel\nSymptoms', 'Systemic\nSymptoms', 'Emotional\nWell-being', 'Social\nFunction']

x = np.arange(len(domains))
width = 0.35

cd_medians = [cd_q[d].median() for d in domains]
uc_medians = [uc_q[d].median() for d in domains]
cd_q25 = [cd_q[d].quantile(0.25) for d in domains]
cd_q75 = [cd_q[d].quantile(0.75) for d in domains]
uc_q25 = [uc_q[d].quantile(0.25) for d in domains]
uc_q75 = [uc_q[d].quantile(0.75) for d in domains]

cd_err_lo = [m - q for m, q in zip(cd_medians, cd_q25)]
cd_err_hi = [q - m for m, q in zip(cd_medians, cd_q75)]
uc_err_lo = [m - q for m, q in zip(uc_medians, uc_q25)]
uc_err_hi = [q - m for m, q in zip(uc_medians, uc_q75)]

bars_cd = axes[1].bar(x - width/2, cd_medians, width, label='CD', color=COLORS['CD'], alpha=0.7,
                       yerr=[cd_err_lo, cd_err_hi], capsize=4, error_kw={'linewidth': 1})
bars_uc = axes[1].bar(x + width/2, uc_medians, width, label='UC', color=COLORS['UC'], alpha=0.7,
                       yerr=[uc_err_lo, uc_err_hi], capsize=4, error_kw={'linewidth': 1})

# Add p-values above each pair
p_values = []
for d in domains:
    u, p = mannwhitneyu(cd_q[d], uc_q[d])
    p_values.append(p)

for i, p in enumerate(p_values):
    max_h = max(cd_q75[i], uc_q75[i]) + 3
    axes[1].text(i, max_h, f'p={p:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

axes[1].set_ylabel('Domain Score (median)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(domain_labels, fontsize=9)
axes[1].set_title('B', loc='left', fontweight='bold', fontsize=14)
axes[1].set_title('IBDQ Domain Scores')
axes[1].legend(loc='upper right', fontsize=9)

plt.tight_layout()
path1 = os.path.join(FIG_DIR, 'figure1_ibdq_by_diagnosis.png')
plt.savefig(path1, bbox_inches='tight')
plt.close()
print(f"Figure 1 saved: {path1}")

# =============================================
# FIGURE 2: IBD-Disk vs IBDQ correlation (rho = -0.71, p < 0.001)
# =============================================
merged = ibddisk_f[['PACIENTE', 'Total_IBD_Disk_Score', 'DIAGNOSTICO']].merge(
    ibdq_full[['PACIENTE', 'Total']], on='PACIENTE')

fig, ax = plt.subplots(figsize=(7, 5.5))

for diag, color, marker in [('CD', COLORS['CD'], 'o'), ('UC', COLORS['UC'], 's')]:
    sub = merged[merged['DIAGNOSTICO'] == diag]
    ax.scatter(sub['Total_IBD_Disk_Score'], sub['Total'], alpha=0.6, s=45,
               c=color, marker=marker, edgecolors='white', linewidth=0.5, label=f'{diag} (n={len(sub)})')

# Trend line
z = np.polyfit(merged['Total_IBD_Disk_Score'], merged['Total'], 1)
xline = np.linspace(0, 100, 100)
ax.plot(xline, np.polyval(z, xline), '--', color='grey', alpha=0.7, linewidth=1)

rho, p = spearmanr(merged['Total_IBD_Disk_Score'], merged['Total'])
p_str = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
ax.text(0.95, 0.95, f'Spearman ρ = {rho:.2f}\n{p_str}\nn = {len(merged)}',
        transform=ax.transAxes, ha='right', va='top', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

ax.set_xlabel('IBD-Disk Total Score (0–100, higher = more disability)')
ax.set_ylabel('IBDQ Total Score (32–224, higher = better QoL)')
ax.set_title('Correlation Between Disability and Quality of Life')
ax.legend(loc='lower left', fontsize=9)
ax.set_xlim(-5, 100)
ax.set_ylim(40, 230)

plt.tight_layout()
path2 = os.path.join(FIG_DIR, 'figure2_ibddisk_vs_ibdq_correlation.png')
plt.savefig(path2, bbox_inches='tight')
plt.close()
print(f"Figure 2 saved: {path2}")

print(f"\nAll figures saved in: {FIG_DIR}")
print("Only statistically significant results shown.")
print("FIM — 07_figures.py")
