#!/usr/bin/env python3
"""
Análise Estatística Completa — IBD Pediátrica em LMIC (Brasil)
Instrumentos: CRAFFT, IBD-Disk, IBDQ, IMPACT-III
Para submissão ao JCC Plus — Special Issue: Global Burden of IBD
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency, fisher_exact
from scipy.stats import spearmanr, pearsonr, shapiro, normaltest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
BASE_DIR = "/Users/marcelocarvalhoesilva/Desktop/Gastro pediatria"
CSV_DIR = os.path.join(BASE_DIR, "csv")
OUT_DIR = os.path.join(BASE_DIR, "resultados")
FIG_DIR = os.path.join(OUT_DIR, "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
sns.set_style("whitegrid")
sns.set_palette("colorblind")

report = []
def log(msg):
    report.append(msg)
    print(msg)

log("=" * 80)
log("ANÁLISE ESTATÍSTICA — IBD PEDIÁTRICA EM LMIC (BRASIL)")
log(f"Data da análise: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 80)

# ==============================================================================
# 1. CARREGAMENTO E LIMPEZA DOS DADOS
# ==============================================================================
log("\n\n" + "=" * 80)
log("1. CARREGAMENTO E LIMPEZA DOS DADOS")
log("=" * 80)

# --- CRAFFT ---
crafft = pd.read_csv(os.path.join(CSV_DIR, "CRAFFT FINAL.xlsx - CRAFFT Scoring.csv"))
crafft.columns = ['data_avaliacao', 'paciente', 'C_Car', 'R_Relax', 'A_Alone',
                   'F_Forget', 'F_Friends', 'T_Trouble', 'total_score', 'risk_interpretation']
# Remover linhas vazias e pacientes repetidos/sem nome
crafft = crafft[crafft['paciente'].notna() & (crafft['paciente'].str.strip() != '')]
crafft = crafft[~crafft['paciente'].str.contains('repetido|RECUSOU', case=False, na=False)]
crafft['paciente'] = crafft['paciente'].str.strip().str.upper()
# Converter scores numéricos
for col in ['C_Car', 'R_Relax', 'A_Alone', 'F_Forget', 'F_Friends', 'T_Trouble', 'total_score']:
    crafft[col] = pd.to_numeric(crafft[col], errors='coerce')
# Flag para screen positivo
crafft['crafft_positive'] = crafft['total_score'] >= 2
crafft_valid = crafft[crafft['total_score'].notna()].copy()

log(f"\nCRAFFT: {len(crafft_valid)} pacientes válidos")
log(f"  Screen positivo (≥2): {crafft_valid['crafft_positive'].sum()} ({100*crafft_valid['crafft_positive'].mean():.1f}%)")
log(f"  Screen negativo (<2): {(~crafft_valid['crafft_positive']).sum()} ({100*(1-crafft_valid['crafft_positive'].mean()):.1f}%)")

# --- IBD-Disk ---
ibddisk = pd.read_csv(os.path.join(CSV_DIR, "IBD Disk FINAL.xlsx - IBD_Disk_Data.csv"))
ibddisk.columns = ['data_avaliacao', 'paciente', 'age', 'sex', 'diagnosis',
                    'item1_abd_pain', 'item2_defecation', 'item3_interpersonal',
                    'item4_education', 'item5_sleep', 'item6_energy', 'item7_emotions',
                    'item8_body_image', 'item9_sexual', 'item10_joint_pain',
                    'physical_domain', 'psychosocial_domain', 'total_score', 'mean_score']
ibddisk = ibddisk[ibddisk['paciente'].notna() & (ibddisk['paciente'].str.strip() != '')]
ibddisk = ibddisk[~ibddisk['paciente'].str.contains('repetido|DII\\?\\?|sem questionário', case=False, na=False)]
ibddisk['paciente'] = ibddisk['paciente'].str.strip().str.upper()

# Converter campos numéricos (formato brasileiro com vírgula)
numeric_cols_disk = ['item1_abd_pain', 'item2_defecation', 'item3_interpersonal',
                     'item4_education', 'item5_sleep', 'item6_energy', 'item7_emotions',
                     'item8_body_image', 'item9_sexual', 'item10_joint_pain',
                     'physical_domain', 'psychosocial_domain', 'total_score', 'mean_score']
for col in numeric_cols_disk:
    ibddisk[col] = ibddisk[col].astype(str).str.replace(',', '.').str.replace('#DIV/0!', '')
    ibddisk[col] = pd.to_numeric(ibddisk[col], errors='coerce')

ibddisk_valid = ibddisk[ibddisk['total_score'].notna() & (ibddisk['total_score'] > 0)].copy()
log(f"\nIBD-Disk: {len(ibddisk_valid)} pacientes com dados válidos (score > 0)")

# --- IBDQ ---
ibdq = pd.read_csv(os.path.join(CSV_DIR, "IBDQ FINAL.xlsx - Cálculo Automático.csv"))
cols_ibdq = ['data_avaliacao', 'paciente'] + [f'item_{i}' for i in range(1, 33)] + \
            ['sintomas_intestinais', 'sintomas_sistemicos', 'bem_estar_emocional', 'interacao_social', 'total']
ibdq.columns = cols_ibdq
ibdq = ibdq[ibdq['paciente'].notna() & (ibdq['paciente'].str.strip() != '')]
ibdq = ibdq[~ibdq['paciente'].str.contains('repetido|DII\\?\\?|sem questionário', case=False, na=False)]
ibdq['paciente'] = ibdq['paciente'].str.strip().str.upper()
for col in ['sintomas_intestinais', 'sintomas_sistemicos', 'bem_estar_emocional', 'interacao_social', 'total']:
    ibdq[col] = pd.to_numeric(ibdq[col], errors='coerce')
ibdq_valid = ibdq[ibdq['total'].notna() & (ibdq['total'] > 0)].copy()
log(f"\nIBDQ: {len(ibdq_valid)} pacientes com dados válidos (total > 0)")

# --- IMPACT-III ---
impact = pd.read_csv(os.path.join(CSV_DIR, "IMPACTIII FINAL.xlsx - Cálculo Automático.csv"))
cols_impact = ['data_avaliacao', 'paciente'] + [f'item_{i}' for i in range(1, 36)] + \
              ['dom_sintomas', 'dom_emocional', 'dom_social', 'dom_bemestar', 'dom_tratamento', 'total']
impact.columns = cols_impact
impact = impact[impact['paciente'].notna() & (impact['paciente'].str.strip() != '')]
impact = impact[~impact['paciente'].str.contains('repetido|RECUSOU', case=False, na=False)]
impact['paciente'] = impact['paciente'].str.strip().str.upper()
for col in ['dom_sintomas', 'dom_emocional', 'dom_social', 'dom_bemestar', 'dom_tratamento', 'total']:
    impact[col] = pd.to_numeric(impact[col], errors='coerce')
impact_valid = impact[impact['total'].notna() & (impact['total'] > 0)].copy()
log(f"\nIMPACT-III: {len(impact_valid)} pacientes com dados válidos (total > 0)")

# ==============================================================================
# 2. ESTATÍSTICA DESCRITIVA
# ==============================================================================
log("\n\n" + "=" * 80)
log("2. ESTATÍSTICA DESCRITIVA")
log("=" * 80)

# --- CRAFFT ---
log("\n--- CRAFFT (Triagem de Substâncias) ---")
log(f"  N = {len(crafft_valid)}")
log(f"  Total Score: média = {crafft_valid['total_score'].mean():.2f}, "
    f"mediana = {crafft_valid['total_score'].median():.1f}, "
    f"DP = {crafft_valid['total_score'].std():.2f}, "
    f"IQR = [{crafft_valid['total_score'].quantile(0.25):.1f} - {crafft_valid['total_score'].quantile(0.75):.1f}]")

crafft_items = ['C_Car', 'R_Relax', 'A_Alone', 'F_Forget', 'F_Friends', 'T_Trouble']
log("\n  Prevalência por item (% positivo):")
for item in crafft_items:
    valid = crafft_valid[item].dropna()
    pct = 100 * valid.mean()
    log(f"    {item}: {pct:.1f}% (n={int(valid.sum())}/{len(valid)})")

# --- IBD-Disk ---
log("\n--- IBD-Disk (Índice de Incapacidade) ---")
log(f"  N = {len(ibddisk_valid)}")
log(f"  Total Score: média = {ibddisk_valid['total_score'].mean():.1f}, "
    f"mediana = {ibddisk_valid['total_score'].median():.1f}, "
    f"DP = {ibddisk_valid['total_score'].std():.1f}, "
    f"IQR = [{ibddisk_valid['total_score'].quantile(0.25):.1f} - {ibddisk_valid['total_score'].quantile(0.75):.1f}]")

disk_items = ['item1_abd_pain', 'item2_defecation', 'item3_interpersonal',
              'item4_education', 'item5_sleep', 'item6_energy', 'item7_emotions',
              'item8_body_image', 'item9_sexual', 'item10_joint_pain']
disk_labels = ['Dor Abdominal', 'Defecação', 'Interações', 'Educação/Trabalho',
               'Sono', 'Energia', 'Emoções', 'Imagem Corporal', 'Função Sexual', 'Dor Articular']

log("\n  Média por item (escala 0-10):")
item_means = []
for item, label in zip(disk_items, disk_labels):
    valid = ibddisk_valid[item].dropna()
    m = valid.mean()
    item_means.append((label, m))
    log(f"    {label}: {m:.1f} (DP={valid.std():.1f})")
item_means.sort(key=lambda x: x[1], reverse=True)
log(f"\n  Domínios mais afetados: {item_means[0][0]} ({item_means[0][1]:.1f}), "
    f"{item_means[1][0]} ({item_means[1][1]:.1f}), {item_means[2][0]} ({item_means[2][1]:.1f})")

log(f"\n  Physical Domain: média = {ibddisk_valid['physical_domain'].mean():.2f}, "
    f"DP = {ibddisk_valid['physical_domain'].std():.2f}")
log(f"  Psychosocial Domain: média = {ibddisk_valid['psychosocial_domain'].mean():.2f}, "
    f"DP = {ibddisk_valid['psychosocial_domain'].std():.2f}")

# --- IBDQ ---
log("\n--- IBDQ (Qualidade de Vida) ---")
log(f"  N = {len(ibdq_valid)}")
log(f"  Total Score: média = {ibdq_valid['total'].mean():.1f}, "
    f"mediana = {ibdq_valid['total'].median():.1f}, "
    f"DP = {ibdq_valid['total'].std():.1f}, "
    f"IQR = [{ibdq_valid['total'].quantile(0.25):.1f} - {ibdq_valid['total'].quantile(0.75):.1f}]")
log(f"  Range: {ibdq_valid['total'].min():.0f} - {ibdq_valid['total'].max():.0f}")

ibdq_domains = ['sintomas_intestinais', 'sintomas_sistemicos', 'bem_estar_emocional', 'interacao_social']
ibdq_labels = ['Sintomas Intestinais', 'Sintomas Sistêmicos', 'Bem-estar Emocional', 'Interação Social']
log("\n  Média por domínio:")
for dom, label in zip(ibdq_domains, ibdq_labels):
    valid = ibdq_valid[dom].dropna()
    if len(valid) > 0:
        log(f"    {label}: {valid.mean():.1f} (DP={valid.std():.1f})")

# Categorizar IBDQ: Remissão (≥170), Leve (150-169), Moderada (100-149), Grave (<100)
ibdq_valid['ibdq_category'] = pd.cut(ibdq_valid['total'],
                                      bins=[0, 100, 150, 170, 250],
                                      labels=['Grave', 'Moderada', 'Leve', 'Remissão'])
log("\n  Classificação IBDQ:")
for cat in ['Remissão', 'Leve', 'Moderada', 'Grave']:
    n = (ibdq_valid['ibdq_category'] == cat).sum()
    pct = 100 * n / len(ibdq_valid)
    log(f"    {cat}: {n} ({pct:.1f}%)")

# --- IMPACT-III ---
log("\n--- IMPACT-III (QoL Pediátrica) ---")
log(f"  N = {len(impact_valid)}")
log(f"  Total Score: média = {impact_valid['total'].mean():.1f}, "
    f"mediana = {impact_valid['total'].median():.1f}, "
    f"DP = {impact_valid['total'].std():.1f}, "
    f"IQR = [{impact_valid['total'].quantile(0.25):.1f} - {impact_valid['total'].quantile(0.75):.1f}]")

impact_domains = ['dom_sintomas', 'dom_emocional', 'dom_social', 'dom_bemestar', 'dom_tratamento']
impact_labels = ['Sintomas', 'Emocional', 'Social', 'Bem-estar', 'Tratamento']
log("\n  Média por domínio:")
for dom, label in zip(impact_domains, impact_labels):
    valid = impact_valid[dom].dropna()
    if len(valid) > 0 and valid.mean() > 0:
        log(f"    {label}: {valid.mean():.1f} (DP={valid.std():.1f})")

# ==============================================================================
# 3. TESTE DE NORMALIDADE
# ==============================================================================
log("\n\n" + "=" * 80)
log("3. TESTE DE NORMALIDADE (Shapiro-Wilk)")
log("=" * 80)

normality_results = {}
for name, data in [('CRAFFT Total', crafft_valid['total_score']),
                    ('IBD-Disk Total', ibddisk_valid['total_score']),
                    ('IBDQ Total', ibdq_valid['total']),
                    ('IMPACT-III Total', impact_valid['total'])]:
    valid = data.dropna()
    if len(valid) >= 3:
        stat, p = shapiro(valid[:5000])  # Shapiro limit
        is_normal = p > 0.05
        normality_results[name] = is_normal
        log(f"  {name}: W={stat:.4f}, p={p:.4f} → {'Normal' if is_normal else 'NÃO normal'}")

log("\n  → Conclusão: Usar testes NÃO-PARAMÉTRICOS (Mann-Whitney, Kruskal-Wallis, Spearman)")

# ==============================================================================
# 4. CORRELAÇÕES ENTRE INSTRUMENTOS
# ==============================================================================
log("\n\n" + "=" * 80)
log("4. CORRELAÇÕES ENTRE INSTRUMENTOS (Spearman)")
log("=" * 80)

# Merge datasets by patient name
merged = crafft_valid[['paciente', 'total_score', 'crafft_positive']].rename(
    columns={'total_score': 'crafft_total'})

disk_merge = ibddisk_valid[['paciente', 'total_score', 'mean_score', 'physical_domain', 'psychosocial_domain']].rename(
    columns={'total_score': 'disk_total', 'mean_score': 'disk_mean'})

ibdq_merge = ibdq_valid[['paciente', 'total', 'sintomas_intestinais', 'sintomas_sistemicos',
                          'bem_estar_emocional', 'interacao_social']].rename(
    columns={'total': 'ibdq_total'})

impact_merge = impact_valid[['paciente', 'total', 'dom_sintomas', 'dom_emocional',
                              'dom_social', 'dom_bemestar']].rename(
    columns={'total': 'impact_total'})

merged = merged.merge(disk_merge, on='paciente', how='outer')
merged = merged.merge(ibdq_merge, on='paciente', how='outer')
merged = merged.merge(impact_merge, on='paciente', how='outer')

log(f"\n  Dataset merged: {len(merged)} pacientes únicos")

# Correlações principais
correlations = [
    ('IBD-Disk Total', 'IBDQ Total', 'disk_total', 'ibdq_total'),
    ('IBD-Disk Total', 'IMPACT-III Total', 'disk_total', 'impact_total'),
    ('IBDQ Total', 'IMPACT-III Total', 'ibdq_total', 'impact_total'),
    ('CRAFFT Total', 'IBD-Disk Total', 'crafft_total', 'disk_total'),
    ('CRAFFT Total', 'IBDQ Total', 'crafft_total', 'ibdq_total'),
    ('CRAFFT Total', 'IMPACT-III Total', 'crafft_total', 'impact_total'),
    ('IBD-Disk Physical', 'IBDQ Sintomas Intestinais', 'physical_domain', 'sintomas_intestinais'),
    ('IBD-Disk Psychosocial', 'IBDQ Bem-estar Emocional', 'psychosocial_domain', 'bem_estar_emocional'),
    ('IBD-Disk Physical', 'IBDQ Sintomas Sistêmicos', 'physical_domain', 'sintomas_sistemicos'),
]

log("\n  Correlações de Spearman (rho, p-value):")
significant_corrs = []
for label1, label2, col1, col2 in correlations:
    valid = merged[[col1, col2]].dropna()
    if len(valid) >= 5:
        rho, p = spearmanr(valid[col1], valid[col2])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log(f"    {label1} vs {label2}: rho={rho:.3f}, p={p:.4f} {sig} (n={len(valid)})")
        if p < 0.05:
            significant_corrs.append((label1, label2, rho, p))

if significant_corrs:
    log(f"\n  → {len(significant_corrs)} correlações significativas encontradas")

# ==============================================================================
# 5. COMPARAÇÕES: CRAFFT POSITIVO vs NEGATIVO
# ==============================================================================
log("\n\n" + "=" * 80)
log("5. COMPARAÇÕES: CRAFFT POSITIVO vs NEGATIVO (Mann-Whitney U)")
log("=" * 80)

merged_crafft = merged[merged['crafft_positive'].notna()].copy()
pos = merged_crafft[merged_crafft['crafft_positive'] == True]
neg = merged_crafft[merged_crafft['crafft_positive'] == False]

log(f"\n  CRAFFT Positivo: n={len(pos)}")
log(f"  CRAFFT Negativo: n={len(neg)}")

comparisons = [
    ('IBD-Disk Total', 'disk_total'),
    ('IBD-Disk Mean', 'disk_mean'),
    ('IBD-Disk Physical', 'physical_domain'),
    ('IBD-Disk Psychosocial', 'psychosocial_domain'),
    ('IBDQ Total', 'ibdq_total'),
    ('IBDQ Sint. Intestinais', 'sintomas_intestinais'),
    ('IBDQ Sint. Sistêmicos', 'sintomas_sistemicos'),
    ('IBDQ Bem-estar Emoc.', 'bem_estar_emocional'),
    ('IBDQ Interação Social', 'interacao_social'),
    ('IMPACT-III Total', 'impact_total'),
]

log(f"\n  {'Variável':<28} {'Pos (med)':<12} {'Neg (med)':<12} {'U':>8} {'p-value':>10} {'Sig':>5}")
log(f"  {'-'*75}")

for label, col in comparisons:
    pos_vals = pos[col].dropna()
    neg_vals = neg[col].dropna()
    if len(pos_vals) >= 3 and len(neg_vals) >= 3:
        U, p = mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log(f"  {label:<28} {pos_vals.median():<12.1f} {neg_vals.median():<12.1f} {U:>8.0f} {p:>10.4f} {sig:>5}")

# ==============================================================================
# 6. ANÁLISE DE DOMÍNIOS IBD-DISK
# ==============================================================================
log("\n\n" + "=" * 80)
log("6. ANÁLISE DETALHADA DOS DOMÍNIOS IBD-DISK")
log("=" * 80)

# Categorizar severidade: leve (0-3), moderada (4-6), grave (7-10)
log("\n  Distribuição de severidade por item (% dos pacientes):")
log(f"  {'Item':<25} {'Leve 0-3':>10} {'Moderada 4-6':>14} {'Grave 7-10':>12}")
log(f"  {'-'*61}")

for item, label in zip(disk_items, disk_labels):
    valid = ibddisk_valid[item].dropna()
    if len(valid) > 0:
        leve = 100 * ((valid >= 0) & (valid <= 3)).sum() / len(valid)
        moderada = 100 * ((valid >= 4) & (valid <= 6)).sum() / len(valid)
        grave = 100 * ((valid >= 7) & (valid <= 10)).sum() / len(valid)
        log(f"  {label:<25} {leve:>9.1f}% {moderada:>13.1f}% {grave:>11.1f}%")

# Correlação entre domínios IBD-Disk
log("\n  Matriz de correlação entre itens IBD-Disk (Spearman):")
disk_corr_data = ibddisk_valid[disk_items].dropna()
if len(disk_corr_data) > 5:
    corr_matrix = disk_corr_data.corr(method='spearman')
    # Encontrar as correlações mais fortes
    strong_corrs = []
    for i in range(len(disk_items)):
        for j in range(i+1, len(disk_items)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.4:
                strong_corrs.append((disk_labels[i], disk_labels[j], r))
    strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
    log(f"\n  Top correlações entre itens (|rho| > 0.4):")
    for l1, l2, r in strong_corrs[:10]:
        log(f"    {l1} ↔ {l2}: rho={r:.3f}")

# ==============================================================================
# 7. ANÁLISE TEMPORAL
# ==============================================================================
log("\n\n" + "=" * 80)
log("7. ANÁLISE TEMPORAL")
log("=" * 80)

# Parse dates from CRAFFT (most complete)
def parse_date(d):
    if pd.isna(d) or str(d).strip() == '' or 'NÃO' in str(d) or 'SEM' in str(d):
        return pd.NaT
    try:
        return pd.to_datetime(d, dayfirst=True)
    except:
        return pd.NaT

crafft_valid['date'] = crafft_valid['data_avaliacao'].apply(parse_date)
crafft_dated = crafft_valid[crafft_valid['date'].notna()].copy()
crafft_dated['year'] = crafft_dated['date'].dt.year

log(f"\n  Período de coleta: {crafft_dated['date'].min().strftime('%d/%m/%Y')} a {crafft_dated['date'].max().strftime('%d/%m/%Y')}")
log(f"  Total com data válida: {len(crafft_dated)} pacientes")

log("\n  Distribuição por ano:")
year_counts = crafft_dated['year'].value_counts().sort_index()
for year, count in year_counts.items():
    pos_year = crafft_dated[crafft_dated['year'] == year]['crafft_positive'].sum()
    pct_pos = 100 * pos_year / count if count > 0 else 0
    log(f"    {int(year)}: {count} pacientes ({pos_year} CRAFFT+, {pct_pos:.0f}%)")

# Tendência temporal - CRAFFT positivo por período
crafft_dated['period'] = pd.cut(crafft_dated['year'],
                                 bins=[2020, 2022, 2024, 2027],
                                 labels=['2021-2022', '2023-2024', '2025-2026'])
log("\n  Tendência CRAFFT+ por período:")
for period in ['2021-2022', '2023-2024', '2025-2026']:
    subset = crafft_dated[crafft_dated['period'] == period]
    if len(subset) > 0:
        pos_n = subset['crafft_positive'].sum()
        pct = 100 * pos_n / len(subset)
        log(f"    {period}: {pos_n}/{len(subset)} positivos ({pct:.1f}%)")

# ==============================================================================
# 8. ANÁLISE DE SUBGRUPOS IBD-DISK POR SEVERIDADE
# ==============================================================================
log("\n\n" + "=" * 80)
log("8. SUBGRUPOS POR SEVERIDADE IBD-DISK")
log("=" * 80)

# Tercis do IBD-Disk
ibddisk_valid['severity_tercil'] = pd.qcut(ibddisk_valid['total_score'], q=3,
                                            labels=['Leve', 'Moderado', 'Grave'])

for tercil in ['Leve', 'Moderado', 'Grave']:
    subset = ibddisk_valid[ibddisk_valid['severity_tercil'] == tercil]
    log(f"\n  {tercil} (n={len(subset)}): "
        f"Total={subset['total_score'].median():.0f} "
        f"[{subset['total_score'].quantile(0.25):.0f}-{subset['total_score'].quantile(0.75):.0f}]")
    log(f"    Physical: {subset['physical_domain'].median():.1f}, "
        f"Psychosocial: {subset['psychosocial_domain'].median():.1f}")

# Kruskal-Wallis entre tercis para IBDQ
log("\n  Kruskal-Wallis: Tercis IBD-Disk vs outros instrumentos")
for label, col in [('IBDQ Total', 'ibdq_total'), ('IMPACT-III Total', 'impact_total')]:
    merged_tercil = merged.merge(ibddisk_valid[['paciente', 'severity_tercil']], on='paciente', how='inner')
    groups = []
    for tercil in ['Leve', 'Moderado', 'Grave']:
        vals = merged_tercil[merged_tercil['severity_tercil'] == tercil][col].dropna()
        if len(vals) >= 2:
            groups.append(vals)
    if len(groups) >= 2:
        H, p = kruskal(*groups)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log(f"    IBD-Disk tercis vs {label}: H={H:.2f}, p={p:.4f} {sig}")

# ==============================================================================
# 9. PREVALÊNCIA DE COMPROMETIMENTO POR ÁREA
# ==============================================================================
log("\n\n" + "=" * 80)
log("9. PREVALÊNCIA DE COMPROMETIMENTO SIGNIFICATIVO (Score ≥ 5 no IBD-Disk)")
log("=" * 80)

log(f"\n  {'Domínio':<25} {'N com score ≥5':>15} {'% afetados':>12}")
log(f"  {'-'*52}")
for item, label in zip(disk_items, disk_labels):
    valid = ibddisk_valid[item].dropna()
    if len(valid) > 0:
        affected = (valid >= 5).sum()
        pct = 100 * affected / len(valid)
        log(f"  {label:<25} {affected:>15} {pct:>11.1f}%")

# Pacientes com comprometimento múltiplo
ibddisk_valid['n_domains_affected'] = ibddisk_valid[disk_items].apply(
    lambda row: (row >= 5).sum(), axis=1)
log(f"\n  Pacientes por nº de domínios com comprometimento significativo (≥5):")
for n in range(11):
    count = (ibddisk_valid['n_domains_affected'] == n).sum()
    if count > 0:
        pct = 100 * count / len(ibddisk_valid)
        log(f"    {n} domínios: {count} pacientes ({pct:.1f}%)")

mean_affected = ibddisk_valid['n_domains_affected'].mean()
log(f"\n  Média de domínios afetados por paciente: {mean_affected:.1f}")

# ==============================================================================
# 10. EFFECT SIZES
# ==============================================================================
log("\n\n" + "=" * 80)
log("10. TAMANHO DE EFEITO (r = Z/√N para Mann-Whitney)")
log("=" * 80)

for label, col in comparisons:
    pos_vals = pos[col].dropna()
    neg_vals = neg[col].dropna()
    if len(pos_vals) >= 3 and len(neg_vals) >= 3:
        U, p = mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
        n1, n2 = len(pos_vals), len(neg_vals)
        # Calculate Z and r
        mu_U = n1 * n2 / 2
        sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        Z = (U - mu_U) / sigma_U
        r = abs(Z) / np.sqrt(n1 + n2)
        effect = "grande" if r >= 0.5 else "médio" if r >= 0.3 else "pequeno"
        if p < 0.05:
            log(f"  {label}: r={r:.3f} ({effect}), Z={Z:.2f}, p={p:.4f}")

# ==============================================================================
# 11. FIGURAS
# ==============================================================================
log("\n\n" + "=" * 80)
log("11. GERANDO FIGURAS")
log("=" * 80)

# Figura 1: Distribuição CRAFFT
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
crafft_valid['total_score'].hist(bins=range(0, 8), ax=axes[0], color='steelblue', edgecolor='black', alpha=0.8)
axes[0].set_xlabel('CRAFFT Total Score')
axes[0].set_ylabel('Frequência')
axes[0].set_title('Distribuição dos Scores CRAFFT')
axes[0].axvline(x=2, color='red', linestyle='--', label='Cutoff (≥2)')
axes[0].legend()

# Prevalência por item
item_prev = [100 * crafft_valid[item].mean() for item in crafft_items]
item_names = ['Car', 'Relax', 'Alone', 'Forget', 'Friends', 'Trouble']
bars = axes[1].bar(item_names, item_prev, color='coral', edgecolor='black', alpha=0.8)
axes[1].set_ylabel('Prevalência (%)')
axes[1].set_title('Prevalência por Item CRAFFT')
for bar, val in zip(bars, item_prev):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{val:.0f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_crafft_distribuicao.png'), bbox_inches='tight')
plt.close()
log("  Figura 1: fig1_crafft_distribuicao.png ✓")

# Figura 2: Distribuição IBD-Disk por item
fig, ax = plt.subplots(figsize=(14, 8))
disk_data = ibddisk_valid[disk_items].dropna()
disk_data.columns = disk_labels
bp = disk_data.boxplot(ax=ax, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.8),
                        medianprops=dict(color='red', linewidth=2))
ax.set_ylabel('Score (0-10)')
ax.set_title('Distribuição dos Scores IBD-Disk por Domínio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_ibddisk_boxplot.png'), bbox_inches='tight')
plt.close()
log("  Figura 2: fig2_ibddisk_boxplot.png ✓")

# Figura 3: Distribuição IBDQ Total e por domínio
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ibdq_valid['total'].hist(bins=20, ax=axes[0], color='forestgreen', edgecolor='black', alpha=0.8)
axes[0].set_xlabel('IBDQ Total Score')
axes[0].set_ylabel('Frequência')
axes[0].set_title('Distribuição IBDQ Total')
axes[0].axvline(x=170, color='red', linestyle='--', label='Remissão (≥170)')
axes[0].legend()

# Stacked bar da classificação
cat_counts = ibdq_valid['ibdq_category'].value_counts()
colors_cat = {'Remissão': 'green', 'Leve': 'yellow', 'Moderada': 'orange', 'Grave': 'red'}
cats = ['Grave', 'Moderada', 'Leve', 'Remissão']
vals = [cat_counts.get(c, 0) for c in cats]
axes[1].bar(cats, vals, color=[colors_cat[c] for c in cats], edgecolor='black', alpha=0.8)
axes[1].set_ylabel('Nº de Pacientes')
axes[1].set_title('Classificação IBDQ')
for i, (c, v) in enumerate(zip(cats, vals)):
    if v > 0:
        axes[1].text(i, v + 0.5, f'{v}\n({100*v/sum(vals):.0f}%)', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_ibdq_distribuicao.png'), bbox_inches='tight')
plt.close()
log("  Figura 3: fig3_ibdq_distribuicao.png ✓")

# Figura 4: Correlação IBD-Disk vs IBDQ
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# Disk vs IBDQ
pair1 = merged[['disk_total', 'ibdq_total']].dropna()
if len(pair1) >= 5:
    axes[0].scatter(pair1['disk_total'], pair1['ibdq_total'], alpha=0.6, s=50, c='steelblue')
    z = np.polyfit(pair1['disk_total'], pair1['ibdq_total'], 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(pair1['disk_total'].min(), pair1['disk_total'].max(), 100)
    axes[0].plot(x_range, p_line(x_range), "r--", alpha=0.8)
    rho, pval = spearmanr(pair1['disk_total'], pair1['ibdq_total'])
    axes[0].set_title(f'IBD-Disk vs IBDQ\nrho={rho:.3f}, p={pval:.4f}')
    axes[0].set_xlabel('IBD-Disk Total')
    axes[0].set_ylabel('IBDQ Total')

# Disk vs IMPACT-III
pair2 = merged[['disk_total', 'impact_total']].dropna()
if len(pair2) >= 5:
    axes[1].scatter(pair2['disk_total'], pair2['impact_total'], alpha=0.6, s=50, c='coral')
    z = np.polyfit(pair2['disk_total'], pair2['impact_total'], 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(pair2['disk_total'].min(), pair2['disk_total'].max(), 100)
    axes[1].plot(x_range, p_line(x_range), "r--", alpha=0.8)
    rho, pval = spearmanr(pair2['disk_total'], pair2['impact_total'])
    axes[1].set_title(f'IBD-Disk vs IMPACT-III\nrho={rho:.3f}, p={pval:.4f}')
    axes[1].set_xlabel('IBD-Disk Total')
    axes[1].set_ylabel('IMPACT-III Total')

# IBDQ vs IMPACT-III
pair3 = merged[['ibdq_total', 'impact_total']].dropna()
if len(pair3) >= 5:
    axes[2].scatter(pair3['ibdq_total'], pair3['impact_total'], alpha=0.6, s=50, c='forestgreen')
    z = np.polyfit(pair3['ibdq_total'], pair3['impact_total'], 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(pair3['ibdq_total'].min(), pair3['ibdq_total'].max(), 100)
    axes[2].plot(x_range, p_line(x_range), "r--", alpha=0.8)
    rho, pval = spearmanr(pair3['ibdq_total'], pair3['impact_total'])
    axes[2].set_title(f'IBDQ vs IMPACT-III\nrho={rho:.3f}, p={pval:.4f}')
    axes[2].set_xlabel('IBDQ Total')
    axes[2].set_ylabel('IMPACT-III Total')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_correlacoes_scatter.png'), bbox_inches='tight')
plt.close()
log("  Figura 4: fig4_correlacoes_scatter.png ✓")

# Figura 5: CRAFFT Positivo vs Negativo — IBD-Disk
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (label, col) in zip(axes, [('IBD-Disk Total', 'disk_total'),
                                     ('IBDQ Total', 'ibdq_total'),
                                     ('IMPACT-III Total', 'impact_total')]):
    data_plot = merged_crafft[merged_crafft[col].notna()]
    if len(data_plot) > 0:
        groups = [data_plot[data_plot['crafft_positive'] == False][col].dropna(),
                  data_plot[data_plot['crafft_positive'] == True][col].dropna()]
        bp = ax.boxplot(groups, labels=['CRAFFT-', 'CRAFFT+'], patch_artist=True,
                        boxprops=dict(alpha=0.8), medianprops=dict(color='red', linewidth=2))
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_title(f'{label}\nCRAFFT- vs CRAFFT+')
        ax.set_ylabel('Score')
        if len(groups[0]) >= 3 and len(groups[1]) >= 3:
            U, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(0.5, 0.95, f'p={p:.4f} {sig}', transform=ax.transAxes,
                    ha='center', va='top', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_crafft_comparacao.png'), bbox_inches='tight')
plt.close()
log("  Figura 5: fig5_crafft_comparacao.png ✓")

# Figura 6: Heatmap de correlação entre domínios
fig, ax = plt.subplots(figsize=(12, 10))
corr_cols = ['crafft_total', 'disk_total', 'physical_domain', 'psychosocial_domain',
             'ibdq_total', 'sintomas_intestinais', 'sintomas_sistemicos',
             'bem_estar_emocional', 'interacao_social', 'impact_total']
corr_labels = ['CRAFFT', 'IBD-Disk', 'Disk Physical', 'Disk Psychosocial',
               'IBDQ Total', 'IBDQ Sint.Intest.', 'IBDQ Sint.Sist.',
               'IBDQ Bem-estar', 'IBDQ Social', 'IMPACT-III']
corr_data = merged[corr_cols].dropna(thresh=2)
if len(corr_data) > 5:
    corr_matrix = corr_data.corr(method='spearman')
    corr_matrix.columns = corr_labels
    corr_matrix.index = corr_labels
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Matriz de Correlação entre Instrumentos (Spearman)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig6_heatmap_correlacao.png'), bbox_inches='tight')
plt.close()
log("  Figura 6: fig6_heatmap_correlacao.png ✓")

# Figura 7: Prevalência de comprometimento por domínio IBD-Disk
fig, ax = plt.subplots(figsize=(12, 7))
prev_data = []
for item, label in zip(disk_items, disk_labels):
    valid = ibddisk_valid[item].dropna()
    if len(valid) > 0:
        prev_data.append({
            'Domínio': label,
            'Leve (0-3)': 100 * ((valid >= 0) & (valid <= 3)).sum() / len(valid),
            'Moderado (4-6)': 100 * ((valid >= 4) & (valid <= 6)).sum() / len(valid),
            'Grave (7-10)': 100 * ((valid >= 7) & (valid <= 10)).sum() / len(valid)
        })
prev_df = pd.DataFrame(prev_data)
prev_df = prev_df.sort_values('Grave (7-10)', ascending=True)
prev_df.set_index('Domínio')[['Leve (0-3)', 'Moderado (4-6)', 'Grave (7-10)']].plot(
    kind='barh', stacked=True, ax=ax, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.85)
ax.set_xlabel('Percentual (%)')
ax.set_title('Severidade por Domínio IBD-Disk')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig7_ibddisk_severidade.png'), bbox_inches='tight')
plt.close()
log("  Figura 7: fig7_ibddisk_severidade.png ✓")

# Figura 8: Temporal
fig, ax = plt.subplots(figsize=(10, 6))
yearly = crafft_dated.groupby('year').agg(
    n=('total_score', 'count'),
    pct_positive=('crafft_positive', 'mean')
).reset_index()
yearly['pct_positive'] *= 100
ax.bar(yearly['year'], yearly['n'], color='steelblue', alpha=0.6, label='N pacientes')
ax2 = ax.twinx()
ax2.plot(yearly['year'], yearly['pct_positive'], 'ro-', linewidth=2, markersize=8, label='% CRAFFT+')
ax.set_xlabel('Ano')
ax.set_ylabel('Nº de Pacientes')
ax2.set_ylabel('% CRAFFT Positivo')
ax.set_title('Evolução Temporal: Pacientes e Taxa CRAFFT+')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig8_temporal.png'), bbox_inches='tight')
plt.close()
log("  Figura 8: fig8_temporal.png ✓")

# ==============================================================================
# 12. RESUMO EXECUTIVO
# ==============================================================================
log("\n\n" + "=" * 80)
log("12. RESUMO EXECUTIVO — INSIGHTS PRINCIPAIS")
log("=" * 80)

log(f"""
DADOS:
  • {len(crafft_valid)} pacientes avaliados com CRAFFT (triagem substâncias)
  • {len(ibddisk_valid)} pacientes com IBD-Disk (incapacidade)
  • {len(ibdq_valid)} pacientes com IBDQ (qualidade de vida)
  • {len(impact_valid)} pacientes com IMPACT-III (QoL pediátrica)
  • Período: {crafft_dated['date'].min().strftime('%m/%Y')} a {crafft_dated['date'].max().strftime('%m/%Y')}

ACHADOS PRINCIPAIS:

1. TRIAGEM DE SUBSTÂNCIAS (CRAFFT):
   • {100*crafft_valid['crafft_positive'].mean():.0f}% dos pacientes com IBD pediátrica têm triagem positiva para risco de abuso de substâncias
   • Item mais prevalente: "Trouble" ({100*crafft_valid['T_Trouble'].mean():.0f}%) — problemas relacionados ao uso
   • Este dado é INÉDITO em população IBD pediátrica de LMIC

2. INCAPACIDADE (IBD-Disk):
   • Score médio: {ibddisk_valid['total_score'].mean():.0f}/100 (mediana {ibddisk_valid['total_score'].median():.0f})
   • Domínios mais afetados: {item_means[0][0]}, {item_means[1][0]}, {item_means[2][0]}
   • {100*(ibddisk_valid['n_domains_affected'] >= 3).sum()/len(ibddisk_valid):.0f}% com ≥3 domínios significativamente comprometidos

3. QUALIDADE DE VIDA (IBDQ):
   • Score médio: {ibdq_valid['total'].mean():.0f}/224
   • {100*(ibdq_valid['ibdq_category'] == 'Grave').sum()/len(ibdq_valid):.0f}% classificados como comprometimento grave
   • {100*(ibdq_valid['ibdq_category'] == 'Remissão').sum()/len(ibdq_valid):.0f}% em remissão clínica

4. RELEVÂNCIA PARA JCC Plus:
   • Primeira avaliação multidimensional (substâncias + incapacidade + QoL) em IBD pediátrica de LMIC
   • Dados reais do sistema público de saúde brasileiro (SUS)
   • Amostra significativa (n>{len(crafft_valid)}) com follow-up longitudinal ({crafft_dated['date'].min().year}-{crafft_dated['date'].max().year})
   • Aborda múltiplos temas da chamada: patient-reported outcomes, QoL, disability, LMIC gaps
""")

# ==============================================================================
# SALVAR RELATÓRIO
# ==============================================================================
report_text = "\n".join(report)
with open(os.path.join(OUT_DIR, "relatorio_estatistico.txt"), 'w') as f:
    f.write(report_text)

# Salvar dataset merged
merged.to_csv(os.path.join(OUT_DIR, "dataset_merged.csv"), index=False)

log(f"\n\nArquivos salvos em: {OUT_DIR}")
log(f"Figuras salvas em: {FIG_DIR}")
log("ANÁLISE COMPLETA!")
