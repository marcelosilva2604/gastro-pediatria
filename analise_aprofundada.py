#!/usr/bin/env python3
"""
ANÁLISE ESTATÍSTICA APROFUNDADA — IBD Pediátrica em LMIC (Brasil)
Instrumentos: CRAFFT, IBD-Disk, IBDQ, IMPACT-III
JCC Plus — Special Issue: Global Burden of IBD

PARTE 1: Qualidade dos Dados e Confiabilidade dos Instrumentos
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (mannwhitneyu, kruskal, chi2_contingency, fisher_exact,
                          spearmanr, pearsonr, shapiro, wilcoxon, kstest,
                          pointbiserialr, kendalltau)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
from itertools import combinations

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
log("ANÁLISE ESTATÍSTICA APROFUNDADA — IBD PEDIÁTRICA EM LMIC (BRASIL)")
log(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 100)

# ==============================================================================
# CARREGAMENTO (mesmo código de limpeza)
# ==============================================================================

# --- CRAFFT ---
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

def parse_date(d):
    if pd.isna(d) or str(d).strip() == '' or 'NÃO' in str(d).upper() or 'SEM' in str(d).upper():
        return pd.NaT
    try:
        return pd.to_datetime(d, dayfirst=True)
    except:
        return pd.NaT

crafft['date'] = crafft['data_avaliacao'].apply(parse_date)

# Separar válidos com dados completos vs incompletos
crafft_with_items = crafft[crafft[crafft_items].notna().any(axis=1)].copy()
crafft_valid = crafft[crafft['total_score'].notna()].copy()

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

ibddisk_all = ibddisk.copy()
ibddisk_valid = ibddisk[ibddisk['total_score'].notna() & (ibddisk['total_score'] > 0)].copy()

# --- IBDQ ---
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
ibdq_all = ibdq.copy()
ibdq_valid = ibdq[ibdq['total'].notna() & (ibdq['total'] > 0)].copy()

# --- IMPACT-III ---
impact = pd.read_csv(os.path.join(CSV_DIR, "IMPACTIII FINAL.xlsx - Cálculo Automático.csv"))
cols_impact = ['data_avaliacao', 'paciente'] + [f'item_{i}' for i in range(1, 36)] + \
              ['dom_sintomas', 'dom_emocional', 'dom_social', 'dom_bemestar', 'dom_tratamento', 'total']
impact.columns = cols_impact
impact = impact[impact['paciente'].notna() & (impact['paciente'].str.strip() != '')]
impact = impact[~impact['paciente'].str.contains('repetido|RECUSOU', case=False, na=False)]
impact['paciente'] = impact['paciente'].str.strip().str.upper()
impact['date'] = impact['data_avaliacao'].apply(parse_date)
impact_item_cols = [f'item_{i}' for i in range(1, 36)]
impact_domain_cols = ['dom_sintomas', 'dom_emocional', 'dom_social', 'dom_bemestar', 'dom_tratamento', 'total']
for col in impact_item_cols + impact_domain_cols:
    impact[col] = pd.to_numeric(impact[col], errors='coerce')
impact_all = impact.copy()
impact_valid = impact[impact['total'].notna() & (impact['total'] > 0)].copy()


# ==============================================================================
# PARTE A: ANÁLISE DE DADOS FALTANTES (Missing Data Analysis)
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE A: ANÁLISE DE DADOS FALTANTES")
log("=" * 100)

total_patients = len(crafft)
log(f"\nTotal de registros no dataset: {total_patients}")

log("\n--- Taxa de completude por instrumento ---")
datasets = {
    'CRAFFT': (crafft, crafft_items, crafft_valid),
    'IBD-Disk': (ibddisk_all, disk_items, ibddisk_valid),
    'IBDQ': (ibdq_all, ibdq_item_cols, ibdq_valid),
    'IMPACT-III': (impact_all, impact_item_cols, impact_valid),
}

for name, (df_all, items, df_valid) in datasets.items():
    total = len(df_all)
    valid = len(df_valid)
    missing = total - valid
    pct_valid = 100 * valid / total if total > 0 else 0
    log(f"\n  {name}:")
    log(f"    Total registros: {total}")
    log(f"    Com dados válidos: {valid} ({pct_valid:.1f}%)")
    log(f"    Sem dados/incompletos: {missing} ({100-pct_valid:.1f}%)")

    # Missing por item
    if valid > 5:
        item_missing = df_valid[items].isna().sum()
        item_missing_pct = 100 * item_missing / valid
        if item_missing_pct.max() > 0:
            log(f"    Itens com dados faltantes (entre pacientes válidos):")
            for item_name, pct in item_missing_pct[item_missing_pct > 0].items():
                log(f"      {item_name}: {pct:.1f}% missing")

# Análise de padrão: quem respondeu quais instrumentos?
log("\n--- Completude cruzada entre instrumentos ---")
all_patients = set(crafft['paciente'].unique())
crafft_pts = set(crafft_valid['paciente'].unique())
disk_pts = set(ibddisk_valid['paciente'].unique())
ibdq_pts = set(ibdq_valid['paciente'].unique())
impact_pts = set(impact_valid['paciente'].unique())

log(f"  Pacientes únicos total: {len(all_patients)}")
log(f"  CRAFFT + IBD-Disk: {len(crafft_pts & disk_pts)}")
log(f"  CRAFFT + IBDQ: {len(crafft_pts & ibdq_pts)}")
log(f"  CRAFFT + IMPACT-III: {len(crafft_pts & impact_pts)}")
log(f"  IBD-Disk + IBDQ: {len(disk_pts & ibdq_pts)}")
log(f"  IBD-Disk + IMPACT-III: {len(disk_pts & impact_pts)}")
log(f"  IBDQ + IMPACT-III: {len(ibdq_pts & impact_pts)}")
log(f"  Todos 4 instrumentos: {len(crafft_pts & disk_pts & ibdq_pts & impact_pts)}")
log(f"  CRAFFT + IBD-Disk + IBDQ (sem IMPACT): {len(crafft_pts & disk_pts & ibdq_pts)}")


# ==============================================================================
# PARTE B: CONFIABILIDADE DOS INSTRUMENTOS (Cronbach's Alpha)
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE B: CONFIABILIDADE DOS INSTRUMENTOS — Consistência Interna")
log("=" * 100)

def cronbach_alpha(df_items):
    """Calcula Cronbach's alpha para um DataFrame de itens."""
    df_clean = df_items.dropna()
    if len(df_clean) < 3:
        return np.nan, len(df_clean)
    n_items = df_clean.shape[1]
    if n_items < 2:
        return np.nan, len(df_clean)
    item_variances = df_clean.var(axis=0, ddof=1)
    total_variance = df_clean.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return np.nan, len(df_clean)
    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha, len(df_clean)

def alpha_if_deleted(df_items):
    """Calcula alpha se cada item for removido."""
    results = []
    cols = df_items.columns.tolist()
    for col in cols:
        remaining = [c for c in cols if c != col]
        alpha, n = cronbach_alpha(df_items[remaining])
        results.append((col, alpha))
    return results

# CRAFFT
log("\n--- CRAFFT ---")
crafft_item_data = crafft_with_items[crafft_items].dropna()
alpha, n = cronbach_alpha(crafft_item_data)
log(f"  Cronbach's Alpha: {alpha:.4f} (n={n})")
log(f"  Interpretação: {'Excelente' if alpha >= 0.9 else 'Bom' if alpha >= 0.8 else 'Aceitável' if alpha >= 0.7 else 'Questionável' if alpha >= 0.6 else 'Pobre' if alpha >= 0.5 else 'Inaceitável'}")

log("  Alpha se item deletado:")
for item, a in alpha_if_deleted(crafft_item_data):
    change = a - alpha
    direction = "↑" if change > 0.01 else "↓" if change < -0.01 else "≈"
    log(f"    {item}: α={a:.4f} ({direction}{abs(change):.4f})")

# Correlação item-total corrigida
log("  Correlação item-total corrigida:")
for item in crafft_items:
    remaining = [c for c in crafft_items if c != item]
    valid_data = crafft_item_data.dropna()
    if len(valid_data) > 5:
        total_remaining = valid_data[remaining].sum(axis=1)
        r, p = spearmanr(valid_data[item], total_remaining)
        log(f"    {item}: r={r:.3f} (p={p:.4f})")

# IBD-Disk
log("\n--- IBD-Disk ---")
disk_item_data = ibddisk_valid[disk_items].dropna()
alpha, n = cronbach_alpha(disk_item_data)
log(f"  Cronbach's Alpha: {alpha:.4f} (n={n})")
log(f"  Interpretação: {'Excelente' if alpha >= 0.9 else 'Bom' if alpha >= 0.8 else 'Aceitável' if alpha >= 0.7 else 'Questionável' if alpha >= 0.6 else 'Pobre' if alpha >= 0.5 else 'Inaceitável'}")

log("  Alpha se item deletado:")
for item, a in alpha_if_deleted(disk_item_data):
    label = disk_labels[disk_items.index(item)]
    change = a - alpha
    direction = "↑" if change > 0.01 else "↓" if change < -0.01 else "≈"
    log(f"    {label}: α={a:.4f} ({direction}{abs(change):.4f})")

log("  Correlação item-total corrigida:")
for item, label in zip(disk_items, disk_labels):
    remaining = [c for c in disk_items if c != item]
    valid_data = disk_item_data.dropna()
    if len(valid_data) > 5:
        total_remaining = valid_data[remaining].sum(axis=1)
        r, p = spearmanr(valid_data[item], total_remaining)
        adequacy = "adequada" if r >= 0.3 else "BAIXA"
        log(f"    {label}: r={r:.3f} (p={p:.4f}) — {adequacy}")

# IBDQ
log("\n--- IBDQ ---")
ibdq_item_data = ibdq_valid[ibdq_item_cols]
# Muitos itens podem estar missing para pacientes com versão curta
ibdq_complete = ibdq_item_data.dropna(thresh=int(0.8 * len(ibdq_item_cols)))
alpha, n = cronbach_alpha(ibdq_complete.dropna())
log(f"  Cronbach's Alpha (itens completos): {alpha:.4f} (n={n})")
log(f"  Interpretação: {'Excelente' if alpha >= 0.9 else 'Bom' if alpha >= 0.8 else 'Aceitável' if alpha >= 0.7 else 'Questionável' if alpha >= 0.6 else 'Pobre' if alpha >= 0.5 else 'Inaceitável'}")

# Alpha por domínio IBDQ
# Domínios IBDQ: Sintomas Intestinais (1-10), Sistêmicos (11), Emocional (12-23), Social (24-32)
ibdq_bowel = [f'item_{i}' for i in range(1, 11)]
ibdq_systemic = [f'item_{i}' for i in range(11, 12)]  # Só 1 item = não dá alpha
ibdq_emotional = [f'item_{i}' for i in range(12, 24)]
ibdq_social = [f'item_{i}' for i in range(24, 33)]

for domain_name, domain_items in [('Sintomas Intestinais (1-10)', ibdq_bowel),
                                    ('Bem-estar Emocional (12-23)', ibdq_emotional),
                                    ('Interação Social (24-32)', ibdq_social)]:
    domain_data = ibdq_valid[domain_items].dropna()
    a, n = cronbach_alpha(domain_data)
    if not np.isnan(a):
        log(f"  Alpha domínio {domain_name}: {a:.4f} (n={n})")

# IMPACT-III
log("\n--- IMPACT-III ---")
impact_item_data = impact_valid[impact_item_cols].dropna(thresh=int(0.8 * len(impact_item_cols)))
alpha, n = cronbach_alpha(impact_item_data.dropna())
if not np.isnan(alpha):
    log(f"  Cronbach's Alpha: {alpha:.4f} (n={n})")
    log(f"  Interpretação: {'Excelente' if alpha >= 0.9 else 'Bom' if alpha >= 0.8 else 'Aceitável' if alpha >= 0.7 else 'Questionável' if alpha >= 0.6 else 'Pobre' if alpha >= 0.5 else 'Inaceitável'}")
else:
    log(f"  N insuficiente para Cronbach's Alpha (n={n})")


# ==============================================================================
# PARTE C: FLOOR E CEILING EFFECTS
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE C: FLOOR E CEILING EFFECTS")
log("=" * 100)
log("  (Floor/ceiling >15% indica problema na sensibilidade do instrumento)")

# CRAFFT
log("\n--- CRAFFT ---")
crafft_floor = 100 * (crafft_valid['total_score'] == 0).sum() / len(crafft_valid)
crafft_ceil = 100 * (crafft_valid['total_score'] == 6).sum() / len(crafft_valid)
log(f"  Floor (score=0): {crafft_floor:.1f}% {'⚠️ >15%' if crafft_floor > 15 else '✓'}")
log(f"  Ceiling (score=6): {crafft_ceil:.1f}% {'⚠️ >15%' if crafft_ceil > 15 else '✓'}")

# IBD-Disk por item
log("\n--- IBD-Disk (por item) ---")
log(f"  {'Item':<25} {'Floor (0)':>10} {'Ceiling (10)':>13} {'Problema':>10}")
log(f"  {'-'*58}")
for item, label in zip(disk_items, disk_labels):
    valid = ibddisk_valid[item].dropna()
    if len(valid) > 0:
        floor = 100 * (valid == 0).sum() / len(valid)
        ceil = 100 * (valid == 10).sum() / len(valid)
        prob = "FLOOR" if floor > 15 else "CEIL" if ceil > 15 else "ok"
        log(f"  {label:<25} {floor:>9.1f}% {ceil:>12.1f}% {prob:>10}")

# IBD-Disk total
disk_floor = 100 * (ibddisk_valid['total_score'] <= 5).sum() / len(ibddisk_valid)
disk_ceil = 100 * (ibddisk_valid['total_score'] >= 90).sum() / len(ibddisk_valid)
log(f"\n  Total Score Floor (≤5): {disk_floor:.1f}%")
log(f"  Total Score Ceiling (≥90): {disk_ceil:.1f}%")

# IBDQ
log("\n--- IBDQ ---")
# Score range: 32-224
ibdq_floor = 100 * (ibdq_valid['total'] <= 50).sum() / len(ibdq_valid)
ibdq_ceil = 100 * (ibdq_valid['total'] >= 210).sum() / len(ibdq_valid)
log(f"  Floor (≤50): {ibdq_floor:.1f}%")
log(f"  Ceiling (≥210): {ibdq_ceil:.1f}%")


# ==============================================================================
# PARTE D: DISTRIBUIÇÕES DETALHADAS E TESTES DE NORMALIDADE
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE D: DISTRIBUIÇÕES DETALHADAS")
log("=" * 100)

for name, data, total_col in [
    ('CRAFFT Total', crafft_valid, 'total_score'),
    ('IBD-Disk Total', ibddisk_valid, 'total_score'),
    ('IBD-Disk Mean', ibddisk_valid, 'mean_score'),
    ('IBDQ Total', ibdq_valid, 'total'),
    ('IMPACT-III Total', impact_valid, 'total')
]:
    vals = data[total_col].dropna()
    if len(vals) < 3:
        continue
    log(f"\n  {name} (n={len(vals)}):")
    log(f"    Média ± DP: {vals.mean():.2f} ± {vals.std():.2f}")
    log(f"    Mediana [IQR]: {vals.median():.1f} [{vals.quantile(0.25):.1f} - {vals.quantile(0.75):.1f}]")
    log(f"    Min - Max: {vals.min():.1f} - {vals.max():.1f}")
    log(f"    Skewness: {vals.skew():.3f}")
    log(f"    Kurtosis: {vals.kurtosis():.3f}")
    if len(vals) >= 8:
        sw_stat, sw_p = shapiro(vals[:5000])
        ks_stat, ks_p = kstest(vals, 'norm', args=(vals.mean(), vals.std()))
        log(f"    Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.6f}")
        log(f"    Kolmogorov-Smirnov: D={ks_stat:.4f}, p={ks_p:.6f}")
        log(f"    Conclusão: {'Distribuição normal' if sw_p > 0.05 else 'NÃO normal → testes não-paramétricos'}")


# ==============================================================================
# PARTE E: CORRELAÇÕES COMPLETAS COM IC 95% (Bootstrap)
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE E: CORRELAÇÕES COMPLETAS COM IC 95% (Bootstrap, 1000 iterações)")
log("=" * 100)

# Merge todos os datasets
merged = crafft_valid[['paciente', 'total_score', 'crafft_positive'] + crafft_items].rename(
    columns={'total_score': 'crafft_total'})
disk_merge = ibddisk_valid[['paciente', 'total_score', 'mean_score', 'physical_domain',
                             'psychosocial_domain'] + disk_items].rename(
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

def bootstrap_spearman_ci(x, y, n_boot=1000, ci=0.95):
    """Bootstrap IC para correlação de Spearman."""
    valid = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(valid) < 5:
        return np.nan, np.nan, np.nan, np.nan, len(valid)
    rho, p = spearmanr(valid['x'], valid['y'])
    boot_rhos = []
    np.random.seed(42)
    for _ in range(n_boot):
        idx = np.random.choice(len(valid), size=len(valid), replace=True)
        boot_sample = valid.iloc[idx]
        try:
            r, _ = spearmanr(boot_sample['x'], boot_sample['y'])
            boot_rhos.append(r)
        except:
            pass
    boot_rhos = np.array(boot_rhos)
    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_rhos, 100 * alpha)
    ci_high = np.percentile(boot_rhos, 100 * (1 - alpha))
    return rho, p, ci_low, ci_high, len(valid)

# Todas as correlações entre totais
log("\n--- Correlações entre scores totais ---")
total_pairs = [
    ('CRAFFT', 'crafft_total', 'IBD-Disk', 'disk_total'),
    ('CRAFFT', 'crafft_total', 'IBDQ', 'ibdq_total'),
    ('CRAFFT', 'crafft_total', 'IMPACT-III', 'impact_total'),
    ('IBD-Disk', 'disk_total', 'IBDQ', 'ibdq_total'),
    ('IBD-Disk', 'disk_total', 'IMPACT-III', 'impact_total'),
    ('IBDQ', 'ibdq_total', 'IMPACT-III', 'impact_total'),
]

log(f"\n  {'Par':<40} {'rho':>6} {'IC 95%':>18} {'p':>10} {'n':>5} {'Força':>12}")
log(f"  {'-'*95}")
for l1, c1, l2, c2 in total_pairs:
    rho, p, ci_l, ci_h, n = bootstrap_spearman_ci(merged[c1], merged[c2])
    if not np.isnan(rho):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        strength = "Forte" if abs(rho) >= 0.7 else "Moderada" if abs(rho) >= 0.4 else "Fraca" if abs(rho) >= 0.2 else "Negligível"
        log(f"  {l1+' vs '+l2:<40} {rho:>6.3f} [{ci_l:.3f}, {ci_h:.3f}] {p:>10.4f}{sig:>3} {n:>5} {strength:>12}")

# Correlações entre domínios
log("\n--- Correlações entre domínios (IBD-Disk vs IBDQ) ---")
domain_pairs = [
    ('Physical Domain', 'physical_domain', 'Sint. Intestinais', 'sintomas_intestinais'),
    ('Physical Domain', 'physical_domain', 'Sint. Sistêmicos', 'sintomas_sistemicos'),
    ('Physical Domain', 'physical_domain', 'Bem-estar Emocional', 'bem_estar_emocional'),
    ('Physical Domain', 'physical_domain', 'Interação Social', 'interacao_social'),
    ('Psychosocial Domain', 'psychosocial_domain', 'Sint. Intestinais', 'sintomas_intestinais'),
    ('Psychosocial Domain', 'psychosocial_domain', 'Sint. Sistêmicos', 'sintomas_sistemicos'),
    ('Psychosocial Domain', 'psychosocial_domain', 'Bem-estar Emocional', 'bem_estar_emocional'),
    ('Psychosocial Domain', 'psychosocial_domain', 'Interação Social', 'interacao_social'),
]

log(f"\n  {'Par':<50} {'rho':>6} {'IC 95%':>18} {'p':>10} {'n':>5}")
log(f"  {'-'*95}")
for l1, c1, l2, c2 in domain_pairs:
    rho, p, ci_l, ci_h, n = bootstrap_spearman_ci(merged[c1], merged[c2])
    if not np.isnan(rho):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log(f"  {l1+' vs '+l2:<50} {rho:>6.3f} [{ci_l:.3f}, {ci_h:.3f}] {p:>10.4f}{sig:>3} {n:>5}")

# Correlações IBD-Disk itens individuais vs IBDQ total
log("\n--- Correlações: Itens individuais IBD-Disk vs IBDQ Total ---")
for item, label in zip(disk_items, disk_labels):
    rho, p, ci_l, ci_h, n = bootstrap_spearman_ci(merged[item], merged['ibdq_total'])
    if not np.isnan(rho):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log(f"  {label:<25}: rho={rho:.3f} [{ci_l:.3f}, {ci_h:.3f}] p={p:.4f}{sig} (n={n})")

# Kendall's tau como medida alternativa (mais robusta para amostras pequenas)
log("\n--- Kendall's tau (robustez adicional) — scores totais ---")
for l1, c1, l2, c2 in total_pairs:
    valid = merged[[c1, c2]].dropna()
    if len(valid) >= 5:
        tau, p = kendalltau(valid[c1], valid[c2])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log(f"  {l1+' vs '+l2:<40}: tau={tau:.3f}, p={p:.4f} {sig} (n={len(valid)})")


# ==============================================================================
# PARTE F: COMPARAÇÕES CRAFFT+ vs CRAFFT- COM EFFECT SIZES
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE F: COMPARAÇÕES CRAFFT+ vs CRAFFT- (Mann-Whitney U + Effect Sizes)")
log("=" * 100)

merged_c = merged[merged['crafft_positive'].notna()].copy()
pos = merged_c[merged_c['crafft_positive'] == True]
neg = merged_c[merged_c['crafft_positive'] == False]

log(f"\n  CRAFFT+ (n={len(pos)}) vs CRAFFT- (n={len(neg)})")

vars_compare = [
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

log(f"\n  {'Variável':<28} {'Pos med(IQR)':<20} {'Neg med(IQR)':<20} {'U':>8} {'p':>8} {'r':>6} {'Efeito':>10}")
log(f"  {'-'*100}")

for label, col in vars_compare:
    pos_vals = pos[col].dropna()
    neg_vals = neg[col].dropna()
    if len(pos_vals) >= 3 and len(neg_vals) >= 3:
        U, p = mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
        # Effect size r = Z/sqrt(N)
        n1, n2 = len(pos_vals), len(neg_vals)
        mu_U = n1 * n2 / 2
        sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        Z = (U - mu_U) / sigma_U
        r = abs(Z) / np.sqrt(n1 + n2)
        effect = "grande" if r >= 0.5 else "médio" if r >= 0.3 else "pequeno" if r >= 0.1 else "nulo"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        pos_iqr = f"{pos_vals.median():.0f} ({pos_vals.quantile(0.25):.0f}-{pos_vals.quantile(0.75):.0f})"
        neg_iqr = f"{neg_vals.median():.0f} ({neg_vals.quantile(0.25):.0f}-{neg_vals.quantile(0.75):.0f})"
        log(f"  {label:<28} {pos_iqr:<20} {neg_iqr:<20} {U:>8.0f} {p:>7.4f}{sig} {r:>5.3f} {effect:>10}")

# Point-biserial correlation CRAFFT positivo vs contínuas
log("\n--- Correlação ponto-bisserial: CRAFFT positivo (dicotômico) vs scores ---")
for label, col in vars_compare:
    valid = merged_c[['crafft_positive', col]].dropna()
    if len(valid) >= 5:
        rpb, p = pointbiserialr(valid['crafft_positive'].astype(int), valid[col])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log(f"  CRAFFT+ vs {label}: rpb={rpb:.3f}, p={p:.4f} {sig} (n={len(valid)})")


# ==============================================================================
# PARTE G: ANÁLISE ITEM-A-ITEM DO CRAFFT vs IBD-DISK
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE G: ANÁLISE ITEM-A-ITEM — CRAFFT items vs IBD-Disk items")
log("=" * 100)
log("  (Identificar quais domínios de incapacidade se associam a quais comportamentos de risco)")

log(f"\n  {'CRAFFT item':<15} {'IBD-Disk item':<25} {'rho':>6} {'p':>8} {'Sig':>5}")
log(f"  {'-'*65}")
significant_item_pairs = []
for c_item in crafft_items:
    for d_item, d_label in zip(disk_items, disk_labels):
        valid = merged[[c_item, d_item]].dropna()
        if len(valid) >= 10:
            rho, p = spearmanr(valid[c_item], valid[d_item])
            if p < 0.05:
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                log(f"  {c_item:<15} {d_label:<25} {rho:>6.3f} {p:>8.4f} {sig:>5}")
                significant_item_pairs.append((c_item, d_label, rho, p))

if not significant_item_pairs:
    log("  Nenhuma correlação item-a-item significativa encontrada")
else:
    log(f"\n  → {len(significant_item_pairs)} pares significativos identificados")


# ==============================================================================
# PARTE H: REGRESSÃO LINEAR MÚLTIPLA — Preditores de QoL
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE H: REGRESSÃO LINEAR — Preditores de Qualidade de Vida (IBDQ)")
log("=" * 100)

from numpy.linalg import lstsq

# Preditores: itens do IBD-Disk, CRAFFT positivo
reg_data = merged[['ibdq_total', 'crafft_positive', 'disk_total',
                    'physical_domain', 'psychosocial_domain']].dropna()

if len(reg_data) >= 20:
    log(f"\n  n = {len(reg_data)} pacientes com dados completos para regressão")

    # Modelo 1: IBD-Disk total → IBDQ
    X = reg_data[['disk_total']].values
    X = np.column_stack([np.ones(len(X)), X])
    y = reg_data['ibdq_total'].values
    beta, residuals, rank, sv = lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    log(f"\n  Modelo 1: IBDQ = β0 + β1×IBD-Disk_Total")
    log(f"    β0 (intercept) = {beta[0]:.2f}")
    log(f"    β1 (IBD-Disk) = {beta[1]:.3f}")
    log(f"    R² = {r_squared:.4f}, R² ajustado = {adj_r2:.4f}")
    log(f"    Interpretação: IBD-Disk explica {100*r_squared:.1f}% da variância do IBDQ")

    # Modelo 2: Physical + Psychosocial → IBDQ
    X2 = reg_data[['physical_domain', 'psychosocial_domain']].values
    X2 = np.column_stack([np.ones(len(X2)), X2])
    beta2, _, _, _ = lstsq(X2, y, rcond=None)
    y_pred2 = X2 @ beta2
    ss_res2 = np.sum((y - y_pred2) ** 2)
    r2_2 = 1 - ss_res2 / ss_tot
    adj_r2_2 = 1 - (1 - r2_2) * (len(y) - 1) / (len(y) - X2.shape[1] - 1)
    log(f"\n  Modelo 2: IBDQ = β0 + β1×Physical + β2×Psychosocial")
    log(f"    β0 = {beta2[0]:.2f}")
    log(f"    β1 (Physical) = {beta2[1]:.3f}")
    log(f"    β2 (Psychosocial) = {beta2[2]:.3f}")
    log(f"    R² = {r2_2:.4f}, R² ajustado = {adj_r2_2:.4f}")

    # Modelo 3: IBD-Disk + CRAFFT → IBDQ
    X3 = reg_data[['disk_total', 'crafft_positive']].values.astype(float)
    X3 = np.column_stack([np.ones(len(X3)), X3])
    beta3, _, _, _ = lstsq(X3, y, rcond=None)
    y_pred3 = X3 @ beta3
    ss_res3 = np.sum((y - y_pred3) ** 2)
    r2_3 = 1 - ss_res3 / ss_tot
    adj_r2_3 = 1 - (1 - r2_3) * (len(y) - 1) / (len(y) - X3.shape[1] - 1)
    log(f"\n  Modelo 3: IBDQ = β0 + β1×IBD-Disk + β2×CRAFFT_positivo")
    log(f"    β0 = {beta3[0]:.2f}")
    log(f"    β1 (IBD-Disk) = {beta3[1]:.3f}")
    log(f"    β2 (CRAFFT+) = {beta3[2]:.3f}")
    log(f"    R² = {r2_3:.4f}, R² ajustado = {adj_r2_3:.4f}")
    delta_r2 = r2_3 - r_squared
    log(f"    ΔR² ao adicionar CRAFFT: {delta_r2:.4f} ({100*delta_r2:.1f}% variância adicional)")


# ==============================================================================
# PARTE I: ANÁLISE DE CLUSTER — Perfis de Pacientes IBD-Disk
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE I: PERFIS DE PACIENTES POR PADRÃO DE COMPROMETIMENTO (IBD-Disk)")
log("=" * 100)

# Classificar cada paciente pelo padrão predominante
disk_complete = ibddisk_valid[disk_items].dropna()
if len(disk_complete) >= 20:
    # Definir domínios compostos
    disk_complete_copy = disk_complete.copy()
    disk_complete_copy['physical'] = disk_complete_copy[['item1_abd_pain', 'item2_defecation',
                                                          'item5_sleep', 'item6_energy',
                                                          'item10_joint_pain']].mean(axis=1)
    disk_complete_copy['psychosocial'] = disk_complete_copy[['item3_interpersonal', 'item4_education',
                                                              'item7_emotions', 'item8_body_image',
                                                              'item9_sexual']].mean(axis=1)

    # Categorizar perfis
    conditions = [
        (disk_complete_copy['physical'] >= 5) & (disk_complete_copy['psychosocial'] >= 5),
        (disk_complete_copy['physical'] >= 5) & (disk_complete_copy['psychosocial'] < 5),
        (disk_complete_copy['physical'] < 5) & (disk_complete_copy['psychosocial'] >= 5),
        (disk_complete_copy['physical'] < 5) & (disk_complete_copy['psychosocial'] < 5),
    ]
    labels_profile = ['Misto (alto ambos)', 'Predominante Físico', 'Predominante Psicossocial', 'Baixo (ambos)']
    disk_complete_copy['perfil'] = np.select(conditions, labels_profile, default='Indefinido')

    log(f"\n  Perfis de comprometimento (n={len(disk_complete_copy)}):")
    for perfil in labels_profile:
        n = (disk_complete_copy['perfil'] == perfil).sum()
        pct = 100 * n / len(disk_complete_copy)
        subset = disk_complete_copy[disk_complete_copy['perfil'] == perfil]
        log(f"\n    {perfil}: {n} ({pct:.1f}%)")
        log(f"      Physical mean: {subset['physical'].mean():.1f} ± {subset['physical'].std():.1f}")
        log(f"      Psychosocial mean: {subset['psychosocial'].mean():.1f} ± {subset['psychosocial'].std():.1f}")

    # Verificar se perfis diferem em IBDQ
    merged_profile = merged.merge(
        disk_complete_copy[['perfil']].reset_index().rename(columns={'index': 'idx'}),
        left_index=True, right_on='idx', how='inner'
    ) if False else None

    # Usar abordagem diferente para merge
    ibddisk_valid_profile = ibddisk_valid.copy()
    ibddisk_valid_profile['physical_comp'] = ibddisk_valid_profile[['item1_abd_pain', 'item2_defecation',
                                                                     'item5_sleep', 'item6_energy',
                                                                     'item10_joint_pain']].mean(axis=1)
    ibddisk_valid_profile['psychosocial_comp'] = ibddisk_valid_profile[['item3_interpersonal', 'item4_education',
                                                                         'item7_emotions', 'item8_body_image',
                                                                         'item9_sexual']].mean(axis=1)
    conditions_p = [
        (ibddisk_valid_profile['physical_comp'] >= 5) & (ibddisk_valid_profile['psychosocial_comp'] >= 5),
        (ibddisk_valid_profile['physical_comp'] >= 5) & (ibddisk_valid_profile['psychosocial_comp'] < 5),
        (ibddisk_valid_profile['physical_comp'] < 5) & (ibddisk_valid_profile['psychosocial_comp'] >= 5),
        (ibddisk_valid_profile['physical_comp'] < 5) & (ibddisk_valid_profile['psychosocial_comp'] < 5),
    ]
    ibddisk_valid_profile['perfil'] = np.select(conditions_p, labels_profile, default='Indefinido')

    profile_merge = merged.merge(ibddisk_valid_profile[['paciente', 'perfil']], on='paciente', how='inner')
    log("\n  Kruskal-Wallis: Perfis vs IBDQ Total")
    groups_kw = []
    for perfil in labels_profile:
        vals = profile_merge[profile_merge['perfil'] == perfil]['ibdq_total'].dropna()
        if len(vals) >= 2:
            groups_kw.append((perfil, vals))
    if len(groups_kw) >= 2:
        H, p = kruskal(*[g[1] for g in groups_kw])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log(f"    H={H:.2f}, p={p:.4f} {sig}")
        for perfil, vals in groups_kw:
            log(f"    {perfil}: mediana IBDQ = {vals.median():.0f} (n={len(vals)})")


# ==============================================================================
# PARTE J: ANÁLISE DOS DOMÍNIOS IBDQ DETALHADA
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE J: ANÁLISE DETALHADA DOS DOMÍNIOS IBDQ")
log("=" * 100)

# Nota: IBDQ tem 2 padrões — versão curta (11 itens → só sint intest e sist) e longa (32 itens)
ibdq_short = ibdq_valid[ibdq_valid['bem_estar_emocional'] == 0].copy()
ibdq_long = ibdq_valid[ibdq_valid['bem_estar_emocional'] > 0].copy()

log(f"\n  IBDQ versão curta (11 itens): {len(ibdq_short)} pacientes")
log(f"  IBDQ versão longa (32 itens): {len(ibdq_long)} pacientes")

if len(ibdq_long) >= 5:
    log(f"\n  --- Estatísticas da versão longa (32 itens, n={len(ibdq_long)}) ---")
    log(f"  Total: {ibdq_long['total'].mean():.1f} ± {ibdq_long['total'].std():.1f}")
    log(f"    Sint. Intestinais: {ibdq_long['sintomas_intestinais'].mean():.1f} ± {ibdq_long['sintomas_intestinais'].std():.1f}")
    log(f"    Sint. Sistêmicos: {ibdq_long['sintomas_sistemicos'].mean():.1f} ± {ibdq_long['sintomas_sistemicos'].std():.1f}")
    log(f"    Bem-estar Emocional: {ibdq_long['bem_estar_emocional'].mean():.1f} ± {ibdq_long['bem_estar_emocional'].std():.1f}")
    log(f"    Interação Social: {ibdq_long['interacao_social'].mean():.1f} ± {ibdq_long['interacao_social'].std():.1f}")

    # Proporção de cada domínio no total
    dom_means = {
        'Sint. Intestinais': ibdq_long['sintomas_intestinais'].mean(),
        'Sint. Sistêmicos': ibdq_long['sintomas_sistemicos'].mean(),
        'Bem-estar Emocional': ibdq_long['bem_estar_emocional'].mean(),
        'Interação Social': ibdq_long['interacao_social'].mean(),
    }
    total_mean = ibdq_long['total'].mean()
    log(f"\n  Contribuição de cada domínio para o total:")
    for dom, mean in dom_means.items():
        pct = 100 * mean / total_mean if total_mean > 0 else 0
        log(f"    {dom}: {pct:.1f}%")

    # Correlação entre domínios IBDQ
    log(f"\n  Correlações inter-domínios IBDQ (Spearman):")
    ibdq_doms = ['sintomas_intestinais', 'sintomas_sistemicos', 'bem_estar_emocional', 'interacao_social']
    ibdq_dom_labels = ['Sint.Int.', 'Sint.Sist.', 'Bem-estar', 'Social']
    for i in range(len(ibdq_doms)):
        for j in range(i+1, len(ibdq_doms)):
            valid = ibdq_long[[ibdq_doms[i], ibdq_doms[j]]].dropna()
            if len(valid) >= 5:
                rho, p = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                log(f"    {ibdq_dom_labels[i]} vs {ibdq_dom_labels[j]}: rho={rho:.3f}, p={p:.4f} {sig}")

if len(ibdq_short) >= 5:
    log(f"\n  --- Estatísticas da versão curta (11 itens, n={len(ibdq_short)}) ---")
    log(f"  Total: {ibdq_short['total'].mean():.1f} ± {ibdq_short['total'].std():.1f}")
    log(f"    Sint. Intestinais: {ibdq_short['sintomas_intestinais'].mean():.1f} ± {ibdq_short['sintomas_intestinais'].std():.1f}")
    log(f"    Sint. Sistêmicos: {ibdq_short['sintomas_sistemicos'].mean():.1f} ± {ibdq_short['sintomas_sistemicos'].std():.1f}")

    # Comparar versão curta vs longa nos domínios em comum
    log(f"\n  --- Comparação versão curta vs longa (domínios comuns) ---")
    for dom, label in [('sintomas_intestinais', 'Sint. Intestinais'), ('sintomas_sistemicos', 'Sint. Sistêmicos')]:
        short_vals = ibdq_short[dom].dropna()
        long_vals = ibdq_long[dom].dropna()
        if len(short_vals) >= 3 and len(long_vals) >= 3:
            U, p = mannwhitneyu(short_vals, long_vals, alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            log(f"  {label}: Curta med={short_vals.median():.0f} vs Longa med={long_vals.median():.0f}, p={p:.4f} {sig}")


# ==============================================================================
# PARTE K: ANÁLISE TEMPORAL DETALHADA
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE K: ANÁLISE TEMPORAL DETALHADA")
log("=" * 100)

# IBD-Disk temporal
ibddisk_valid['date'] = ibddisk_valid['data_avaliacao'].apply(parse_date)
ibddisk_dated = ibddisk_valid[ibddisk_valid['date'].notna()].copy()
ibddisk_dated['year'] = ibddisk_dated['date'].dt.year

log(f"\n  IBD-Disk com data válida: {len(ibddisk_dated)}")
if len(ibddisk_dated) >= 10:
    # Correlação temporal: score vs tempo (tendência)
    ibddisk_dated['days_since_start'] = (ibddisk_dated['date'] - ibddisk_dated['date'].min()).dt.days
    rho, p = spearmanr(ibddisk_dated['days_since_start'], ibddisk_dated['total_score'])
    log(f"  Tendência temporal IBD-Disk Total: rho={rho:.3f}, p={p:.4f}")
    log(f"    {'Scores aumentando ao longo do tempo' if rho > 0 else 'Scores diminuindo ao longo do tempo' if rho < 0 else 'Sem tendência'}")

    # Comparar períodos
    ibddisk_dated['period'] = pd.cut(ibddisk_dated['year'],
                                      bins=[2020, 2022, 2024, 2027],
                                      labels=['2021-2022', '2023-2024', '2025-2026'])
    log(f"\n  IBD-Disk por período:")
    period_groups = []
    for period in ['2021-2022', '2023-2024', '2025-2026']:
        subset = ibddisk_dated[ibddisk_dated['period'] == period]['total_score']
        if len(subset) >= 3:
            period_groups.append(subset)
            log(f"    {period}: n={len(subset)}, mediana={subset.median():.0f} [{subset.quantile(0.25):.0f}-{subset.quantile(0.75):.0f}]")

    if len(period_groups) >= 2:
        H, p = kruskal(*period_groups)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log(f"    Kruskal-Wallis entre períodos: H={H:.2f}, p={p:.4f} {sig}")

# CRAFFT temporal
crafft_dated = crafft_valid[crafft_valid['date'].notna()].copy()
crafft_dated['year'] = crafft_dated['date'].dt.year
crafft_dated['days_since_start'] = (crafft_dated['date'] - crafft_dated['date'].min()).dt.days

log(f"\n  CRAFFT tendência temporal:")
rho, p = spearmanr(crafft_dated['days_since_start'], crafft_dated['total_score'])
log(f"    Score total vs tempo: rho={rho:.3f}, p={p:.4f}")

# Tendência por item
log(f"    Tendência por item CRAFFT:")
for item in crafft_items:
    valid = crafft_dated[crafft_dated[item].notna()]
    if len(valid) >= 10:
        rho, p = spearmanr(valid['days_since_start'], valid[item])
        if p < 0.1:  # Report marginally significant
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.1 else ""
            log(f"      {item}: rho={rho:.3f}, p={p:.4f} {sig}")


# ==============================================================================
# PARTE L: ANÁLISE DE CONCORDÂNCIA ENTRE INSTRUMENTOS
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE L: CONCORDÂNCIA ENTRE INSTRUMENTOS (Convergent Validity)")
log("=" * 100)

# IBD-Disk alto vs IBDQ baixo — concordância
merged_for_agreement = merged[['disk_total', 'ibdq_total']].dropna()
if len(merged_for_agreement) >= 10:
    disk_median = merged_for_agreement['disk_total'].median()
    ibdq_median = merged_for_agreement['ibdq_total'].median()

    # IBDQ invertido: score ALTO = melhor QoL. IBD-Disk: score ALTO = mais incapacidade
    # Portanto esperamos: disk alto + ibdq baixo (concordância)
    merged_for_agreement['disk_high'] = merged_for_agreement['disk_total'] >= disk_median
    merged_for_agreement['ibdq_low'] = merged_for_agreement['ibdq_total'] < ibdq_median

    # Tabela 2x2
    concordant = ((merged_for_agreement['disk_high'] & merged_for_agreement['ibdq_low']) |
                   (~merged_for_agreement['disk_high'] & ~merged_for_agreement['ibdq_low'])).sum()
    discordant = len(merged_for_agreement) - concordant
    pct_agreement = 100 * concordant / len(merged_for_agreement)

    log(f"\n  IBD-Disk (alta incapacidade) vs IBDQ (baixa QoL):")
    log(f"    Concordantes: {concordant}/{len(merged_for_agreement)} ({pct_agreement:.1f}%)")
    log(f"    Discordantes: {discordant}/{len(merged_for_agreement)} ({100-pct_agreement:.1f}%)")

    # Tabela cruzada
    ct = pd.crosstab(merged_for_agreement['disk_high'], merged_for_agreement['ibdq_low'])
    log(f"\n    Tabela cruzada:")
    log(f"    {'':>20} {'IBDQ baixo':>12} {'IBDQ alto':>12}")
    log(f"    {'Disk alto':<20} {ct.iloc[1,1] if ct.shape[0]>1 and ct.shape[1]>1 else 0:>12} {ct.iloc[1,0] if ct.shape[0]>1 else 0:>12}")
    log(f"    {'Disk baixo':<20} {ct.iloc[0,1] if ct.shape[1]>1 else 0:>12} {ct.iloc[0,0]:>12}")

    # Chi-squared
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p_chi, dof, expected = chi2_contingency(ct)
        sig = "***" if p_chi < 0.001 else "**" if p_chi < 0.01 else "*" if p_chi < 0.05 else "ns"
        log(f"\n    Chi²={chi2:.2f}, df={dof}, p={p_chi:.4f} {sig}")

        # Odds ratio
        if ct.iloc[0,0] > 0 and ct.iloc[1,0] > 0:
            or_val = (ct.iloc[1,1] * ct.iloc[0,0]) / (ct.iloc[1,0] * ct.iloc[0,1]) if ct.iloc[1,0] > 0 and ct.iloc[0,1] > 0 else np.inf
            log(f"    Odds Ratio: {or_val:.2f}")


# ==============================================================================
# PARTE M: BURDEN COMPOSTO — Índice Multidimensional
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE M: ÍNDICE DE BURDEN COMPOSTO")
log("=" * 100)
log("  (Combinando informação de múltiplos instrumentos num único indicador)")

# Para pacientes com IBD-Disk + IBDQ + CRAFFT
burden_data = merged[['paciente', 'crafft_total', 'crafft_positive',
                       'disk_total', 'ibdq_total']].dropna(subset=['disk_total', 'ibdq_total', 'crafft_total'])

if len(burden_data) >= 10:
    # Z-scores normalizados
    burden_data['disk_z'] = (burden_data['disk_total'] - burden_data['disk_total'].mean()) / burden_data['disk_total'].std()
    # IBDQ invertido (score alto = melhor)
    burden_data['ibdq_z_inv'] = -(burden_data['ibdq_total'] - burden_data['ibdq_total'].mean()) / burden_data['ibdq_total'].std()
    burden_data['crafft_z'] = (burden_data['crafft_total'] - burden_data['crafft_total'].mean()) / burden_data['crafft_total'].std()

    # Índice composto (média dos z-scores)
    burden_data['burden_index'] = (burden_data['disk_z'] + burden_data['ibdq_z_inv'] + burden_data['crafft_z']) / 3

    log(f"\n  Burden Index (n={len(burden_data)}):")
    log(f"    Média ± DP: {burden_data['burden_index'].mean():.3f} ± {burden_data['burden_index'].std():.3f}")
    log(f"    Mediana [IQR]: {burden_data['burden_index'].median():.3f} "
        f"[{burden_data['burden_index'].quantile(0.25):.3f} - {burden_data['burden_index'].quantile(0.75):.3f}]")
    log(f"    Min - Max: {burden_data['burden_index'].min():.3f} - {burden_data['burden_index'].max():.3f}")

    # Categorizar burden
    burden_data['burden_cat'] = pd.qcut(burden_data['burden_index'], q=3,
                                         labels=['Baixo', 'Moderado', 'Alto'])
    for cat in ['Baixo', 'Moderado', 'Alto']:
        subset = burden_data[burden_data['burden_cat'] == cat]
        log(f"\n    Burden {cat} (n={len(subset)}):")
        log(f"      IBD-Disk: {subset['disk_total'].median():.0f}")
        log(f"      IBDQ: {subset['ibdq_total'].median():.0f}")
        log(f"      CRAFFT+: {100*subset['crafft_positive'].mean():.0f}%")
        log(f"      Burden Index: {subset['burden_index'].median():.3f}")

    # CRAFFT+ vs burden category
    ct_burden = pd.crosstab(burden_data['burden_cat'], burden_data['crafft_positive'])
    log(f"\n    CRAFFT+ por categoria de burden:")
    for cat in ['Baixo', 'Moderado', 'Alto']:
        subset = burden_data[burden_data['burden_cat'] == cat]
        n_pos = subset['crafft_positive'].sum()
        pct = 100 * n_pos / len(subset) if len(subset) > 0 else 0
        log(f"      {cat}: {n_pos}/{len(subset)} CRAFFT+ ({pct:.0f}%)")


# ==============================================================================
# PARTE N: FIGURAS ADICIONAIS
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE N: FIGURAS ADICIONAIS")
log("=" * 100)

# Figura 9: Violin plots IBD-Disk por domínio
fig, ax = plt.subplots(figsize=(16, 8))
disk_melt = ibddisk_valid[disk_items].melt(var_name='Item', value_name='Score')
disk_melt['Item'] = disk_melt['Item'].map(dict(zip(disk_items, disk_labels)))
disk_melt = disk_melt.dropna()
sns.violinplot(data=disk_melt, x='Item', y='Score', ax=ax, inner='quartile', palette='Set2')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Score (0-10)')
ax.set_title('Distribuição dos Scores IBD-Disk por Domínio (Violin Plot)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig9_ibddisk_violin.png'), bbox_inches='tight')
plt.close()
log("  Figura 9: fig9_ibddisk_violin.png ✓")

# Figura 10: Heatmap IBD-Disk itens
fig, ax = plt.subplots(figsize=(10, 10))
disk_corr = ibddisk_valid[disk_items].corr(method='spearman')
disk_corr.columns = disk_labels
disk_corr.index = disk_labels
sns.heatmap(disk_corr, annot=True, fmt='.2f', cmap='YlOrRd', square=True,
            linewidths=.5, ax=ax, vmin=0, vmax=1)
ax.set_title('Correlação entre Itens IBD-Disk (Spearman)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig10_ibddisk_item_corr.png'), bbox_inches='tight')
plt.close()
log("  Figura 10: fig10_ibddisk_item_corr.png ✓")

# Figura 11: IBDQ por versão (curta vs longa)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
if len(ibdq_short) > 0 and len(ibdq_long) > 0:
    axes[0].hist(ibdq_short['total'], bins=15, alpha=0.7, color='steelblue', edgecolor='black', label=f'Curta (n={len(ibdq_short)})')
    axes[0].hist(ibdq_long['total'], bins=15, alpha=0.7, color='coral', edgecolor='black', label=f'Longa (n={len(ibdq_long)})')
    axes[0].set_xlabel('IBDQ Total')
    axes[0].set_ylabel('Frequência')
    axes[0].set_title('Distribuição IBDQ: Versão Curta vs Longa')
    axes[0].legend()

# Domínios da versão longa
if len(ibdq_long) >= 5:
    domain_data = ibdq_long[['sintomas_intestinais', 'sintomas_sistemicos', 'bem_estar_emocional', 'interacao_social']]
    domain_data.columns = ['Sint.Intest.', 'Sint.Sist.', 'Bem-estar', 'Social']
    domain_data.boxplot(ax=axes[1], patch_artist=True,
                         boxprops=dict(facecolor='lightyellow', alpha=0.8),
                         medianprops=dict(color='red', linewidth=2))
    axes[1].set_title(f'Domínios IBDQ — Versão Longa (n={len(ibdq_long)})')
    axes[1].set_ylabel('Score')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig11_ibdq_versoes.png'), bbox_inches='tight')
plt.close()
log("  Figura 11: fig11_ibdq_versoes.png ✓")

# Figura 12: Perfis de comprometimento (scatter physical vs psychosocial)
fig, ax = plt.subplots(figsize=(10, 10))
if len(ibddisk_valid) > 0:
    phys = ibddisk_valid[['item1_abd_pain', 'item2_defecation', 'item5_sleep',
                           'item6_energy', 'item10_joint_pain']].mean(axis=1)
    psych = ibddisk_valid[['item3_interpersonal', 'item4_education', 'item7_emotions',
                            'item8_body_image', 'item9_sexual']].mean(axis=1)

    ax.scatter(phys, psych, alpha=0.6, s=60, c=ibddisk_valid['total_score'],
               cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='IBD-Disk Total Score')
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Domínio Físico (média)')
    ax.set_ylabel('Domínio Psicossocial (média)')
    ax.set_title('Perfis de Comprometimento IBD-Disk')
    ax.text(7.5, 7.5, 'Misto', fontsize=12, ha='center', fontweight='bold', color='red')
    ax.text(7.5, 2.5, 'Físico', fontsize=12, ha='center', fontweight='bold', color='orange')
    ax.text(2.5, 7.5, 'Psicossocial', fontsize=12, ha='center', fontweight='bold', color='purple')
    ax.text(2.5, 2.5, 'Baixo', fontsize=12, ha='center', fontweight='bold', color='green')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig12_perfis_comprometimento.png'), bbox_inches='tight')
plt.close()
log("  Figura 12: fig12_perfis_comprometimento.png ✓")

# Figura 13: Burden index distribution
if len(burden_data) >= 10:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    burden_data['burden_index'].hist(bins=20, ax=axes[0], color='mediumpurple', edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Burden Index')
    axes[0].set_ylabel('Frequência')
    axes[0].set_title('Distribuição do Índice de Burden Composto')

    cat_data = burden_data.groupby('burden_cat').agg(
        disk=('disk_total', 'median'),
        ibdq=('ibdq_total', 'median'),
        crafft_pct=('crafft_positive', lambda x: 100*x.mean())
    )
    x = np.arange(3)
    width = 0.25
    axes[1].bar(x - width, cat_data['disk'], width, label='IBD-Disk (med)', color='steelblue', alpha=0.8)
    axes[1].bar(x, cat_data['ibdq'], width, label='IBDQ (med)', color='forestgreen', alpha=0.8)
    axes[1].bar(x + width, cat_data['crafft_pct'], width, label='% CRAFFT+', color='coral', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Baixo', 'Moderado', 'Alto'])
    axes[1].set_title('Indicadores por Categoria de Burden')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig13_burden_index.png'), bbox_inches='tight')
    plt.close()
    log("  Figura 13: fig13_burden_index.png ✓")

# Figura 14: Missing data pattern
fig, ax = plt.subplots(figsize=(10, 6))
instruments = ['CRAFFT', 'IBD-Disk', 'IBDQ', 'IMPACT-III']
valid_counts = [len(crafft_valid), len(ibddisk_valid), len(ibdq_valid), len(impact_valid)]
total_counts = [len(crafft), len(ibddisk_all), len(ibdq_all), len(impact_all)]
missing_counts = [t - v for t, v in zip(total_counts, valid_counts)]

x = np.arange(len(instruments))
ax.bar(x, valid_counts, color='steelblue', alpha=0.8, label='Dados válidos')
ax.bar(x, missing_counts, bottom=valid_counts, color='lightcoral', alpha=0.8, label='Dados faltantes')
ax.set_xticks(x)
ax.set_xticklabels(instruments)
ax.set_ylabel('Nº de Pacientes')
ax.set_title('Completude dos Dados por Instrumento')
ax.legend()
for i, (v, t) in enumerate(zip(valid_counts, total_counts)):
    ax.text(i, t + 1, f'{100*v/t:.0f}%', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig14_missing_data.png'), bbox_inches='tight')
plt.close()
log("  Figura 14: fig14_missing_data.png ✓")


# ==============================================================================
# PARTE O: TABELAS PARA O ARTIGO
# ==============================================================================
log("\n\n" + "=" * 100)
log("PARTE O: TABELAS FORMATADAS PARA O ARTIGO")
log("=" * 100)

# Tabela 1: Características descritivas
log("\n--- TABELA 1: Estatísticas Descritivas dos Instrumentos ---")
log(f"{'Instrumento':<20} {'N':>5} {'Média':>8} {'DP':>8} {'Mediana':>8} {'IQR':>15} {'Min':>6} {'Max':>6}")
log("-" * 80)
for name, df, col in [('CRAFFT Total', crafft_valid, 'total_score'),
                       ('IBD-Disk Total', ibddisk_valid, 'total_score'),
                       ('IBD-Disk Mean', ibddisk_valid, 'mean_score'),
                       ('IBDQ Total', ibdq_valid, 'total'),
                       ('IMPACT-III Total', impact_valid, 'total')]:
    v = df[col].dropna()
    if len(v) > 0:
        iqr = f"{v.quantile(0.25):.0f}-{v.quantile(0.75):.0f}"
        log(f"{name:<20} {len(v):>5} {v.mean():>8.1f} {v.std():>8.1f} {v.median():>8.1f} {iqr:>15} {v.min():>6.0f} {v.max():>6.0f}")

# Tabela 2: IBD-Disk por item
log(f"\n--- TABELA 2: IBD-Disk — Scores por Item ---")
log(f"{'Item':<25} {'N':>5} {'Média':>7} {'DP':>7} {'Med':>5} {'IQR':>12} {'≥5 (%)':>10}")
log("-" * 75)
for item, label in zip(disk_items, disk_labels):
    v = ibddisk_valid[item].dropna()
    if len(v) > 0:
        iqr = f"{v.quantile(0.25):.0f}-{v.quantile(0.75):.0f}"
        pct5 = f"{100*(v>=5).sum()/len(v):.1f}"
        log(f"{label:<25} {len(v):>5} {v.mean():>7.1f} {v.std():>7.1f} {v.median():>5.0f} {iqr:>12} {pct5:>10}")

# Tabela 3: CRAFFT prevalência
log(f"\n--- TABELA 3: CRAFFT — Prevalência por Item ---")
log(f"{'Item':<20} {'N resp':>8} {'Positivo':>10} {'%':>8}")
log("-" * 50)
for item in crafft_items:
    v = crafft_with_items[item].dropna()
    n_pos = int(v.sum())
    pct = 100 * n_pos / len(v) if len(v) > 0 else 0
    log(f"{item:<20} {len(v):>8} {n_pos:>10} {pct:>7.1f}%")
v_total = crafft_valid[crafft_valid['total_score'] >= 2]
log(f"{'Screen Positivo':<20} {len(crafft_valid):>8} {len(v_total):>10} {100*len(v_total)/len(crafft_valid):>7.1f}%")


# ==============================================================================
# SALVAR
# ==============================================================================
report_text = "\n".join(report)
with open(os.path.join(OUT_DIR, "relatorio_aprofundado.txt"), 'w') as f:
    f.write(report_text)

# Salvar merged atualizado
merged.to_csv(os.path.join(OUT_DIR, "dataset_merged_completo.csv"), index=False)

if 'burden_data' in dir() and len(burden_data) >= 10:
    burden_data.to_csv(os.path.join(OUT_DIR, "burden_index.csv"), index=False)

log(f"\n\nTodos os arquivos salvos em: {OUT_DIR}")
log(f"Figuras: {FIG_DIR}")
log("=" * 100)
log("ANÁLISE APROFUNDADA COMPLETA!")
log("=" * 100)
