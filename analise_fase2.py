#!/usr/bin/env python3
"""
FASE 2 — ANÁLISE ESTATÍSTICA AVANÇADA
IBD Pediátrica em LMIC (Brasil)
JCC Plus — Special Issue: Global Burden of IBD

Inclui:
  P1: Correção para comparações múltiplas (Benjamini-Hochberg FDR)
  P2: Regressão logística — preditores de CRAFFT+
  P3: Análise dose-resposta CRAFFT score vs IBD-Disk/IBDQ
  P4: Subgrupo IBDQ curta vs longa comparado em IBD-Disk
  P5: Análise de sensibilidade (casos completos)
  P6: Power analysis post-hoc
  P7: ROC curve — IBD-Disk predizendo IBDQ grave
  P8: Multicolinearidade e diagnóstico de regressão
  P9: Effect sizes com IC bootstrapped
  P10: Figuras publicação (forest plot, ROC, etc.)
  P11: Tabelas STROBE adicionais
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (mannwhitneyu, kruskal, chi2_contingency, fisher_exact,
                          spearmanr, pearsonr, shapiro, kendalltau,
                          pointbiserialr, norm, binom)
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
log("FASE 2 — ANÁLISE ESTATÍSTICA AVANÇADA — IBD PEDIÁTRICA EM LMIC (BRASIL)")
log(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
log("=" * 100)

# ==============================================================================
# CARREGAMENTO DE DADOS (idêntico à Fase 1)
# ==============================================================================

def parse_date(d):
    if pd.isna(d) or str(d).strip() == '' or 'NÃO' in str(d).upper() or 'SEM' in str(d).upper():
        return pd.NaT
    try:
        return pd.to_datetime(d, dayfirst=True)
    except:
        return pd.NaT

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
crafft['date'] = crafft['data_avaliacao'].apply(parse_date)
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
impact_valid = impact[impact['total'].notna() & (impact['total'] > 0)].copy()

# ==============================================================================
# MERGE DATASETS
# ==============================================================================
crafft_merge = crafft_valid[['paciente', 'total_score', 'crafft_positive'] + crafft_items].rename(
    columns={'total_score': 'crafft_total'})
disk_merge = ibddisk_valid[['paciente'] + disk_items + ['physical_domain', 'psychosocial_domain',
    'total_score', 'mean_score']].rename(
    columns={'total_score': 'disk_total', 'mean_score': 'disk_mean'})
ibdq_merge = ibdq_valid[['paciente', 'sintomas_intestinais', 'sintomas_sistemicos',
    'bem_estar_emocional', 'interacao_social', 'total']].rename(
    columns={'total': 'ibdq_total'})
impact_merge = impact_valid[['paciente', 'dom_sintomas', 'dom_emocional', 'dom_social',
    'dom_bemestar', 'dom_tratamento', 'total']].rename(
    columns={'total': 'impact_total'})

merged = crafft_merge.merge(disk_merge, on='paciente', how='outer')
merged = merged.merge(ibdq_merge, on='paciente', how='outer')
merged = merged.merge(impact_merge, on='paciente', how='outer')

# IBDQ version classification
ibdq_long_items = [f'item_{i}' for i in range(12, 33)]
ibdq_valid['ibdq_version'] = ibdq_valid[ibdq_long_items].notna().sum(axis=1).apply(
    lambda x: 'longa' if x > 5 else 'curta')

# Merge version info
version_info = ibdq_valid[['paciente', 'ibdq_version']].copy()
merged = merged.merge(version_info, on='paciente', how='left')

# IBDQ severity categories
merged['ibdq_severity'] = pd.cut(merged['ibdq_total'],
    bins=[0, 100, 150, 224], labels=['Grave', 'Moderado', 'Leve'], right=True)

log(f"\nDataset merged: {len(merged)} pacientes")
log(f"  Com CRAFFT + IBD-Disk: {merged[['crafft_total','disk_total']].dropna().shape[0]}")
log(f"  Com CRAFFT + IBDQ: {merged[['crafft_total','ibdq_total']].dropna().shape[0]}")
log(f"  Com IBD-Disk + IBDQ: {merged[['disk_total','ibdq_total']].dropna().shape[0]}")
log(f"  Com todos 3 (CRAFFT+Disk+IBDQ): {merged[['crafft_total','disk_total','ibdq_total']].dropna().shape[0]}")


# ==============================================================================
# P1: CORREÇÃO PARA COMPARAÇÕES MÚLTIPLAS (Benjamini-Hochberg FDR)
# ==============================================================================
log("\n\n" + "=" * 100)
log("P1: CORREÇÃO PARA COMPARAÇÕES MÚLTIPLAS — Benjamini-Hochberg FDR")
log("=" * 100)

def benjamini_hochberg(p_values):
    """Aplica correção Benjamini-Hochberg FDR."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / (i + 1))
    result = np.zeros(n)
    result[sorted_indices] = adjusted
    return result

# Collect all p-values from correlation analyses
log("\n--- Correlações entre scores totais (com correção FDR) ---")
correlation_tests = []
pairs = [
    ('crafft_total', 'disk_total', 'CRAFFT vs IBD-Disk'),
    ('crafft_total', 'ibdq_total', 'CRAFFT vs IBDQ'),
    ('crafft_total', 'impact_total', 'CRAFFT vs IMPACT-III'),
    ('disk_total', 'ibdq_total', 'IBD-Disk vs IBDQ'),
    ('disk_total', 'impact_total', 'IBD-Disk vs IMPACT-III'),
]

for var1, var2, label in pairs:
    df_pair = merged[[var1, var2]].dropna()
    if len(df_pair) >= 5:
        rho, p = spearmanr(df_pair[var1], df_pair[var2])
        correlation_tests.append({'label': label, 'rho': rho, 'p_raw': p, 'n': len(df_pair)})

p_raw = [t['p_raw'] for t in correlation_tests]
p_adj = benjamini_hochberg(p_raw)
for i, t in enumerate(correlation_tests):
    t['p_fdr'] = p_adj[i]
    sig_raw = '***' if t['p_raw'] < 0.001 else '**' if t['p_raw'] < 0.01 else '*' if t['p_raw'] < 0.05 else 'ns'
    sig_fdr = '***' if t['p_fdr'] < 0.001 else '**' if t['p_fdr'] < 0.01 else '*' if t['p_fdr'] < 0.05 else 'ns'
    log(f"  {t['label']:35s}  rho={t['rho']:+.3f}  p_raw={t['p_raw']:.4f} {sig_raw:3s}  p_FDR={t['p_fdr']:.4f} {sig_fdr:3s}  n={t['n']}")

# Domain-level correlations with FDR
log("\n--- Correlações domínios IBD-Disk vs IBDQ (com FDR) ---")
domain_tests = []
disk_domains = [('physical_domain', 'Physical'), ('psychosocial_domain', 'Psychosocial')]
ibdq_domains = [('sintomas_intestinais', 'Sint.Intest.'), ('sintomas_sistemicos', 'Sint.Sist.'),
                ('bem_estar_emocional', 'Bem-estar'), ('interacao_social', 'Social')]

for d_col, d_name in disk_domains:
    for q_col, q_name in ibdq_domains:
        df_pair = merged[[d_col, q_col]].dropna()
        if len(df_pair) >= 5:
            rho, p = spearmanr(df_pair[d_col], df_pair[q_col])
            domain_tests.append({'label': f'{d_name} vs {q_name}', 'rho': rho, 'p_raw': p, 'n': len(df_pair)})

p_raw_d = [t['p_raw'] for t in domain_tests]
p_adj_d = benjamini_hochberg(p_raw_d)
for i, t in enumerate(domain_tests):
    t['p_fdr'] = p_adj_d[i]
    sig_raw = '***' if t['p_raw'] < 0.001 else '**' if t['p_raw'] < 0.01 else '*' if t['p_raw'] < 0.05 else 'ns'
    sig_fdr = '***' if t['p_fdr'] < 0.001 else '**' if t['p_fdr'] < 0.01 else '*' if t['p_fdr'] < 0.05 else 'ns'
    log(f"  {t['label']:35s}  rho={t['rho']:+.3f}  p_raw={t['p_raw']:.6f} {sig_raw:3s}  p_FDR={t['p_fdr']:.6f} {sig_fdr:3s}  n={t['n']}")

# Item-level CRAFFT vs IBD-Disk with FDR
log("\n--- Itens CRAFFT vs IBD-Disk (com FDR) ---")
item_tests = []
for c_item in crafft_items:
    for d_item in disk_items:
        df_pair = merged[[c_item, d_item]].dropna()
        if len(df_pair) >= 10:
            rho, p = spearmanr(df_pair[c_item], df_pair[d_item])
            item_tests.append({'crafft': c_item, 'disk': d_item, 'rho': rho, 'p_raw': p, 'n': len(df_pair)})

p_raw_items = [t['p_raw'] for t in item_tests]
p_adj_items = benjamini_hochberg(p_raw_items)
sig_after_fdr = 0
for i, t in enumerate(item_tests):
    t['p_fdr'] = p_adj_items[i]
    if t['p_fdr'] < 0.05:
        sig_after_fdr += 1
        log(f"  {t['crafft']:12s} vs {t['disk']:25s}  rho={t['rho']:+.3f}  p_raw={t['p_raw']:.4f}  p_FDR={t['p_fdr']:.4f} *  n={t['n']}")

log(f"\n  Total de testes item-a-item: {len(item_tests)}")
log(f"  Significativos antes de FDR (p<0.05): {sum(1 for t in item_tests if t['p_raw'] < 0.05)}")
log(f"  Significativos após FDR (q<0.05): {sig_after_fdr}")
log(f"  → Conclusão: {'Associações item-nível sobrevivem à correção' if sig_after_fdr > 0 else 'Nenhuma associação item-nível sobrevive à correção FDR — achados exploratórios'}")


# ==============================================================================
# P2: REGRESSÃO LOGÍSTICA — Preditores de CRAFFT+
# ==============================================================================
log("\n\n" + "=" * 100)
log("P2: REGRESSÃO LOGÍSTICA — Preditores de CRAFFT Positivo")
log("=" * 100)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Model 1: IBD-Disk domains predicting CRAFFT+
log("\n--- Modelo 1: IBD-Disk Domains → CRAFFT+ ---")
df_log1 = merged[['crafft_positive', 'physical_domain', 'psychosocial_domain']].dropna()
if len(df_log1) >= 20:
    X = df_log1[['physical_domain', 'psychosocial_domain']].values
    y = df_log1['crafft_positive'].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    y_pred_prob = model.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, y_pred_prob)

    log(f"  n = {len(df_log1)}, CRAFFT+ = {y.sum()} ({100*y.mean():.1f}%)")
    log(f"  Coeficientes (padronizados):")
    log(f"    Physical Domain:      β = {model.coef_[0][0]:+.4f}")
    log(f"    Psychosocial Domain:  β = {model.coef_[0][1]:+.4f}")
    log(f"    Intercept:            β₀ = {model.intercept_[0]:+.4f}")
    log(f"  AUC-ROC: {auc:.3f}")

    # Odds ratios
    or_phys = np.exp(model.coef_[0][0])
    or_psych = np.exp(model.coef_[0][1])
    log(f"  Odds Ratios (por 1 DP de aumento):")
    log(f"    Physical:      OR = {or_phys:.3f}")
    log(f"    Psychosocial:  OR = {or_psych:.3f}")

# Model 2: IBD-Disk individual items predicting CRAFFT+
log("\n--- Modelo 2: IBD-Disk Items individuais → CRAFFT+ (univariados) ---")
df_log2 = merged[['crafft_positive'] + disk_items].dropna()
if len(df_log2) >= 20:
    y = df_log2['crafft_positive'].astype(int).values
    log(f"  n = {len(df_log2)}, CRAFFT+ = {y.sum()} ({100*y.mean():.1f}%)")

    univariate_results = []
    for item, label in zip(disk_items, disk_labels):
        X_item = df_log2[[item]].values
        model_uni = LogisticRegression(random_state=42, max_iter=1000)
        model_uni.fit(X_item, y)
        y_prob = model_uni.predict_proba(X_item)[:, 1]
        auc_uni = roc_auc_score(y, y_prob)
        or_val = np.exp(model_uni.coef_[0][0])
        p_val = stats.norm.sf(abs(model_uni.coef_[0][0] / 0.1)) * 2  # Wald approximation

        # Better p-value via likelihood ratio test
        from sklearn.metrics import log_loss
        null_prob = y.mean()
        ll_null = -log_loss(y, [null_prob]*len(y), normalize=False)
        ll_model = -log_loss(y, y_prob, normalize=False)
        lr_stat = 2 * (ll_model - ll_null)
        p_lr = stats.chi2.sf(lr_stat, df=1) if lr_stat > 0 else 1.0

        univariate_results.append({
            'label': label, 'item': item, 'beta': model_uni.coef_[0][0],
            'OR': or_val, 'AUC': auc_uni, 'p': p_lr, 'LR': lr_stat
        })

    log(f"\n  {'Item':<22s}  {'β':>8s}  {'OR':>8s}  {'AUC':>6s}  {'LR χ²':>8s}  {'p':>8s}  Sig")
    log(f"  {'-'*80}")
    for r in sorted(univariate_results, key=lambda x: x['p']):
        sig = '***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*' if r['p'] < 0.05 else 'ns'
        log(f"  {r['label']:<22s}  {r['beta']:+8.4f}  {r['OR']:8.3f}  {r['AUC']:6.3f}  {r['LR']:8.3f}  {r['p']:8.4f}  {sig}")

# Model 3: IBD-Disk + IBDQ domains predicting CRAFFT+
log("\n--- Modelo 3: IBD-Disk Domains + IBDQ Total → CRAFFT+ ---")
df_log3 = merged[['crafft_positive', 'physical_domain', 'psychosocial_domain', 'ibdq_total']].dropna()
if len(df_log3) >= 20:
    X = df_log3[['physical_domain', 'psychosocial_domain', 'ibdq_total']].values
    y = df_log3['crafft_positive'].astype(int).values

    scaler3 = StandardScaler()
    X_scaled = scaler3.fit_transform(X)

    model3 = LogisticRegression(random_state=42, max_iter=1000)
    model3.fit(X_scaled, y)

    y_pred_prob = model3.predict_proba(X_scaled)[:, 1]
    auc3 = roc_auc_score(y, y_pred_prob)

    log(f"  n = {len(df_log3)}, CRAFFT+ = {y.sum()} ({100*y.mean():.1f}%)")
    log(f"  Coeficientes (padronizados):")
    log(f"    Physical Domain:  β = {model3.coef_[0][0]:+.4f}, OR = {np.exp(model3.coef_[0][0]):.3f}")
    log(f"    Psychosocial:     β = {model3.coef_[0][1]:+.4f}, OR = {np.exp(model3.coef_[0][1]):.3f}")
    log(f"    IBDQ Total:       β = {model3.coef_[0][2]:+.4f}, OR = {np.exp(model3.coef_[0][2]):.3f}")
    log(f"  AUC-ROC: {auc3:.3f}")


# ==============================================================================
# P3: ANÁLISE DOSE-RESPOSTA — CRAFFT Score vs Disability/QoL
# ==============================================================================
log("\n\n" + "=" * 100)
log("P3: ANÁLISE DOSE-RESPOSTA — CRAFFT Score vs Disability/QoL")
log("=" * 100)

log("\n--- IBD-Disk Total por nível de CRAFFT ---")
df_dose = merged[['crafft_total', 'disk_total', 'ibdq_total']].dropna(subset=['crafft_total'])

crafft_groups = {0: 'Nenhum', 1: 'Baixo (1)', '2+': 'Alto (≥2)'}
df_dose['crafft_cat'] = df_dose['crafft_total'].apply(
    lambda x: 0 if x == 0 else (1 if x == 1 else 2))

for outcome, outcome_name in [('disk_total', 'IBD-Disk'), ('ibdq_total', 'IBDQ')]:
    df_sub = df_dose[df_dose[outcome].notna()]
    if len(df_sub) < 10:
        continue

    log(f"\n  {outcome_name} por nível de CRAFFT:")
    for cat_val, cat_name in [(0, 'CRAFFT=0'), (1, 'CRAFFT=1'), (2, 'CRAFFT≥2')]:
        subset = df_sub[df_sub['crafft_cat'] == cat_val][outcome]
        if len(subset) > 0:
            log(f"    {cat_name}: n={len(subset)}, mediana={subset.median():.1f}, "
                f"IQR=[{subset.quantile(0.25):.1f}-{subset.quantile(0.75):.1f}]")

    # Jonckheere-Terpstra trend test (approximated via Spearman on ordinal groups)
    rho, p = spearmanr(df_sub['crafft_cat'], df_sub[outcome])
    log(f"    Tendência (Spearman ordinal): rho={rho:+.3f}, p={p:.4f}")

    # Kruskal-Wallis entre os 3 grupos
    groups = [df_sub[df_sub['crafft_cat'] == i][outcome].dropna() for i in [0, 1, 2]]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) >= 2:
        H, p_kw = kruskal(*groups)
        log(f"    Kruskal-Wallis: H={H:.3f}, p={p_kw:.4f}")

        # Post-hoc: pairwise Mann-Whitney com Bonferroni
        if p_kw < 0.05 and len(groups) == 3:
            log(f"    Post-hoc (Bonferroni-corrigido):")
            comparisons = [(0, 1, 'CRAFFT=0 vs 1'), (0, 2, 'CRAFFT=0 vs ≥2'), (1, 2, 'CRAFFT=1 vs ≥2')]
            for i, j, label in comparisons:
                g1 = df_sub[df_sub['crafft_cat'] == i][outcome].dropna()
                g2 = df_sub[df_sub['crafft_cat'] == j][outcome].dropna()
                if len(g1) >= 2 and len(g2) >= 2:
                    U, p_mw = mannwhitneyu(g1, g2, alternative='two-sided')
                    p_bonf = min(p_mw * 3, 1.0)
                    r_eff = abs(stats.norm.isf(p_mw/2)) / np.sqrt(len(g1) + len(g2))
                    sig = '***' if p_bonf < 0.001 else '**' if p_bonf < 0.01 else '*' if p_bonf < 0.05 else 'ns'
                    log(f"      {label}: U={U:.0f}, p_raw={p_mw:.4f}, p_Bonf={p_bonf:.4f} {sig}, r={r_eff:.3f}")


# ==============================================================================
# P4: SUBGRUPO — IBDQ Versão Curta vs Longa
# ==============================================================================
log("\n\n" + "=" * 100)
log("P4: ANÁLISE DE SUBGRUPO — IBDQ Versão Curta vs Longa")
log("=" * 100)

df_version = merged[merged['ibdq_version'].notna()].copy()
curta = df_version[df_version['ibdq_version'] == 'curta']
longa = df_version[df_version['ibdq_version'] == 'longa']

log(f"\n  Versão curta (11 itens): n={len(curta)}")
log(f"  Versão longa (32 itens): n={len(longa)}")

# Compare IBD-Disk between groups
log(f"\n--- Comparação IBD-Disk entre versões IBDQ ---")
for var, label in [('disk_total', 'IBD-Disk Total'), ('physical_domain', 'Physical Domain'),
                   ('psychosocial_domain', 'Psychosocial Domain'),
                   ('disk_mean', 'IBD-Disk Mean')]:
    g1 = curta[var].dropna()
    g2 = longa[var].dropna()
    if len(g1) >= 3 and len(g2) >= 3:
        U, p = mannwhitneyu(g1, g2, alternative='two-sided')
        r_eff = abs(stats.norm.isf(p/2)) / np.sqrt(len(g1) + len(g2))
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        log(f"  {label:25s}: Curta med={g1.median():.1f} vs Longa med={g2.median():.1f}  "
            f"U={U:.0f}, p={p:.4f} {sig}, r={r_eff:.3f}")

# Compare CRAFFT between groups
log(f"\n--- Comparação CRAFFT entre versões IBDQ ---")
for var, label in [('crafft_total', 'CRAFFT Total')]:
    g1 = curta[var].dropna()
    g2 = longa[var].dropna()
    if len(g1) >= 3 and len(g2) >= 3:
        U, p = mannwhitneyu(g1, g2, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        log(f"  {label:25s}: Curta med={g1.median():.1f} vs Longa med={g2.median():.1f}  "
            f"U={U:.0f}, p={p:.4f} {sig}")

# CRAFFT+ prevalence by version
crafft_pos_curta = curta['crafft_positive'].sum()
crafft_pos_longa = longa['crafft_positive'].sum()
n_curta_c = curta['crafft_positive'].notna().sum()
n_longa_c = longa['crafft_positive'].notna().sum()
if n_curta_c > 0 and n_longa_c > 0:
    table = np.array([[crafft_pos_curta, n_curta_c - crafft_pos_curta],
                      [crafft_pos_longa, n_longa_c - crafft_pos_longa]])
    if table.min() >= 0:
        if table.min() < 5:
            _, p_fisher = fisher_exact(table)
            log(f"\n  CRAFFT+ prevalência: Curta={100*crafft_pos_curta/n_curta_c:.1f}% vs Longa={100*crafft_pos_longa/n_longa_c:.1f}%")
            log(f"  Fisher exact: p={p_fisher:.4f}")
        else:
            chi2, p_chi, _, _ = chi2_contingency(table)
            log(f"\n  CRAFFT+ prevalência: Curta={100*crafft_pos_curta/n_curta_c:.1f}% vs Longa={100*crafft_pos_longa/n_longa_c:.1f}%")
            log(f"  Chi²={chi2:.3f}, p={p_chi:.4f}")

# Correlations within each version subgroup
log(f"\n--- Correlações IBD-Disk vs IBDQ por versão ---")
for version_name, version_df in [('Curta', curta), ('Longa', longa)]:
    df_sub = version_df[['disk_total', 'ibdq_total']].dropna()
    if len(df_sub) >= 5:
        rho, p = spearmanr(df_sub['disk_total'], df_sub['ibdq_total'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        log(f"  {version_name}: rho={rho:+.3f}, p={p:.4f} {sig}, n={len(df_sub)}")


# ==============================================================================
# P5: ANÁLISE DE SENSIBILIDADE — Casos Completos
# ==============================================================================
log("\n\n" + "=" * 100)
log("P5: ANÁLISE DE SENSIBILIDADE — Apenas Casos Completos (3 instrumentos)")
log("=" * 100)

complete_cases = merged[['crafft_total', 'disk_total', 'ibdq_total', 'crafft_positive',
                          'physical_domain', 'psychosocial_domain']].dropna()
log(f"\n  Casos completos (CRAFFT + IBD-Disk + IBDQ): n={len(complete_cases)}")

if len(complete_cases) >= 10:
    log(f"\n--- Repetindo correlações principais em casos completos ---")
    pairs_sens = [
        ('crafft_total', 'disk_total', 'CRAFFT vs IBD-Disk'),
        ('crafft_total', 'ibdq_total', 'CRAFFT vs IBDQ'),
        ('disk_total', 'ibdq_total', 'IBD-Disk vs IBDQ'),
    ]
    for v1, v2, label in pairs_sens:
        rho, p = spearmanr(complete_cases[v1], complete_cases[v2])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        log(f"  {label:30s}: rho={rho:+.3f}, p={p:.4f} {sig}, n={len(complete_cases)}")

    # CRAFFT+ comparisons in complete cases
    log(f"\n--- CRAFFT+ vs CRAFFT- em casos completos ---")
    pos = complete_cases[complete_cases['crafft_positive'] == True]
    neg = complete_cases[complete_cases['crafft_positive'] == False]
    log(f"  CRAFFT+ = {len(pos)}, CRAFFT- = {len(neg)}")

    for var, label in [('disk_total', 'IBD-Disk'), ('ibdq_total', 'IBDQ')]:
        g1 = pos[var].dropna()
        g2 = neg[var].dropna()
        if len(g1) >= 3 and len(g2) >= 3:
            U, p = mannwhitneyu(g1, g2, alternative='two-sided')
            r_eff = abs(stats.norm.isf(p/2)) / np.sqrt(len(g1) + len(g2))
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            log(f"  {label}: CRAFFT+ med={g1.median():.1f} vs CRAFFT- med={g2.median():.1f}, "
                f"U={U:.0f}, p={p:.4f} {sig}, r={r_eff:.3f}")

    # Compare complete vs incomplete cases on available variables
    log(f"\n--- Viés de seleção: casos completos vs incompletos ---")
    has_crafft = merged['crafft_total'].notna()
    has_disk = merged['disk_total'].notna()
    has_ibdq = merged['ibdq_total'].notna()

    complete_mask = has_crafft & has_disk & has_ibdq
    incomplete_mask = has_crafft & ~(has_disk & has_ibdq)

    comp = merged[complete_mask]
    incomp = merged[incomplete_mask]

    for var, label in [('crafft_total', 'CRAFFT Total')]:
        g1 = comp[var].dropna()
        g2 = incomp[var].dropna()
        if len(g1) >= 3 and len(g2) >= 3:
            U, p = mannwhitneyu(g1, g2, alternative='two-sided')
            sig = '*' if p < 0.05 else 'ns'
            log(f"  {label}: Completos med={g1.median():.1f} vs Incompletos med={g2.median():.1f}, p={p:.4f} {sig}")

    # CRAFFT+ prevalence comparison
    pos_comp = comp['crafft_positive'].sum()
    pos_incomp = incomp['crafft_positive'].sum()
    n_comp = comp['crafft_positive'].notna().sum()
    n_incomp = incomp['crafft_positive'].notna().sum()
    if n_comp > 0 and n_incomp > 0:
        log(f"  CRAFFT+ prevalência: Completos={100*pos_comp/n_comp:.1f}% vs Incompletos={100*pos_incomp/n_incomp:.1f}%")


# ==============================================================================
# P6: POWER ANALYSIS POST-HOC
# ==============================================================================
log("\n\n" + "=" * 100)
log("P6: POWER ANALYSIS POST-HOC")
log("=" * 100)

def power_spearman(n, rho, alpha=0.05):
    """Calcula poder estatístico para correlação de Spearman."""
    if abs(rho) < 0.001:
        return alpha
    z_rho = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1.0 / np.sqrt(n - 3)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = abs(z_rho) / se - z_alpha
    power = norm.cdf(z_power)
    return power

def power_mann_whitney(n1, n2, effect_r, alpha=0.05):
    """Power para Mann-Whitney U (aproximação normal)."""
    n = n1 + n2
    se = np.sqrt(n1 * n2 * (n + 1) / 12)
    # Effect size r translates to Z
    z_eff = effect_r * np.sqrt(n)
    z_alpha = norm.ppf(1 - alpha / 2)
    power = norm.cdf(z_eff - z_alpha) + norm.cdf(-z_eff - z_alpha)
    return max(power, alpha)

def sample_needed_spearman(rho, power=0.80, alpha=0.05):
    """N mínimo para detectar correlação com poder desejado."""
    if abs(rho) < 0.001:
        return float('inf')
    z_rho = 0.5 * np.log((1 + rho) / (1 - rho))
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    n = ((z_alpha + z_power) / z_rho) ** 2 + 3
    return int(np.ceil(n))

log("\n--- Poder estatístico para correlações observadas ---")
power_results = [
    ('CRAFFT vs IBD-Disk', -0.006, 143),
    ('CRAFFT vs IBDQ', 0.014, 127),
    ('CRAFFT vs IMPACT-III', -0.090, 23),
    ('IBD-Disk vs IBDQ', -0.210, 123),
    ('IBD-Disk vs IMPACT-III', 0.810, 22),
    ('Phys.Domain vs Sint.Intest.', -0.708, 123),
    ('Phys.Domain vs Sint.Sist.', -0.629, 123),
    ('Psycho vs Sint.Intest.', -0.606, 123),
]

log(f"\n  {'Par':<35s}  {'rho':>6s}  {'n':>5s}  {'Poder':>7s}  {'N p/ 80%':>10s}")
log(f"  {'-'*75}")
for label, rho, n in power_results:
    pwr = power_spearman(n, rho)
    n_needed = sample_needed_spearman(rho)
    n_str = f"{n_needed}" if n_needed < 10000 else ">10000"
    log(f"  {label:<35s}  {rho:+.3f}  {n:5d}  {pwr:7.1%}  {n_str:>10s}")

log("\n--- Poder para comparações CRAFFT+ vs CRAFFT- ---")
log(f"\n  {'Outcome':<25s}  {'r':>6s}  {'n1':>4s}  {'n2':>4s}  {'Poder':>7s}")
log(f"  {'-'*55}")
comparisons_power = [
    ('IBD-Disk Total', 0.072, 33, 110),
    ('IBDQ Total', 0.062, 30, 97),
    ('IBDQ Bem-estar Emoc.', 0.100, 30, 97),
    ('IBDQ Interação Social', 0.113, 30, 97),
]
for label, r, n1, n2 in comparisons_power:
    pwr = power_mann_whitney(n1, n2, r)
    log(f"  {label:<25s}  {r:6.3f}  {n1:4d}  {n2:4d}  {pwr:7.1%}")

# Minimum detectable effect size
log(f"\n--- Tamanho mínimo de efeito detectável (poder=80%, α=0.05) ---")
for n_val, label in [(143, 'CRAFFT vs IBD-Disk (n=143)'), (123, 'IBD-Disk vs IBDQ (n=123)'),
                      (22, 'IBD-Disk vs IMPACT-III (n=22)')]:
    for rho_test in np.arange(0.05, 0.80, 0.01):
        if power_spearman(n_val, rho_test) >= 0.80:
            log(f"  {label}: |rho| mínimo detectável = {rho_test:.2f}")
            break


# ==============================================================================
# P7: ROC CURVE — IBD-Disk predizendo IBDQ Grave
# ==============================================================================
log("\n\n" + "=" * 100)
log("P7: ROC CURVE — IBD-Disk Predizendo IBDQ Grave")
log("=" * 100)

df_roc = merged[['disk_total', 'ibdq_total', 'ibdq_severity']].dropna()
if len(df_roc) >= 20:
    # Binary: Grave vs Não-Grave
    df_roc['grave'] = (df_roc['ibdq_severity'] == 'Grave').astype(int)

    log(f"\n  n = {len(df_roc)}")
    log(f"  IBDQ Grave (≤100): {df_roc['grave'].sum()} ({100*df_roc['grave'].mean():.1f}%)")
    log(f"  IBDQ Não-grave: {(1-df_roc['grave']).sum()} ({100*(1-df_roc['grave'].mean()):.1f}%)")

    # ROC manual
    thresholds = np.arange(0, 100, 1)
    tpr_list, fpr_list = [], []

    for thresh in thresholds:
        predicted_grave = (df_roc['disk_total'] >= thresh).astype(int)
        tp = ((predicted_grave == 1) & (df_roc['grave'] == 1)).sum()
        fp = ((predicted_grave == 1) & (df_roc['grave'] == 0)).sum()
        fn = ((predicted_grave == 0) & (df_roc['grave'] == 1)).sum()
        tn = ((predicted_grave == 0) & (df_roc['grave'] == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # AUC via trapezoidal
    sorted_idx = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[sorted_idx]
    tpr_sorted = np.array(tpr_list)[sorted_idx]
    auc_roc = np.trapz(tpr_sorted, fpr_sorted)

    log(f"  AUC-ROC (IBD-Disk → IBDQ Grave): {auc_roc:.3f}")

    # Optimal cutoff (Youden's J)
    j_scores = np.array(tpr_list) - np.array(fpr_list)
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    best_sens = tpr_list[best_idx]
    best_spec = 1 - fpr_list[best_idx]

    log(f"  Cutoff ótimo (Youden's J): IBD-Disk ≥ {best_thresh}")
    log(f"    Sensibilidade: {best_sens:.1%}")
    log(f"    Especificidade: {best_spec:.1%}")
    log(f"    Youden's J: {j_scores[best_idx]:.3f}")

    # Additional cutoffs
    log(f"\n  Cutoffs alternativos:")
    for t in [20, 30, 40, 50, 60]:
        predicted = (df_roc['disk_total'] >= t).astype(int)
        tp = ((predicted == 1) & (df_roc['grave'] == 1)).sum()
        fp = ((predicted == 1) & (df_roc['grave'] == 0)).sum()
        fn = ((predicted == 0) & (df_roc['grave'] == 1)).sum()
        tn = ((predicted == 0) & (df_roc['grave'] == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        log(f"    IBD-Disk ≥{t}: Sens={sens:.1%}, Spec={spec:.1%}, PPV={ppv:.1%}, NPV={npv:.1%}")

    # ROC Figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(fpr_sorted, tpr_sorted, 'b-', linewidth=2, label=f'IBD-Disk (AUC={auc_roc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Referência (AUC=0.5)')
    ax.plot(1 - best_spec, best_sens, 'ro', markersize=10, label=f'Cutoff ótimo (≥{best_thresh})')
    ax.set_xlabel('1 - Especificidade (Taxa Falso Positivo)', fontsize=12)
    ax.set_ylabel('Sensibilidade (Taxa Verdadeiro Positivo)', fontsize=12)
    ax.set_title('ROC: IBD-Disk Predizendo IBDQ Grave (≤100)', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig15_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"\n  → Figura salva: fig15_roc_curve.png")


# ==============================================================================
# P8: DIAGNÓSTICO DE REGRESSÃO E MULTICOLINEARIDADE
# ==============================================================================
log("\n\n" + "=" * 100)
log("P8: DIAGNÓSTICO DE REGRESSÃO E MULTICOLINEARIDADE")
log("=" * 100)

# VIF for IBD-Disk domains
log("\n--- VIF (Variance Inflation Factor) para domínios IBD-Disk ---")
df_vif = merged[['physical_domain', 'psychosocial_domain']].dropna()
if len(df_vif) >= 10:
    corr_domains = np.corrcoef(df_vif['physical_domain'], df_vif['psychosocial_domain'])[0, 1]
    vif = 1 / (1 - corr_domains**2)
    log(f"  Correlação Physical vs Psychosocial: r={corr_domains:.3f}")
    log(f"  VIF: {vif:.3f}")
    log(f"  Interpretação: {'OK (VIF < 5)' if vif < 5 else '⚠️ Multicolinearidade (VIF ≥ 5)' if vif < 10 else '🚫 Multicolinearidade severa (VIF ≥ 10)'}")

# VIF for all IBD-Disk items
log("\n--- VIF para itens individuais IBD-Disk ---")
df_vif_items = merged[disk_items].dropna()
if len(df_vif_items) >= 10:
    corr_matrix = df_vif_items.corr()
    try:
        corr_inv = np.linalg.inv(corr_matrix.values)
        vif_items = np.diag(corr_inv)
        for i, (item, label) in enumerate(zip(disk_items, disk_labels)):
            status = 'OK' if vif_items[i] < 5 else '⚠️ Alto' if vif_items[i] < 10 else '🚫 Severo'
            log(f"  {label:25s}: VIF = {vif_items[i]:.2f}  {status}")
    except np.linalg.LinAlgError:
        log("  Matriz singular — não foi possível calcular VIF")

# Residual analysis for best regression model
log("\n--- Diagnóstico de resíduos — Modelo: IBD-Disk Domains → IBDQ ---")
df_resid = merged[['physical_domain', 'psychosocial_domain', 'ibdq_total']].dropna()
if len(df_resid) >= 10:
    from numpy.linalg import lstsq
    X_mat = np.column_stack([np.ones(len(df_resid)),
                              df_resid['physical_domain'].values,
                              df_resid['psychosocial_domain'].values])
    y_vec = df_resid['ibdq_total'].values
    beta, residuals_ss, rank, sv = lstsq(X_mat, y_vec, rcond=None)
    y_pred = X_mat @ beta
    residuals = y_vec - y_pred

    # Normality of residuals
    W, p_sw = shapiro(residuals)
    log(f"  Normalidade dos resíduos (Shapiro-Wilk): W={W:.4f}, p={p_sw:.4f}")
    log(f"  → Resíduos {'normais' if p_sw > 0.05 else 'NÃO normais'}")

    # Homoscedasticity (simple: correlation between |residuals| and predicted)
    rho_hetero, p_hetero = spearmanr(np.abs(residuals), y_pred)
    log(f"  Heterocedasticidade (|resíduos| vs preditos): rho={rho_hetero:.3f}, p={p_hetero:.4f}")
    log(f"  → {'Homocedasticidade OK' if p_hetero > 0.05 else '⚠️ Possível heterocedasticidade'}")

    # R² and adjusted R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_vec - y_vec.mean())**2)
    r2 = 1 - ss_res / ss_tot
    n_obs = len(df_resid)
    p_params = 2
    r2_adj = 1 - (1 - r2) * (n_obs - 1) / (n_obs - p_params - 1)
    log(f"  R² = {r2:.4f}, R² ajustado = {r2_adj:.4f}")

    # Residual plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_pred, residuals, alpha=0.5, s=30)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Valores Preditos', fontsize=11)
    axes[0].set_ylabel('Resíduos', fontsize=11)
    axes[0].set_title('Resíduos vs Preditos', fontsize=13)

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('QQ-Plot dos Resíduos', fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig16_residual_diagnostics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"  → Figura salva: fig16_residual_diagnostics.png")


# ==============================================================================
# P9: EFFECT SIZES COM IC BOOTSTRAPPED
# ==============================================================================
log("\n\n" + "=" * 100)
log("P9: EFFECT SIZES COM IC BOOTSTRAPPED (2000 iterações)")
log("=" * 100)

def bootstrap_effect_size_mw(group1, group2, n_boot=2000, ci=0.95):
    """Bootstrap IC para r de Mann-Whitney."""
    r_boots = []
    for _ in range(n_boot):
        g1 = np.random.choice(group1, size=len(group1), replace=True)
        g2 = np.random.choice(group2, size=len(group2), replace=True)
        try:
            U, p = mannwhitneyu(g1, g2, alternative='two-sided')
            n_total = len(g1) + len(g2)
            z = abs(norm.isf(p/2))
            r = z / np.sqrt(n_total)
            r_boots.append(r)
        except:
            pass
    if len(r_boots) < 100:
        return np.nan, np.nan, np.nan
    r_boots = np.array(r_boots)
    alpha = (1 - ci) / 2
    return np.median(r_boots), np.percentile(r_boots, 100*alpha), np.percentile(r_boots, 100*(1-alpha))

log("\n--- Effect sizes (r) para CRAFFT+ vs CRAFFT- com IC 95% ---")
pos_mask = merged['crafft_positive'] == True
neg_mask = merged['crafft_positive'] == False

outcomes_es = [
    ('disk_total', 'IBD-Disk Total'),
    ('disk_mean', 'IBD-Disk Mean'),
    ('physical_domain', 'Physical Domain'),
    ('psychosocial_domain', 'Psychosocial Domain'),
    ('ibdq_total', 'IBDQ Total'),
    ('sintomas_intestinais', 'IBDQ Sint. Intestinais'),
    ('sintomas_sistemicos', 'IBDQ Sint. Sistêmicos'),
    ('bem_estar_emocional', 'IBDQ Bem-estar Emoc.'),
    ('interacao_social', 'IBDQ Interação Social'),
]

log(f"\n  {'Outcome':<28s}  {'r median':>8s}  {'IC 95%':>18s}  {'Interp.':>10s}")
log(f"  {'-'*72}")

for var, label in outcomes_es:
    g1 = merged[pos_mask][var].dropna().values
    g2 = merged[neg_mask][var].dropna().values
    if len(g1) >= 5 and len(g2) >= 5:
        r_med, r_lo, r_hi = bootstrap_effect_size_mw(g1, g2, n_boot=2000)
        interp = 'grande' if r_med > 0.5 else 'médio' if r_med > 0.3 else 'pequeno' if r_med > 0.1 else 'nulo'
        log(f"  {label:<28s}  {r_med:8.3f}  [{r_lo:.3f}, {r_hi:.3f}]  {interp:>10s}")


# ==============================================================================
# P10: FIGURAS ADICIONAIS PARA PUBLICAÇÃO
# ==============================================================================
log("\n\n" + "=" * 100)
log("P10: FIGURAS ADICIONAIS PARA PUBLICAÇÃO")
log("=" * 100)

# Fig 17: Forest plot of effect sizes
log("\n--- Fig 17: Forest Plot — Effect Sizes CRAFFT+ vs CRAFFT- ---")
fig, ax = plt.subplots(figsize=(10, 7))

es_data = []
for var, label in outcomes_es:
    g1 = merged[pos_mask][var].dropna().values
    g2 = merged[neg_mask][var].dropna().values
    if len(g1) >= 5 and len(g2) >= 5:
        U, p = mannwhitneyu(g1, g2, alternative='two-sided')
        n_total = len(g1) + len(g2)
        z = norm.isf(p/2)
        r = z / np.sqrt(n_total)  # signed: positive if CRAFFT+ has higher values
        # Determine direction
        if np.median(g1) < np.median(g2):
            r = -r
        r_med, r_lo, r_hi = bootstrap_effect_size_mw(g1, g2, n_boot=1000)
        # Use signed version
        es_data.append({'label': label, 'r': r, 'r_lo': -r_hi, 'r_hi': r_hi, 'p': p})

labels_plot = [d['label'] for d in es_data]
r_vals = [d['r'] for d in es_data]
y_pos = range(len(es_data))

ax.barh(y_pos, r_vals, height=0.5, color=['#e74c3c' if r > 0 else '#3498db' for r in r_vals], alpha=0.7)
ax.axvline(x=0, color='black', linewidth=1)
ax.axvline(x=0.1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.axvline(x=-0.1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_plot, fontsize=10)
ax.set_xlabel('Effect Size (r)', fontsize=12)
ax.set_title('Effect Sizes: CRAFFT+ vs CRAFFT-\n(positivo = CRAFFT+ tem scores maiores)', fontsize=13)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig17_forest_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
log("  → fig17_forest_plot.png salva")

# Fig 18: Dose-response CRAFFT
log("\n--- Fig 18: Dose-Resposta CRAFFT Score ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

df_dose_plot = merged[merged['crafft_total'].notna()].copy()
for ax, (var, label, color) in zip(axes,
    [('disk_total', 'IBD-Disk Total', '#e74c3c'), ('ibdq_total', 'IBDQ Total', '#2ecc71')]):

    df_sub = df_dose_plot[df_dose_plot[var].notna()]
    crafft_levels = sorted(df_sub['crafft_total'].unique())

    medians = []
    q25s = []
    q75s = []
    ns = []
    valid_levels = []

    for level in crafft_levels:
        data = df_sub[df_sub['crafft_total'] == level][var]
        if len(data) >= 2:
            medians.append(data.median())
            q25s.append(data.quantile(0.25))
            q75s.append(data.quantile(0.75))
            ns.append(len(data))
            valid_levels.append(level)

    if valid_levels:
        ax.errorbar(valid_levels, medians,
                     yerr=[np.array(medians) - np.array(q25s), np.array(q75s) - np.array(medians)],
                     fmt='o-', color=color, linewidth=2, markersize=8, capsize=5)
        for i, (lv, med, n) in enumerate(zip(valid_levels, medians, ns)):
            ax.annotate(f'n={n}', (lv, med), textcoords="offset points", xytext=(0, 12),
                       ha='center', fontsize=9, color='gray')
        ax.set_xlabel('CRAFFT Score', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} por CRAFFT Score', fontsize=13)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig18_dose_response.png'), dpi=300, bbox_inches='tight')
plt.close()
log("  → fig18_dose_response.png salva")

# Fig 19: IBDQ version comparison
log("\n--- Fig 19: Comparação Versão Curta vs Longa IBDQ ---")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (var, label) in zip(axes, [('disk_total', 'IBD-Disk Total'),
    ('physical_domain', 'Physical Domain'), ('psychosocial_domain', 'Psychosocial Domain')]):

    data_curta = curta[var].dropna()
    data_longa = longa[var].dropna()

    if len(data_curta) >= 3 and len(data_longa) >= 3:
        bp = ax.boxplot([data_curta, data_longa], labels=['Curta\n(11 itens)', 'Longa\n(32 itens)'],
                       patch_artist=True, widths=0.6)
        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        U, p = mannwhitneyu(data_curta, data_longa, alternative='two-sided')
        sig = f'p={p:.4f}' + (' *' if p < 0.05 else ' ns')
        ax.set_title(f'{label}\n{sig}', fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)

plt.suptitle('IBD-Disk por Versão do IBDQ Aplicada', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig19_version_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
log("  → fig19_version_comparison.png salva")

# Fig 20: Comprehensive correlation matrix (all scores)
log("\n--- Fig 20: Matriz de Correlação Completa ---")
all_scores = ['crafft_total', 'disk_total', 'disk_mean', 'physical_domain', 'psychosocial_domain',
              'ibdq_total', 'sintomas_intestinais', 'sintomas_sistemicos',
              'bem_estar_emocional', 'interacao_social']
score_labels = ['CRAFFT', 'Disk Total', 'Disk Mean', 'Physical', 'Psychosocial',
                'IBDQ Total', 'Sint.Intest.', 'Sint.Sist.', 'Bem-estar', 'Social']

df_corr_all = merged[all_scores].dropna()
if len(df_corr_all) >= 10:
    corr_matrix = df_corr_all.corr(method='spearman')
    p_matrix = pd.DataFrame(np.zeros_like(corr_matrix),
                            index=corr_matrix.index, columns=corr_matrix.columns)
    for i, c1 in enumerate(all_scores):
        for j, c2 in enumerate(all_scores):
            if i != j:
                rho, p = spearmanr(df_corr_all[c1], df_corr_all[c2])
                p_matrix.iloc[i, j] = p

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                xticklabels=score_labels, yticklabels=score_labels, ax=ax,
                annot_kws={'size': 9})

    ax.set_title('Matriz de Correlação (Spearman) — Todos os Scores\n(casos completos)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig20_full_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log("  → fig20_full_correlation_matrix.png salva")


# ==============================================================================
# P11: TABELAS STROBE ADICIONAIS
# ==============================================================================
log("\n\n" + "=" * 100)
log("P11: TABELAS STROBE ADICIONAIS PARA MANUSCRITO")
log("=" * 100)

# Table 4: CRAFFT+ vs CRAFFT- comprehensive comparison
log("\n--- TABELA 4: Comparação CRAFFT+ vs CRAFFT- (com effect sizes e IC) ---")
log(f"\n  {'Variável':<28s}  {'CRAFFT+ med(IQR)':<22s}  {'CRAFFT- med(IQR)':<22s}  {'p':>8s}  {'r [IC95%]':>22s}")
log(f"  {'-'*110}")

for var, label in outcomes_es:
    g1 = merged[pos_mask][var].dropna()
    g2 = merged[neg_mask][var].dropna()
    if len(g1) >= 3 and len(g2) >= 3:
        U, p = mannwhitneyu(g1, g2, alternative='two-sided')
        n_total = len(g1) + len(g2)
        z = abs(norm.isf(p/2))
        r = z / np.sqrt(n_total)
        r_med, r_lo, r_hi = bootstrap_effect_size_mw(g1.values, g2.values, n_boot=1000)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        pos_str = f"{g1.median():.0f} ({g1.quantile(0.25):.0f}-{g1.quantile(0.75):.0f})"
        neg_str = f"{g2.median():.0f} ({g2.quantile(0.25):.0f}-{g2.quantile(0.75):.0f})"
        r_str = f"{r_med:.3f} [{r_lo:.3f}, {r_hi:.3f}]"

        log(f"  {label:<28s}  {pos_str:<22s}  {neg_str:<22s}  {p:8.4f}{sig}  {r_str:>22s}")

# Table 5: Correlation matrix with p-values and FDR
log(f"\n--- TABELA 5: Correlações entre Instrumentos (com p-value e FDR) ---")
log(f"\n  {'Par':<40s}  {'rho':>7s}  {'p_raw':>8s}  {'p_FDR':>8s}  {'Sig FDR':>8s}  {'n':>5s}")
log(f"  {'-'*85}")

all_corr_results = []
score_pairs = [
    ('crafft_total', 'disk_total', 'CRAFFT vs IBD-Disk Total'),
    ('crafft_total', 'ibdq_total', 'CRAFFT vs IBDQ Total'),
    ('crafft_total', 'impact_total', 'CRAFFT vs IMPACT-III'),
    ('disk_total', 'ibdq_total', 'IBD-Disk vs IBDQ Total'),
    ('disk_total', 'impact_total', 'IBD-Disk vs IMPACT-III'),
    ('physical_domain', 'sintomas_intestinais', 'Physical vs Sint. Intestinais'),
    ('physical_domain', 'sintomas_sistemicos', 'Physical vs Sint. Sistêmicos'),
    ('physical_domain', 'bem_estar_emocional', 'Physical vs Bem-estar Emoc.'),
    ('physical_domain', 'interacao_social', 'Physical vs Interação Social'),
    ('psychosocial_domain', 'sintomas_intestinais', 'Psychosocial vs Sint. Intestinais'),
    ('psychosocial_domain', 'sintomas_sistemicos', 'Psychosocial vs Sint. Sistêmicos'),
    ('psychosocial_domain', 'bem_estar_emocional', 'Psychosocial vs Bem-estar Emoc.'),
    ('psychosocial_domain', 'interacao_social', 'Psychosocial vs Interação Social'),
]

for v1, v2, label in score_pairs:
    df_p = merged[[v1, v2]].dropna()
    if len(df_p) >= 5:
        rho, p = spearmanr(df_p[v1], df_p[v2])
        all_corr_results.append({'label': label, 'rho': rho, 'p_raw': p, 'n': len(df_p)})

p_vals = [r['p_raw'] for r in all_corr_results]
p_fdr = benjamini_hochberg(p_vals)
for i, r in enumerate(all_corr_results):
    r['p_fdr'] = p_fdr[i]
    sig = '***' if r['p_fdr'] < 0.001 else '**' if r['p_fdr'] < 0.01 else '*' if r['p_fdr'] < 0.05 else 'ns'
    log(f"  {r['label']:<40s}  {r['rho']:+7.3f}  {r['p_raw']:8.4f}  {r['p_fdr']:8.4f}  {sig:>8s}  {r['n']:5d}")

# Table 6: Regression model comparison
log(f"\n--- TABELA 6: Comparação de Modelos de Regressão ---")
log(f"\n  {'Modelo':<55s}  {'R²':>6s}  {'R²adj':>7s}  {'AIC':>8s}  {'n':>5s}")
log(f"  {'-'*90}")

# Re-run regressions for AIC
from numpy.linalg import lstsq as np_lstsq

def calc_aic(y, y_pred, k):
    """AIC = n*ln(RSS/n) + 2k."""
    n = len(y)
    rss = np.sum((y - y_pred)**2)
    aic = n * np.log(rss / n) + 2 * k
    return aic

models_info = []

# Model A: IBDQ ~ IBD-Disk Total
df_m = merged[['ibdq_total', 'disk_total']].dropna()
if len(df_m) >= 10:
    X = np.column_stack([np.ones(len(df_m)), df_m['disk_total'].values])
    y = df_m['ibdq_total'].values
    beta, _, _, _ = np_lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - 2)
    aic = calc_aic(y, y_pred, 2)
    models_info.append(('IBDQ ~ IBD-Disk Total', r2, r2_adj, aic, len(df_m)))

# Model B: IBDQ ~ Physical + Psychosocial
df_m = merged[['ibdq_total', 'physical_domain', 'psychosocial_domain']].dropna()
if len(df_m) >= 10:
    X = np.column_stack([np.ones(len(df_m)), df_m['physical_domain'].values, df_m['psychosocial_domain'].values])
    y = df_m['ibdq_total'].values
    beta, _, _, _ = np_lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - 3)
    aic = calc_aic(y, y_pred, 3)
    models_info.append(('IBDQ ~ Physical + Psychosocial', r2, r2_adj, aic, len(df_m)))

# Model C: IBDQ ~ IBD-Disk + CRAFFT
df_m = merged[['ibdq_total', 'disk_total', 'crafft_total']].dropna()
if len(df_m) >= 10:
    X = np.column_stack([np.ones(len(df_m)), df_m['disk_total'].values, df_m['crafft_total'].values])
    y = df_m['ibdq_total'].values
    beta, _, _, _ = np_lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - 3)
    aic = calc_aic(y, y_pred, 3)
    models_info.append(('IBDQ ~ IBD-Disk + CRAFFT Total', r2, r2_adj, aic, len(df_m)))

# Model D: IBDQ ~ Physical + Psychosocial + CRAFFT
df_m = merged[['ibdq_total', 'physical_domain', 'psychosocial_domain', 'crafft_total']].dropna()
if len(df_m) >= 10:
    X = np.column_stack([np.ones(len(df_m)), df_m['physical_domain'].values,
                         df_m['psychosocial_domain'].values, df_m['crafft_total'].values])
    y = df_m['ibdq_total'].values
    beta, _, _, _ = np_lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - 4)
    aic = calc_aic(y, y_pred, 4)
    models_info.append(('IBDQ ~ Physical + Psychosocial + CRAFFT', r2, r2_adj, aic, len(df_m)))

# Model E: IBDQ ~ top 3 IBD-Disk items
df_m = merged[['ibdq_total', 'item7_emotions', 'item6_energy', 'item5_sleep']].dropna()
if len(df_m) >= 10:
    X = np.column_stack([np.ones(len(df_m)), df_m['item7_emotions'].values,
                         df_m['item6_energy'].values, df_m['item5_sleep'].values])
    y = df_m['ibdq_total'].values
    beta, _, _, _ = np_lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - 4)
    aic = calc_aic(y, y_pred, 4)
    models_info.append(('IBDQ ~ Emoções + Energia + Sono', r2, r2_adj, aic, len(df_m)))

for name, r2, r2a, aic, n in models_info:
    log(f"  {name:<55s}  {r2:6.4f}  {r2a:7.4f}  {aic:8.1f}  {n:5d}")


# ==============================================================================
# SALVAR RELATÓRIO
# ==============================================================================
log("\n\n" + "=" * 100)
log("FIM DA FASE 2 — ANÁLISE AVANÇADA COMPLETA")
log("=" * 100)

report_path = os.path.join(OUT_DIR, "relatorio_fase2.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print(f"\n✓ Relatório salvo em: {report_path}")
print(f"✓ Figuras salvas em: {FIG_DIR}")
print("✓ Fase 2 concluída com sucesso!")
