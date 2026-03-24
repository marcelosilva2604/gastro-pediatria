#!/usr/bin/env python3
"""
06_regression_crafft.py
Regressão logística: preditores de CRAFFT positivo.
Uso: python3 calculos/06_regression_crafft.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DC = os.path.join(BASE, 'data_clean')

crafft = pd.read_csv(os.path.join(DC, 'crafft_clean.csv'))
pts = pd.read_csv(os.path.join(DC, 'todospacientes_clean.csv'))

# Merge CRAFFT with sex
sex_map = pts[pts['SEXO'].notna()].set_index('NOME')['SEXO'].to_dict()

crafft_f = crafft[crafft['DIAGNOSTICO'].notna()].drop_duplicates('PACIENTE', keep='first').copy()
crafft_f['CRAFFT_Positive'] = (crafft_f['Risk Interpretation'] == 'Positive screen (≥2)').astype(int)
crafft_f['SEXO'] = crafft_f['PACIENTE'].map(sex_map)
crafft_f['IS_CD'] = (crafft_f['DIAGNOSTICO'] == 'CD').astype(int)
crafft_f['IS_MALE'] = (crafft_f['SEXO'] == 'M').astype(int)

print("=" * 70)
print("REGRESSÃO LOGÍSTICA — PREDITORES DE CRAFFT POSITIVO")
print("=" * 70)

print(f"\nTotal pacientes com CRAFFT + diagnóstico: {len(crafft_f)}")
print(f"CRAFFT positivo: {crafft_f['CRAFFT_Positive'].sum()} ({crafft_f['CRAFFT_Positive'].mean()*100:.1f}%)")

# Check if statsmodels is available
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import logit
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("\nAVISO: statsmodels não instalado. Instalando...")
    import subprocess
    subprocess.check_call(['pip3', 'install', 'statsmodels', '-q'])
    import statsmodels.api as sm
    from statsmodels.formula.api import logit
    HAS_SM = True

# Prepare data - drop rows with missing sex
df = crafft_f[['CRAFFT_Positive', 'IS_CD', 'IS_MALE', 'IDADE_DIAGNOSTICO', 'DIAGNOSTICO']].dropna()
print(f"Pacientes no modelo (sem missing): {len(df)}")
print(f"Eventos (positivos): {df['CRAFFT_Positive'].sum()}")
print(f"Variáveis: 3 (diagnóstico CD, sexo masculino, idade ao diagnóstico)")
print(f"Eventos por variável: {df['CRAFFT_Positive'].sum()/3:.1f} (recomendado >=10)")

# Univariate analyses
print(f"\n--- ANÁLISES UNIVARIADAS ---")
for var, label in [('IS_CD', 'CD (vs UC/IBD-U)'), ('IS_MALE', 'Male (vs Female)'), ('IDADE_DIAGNOSTICO', 'Age at diagnosis')]:
    X = sm.add_constant(df[[var]])
    try:
        model = sm.Logit(df['CRAFFT_Positive'], X).fit(disp=0)
        or_val = np.exp(model.params[var])
        ci_lo = np.exp(model.conf_int().loc[var, 0])
        ci_hi = np.exp(model.conf_int().loc[var, 1])
        p = model.pvalues[var]
        print(f"  {label}: OR = {or_val:.2f} (95% CI: {ci_lo:.2f}–{ci_hi:.2f}), p = {p:.3f}")
    except Exception as e:
        print(f"  {label}: ERRO — {e}")

# Multivariable model
print(f"\n--- MODELO MULTIVARIÁVEL ---")
X = sm.add_constant(df[['IS_CD', 'IS_MALE', 'IDADE_DIAGNOSTICO']])
try:
    model = sm.Logit(df['CRAFFT_Positive'], X).fit(disp=0)
    print(f"\nResultados:")
    print(f"  {'Variable':<25} {'aOR':>6} {'95% CI':>15} {'p-value':>10}")
    print(f"  {'-'*60}")
    for var, label in [('IS_CD', 'CD (vs UC/IBD-U)'), ('IS_MALE', 'Male (vs Female)'), ('IDADE_DIAGNOSTICO', 'Age at diagnosis')]:
        or_val = np.exp(model.params[var])
        ci_lo = np.exp(model.conf_int().loc[var, 0])
        ci_hi = np.exp(model.conf_int().loc[var, 1])
        p = model.pvalues[var]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {label:<25} {or_val:>6.2f} {f'{ci_lo:.2f}–{ci_hi:.2f}':>15} {p:>10.3f} {sig}")

    print(f"\n  Model summary:")
    print(f"    Pseudo R²: {model.prsquared:.3f}")
    print(f"    AIC: {model.aic:.1f}")
    print(f"    Log-likelihood: {model.llf:.1f}")

except Exception as e:
    print(f"ERRO no modelo: {e}")

print(f"\n{'=' * 70}")
print("FIM — 06_regression_crafft.py")
