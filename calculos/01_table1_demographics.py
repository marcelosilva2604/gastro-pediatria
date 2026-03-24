#!/usr/bin/env python3
"""
01_table1_demographics.py
Gera estatísticas demográficas da Table 1.
Lê: data_clean/todospacientes_clean.csv
Uso: python3 calculos/01_table1_demographics.py
"""

import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pts = pd.read_csv(os.path.join(BASE, 'data_clean', 'todospacientes_clean.csv'))
inc = pts[pts['STATUS'] == 'INCLUÍDO'].copy()

print("=" * 70)
print("TABLE 1 — DEMOGRAPHICS")
print("=" * 70)

n = len(inc)
print(f"\nTotal incluídos: {n}")

# Sex
print(f"\n--- SEXO ---")
sex_known = inc[inc['SEXO'].notna()]
n_sex = len(sex_known)
for s in ['M', 'F']:
    c = (sex_known['SEXO'] == s).sum()
    print(f"  {s}: {c} ({c/n_sex*100:.1f}%)")
print(f"  Missing: {inc['SEXO'].isna().sum()}")
print(f"  Total com sexo: {n_sex}")

# Sex by diagnosis
print(f"\n--- SEXO POR DIAGNÓSTICO ---")
for d in ['CD', 'UC', 'IBD-U']:
    sub = sex_known[sex_known['DIAGNOSTICO'] == d]
    n_d = len(sub)
    m = (sub['SEXO'] == 'M').sum()
    f = (sub['SEXO'] == 'F').sum()
    miss = inc[(inc['DIAGNOSTICO'] == d) & (inc['SEXO'].isna())].shape[0]
    print(f"  {d} (n={len(inc[inc['DIAGNOSTICO']==d])}): M={m} ({m/n_d*100:.1f}%), F={f} ({f/n_d*100:.1f}%), Missing={miss}")

# Diagnosis
print(f"\n--- DIAGNÓSTICO ---")
for d in ['CD', 'UC', 'IBD-U']:
    c = (inc['DIAGNOSTICO'] == d).sum()
    print(f"  {d}: {c} ({c/n*100:.1f}%)")

# Age at diagnosis
print(f"\n--- IDADE AO DIAGNÓSTICO ---")
ages = inc['IDADE_DIAGNOSTICO'].dropna()
print(f"  n com idade: {len(ages)}")
print(f"  Median (IQR): {ages.median():.0f} ({ages.quantile(0.25):.0f}–{ages.quantile(0.75):.0f})")
print(f"  Range: {ages.min():.0f}–{ages.max():.0f}")
print(f"  Mean ± SD: {ages.mean():.1f} ± {ages.std():.1f}")

for d in ['CD', 'UC', 'IBD-U']:
    a = inc[inc['DIAGNOSTICO'] == d]['IDADE_DIAGNOSTICO'].dropna()
    print(f"  {d} (n={len(a)}): median {a.median():.0f} (IQR {a.quantile(0.25):.0f}–{a.quantile(0.75):.0f}), range {a.min():.0f}–{a.max():.0f}")

# Paris classification
print(f"\n--- CLASSIFICAÇÃO DE PARIS ---")
def paris(age):
    if age < 10: return 'A1a (<10)'
    elif age < 17: return 'A1b (10-16)'
    else: return 'A2 (17-40)'

for d in ['All', 'CD', 'UC', 'IBD-U']:
    a = ages if d == 'All' else inc[inc['DIAGNOSTICO'] == d]['IDADE_DIAGNOSTICO'].dropna()
    n_a = len(a)
    a1a = (a < 10).sum()
    a1b = ((a >= 10) & (a < 17)).sum()
    a2 = (a >= 17).sum()
    print(f"  {d} (n={n_a}): A1a={a1a} ({a1a/n_a*100:.1f}%), A1b={a1b} ({a1b/n_a*100:.1f}%), A2={a2} ({a2/n_a*100:.1f}%)")

print(f"\n{'=' * 70}")
print("FIM — 01_table1_demographics.py")
