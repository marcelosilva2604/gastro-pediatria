"""
Limpeza do IMPACT-III — Versão de 30 itens (domínio Tratamento não administrado)
Critérios de exclusão:
  1. Linhas sem nome de paciente
  2. Pacientes com todos os itens em branco (não ofertado)
  3. Pacientes com >30% dos 30 itens faltantes (>9 itens)
  4. RECUSOU / sem questionário
Critério de imputação:
  - Pacientes com 1-9 itens faltantes (dos 30): mediana do próprio paciente
Itens 31-35 descartados (domínio Tratamento não foi administrado)
Domínios (versão 30 itens):
  - Sintomas: itens 1-9 (9 itens, range 9-45)
  - Bem-estar: itens 10-16 (7 itens, range 7-35)
  - Emocional: itens 17-23 (7 itens, range 7-35)
  - Social: itens 24-30 (7 itens, range 7-35)
  - Total: 30 itens, range 30-150
  - Média por item: range 1-5 (comparável entre versões)
"""

import pandas as pd
import numpy as np

df = pd.read_csv(
    'csv/IMPACTIII FINAL.xlsx - Cálculo Automático.csv',
    dtype=str
)

print(f"Linhas no arquivo original: {len(df)}")
print()

# Renomear colunas
item_cols_all = [f'item_{i}' for i in range(1, 36)]
df.columns = (
    ['data_avaliacao', 'paciente']
    + item_cols_all
    + ['dom_sintomas_orig', 'dom_emocional_orig', 'dom_social_orig',
       'dom_bemestar_orig', 'dom_tratamento_orig', 'total_orig']
)

# Itens que vamos usar (1-30 apenas)
item_cols = [f'item_{i}' for i in range(1, 31)]

# Converter itens para numérico
df_num = df[item_cols].apply(pd.to_numeric, errors='coerce')
items_respondidos = df_num.notna().sum(axis=1)

# ---- EXCLUSÃO 1: Sem nome ----
sem_nome = df['paciente'].isna() | (df['paciente'].str.strip() == '')
print(f"1. Sem nome: {sem_nome.sum()} linhas")

# ---- EXCLUSÃO 2: RECUSOU ----
recusou = df['paciente'].str.contains('RECUSOU', case=False, na=False)
print(f"2. Recusou: {recusou.sum()} linhas")

# ---- EXCLUSÃO 3: Todos itens vazios (não ofertado) ----
todos_vazios = items_respondidos == 0
tem_nome = ~sem_nome & ~recusou
nao_ofertado = tem_nome & todos_vazios
print(f"3. Não ofertado (0 itens): {nao_ofertado.sum()} linhas")

# ---- EXCLUSÃO 4: >30% faltante (>9 itens de 30) ----
muita_falta = tem_nome & ~todos_vazios & (items_respondidos < 21)
print(f"4. Mais de 30% faltante (>9 itens): {muita_falta.sum()} linhas")
for idx in df[muita_falta].index:
    nome = df.loc[idx, 'paciente']
    n = items_respondidos[idx]
    print(f"   - {nome} ({n}/30 itens = {n/30*100:.1f}%)")

print()

# ---- APLICAR EXCLUSÕES ----
excluir = sem_nome | recusou | nao_ofertado | muita_falta
df_clean = df[~excluir].copy()
df_num_clean = df_num.loc[~excluir].copy()
items_resp_clean = items_respondidos[~excluir]

print(f"Total excluídas: {excluir.sum()}")
print(f"Linhas após exclusão: {len(df_clean)}")
print()

# ---- DIAGNÓSTICO DE ITENS FALTANTES ----
faltantes = 30 - items_resp_clean
com_falta = faltantes > 0
print(f"Pacientes com itens faltantes (dos 30): {com_falta.sum()}")
for idx in df_clean[com_falta].index:
    nome = df_clean.loc[idx, 'paciente']
    n_falta = faltantes[idx]
    itens_vazios = [c for c in item_cols if pd.isna(df_num_clean.loc[idx, c])]
    mediana_pac = df_num_clean.loc[idx, item_cols].median()
    print(f"   - {nome}: {n_falta} item(s) faltante(s) {itens_vazios} → mediana={mediana_pac}")
print()

# ---- IMPUTAÇÃO: mediana do próprio paciente ----
for idx in df_clean[com_falta].index:
    mediana_pac = df_num_clean.loc[idx, item_cols].median()
    for col in item_cols:
        if pd.isna(df_num_clean.loc[idx, col]):
            df_num_clean.loc[idx, col] = mediana_pac

assert df_num_clean[item_cols].isna().sum().sum() == 0, "Ainda há NaN!"
print("Imputação concluída — zero NaN restantes.")
print()

# ---- RECALCULAR DOMÍNIOS DO ZERO ----
sintomas = [f'item_{i}' for i in range(1, 10)]       # 1-9
bemestar = [f'item_{i}' for i in range(10, 17)]       # 10-16
emocional = [f'item_{i}' for i in range(17, 24)]      # 17-23
social = [f'item_{i}' for i in range(24, 31)]          # 24-30

df_result = pd.DataFrame()
df_result['data_avaliacao'] = df_clean['data_avaliacao'].values
df_result['paciente'] = df_clean['paciente'].str.strip().str.replace(
    r'\s*\(.*?\)\s*', '', regex=True
).str.strip().values

# Itens
for col in item_cols:
    df_result[col] = df_num_clean[col].astype(int).values

# Domínios
df_result['dom_sintomas'] = df_num_clean[sintomas].sum(axis=1).astype(int).values
df_result['dom_bemestar'] = df_num_clean[bemestar].sum(axis=1).astype(int).values
df_result['dom_emocional'] = df_num_clean[emocional].sum(axis=1).astype(int).values
df_result['dom_social'] = df_num_clean[social].sum(axis=1).astype(int).values
df_result['total_score'] = df_num_clean[item_cols].sum(axis=1).astype(int).values
df_result['mean_per_item'] = (df_result['total_score'] / 30).round(2)
df_result['items_imputed'] = faltantes[~excluir].values

# Corrigir "SEM DATA"
df_result['data_avaliacao'] = df_result['data_avaliacao'].replace('SEM DATA', '')

# ---- VERIFICAÇÃO FINAL ----
print("=== DATASET LIMPO ===")
print(f"N total: {len(df_result)}")
print(f"Com data válida: {(df_result['data_avaliacao'].notna() & (df_result['data_avaliacao'] != '')).sum()}")
print(f"Sem data: {(df_result['data_avaliacao'].isna() | (df_result['data_avaliacao'] == '')).sum()}")
print()
print(f"Itens imputados por paciente:")
print(f"  0 (completo): {(df_result['items_imputed'] == 0).sum()}")
print(f"  1+ itens:     {(df_result['items_imputed'] > 0).sum()}")
print()
print(f"Total score (30 itens): mediana={df_result['total_score'].median()}, "
      f"média={df_result['total_score'].mean():.1f}, "
      f"range={df_result['total_score'].min()}-{df_result['total_score'].max()}")
print(f"Média por item: mediana={df_result['mean_per_item'].median()}, "
      f"média={df_result['mean_per_item'].mean():.2f}, "
      f"range={df_result['mean_per_item'].min()}-{df_result['mean_per_item'].max()}")
print()
print(f"Domínios (média ± DP):")
print(f"  Sintomas (9 itens, 9-45):   {df_result['dom_sintomas'].mean():.1f} ± {df_result['dom_sintomas'].std():.1f}")
print(f"  Bem-estar (7 itens, 7-35):  {df_result['dom_bemestar'].mean():.1f} ± {df_result['dom_bemestar'].std():.1f}")
print(f"  Emocional (7 itens, 7-35):  {df_result['dom_emocional'].mean():.1f} ± {df_result['dom_emocional'].std():.1f}")
print(f"  Social (7 itens, 7-35):     {df_result['dom_social'].mean():.1f} ± {df_result['dom_social'].std():.1f}")

# ---- SALVAR ----
df_result.to_csv('data_clean/impact_clean.csv', index=False)
df_result.to_excel('data_clean/impact_clean.xlsx', index=False)
print()
print("Salvo em: data_clean/impact_clean.csv")
print("Salvo em: data_clean/impact_clean.xlsx")
