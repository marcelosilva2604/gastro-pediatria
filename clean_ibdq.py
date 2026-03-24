"""
Limpeza do IBDQ — Remoção de linhas inválidas e recálculo dos scores
Critérios de exclusão:
  1. Linhas sem nome de paciente (vazias)
  2. Duplicatas explícitas ("pct. repetido")
  3. Paciente que recusou ("RECUSOU") ou "sem questionário"
  4. Pacientes com todos os itens em branco (não ofertado)
  5. Pacientes com apenas 11 itens respondidos (instrumento incompleto)
Critério de imputação:
  - Pacientes com 30-31 de 32 itens: imputar a MEDIANA do próprio paciente
Recálculos:
  - Todos os domínios e total recalculados do zero
  - Domínios IBDQ-32 (Irvine 1994):
    * Bowel Symptoms (10 itens): 1, 5, 9, 13, 17, 20, 22, 24, 26, 29
    * Systemic Symptoms (5 itens): 2, 6, 10, 14, 18
    * Emotional Function (12 itens): 3, 7, 11, 15, 19, 21, 23, 25, 27, 30, 31, 32
    * Social Function (5 itens): 4, 8, 12, 16, 28
"""

import pandas as pd
import numpy as np

# Carregar CSV original
df = pd.read_csv(
    'csv/IBDQ FINAL.xlsx - Cálculo Automático.csv',
    dtype=str
)

print(f"Linhas no arquivo original: {len(df)}")
print()

# Renomear colunas
item_cols = [f'item_{i}' for i in range(1, 33)]
df.columns = (
    ['data_avaliacao', 'paciente']
    + item_cols
    + ['sint_intestinais_orig', 'sint_sistemicos_orig',
       'bem_estar_emocional_orig', 'interacao_social_orig', 'total_orig']
)

# ---- EXCLUSÃO 1: Linhas sem nome ----
sem_nome = df['paciente'].isna() | (df['paciente'].str.strip() == '')
print(f"1. Sem nome: {sem_nome.sum()} linhas")

# ---- EXCLUSÃO 2: Duplicatas ("pct. repetido") ----
duplicata = df['paciente'].str.contains('pct. repetido', case=False, na=False)
print(f"2. Duplicatas: {duplicata.sum()} linhas")
for _, row in df[duplicata].iterrows():
    print(f"   - {row['paciente']}")

# ---- EXCLUSÃO 3: RECUSOU / sem questionário ----
recusou = df['paciente'].str.contains('RECUSOU|sem questionário', case=False, na=False)
print(f"3. Recusou/sem questionário: {recusou.sum()} linhas")
for _, row in df[recusou].iterrows():
    print(f"   - {row['paciente']}")

# ---- Converter itens para numérico ----
df_num = df[item_cols].apply(pd.to_numeric, errors='coerce')
items_respondidos = df_num.notna().sum(axis=1)

# ---- EXCLUSÃO 4: Todos os itens em branco (não ofertado) ----
todos_vazios = items_respondidos == 0
tem_nome = ~sem_nome & ~duplicata & ~recusou
nao_ofertado = tem_nome & todos_vazios
print(f"4. Não ofertado (0 itens): {nao_ofertado.sum()} linhas")

# ---- EXCLUSÃO 5: Apenas 11 itens (instrumento incompleto) ----
# Pacientes com itens 1-11 preenchidos e 12-32 todos vazios
itens_1_11 = df_num[item_cols[:11]].notna().sum(axis=1)
itens_12_32 = df_num[item_cols[11:]].notna().sum(axis=1)
incompleto_11 = tem_nome & ~todos_vazios & (itens_1_11 > 0) & (itens_12_32 == 0)
print(f"5. Apenas 11 itens (incompleto): {incompleto_11.sum()} linhas")
for _, row in df[incompleto_11].iterrows():
    n_resp = items_respondidos[row.name]
    print(f"   - {row['paciente']} ({n_resp} itens)")

# ---- EXCLUSÃO 6: Mais de 30% dos itens faltantes (>9.6 = 10+ itens) ----
muita_falta = tem_nome & ~todos_vazios & ~incompleto_11 & (items_respondidos < 23)
print(f"6. Mais de 30% faltante (>9 itens): {muita_falta.sum()} linhas")
for _, row in df[muita_falta].iterrows():
    n_resp = items_respondidos[row.name]
    print(f"   - {row['paciente']} ({n_resp}/32 itens = {n_resp/32*100:.1f}%)")

print()

# ---- APLICAR EXCLUSÕES ----
excluir = sem_nome | duplicata | recusou | nao_ofertado | incompleto_11 | muita_falta
df_clean = df[~excluir].copy()
df_num_clean = df_num[~excluir].copy()
items_resp_clean = items_respondidos[~excluir]

print(f"Total excluídas: {excluir.sum()}")
print(f"Linhas após exclusão: {len(df_clean)}")
print()

# ---- DIAGNÓSTICO DE ITENS FALTANTES NOS MANTIDOS ----
faltantes = 32 - items_resp_clean
com_falta = faltantes > 0
print(f"Pacientes mantidos com itens faltantes: {com_falta.sum()}")
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

# Verificar que não há mais NaN
assert df_num_clean[item_cols].isna().sum().sum() == 0, "Ainda há NaN após imputação!"
print("Imputação concluída — zero NaN restantes.")
print()

# ---- RECALCULAR DOMÍNIOS DO ZERO ----
# Mapeamento dos domínios IBDQ-32
bowel = [f'item_{i}' for i in [1, 5, 9, 13, 17, 20, 22, 24, 26, 29]]
systemic = [f'item_{i}' for i in [2, 6, 10, 14, 18]]
emotional = [f'item_{i}' for i in [3, 7, 11, 15, 19, 21, 23, 25, 27, 30, 31, 32]]
social = [f'item_{i}' for i in [4, 8, 12, 16, 28]]

df_result = pd.DataFrame()
df_result['data_avaliacao'] = df_clean['data_avaliacao'].values
df_result['paciente'] = df_clean['paciente'].str.strip().str.replace(
    r'\s*\(.*?\)\s*', '', regex=True
).str.strip().values

# Itens
for col in item_cols:
    df_result[col] = df_num_clean[col].astype(int).values

# Domínios
df_result['bowel_symptoms'] = df_num_clean[bowel].sum(axis=1).astype(int).values
df_result['systemic_symptoms'] = df_num_clean[systemic].sum(axis=1).astype(int).values
df_result['emotional_function'] = df_num_clean[emotional].sum(axis=1).astype(int).values
df_result['social_function'] = df_num_clean[social].sum(axis=1).astype(int).values
df_result['total_score'] = df_num_clean[item_cols].sum(axis=1).astype(int).values
df_result['items_imputed'] = faltantes[~excluir].values

# Corrigir data com erro de digitação
df_result['data_avaliacao'] = df_result['data_avaliacao'].str.replace(
    '18/404/2023', '18/04/2023'
)

# ---- VERIFICAÇÃO FINAL ----
print("=== DATASET LIMPO ===")
print(f"N total: {len(df_result)}")
print(f"Com data válida: {(df_result['data_avaliacao'].notna() & (df_result['data_avaliacao'] != '')).sum()}")
print(f"Sem data: {(df_result['data_avaliacao'].isna() | (df_result['data_avaliacao'] == '')).sum()}")
print()
print(f"Itens imputados por paciente:")
print(f"  0 (completo): {(df_result['items_imputed'] == 0).sum()}")
print(f"  1 item:       {(df_result['items_imputed'] == 1).sum()}")
print(f"  2 itens:      {(df_result['items_imputed'] == 2).sum()}")
print(f"  3+ itens:     {(df_result['items_imputed'] >= 3).sum()}")
print()
print(f"Total score: mediana={df_result['total_score'].median()}, "
      f"média={df_result['total_score'].mean():.1f}, "
      f"range={df_result['total_score'].min()}-{df_result['total_score'].max()}")
print()
print(f"Domínios (média ± DP):")
print(f"  Bowel Symptoms:     {df_result['bowel_symptoms'].mean():.1f} ± {df_result['bowel_symptoms'].std():.1f} (range 10-70)")
print(f"  Systemic Symptoms:  {df_result['systemic_symptoms'].mean():.1f} ± {df_result['systemic_symptoms'].std():.1f} (range 5-35)")
print(f"  Emotional Function: {df_result['emotional_function'].mean():.1f} ± {df_result['emotional_function'].std():.1f} (range 12-84)")
print(f"  Social Function:    {df_result['social_function'].mean():.1f} ± {df_result['social_function'].std():.1f} (range 5-35)")

# ---- SALVAR ----
df_result.to_csv('data_clean/ibdq_clean.csv', index=False)
df_result.to_excel('data_clean/ibdq_clean.xlsx', index=False)
print()
print("Salvo em: data_clean/ibdq_clean.csv")
print("Salvo em: data_clean/ibdq_clean.xlsx")
