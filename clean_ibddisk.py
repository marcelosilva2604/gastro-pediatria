"""
Limpeza do IBD-Disk — Remoção de linhas inválidas e recálculo dos scores
Critérios de exclusão:
  1. Linhas sem nome de paciente (vazias)
  2. Linhas com nome mas sem nenhum item preenchido (não responderam, #DIV/0!)
  3. Duplicatas explícitas marcadas como "pct. repetido"
  4. Paciente que recusou participar (RECUSOU)
  5. Marcados como "NÃO RESPONDEU" ou "sem questionário"
  6. Linhas com data mas todos os itens vazios (#DIV/0!)
Critério de manutenção:
  - Pacientes com data + todos os itens = 0 são respostas legítimas
  - Pacientes sem data mas com itens preenchidos (FABIO, CESAR) — mantidos
  - Pacientes com 1-2 itens faltantes — mantidos, média calculada sobre itens respondidos
Recálculos:
  - Physical_Domain_Mean, Psychosocial_Domain_Mean, Total, Mean — recalculados do zero
  - Colunas Age, Sex, Diagnosis removidas (todas vazias)
"""

import pandas as pd
import numpy as np

# Carregar CSV original — tudo como string para não perder nada
df = pd.read_csv(
    'csv/IBD Disk FINAL.xlsx - IBD_Disk_Data.csv',
    dtype=str
)

print(f"Linhas no arquivo original: {len(df)}")
print()

# Renomear colunas
df.columns = [
    'data_avaliacao', 'paciente', 'age', 'sex', 'diagnosis',
    'item1_abd_pain', 'item2_defecation', 'item3_interpersonal',
    'item4_education_work', 'item5_sleep', 'item6_energy',
    'item7_emotions', 'item8_body_image', 'item9_sexual_function',
    'item10_joint_pain',
    'physical_domain_mean_orig', 'psychosocial_domain_mean_orig',
    'total_score_orig', 'mean_score_orig'
]

itens = [
    'item1_abd_pain', 'item2_defecation', 'item3_interpersonal',
    'item4_education_work', 'item5_sleep', 'item6_energy',
    'item7_emotions', 'item8_body_image', 'item9_sexual_function',
    'item10_joint_pain'
]

# ---- EXCLUSÃO 1: Linhas sem nome de paciente ----
sem_nome = df['paciente'].isna() | (df['paciente'].str.strip() == '')
print(f"1. Sem nome de paciente: {sem_nome.sum()} linhas")

# ---- EXCLUSÃO 2: Duplicatas explícitas ("pct. repetido") ----
duplicata = df['paciente'].str.contains('pct. repetido', case=False, na=False)
print(f"2. Duplicatas explícitas: {duplicata.sum()} linhas")
for _, row in df[duplicata].iterrows():
    print(f"   - {row['paciente']}")

# ---- EXCLUSÃO 3: Paciente que recusou ----
recusou = df['paciente'].str.contains('RECUSOU', case=False, na=False)
print(f"3. Recusou participar: {recusou.sum()} linhas")
for _, row in df[recusou].iterrows():
    print(f"   - {row['paciente']}")

# ---- EXCLUSÃO 4: Marcados como "sem questionário" ----
sem_quest = df['paciente'].str.contains('sem questionário', case=False, na=False)
print(f"4. Sem questionário: {sem_quest.sum()} linhas")
for _, row in df[sem_quest].iterrows():
    print(f"   - {row['paciente']}")

# ---- EXCLUSÃO 5: "NÃO RESPONDEU" no campo data ----
nao_respondeu_data = df['data_avaliacao'].str.contains('NÃO RESPONDEU', case=False, na=False)
print(f"5. 'NÃO RESPONDEU' na data: {nao_respondeu_data.sum()} linhas")
for _, row in df[nao_respondeu_data].iterrows():
    print(f"   - {row['paciente']}")

# ---- EXCLUSÃO 6: Com nome, sem data, sem itens preenchidos ----
# Converter itens para numérico temporariamente para checar
itens_num = df[itens].apply(pd.to_numeric, errors='coerce')
sem_data = df['data_avaliacao'].isna() | (df['data_avaliacao'].str.strip() == '')
sem_itens = itens_num.isna().all(axis=1)
tem_nome = ~sem_nome
nao_respondeu_sem_data = (
    tem_nome & sem_data & sem_itens
    & ~duplicata & ~recusou & ~sem_quest
)
print(f"6. Com nome, sem data, sem itens: {nao_respondeu_sem_data.sum()} linhas")
for _, row in df[nao_respondeu_sem_data].iterrows():
    print(f"   - {row['paciente']}")

# ---- EXCLUSÃO 7: Com data mas itens todos vazios (#DIV/0!) ----
com_data = ~sem_data & ~nao_respondeu_data
com_data_sem_itens = com_data & tem_nome & sem_itens
print(f"7. Com data mas itens todos vazios: {com_data_sem_itens.sum()} linhas")
for _, row in df[com_data_sem_itens].iterrows():
    print(f"   - {row['paciente']} (data: {row['data_avaliacao']})")

print()

# ---- APLICAR TODAS AS EXCLUSÕES ----
excluir = (
    sem_nome | duplicata | recusou | sem_quest
    | nao_respondeu_data | nao_respondeu_sem_data | com_data_sem_itens
)
df_clean = df[~excluir].copy()

print(f"Total excluídas: {excluir.sum()}")
print(f"Linhas após limpeza: {len(df_clean)}")
print()

# ---- LIMPAR NOMES ----
df_clean['paciente'] = df_clean['paciente'].str.strip()
df_clean['paciente'] = df_clean['paciente'].str.replace(
    r'\s*\(.*?\)\s*', '', regex=True
).str.strip()

# ---- CORRIGIR DATA COM ERRO DE DIGITAÇÃO ----
df_clean['data_avaliacao'] = df_clean['data_avaliacao'].str.replace(
    '18/404/2023', '18/04/2023'
)
# "SEM DATA" → vazio
df_clean['data_avaliacao'] = df_clean['data_avaliacao'].replace('SEM DATA', '')

# ---- CONVERTER ITENS PARA NUMÉRICO ----
for col in itens:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# ---- REMOVER COLUNAS VAZIAS E CALCULADAS DO EXCEL ----
df_clean = df_clean.drop(columns=[
    'age', 'sex', 'diagnosis',
    'physical_domain_mean_orig', 'psychosocial_domain_mean_orig',
    'total_score_orig', 'mean_score_orig'
])

# ---- RECALCULAR SCORES DO ZERO ----
# IBD-Disk: Physical = itens 1,2,5,6,10 (dor abd, defecação, sono, energia, dor articular)
# IBD-Disk: Psychosocial = itens 3,4,7,8,9 (interpessoal, educação, emoções, imagem, sexual)
physical_cols = [
    'item1_abd_pain', 'item2_defecation', 'item5_sleep',
    'item6_energy', 'item10_joint_pain'
]
psychosocial_cols = [
    'item3_interpersonal', 'item4_education_work', 'item7_emotions',
    'item8_body_image', 'item9_sexual_function'
]

# Média sobre itens respondidos (ignora NaN)
df_clean['physical_domain_mean'] = df_clean[physical_cols].mean(axis=1).round(2)
df_clean['psychosocial_domain_mean'] = df_clean[psychosocial_cols].mean(axis=1).round(2)
df_clean['total_score'] = df_clean[itens].sum(axis=1, min_count=1).astype('Int64')
df_clean['items_responded'] = df_clean[itens].notna().sum(axis=1)
df_clean['mean_score'] = (df_clean[itens].sum(axis=1, min_count=1) / df_clean['items_responded']).round(2)

# ---- VERIFICAÇÃO FINAL ----
print("=== DATASET LIMPO ===")
print(f"N total: {len(df_clean)}")
print(f"Com data válida: {(df_clean['data_avaliacao'].notna() & (df_clean['data_avaliacao'] != '')).sum()}")
print(f"Sem data (com itens): {(df_clean['data_avaliacao'].isna() | (df_clean['data_avaliacao'] == '')).sum()}")
print()
print(f"Itens respondidos por paciente:")
print(f"  10/10 itens: {(df_clean['items_responded'] == 10).sum()}")
print(f"  9/10 itens:  {(df_clean['items_responded'] == 9).sum()}")
print(f"  8/10 itens:  {(df_clean['items_responded'] == 8).sum()}")
print()
print(f"Total score: mediana={df_clean['total_score'].median()}, "
      f"média={df_clean['total_score'].mean():.1f}, "
      f"range={df_clean['total_score'].min()}-{df_clean['total_score'].max()}")
print()
print(f"Pacientes com total = 0 (legítimos): {(df_clean['total_score'] == 0).sum()}")

# ---- SALVAR ----
df_clean.to_csv('data_clean/ibddisk_clean.csv', index=False)
df_clean.to_excel('data_clean/ibddisk_clean.xlsx', index=False)
print()
print("Salvo em: data_clean/ibddisk_clean.csv")
print("Salvo em: data_clean/ibddisk_clean.xlsx")
