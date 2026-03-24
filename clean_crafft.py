"""
Limpeza do CRAFFT — Remoção de linhas inválidas
Critérios de exclusão:
  1. Linhas sem nome de paciente (vazias)
  2. Linhas com nome mas sem nenhum item preenchido e sem data (não responderam)
  3. Duplicatas explícitas marcadas como "pct. repetido"
  4. Paciente que recusou participar (RECUSOU)
Critério de manutenção:
  - Pacientes com data + todos os itens = 0 são respostas legítimas
  - FABIO GONÇALVES SUZART: itens preenchidos mas sem data — mantido
"""

import pandas as pd

# Carregar CSV original
df = pd.read_csv(
    'csv/CRAFFT FINAL.xlsx - CRAFFT Scoring.csv',
    dtype=str  # tudo como string primeiro para não perder nada
)

print(f"Linhas no arquivo original: {len(df)}")
print()

# Renomear colunas para facilitar
df.columns = [
    'data_avaliacao', 'paciente',
    'C_Car', 'R_Relax', 'A_Alone', 'F_Forget', 'F_Friends', 'T_Trouble',
    'total_score', 'risk_interpretation'
]

# ---- EXCLUSÃO 1: Linhas sem nome de paciente ----
sem_nome = df['paciente'].isna() | (df['paciente'].str.strip() == '')
print(f"1. Sem nome de paciente: {sem_nome.sum()} linhas")
print()

# ---- EXCLUSÃO 2: Duplicatas explícitas ("pct. repetido") ----
duplicata = df['paciente'].str.contains('pct. repetido', case=False, na=False)
print(f"2. Duplicatas explícitas: {duplicata.sum()} linhas")
for _, row in df[duplicata].iterrows():
    print(f"   - {row['paciente']}")
print()

# ---- EXCLUSÃO 3: Paciente que recusou ----
recusou = df['paciente'].str.contains('RECUSOU', case=False, na=False)
print(f"3. Recusou participar: {recusou.sum()} linhas")
for _, row in df[recusou].iterrows():
    print(f"   - {row['paciente']}")
print()

# ---- EXCLUSÃO 4: Nome presente mas sem data E sem itens preenchidos ----
itens = ['C_Car', 'R_Relax', 'A_Alone', 'F_Forget', 'F_Friends', 'T_Trouble']
sem_data = df['data_avaliacao'].isna() | (df['data_avaliacao'].str.strip() == '')
sem_itens = df[itens].isna().all(axis=1) | (df[itens] == '').all(axis=1)
tem_nome = ~sem_nome
nao_respondeu = tem_nome & sem_data & sem_itens & ~duplicata & ~recusou

print(f"4. Com nome mas sem data e sem itens (não responderam): {nao_respondeu.sum()} linhas")
for _, row in df[nao_respondeu].iterrows():
    print(f"   - {row['paciente']}")
print()

# ---- APLICAR EXCLUSÕES ----
excluir = sem_nome | duplicata | recusou | nao_respondeu
df_clean = df[~excluir].copy()

print(f"Total excluídas: {excluir.sum()}")
print(f"Linhas após limpeza: {len(df_clean)}")
print()

# ---- CONVERTER TIPOS ----
for col in itens:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('Int64')

df_clean['total_score'] = pd.to_numeric(df_clean['total_score'], errors='coerce').astype('Int64')

# Limpar espaços nos nomes
df_clean['paciente'] = df_clean['paciente'].str.strip()

# Remover anotações dos nomes (DII??, etc.)
df_clean['paciente'] = df_clean['paciente'].str.replace(r'\s*\(.*?\)\s*', '', regex=True).str.strip()

# ---- VERIFICAÇÃO FINAL ----
print("=== DATASET LIMPO ===")
print(f"N total: {len(df_clean)}")
print(f"Com data: {df_clean['data_avaliacao'].notna().sum()}")
print(f"Sem data (FABIO GONÇALVES SUZART): {df_clean['data_avaliacao'].isna().sum() | (df_clean['data_avaliacao'] == '').sum()}")
print()
print(f"Score 0 (respondeu tudo zero): {(df_clean['total_score'] == 0).sum()}")
print(f"Negative screen (<2): {(df_clean['risk_interpretation'].str.contains('Negative', na=False)).sum()}")
print(f"Positive screen (>=2): {(df_clean['risk_interpretation'].str.contains('Positive', na=False)).sum()}")
print()
print(f"Período: {df_clean['data_avaliacao'].dropna().min()} a {df_clean['data_avaliacao'].dropna().max()}")

# ---- SALVAR ----
df_clean.to_csv('data_clean/crafft_clean.csv', index=False)
print()
print("Salvo em: data_clean/crafft_clean.csv")
