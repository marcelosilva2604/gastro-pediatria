"""
Adiciona coluna 'sex' (1=M, 0=F) nos datasets limpos, inferido pelo primeiro nome.
Nomes brasileiros são altamente confiáveis para inferência de sexo.
"""

import pandas as pd

# Mapeamento por primeiro nome
femininos = {
    'ADRIELI', 'AMANDA', 'ANA', 'ANNA', 'BEATRIZ', 'BRUNA', 'CARLA',
    'DAIENE', 'ELLEN', 'ELOA', 'EMILY', 'ESTHEPHANY', 'EVELYN',
    'FABRICIA', 'FABRINA', 'FERNANDA', 'GABRIELA', 'GIOVANNA',
    'HORANA', 'ISA', 'JESSICA', 'JULIA', 'JULIANE', 'KARISSA',
    'KELEN', 'LAIS', 'LARISSA', 'LETICIA', 'LIGIA', 'MARIA',
    'MARLI', 'MELISSA', 'MONIQUE', 'NATHALIA', 'PATRICIA',
    'PRISCILA', 'RAFAELI', 'RAISSA', 'RAKEL', 'RAPHAELA',
    'RENATA', 'SAFIRA', 'STEFANIE', 'TALITA', 'THAINA', 'THALIA',
    'VITORIA', 'YANDRA', 'YARA', 'YASMIN'
}

def infer_sex(nome):
    if pd.isna(nome):
        return None
    primeiro = nome.strip().split()[0]
    return 0 if primeiro in femininos else 1

# Aplicar em cada dataset limpo
datasets = {
    'crafft_clean': 'data_clean/crafft_clean.csv',
    'ibddisk_clean': 'data_clean/ibddisk_clean.csv',
    'ibdq_clean': 'data_clean/ibdq_clean.csv',
    'impact_clean': 'data_clean/impact_clean.csv',
}

for name, path in datasets.items():
    df = pd.read_csv(path)
    df['sex'] = df['paciente'].apply(infer_sex)

    n_m = (df['sex'] == 1).sum()
    n_f = (df['sex'] == 0).sum()
    print(f"{name}: {len(df)} pacientes → {n_m} M ({n_m/len(df)*100:.0f}%), {n_f} F ({n_f/len(df)*100:.0f}%)")

    # Salvar CSV e XLSX
    df.to_csv(path, index=False)
    df.to_excel(path.replace('.csv', '.xlsx'), index=False)

print()
print("Coluna 'sex' adicionada (1=M, 0=F) em todos os datasets.")
