import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # <-- MUDANÇA: Importa o DecisionTree
from sklearn.preprocessing import StandardScaler # (Não usado, mas mantido para referência se necessário)
from sklearn.pipeline import Pipeline            # (Não usado)
import joblib
import os

print("--- INICIANDO TREINAMENTO DO MODELO FINAL (CART P1) ---")

# Caminho para os dados de treino
# (Usando o caminho do seu projeto)
CSV_TREINO = 'datasets/vict/408v/data.csv' 

# Definições exatas das suas tarefas
colunas = [
    'idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim',
    'gcs', 'avpu', 'tri', 'sobr'
]
features = colunas[:10] # Usa APENAS as 10 primeiras features
target = 'tri'

# --- MUDANÇA: Parâmetros P1 do CART (DecisionTree) ---
# Baseado no seu script de análise do CART:
# p1 = {'max_depth': 3, 'min_samples_leaf': 20, 'criterion': 'entropy'}
params_cart_p1 = {
    'max_depth': 3,
    'min_samples_leaf': 20,
    'criterion': 'entropy',
    'random_state': 42 # Boa prática para reprodutibilidade
}

# Arquivo de saída (O MESMO NOME)
MODELO_ARQUIVO = 'classifier_p1.pkl'

# --- 1. Carregar Dados ---
print(f"Carregando dataset de treino de '{CSV_TREINO}'...")
try:
    dados_treino = pd.read_csv(CSV_TREINO)
    # Garante que o dataset de 408v tenha as colunas corretas
    dados_treino = dados_treino[features + [target]] 
    X_treino = dados_treino[features]
    y_treino = dados_treino[target]
    print(f"{len(dados_treino)} registros carregados.")
except FileNotFoundError:
    print(f"ERRO: Arquivo '{CSV_TREINO}' não encontrado!")
    exit()
except KeyError:
    print(f"ERRO: O arquivo '{CSV_TREINO}' não contém as colunas de 'features' ou 'target' esperadas.")
    exit()


# --- 2. Definir o Pipeline (Modelo Final) ---
# Modelos de Árvore de Decisão não precisam de StandardScaler.
# Vamos treinar o modelo diretamente, como no seu script de análise.
print("Definindo o modelo: DecisionTreeClassifier (P1)")
modelo_final = DecisionTreeClassifier(**params_cart_p1)


# --- 3. Treinar o Modelo Final ---
print(f"Treinando o modelo CART P1 com {len(X_treino)} amostras...")
# Usamos .values para evitar o warning de "feature names" que vimos antes
modelo_final.fit(X_treino.values, y_treino)

# --- 4. Salvar o Modelo Treinado ---
try:
    joblib.dump(modelo_final, MODELO_ARQUIVO)
    print("\n" + "="*50)
    print(f"SUCESSO: Modelo final (CART P1) treinado e salvo como '{MODELO_ARQUIVO}'")
    print("="*50)
except Exception as e:
    print(f"ERRO ao salvar o modelo: {e}")