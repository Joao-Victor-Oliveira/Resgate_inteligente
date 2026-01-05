import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

print("--- INICIANDO TREINAMENTO DO MODELO FINAL (P1) ---")

# Caminho para os dados de treino
# (Ajuste se o script não estiver na pasta raiz do projeto)
CSV_TREINO = 'datasets/vict/408v/data.csv'

# Definições exatas da sua Tarefa 1
colunas = [
    'idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim',
    'gcs', 'avpu', 'tri', 'sobr'
]
features = colunas[:10] # Usa APENAS as 10 primeiras features
target = 'tri'

# Parâmetros P1
params_p1 = {'hidden_layer_sizes': (25,), 'activation': 'relu', 'max_iter': 100, 'random_state': 42}

# Arquivo de saída
MODELO_ARQUIVO = 'classifier_p1.pkl'

# --- 1. Carregar Dados ---
print(f"Carregando dataset de treino de '{CSV_TREINO}'...")
try:
    dados_treino = pd.read_csv(CSV_TREINO)
    X_treino = dados_treino[features]
    y_treino = dados_treino[target]
    print(f"{len(dados_treino)} registros carregados.")
except FileNotFoundError:
    print(f"ERRO: Arquivo '{CSV_TREINO}' não encontrado!")
    exit()

# --- 2. Definir o Pipeline (Modelo Final) ---
modelo_final = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(**params_p1))
])

# --- 3. Treinar o Modelo Final ---
print("Treinando o modelo P1 com todos os dados...")
modelo_final.fit(X_treino.values, y_treino)

# --- 4. Salvar o Modelo Treinado ---
try:
    joblib.dump(modelo_final, MODELO_ARQUIVO)
    print("\n" + "="*50)
    print(f"SUCESSO: Modelo final treinado e salvo como '{MODELO_ARQUIVO}'")
    print("="*50)
except Exception as e:
    print(f"ERRO ao salvar o modelo: {e}")