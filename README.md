# Resgate_inteligente

Projeto acadêmico desenvolvido na disciplina de Inteligência Artificial (UTFPR). O sistema simula um cenário de desastre onde agentes robóticos autônomos cooperam para explorar o ambiente, mapear obstáculos e resgatar vítimas sob restrições de tempo e energia.

## Instalação e Execução

### Pré-requisitos
* Python 3+
* Bibliotecas: `numpy`, `scikit-learn`, `joblib`

### Passo a Passo

1. Clone o repositório:
   ```bash
   git clone [https://github.com/Joao-Victor-Oliveira/Resgate_inteligente.git](https://github.com/Resgate_inteligente.git)
Instale as dependências necessárias:
   pip install numpy scikit-learn joblib
Execute a simulação:
  python main.py


Sobre o Projeto:

O objetivo do projeto é resolver o problema de Busca e Salvamento (SAR) utilizando uma arquitetura multiagentes. O sistema é dividido em duas fases operacionais:

    Exploração: Agentes exploradores varrem o ambiente desconhecido para localizar vítimas e mapear obstáculos.

    Socorro: Agentes socorristas planejam rotas otimizadas para salvar o maior número de vítimas com base na distancia e gravidade dos sinais vitais.

A solução integra algoritmos de Busca Heurística, Machine Learning e Computação Evolutiva para tomada de decisão autônoma.
Arquitetura e Algoritmos

1. Agentes Exploradores (Mapeamento)

    DFS Online (Depth-First Search): Implementação de busca em profundidade para navegação em ambientes desconhecidos sem mapa prévio.

    Estratégia de Zonas: Divisão geométrica do grid para garantir cobertura eficiente e minimizar redundância entre agentes.

    Smart Backtracking: Utilização de pathfinding para retorno rápido a nós de fronteira, otimizando a bateria durante a exploração.

2. Agentes Socorristas (Planejamento e Execução)

    Machine Learning (Triagem): Classificador MLP (Multi-Layer Perceptron) treinado para prever a probabilidade de sobrevivência das vítimas com base em sinais vitais.

    Clustering (K-Means): Algoritmo não-supervisionado para agrupar vítimas geograficamente e distribuir a carga de trabalho entre os socorristas.

    Algoritmo Genético (Sequenciamento):

        Resolve o problema de roteamento (Team Orienteering Problem).

        Função de aptidão multiobjetivo que priorizar vítimas próxima e com mais chances de sobrevivencia e eficiência energética.

        Utilização de Matriz de Custos pré-calculada para otimização de performance.

    *Navegação Robusta (A):**

        Algoritmo A-Star com heurística adaptativa.

        Implementação de verificação de sensores em tempo real para evitar colisões dinâmicas.


Autores

    João Victor de Oliveira

    Ariel Wilson Carvalho
