import os
from vs.abstract_agent import AbstAgent
from vs.constants import VS
import numpy as np
from sklearn.cluster import KMeans
import joblib  # Importe joblib

class RescuerAgent(AbstAgent):
    """
    Agente Socorrista.
    - RESCUER_1 atua como "Mestre".
    - O Mestre recebe os mapas unificados, executa o clustering (K-Means),
      salva os arquivos de cluster e atribui vítimas aos outros socorristas.
    - Os socorristas não se movem.
    """

    def __init__(self, env, config_file):
        super().__init__(env, config_file)
        
        self.is_master = (self.NAME == "RESCUER_1")
        self.unified_victims = {}
        self.unified_obstacles = {}
        self.my_assigned_victims = [] 
        self.maps_received = False
        self.clustering_done = False
        self.internal_state = "WAITING_DATA" 

        # Carrega o classificador treinado ---
        self.classifier_pipeline = None
        if self.is_master: # Só o mestre precisa do modelo
            try:
                self.classifier_pipeline = joblib.load('classifier_p1.pkl')
                print(f"--- {self.NAME} (Mestre): Modelo (classifier_p1.pkl) da Tarefa 1 carregado.")
            except FileNotFoundError:
                print(f"!!! {self.NAME} (Mestre) ERRO: Não encontrou o arquivo 'classifier_p1.pkl'.")
            except Exception as e:
                print(f"!!! {self.NAME} (Mestre) ERRO ao carregar 'classifier_p1.pkl': {e}")

    
    def _get_classifier_data(self, victim_signals):
        """
        Utiliza o classificador da Tarefa 1 para prever 'tri' e 'sobr'.
        """
        if not self.classifier_pipeline:
            print(f"{self.NAME}: Modelo da Tarefa 1 não carregado. Retornando erro.")
            return -1, 0.0 # Retorna valores de erro

        try:
            # --- CORREÇÃO CRÍTICA ---
            # O Explorer passa 'victim_signals' com 13 itens mas apenas 10 são usados pelo modelo:
            
            features = victim_signals[1:11] # Isso pega 10 itens (do índice 1 ao 10)
            
            if len(features) != 10:
                print(f"{self.NAME}: ERRO DE FEATURES! Esperava 10, recebeu {len(features)}")
                return -1, 0.0
            
            features_2d = np.array([features])
            
            # Prever a classe (triagem: 0, 1, 2, ou 3)
            tri_prediction = self.classifier_pipeline.predict(features_2d)[0]
            
            # Prever a 'sobr' (probabilidade de sobrevivência)
            # 'predict_proba' retorna as probs para [classe_0, classe_1, classe_2, classe_3]
            probabilities = self.classifier_pipeline.predict_proba(features_2d)[0]
            
            # Assumindo que 'sobr' é a probabilidade de *não* ser classe 3 (Preto/Morto)
            if len(probabilities) == 4:
                sobr_prediction = probabilities[0] + probabilities[1] + probabilities[2]
            else:
                sobr_prediction = probabilities[int(tri_prediction)]

            return int(tri_prediction), float(sobr_prediction)
        
        except Exception as e:
            print(f"{self.NAME}: Erro ao tentar prever com o modelo da Tarefa 1: {e}")
            print(f"    Sinais recebidos (tamanho {len(victim_signals)}): {victim_signals}")
            return -1, 0.0 # Valores de erro

    
    def receber_mapas_unificados(self, victims, obstacles):
        """
        Método chamado pelo EXPLORER_1 (Chefe) para entregar os mapas.
        Apenas o mestre (RESCUER_1) deve receber e processar.
        """
        if not self.is_master:
            return
            
        self.unified_victims = victims
        self.unified_obstacles = obstacles
        self.maps_received = True
        self.internal_state = "CLUSTERING" # Próximo estado do mestre
        print(f"--- {self.NAME} (Mestre): Mapas unificados recebidos. {len(self.unified_victims)} vítimas.")


    def receive_assignment(self, victim_data_list):
        """
        Método chamado pelo RESCUER_1 (Mestre) para atribuir vítimas.
        """
        self.my_assigned_victims = victim_data_list
        self.internal_state = "ASSIGNED"
        print(f"--- {self.NAME}: Recebi minha atribuição de {len(self.my_assigned_victims)} vítimas.")
        self.set_state(VS.ENDED) # Tarefa concluída


    def deliberate(self) -> bool:
        """
        Ciclo de deliberação do Socorrista.
        Os socorristas não se movem nesta tarefa."
        """
        
        if self.is_master:
            if self.internal_state == "CLUSTERING" and self.maps_received and not self.clustering_done:
                self.run_clustering_and_assignment()
                self.clustering_done = True
                self.internal_state = "DONE"
            
            if self.internal_state == "DONE" or self.internal_state == "ASSIGNED":
                return False 
            
            return True

        else: # Socorristas não-mestres
            if self.internal_state == "ASSIGNED" or self.internal_state == "DONE":
                return False 
            
            return True # Continua ativo, esperando a atribuição


    def run_clustering_and_assignment(self):
        """
        (APENAS MESTRE)
        Tarefa 3: Agrupamento de Vítimas.
        1. Prepara os dados das vítimas.
        2. Roda o K-Means (o "algoritmo de clustering" obrigatório).
        3. Salva os clusters em arquivos .txt (formato obrigatório).
        4. Atribui os clusters aos socorristas.
        """
        print(f"--- {self.NAME} (Mestre): Iniciando Tarefa 3: Agrupamento (K-Means) ---")
        
        if not self.unified_victims:
            print(f"--- {self.NAME} (Mestre): Nenhuma vítima para agrupar.")
            return

        victim_data_points = []
        victim_details_map = {}
        victim_ids_in_order = list(self.unified_victims.keys())

        for victim_id in victim_ids_in_order:
            data = self.unified_victims[victim_id]
            x, y, signals = data
            
            # Chama o método que usa o modelo P1 da Tarefa 1
            tri, sobr = self._get_classifier_data(signals) 
            
            victim_data_points.append((x, y)) 
            
            victim_details_map[victim_id] = {
                'id': victim_id,
                'x': x,
                'y': y,
                'tri': tri,
                'sobr': sobr,
                'raw_data': data 
            }

        X = np.array(victim_data_points)
        
        num_rescuers = 3
        n_clusters = min(num_rescuers, len(X)) 
        
        if n_clusters <= 0:
             print(f"--- {self.NAME} (Mestre): K-Means não pode rodar com 0 vítimas.")
             return

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(X)
        cluster_labels = kmeans.labels_ 

        clusters = [[] for _ in range(n_clusters)]
        
        for i, label in enumerate(cluster_labels):
            victim_id = victim_ids_in_order[i]
            clusters[label].append(victim_details_map[victim_id])

        # Encontrar os socorristas
        rescuers_minds = {}
        for phy in self.get_env().agents:
            if "RESCUER" in phy.mind.NAME:
                mind = phy.mind
                if mind.NAME == "RESCUER_1": rescuers_minds['1'] = mind
                if mind.NAME == "RESCUER_2": rescuers_minds['2'] = mind
                if mind.NAME == "RESCUER_3": rescuers_minds['3'] = mind
        
        # Salvar arquivos
        output_dir = "clusters"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for f in os.listdir(output_dir): # Limpar pasta
            if f.endswith(".txt"):
                os.remove(os.path.join(output_dir, f))

        for cluster_index, victim_list in enumerate(clusters):
            
            victim_list_sorted = sorted(victim_list, key=lambda v: v['sobr'], reverse=True)
            
            file_name = os.path.join(output_dir, f"cluster_{cluster_index + 1}.txt")
            
            with open(file_name, 'w', newline='') as f:
                f.write("id,vict_id,x,y,sobr,tri\n")
                
                for item_id, victim in enumerate(victim_list_sorted):
                    f.write(f"{item_id},{victim['id']},{victim['x']},{victim['y']},{victim['sobr']:.4f},{victim['tri']}\n")
            
            print(f"--- {self.NAME} (Mestre): Cluster {cluster_index + 1} salvo em {file_name} ({len(victim_list)} vítimas).")

            rescuer_num_str = str(cluster_index + 1)
            assignment_data = [v['raw_data'] for v in victim_list_sorted]
            
            if rescuer_num_str in rescuers_minds:
                rescuers_minds[rescuer_num_str].receive_assignment(assignment_data)
            else:
                print(f"AVISO: Socorrista {rescuer_num_str} não encontrado para atribuição.")
