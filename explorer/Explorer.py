from vs.abstract_agent import AbstAgent
from vs.constants import VS
import random
import heapq

class ExplorerAgent(AbstAgent):
    """
    Agente explorador que usa ONLINE-DFS para explorar e A* para estimar e retornar à base.
    """

    def __init__(self, env, config_file):
        super().__init__(env, config_file)

        self.current_pos = (self._AbstAgent__phy.x, self._AbstAgent__phy.y)
        self.base_pos = (self._AbstAgent__phy.x, self._AbstAgent__phy.y)
        self.state = "EXPLORING"

        self.map_visited = set()
        self.map_obstacles = {}
        self.dfs_path_stack = []
        self.unvisited_neighbors = {}
        self.found_victims = {}

        self.is_chief = (self.NAME == "EXPLORER_1")
        if self.is_chief:
            print(f"{self.NAME}: Eu sou o Chefe. Coordenarei a transição.")

        base_margin = 20
        self.SAFETY_MARGIN = base_margin + (self.COST_READ * 2 if self.is_chief else 0)

        self.unified_obstacles = {}
        self.unified_victims = {}

        self.USE_ASTAR = True
        self.astar_return_path = []
        
        # --- LÓGICA DO CONE  ---

        grid_width = self.get_env().dic["GRID_WIDTH"] - 1
        grid_height = self.get_env().dic["GRID_HEIGHT"] - 1
        base_x = self.base_pos[0]
        
        if "EXPLORER_1" in self.NAME:
            self.sector_goal = (base_x, 0) # Norte
        elif "EXPLORER_2" in self.NAME:
            self.sector_goal = (grid_width, grid_height) # Sudeste
        else: # EXPLORER_3
            self.sector_goal = (0, grid_height) # Sudoeste


    def deliberate(self) -> bool:
        self.current_pos = (self._AbstAgent__phy.x, self._AbstAgent__phy.y)

        # --- Lógica do Chefe ---
        if self.state == "WAITING_FOR_OTHERS":
            return self.handle_waiting_state()
        if self.state == "SYNCHRONIZING":
            return self.handle_sync_state()
        # --- Fim da Lógica do Chefe ---
        
        if self.state == "DONE":
            return False
        if self.state == "RETURNING_TO_BASE":
            return self.handle_return_state()
        if self.state == "EXPLORING":
            return self.handle_exploring_state()

        return False

    def handle_exploring_state(self):
        cost_to_return = self.estimate_astar_cost(self.current_pos, self.base_pos)
        current_time = self.get_rtime()

        if current_time < (cost_to_return + self.SAFETY_MARGIN):
            print(f"{self.NAME}: Bateria baixa ({current_time:.1f} < {cost_to_return:.1f} [A* Estimado]). Voltando (A*).")
            path = self.astar_path(self.current_pos, self.base_pos)
            if path:
                self.astar_return_path = path
                self.state = "RETURNING_TO_BASE"
                return True
            else:
                print(f"{self.NAME}: A* falhou, voltando via DFS.")
                self.state = "RETURNING_TO_BASE"
                return True

        if self.current_pos not in self.map_visited:
            self.map_visited.add(self.current_pos)

            self.update_unvisited_neighbors(self.current_pos)

            vic_id = self.check_for_victim()
            if vic_id != VS.NO_VICTIM and vic_id not in self.found_victims:
                signals = self.read_vital_signals()
                if signals == VS.TIME_EXCEEDED:
                    self.state = "DONE"
                    return False
                if signals:
                    self.found_victims[vic_id] = (self.current_pos[0], self.current_pos[1], signals)
                    self.first_aid()

        if self.current_pos in self.unvisited_neighbors and self.unvisited_neighbors[self.current_pos]:
            direction = self.unvisited_neighbors[self.current_pos].pop(0)
            dx, dy = self.AC_INCR[direction]
            result = self.walk(dx, dy)
            if result == VS.EXECUTED:
                self.dfs_path_stack.append((-dx, -dy, 1))
            return True

        if not self.dfs_path_stack:
            print(f"{self.NAME}: Exploração completa. Na base.")
            
            if self.is_chief:
                self.state = "WAITING_FOR_OTHERS"
                print(f"{self.NAME} (Chefe): Exploração própria concluída. Aguardando...")
                return True
            else:
                self.state = "RETURNING_TO_BASE"
                return True

        dx, dy, _ = self.dfs_path_stack.pop()
        self.walk(dx, dy)
        return True

    def handle_return_state(self):
        if self.current_pos == self.base_pos:
            bateria_restante = self.get_rtime()
            print(f"{self.NAME}: Retornou à base com sucesso. [Bateria Restante: {bateria_restante:.2f}]")
            
            if self.is_chief:
                self.state = "WAITING_FOR_OTHERS"
                print(f"{self.NAME} (Chefe): Aguardando outros exploradores...")
                return True
            else:
                self.state = "DONE"
                return False

        if self.astar_return_path:
            dx, dy = self.astar_return_path.pop(0)
            result = self.walk(dx, dy)
            if result == VS.EXECUTED:
                return True
            else:
                self.map_obstacles[(self.current_pos[0] + dx, self.current_pos[1] + dy)] = VS.WALL
                self.astar_return_path = self.astar_path(self.current_pos, self.base_pos)
                return True

        self.astar_return_path = self.astar_path(self.current_pos, self.base_pos)
        if not self.astar_return_path and self.dfs_path_stack:
            dx, dy, _ = self.dfs_path_stack.pop()
            self.walk(dx, dy)
        return True


    def handle_waiting_state(self):
        """
        (APENAS CHEFE) Fica na base e verifica se os outros
        exploradores terminaram.
        """
        all_agents = self.get_env().agents
        for phy in all_agents:
            if "EXPLORER" in phy.mind.NAME and phy.mind.NAME != self.NAME:
                if phy._state == VS.ACTIVE:
                    if random.random() < 0.05:
                         print(f"{self.NAME} (Chefe): Esperando {phy.mind.NAME} terminar...")
                    return True

        print(f"-------------------------------------------------")
        print(f"--- {self.NAME} (CHEFE): FASE 1 (EXPLORAÇÃO) CONCLUÍDA ---")
        self.state = "SYNCHRONIZING"
        return True 

    def handle_sync_state(self):
        """
        (APENAS CHEFE) Função de Unificação de Mapas
        *** ATUALIZAÇÃO: Lógica de loop removida para unificação/contagem ***
        """
        print(f"--- {self.NAME} (CHEFE): FASE 2 - UNIFICAÇÃO ---")
        all_agents = self.get_env().agents
        
        # --- Dicionários para armazenar dados e contagens ---
        explorer_minds = {} # Armazena a 'mente' (mind) de cada explorador
        victim_counts = {}  # Armazena a contagem de vítimas (Ve1, Ve2, Ve3)

        # Encontra os 3 exploradores ---
        for phy in all_agents:
            if phy.mind.NAME == "EXPLORER_1" and phy._state != VS.DEAD:
                explorer_minds['1'] = phy.mind
            elif phy.mind.NAME == "EXPLORER_2" and phy._state != VS.DEAD:
                explorer_minds['2'] = phy.mind
            elif phy.mind.NAME == "EXPLORER_3" and phy._state != VS.DEAD:
                explorer_minds['3'] = phy.mind

        self.unified_obstacles = {}
        self.unified_victims = {}
        ve_total_soma = 0
        
        if '1' in explorer_minds:
            mind = explorer_minds['1']
            self.unified_obstacles.update(mind.map_obstacles)
            self.unified_victims.update(mind.found_victims)
            victim_counts['1'] = len(mind.found_victims)
            ve_total_soma += victim_counts['1']
            print(f"Dados do EXPLORER_1 (Chefe) adicionados.")
        else:
            victim_counts['1'] = 0
            
        if '2' in explorer_minds:
            mind = explorer_minds['2']
            self.unified_obstacles.update(mind.map_obstacles)
            self.unified_victims.update(mind.found_victims)
            victim_counts['2'] = len(mind.found_victims)
            ve_total_soma += victim_counts['2']
            print(f"Dados do EXPLORER_2 adicionados.")
        else:
            victim_counts['2'] = 0

        if '3' in explorer_minds:
            mind = explorer_minds['3']
            self.unified_obstacles.update(mind.map_obstacles)
            self.unified_victims.update(mind.found_victims)
            victim_counts['3'] = len(mind.found_victims)
            ve_total_soma += victim_counts['3']
            print(f"Dados do EXPLORER_3 adicionados.")
        else:
            victim_counts['3'] = 0

        print(f"-------------------------------------------------")
        print(f"UNIFICAÇÃO CONCLUÍDA:")
        ve_total_unico = len(self.unified_victims) # Pega o total único (Ve)
        print(f"  > {ve_total_unico} vítimas únicas encontradas.")
        print(f"  > {len(self.unified_obstacles)} obstáculos únicos mapeados.")
        
        # --- 3. Cálculo de Sobreposição (separado, como pedido) ---
        print(f"\n--- {self.NAME} (CHEFE): CÁLCULO DE SOBREPOSIÇÃO ---")
        
        print(f"  > EXPLORER_1 (Ve1): {victim_counts['1']} vítimas")
        print(f"  > EXPLORER_2 (Ve2): {victim_counts['2']} vítimas")
        print(f"  > EXPLORER_3 (Ve3): {victim_counts['3']} vítimas")

        if ve_total_unico > 0:
            sobreposicao = (ve_total_soma / ve_total_unico) - 1
            print(f"  > Soma (Ve1+Ve2+Ve3): {ve_total_soma}")
            print(f"  > Únicas (Ve): {ve_total_unico}")
            print(f"  > SOBREPOSIÇÃO: ({ve_total_soma} / {ve_total_unico}) - 1 = {sobreposicao:.4f}")
        else:
            print("  > Nenhuma vítima encontrada. Sobreposição = 0.0")
        
        print(f"-------------------------------------------------")
        print(f"--- {self.NAME} (CHEFE): FASE 3 - ATIVAÇÃO DOS SOCORRISTAS ---")
        
        for phy in all_agents:
            if "RESCUER" in phy.mind.NAME:
                if phy._state == VS.IDLE:
                    print(f"ATIVANDO: {phy.mind.NAME}")
                    phy._state = VS.ACTIVE
                
                if phy.mind.NAME == "RESCUER_1": 
                    try:
                        phy.mind.receber_mapas_unificados(self.unified_victims, self.unified_obstacles)
                        print(f"MAPAS UNIFICADOS ENTREGUES para {phy.mind.NAME}.")
                    except AttributeError:
                        print(f"AVISO: {phy.mind.NAME} não tem o método 'receber_mapas_unificados'")
        
        print(f"-------------------------------------------------")
        self.state = "DONE"
        return False


    def estimate_astar_cost(self, start, goal):
        path = self.astar_path(start, goal)
        if not path:
            return self.calculate_dfs_stack_cost()
        return len(path)*1.5

    def astar_neighbors(self, node):
        neighbors = []
        x, y = node
        for dx, dy in self.AC_INCR.values():
            nx, ny = x + dx, y + dy
            npos = (nx, ny)
            if npos in self.map_obstacles and self.map_obstacles[npos] != VS.CLEAR:
                continue
            neighbors.append(npos)
        return neighbors

    def astar_heuristic(self, a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))*1.5

    def astar_move_cost(self, a, b):
        if a[0] != b[0] and a[1] != b[1]:
            return 1.5
        return 1.0

    def astar_path(self, start, goal):
        if start == goal:
            return []

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.astar_heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path_positions = [current]
                while current in came_from:
                    current = came_from[current]
                    path_positions.append(current)
                path_positions.reverse()
                moves = []
                for i in range(len(path_positions) - 1):
                    x0, y0 = path_positions[i]
                    x1, y1 = path_positions[i + 1]
                    moves.append((x1 - x0, y1 - y0))
                return moves

            for neighbor in self.astar_neighbors(current):
                tentative_g = gscore[current] + self.astar_move_cost(current, neighbor)
                if neighbor in gscore and tentative_g >= gscore[neighbor]:
                    continue
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                fscore[neighbor] = tentative_g + self.astar_heuristic(neighbor, goal)
                heapq.heappush(open_set, (fscore[neighbor], neighbor))

        return []

    def calculate_dfs_stack_cost(self):
        return sum(cost for _, _, cost in self.dfs_path_stack)

    def update_unvisited_neighbors(self, pos):
        """
        LIMITES DA PIZZA
        """
        if pos not in self.unvisited_neighbors:
            self.unvisited_neighbors[pos] = []

        obstacles = self.check_walls_and_lim()
        
        scored_directions = []
        base_x, base_y = self.base_pos # Pega as coordenadas da base

        for i in range(8): # Itera por todas as 8 direções
            obs_type = obstacles[i]
            dx, dy = self.AC_INCR[i]
            neighbor_pos = (pos[0] + dx, pos[1] + dy)
            nx, ny = neighbor_pos
            
            is_allowed = True
            
            
            if "EXPLORER_1" in self.NAME:
                # Setor Norte (180°)
                if ny > base_y:
                    is_allowed = False
                    
            elif "EXPLORER_2" in self.NAME:
                # Setor Sudeste (90°)
                if ny < base_y or nx < base_x:
                    is_allowed = False
                    
            else: # EXPLORER_3
                # Setor Sudoeste (90°)
                if ny < base_y or nx > base_x:
                    is_allowed = False
            
            
            if obs_type != VS.CLEAR:
                if neighbor_pos not in self.map_obstacles:
                    self.map_obstacles[neighbor_pos] = obs_type
            
            elif not is_allowed:
                if neighbor_pos not in self.map_obstacles:
                    self.map_obstacles[neighbor_pos] = VS.WALL
            
            elif neighbor_pos not in self.map_visited:
                score = self.astar_heuristic(neighbor_pos, self.sector_goal)
                score += random.random() * 0.1
                scored_directions.append((score, i))
        
        scored_directions.sort()
        
        self.unvisited_neighbors[pos] = [direction for score, direction in scored_directions]