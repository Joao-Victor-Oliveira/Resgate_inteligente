from vs.environment import Env
from vs.constants import VS  # <-- Importante para o VS.IDLE

from explorer.Explorer import ExplorerAgent
from rescuer.Rescuer import RescuerAgent  # <-- 1. IMPORTE A NOVA CLASSE

print("--- Programa Iniciado ---")
vict_path = "datasets/vict/408v"
env_path = "datasets/env/94x94_408v"
env = Env(
    vict_folder=vict_path,
    env_folder=env_path
)

# --- Instancie seus ExplorerAgents ---
print("Criando exploradores...")
explorer_1 = ExplorerAgent(env, "explorer/explorer_1.txt")
explorer_2 = ExplorerAgent(env, "explorer/explorer_2.txt")
explorer_3 = ExplorerAgent(env, "explorer/explorer_3.txt")

# --- Instancie seus RescuerAgents (Eles começam IDLE) ---
print("Criando socorristas (inativos)...")
rescuer_1 = RescuerAgent(env, "rescuer/rescuer_1.txt")
rescuer_2 = RescuerAgent(env, "rescuer/rescuer_2.txt")
rescuer_3 = RescuerAgent(env, "rescuer/rescuer_3.txt")

# --- Adicione os agentes ao ambiente ---
env.add_agent(explorer_1)
env.add_agent(explorer_2)
env.add_agent(explorer_3)

# Define o estado inicial dos exploradores como ACTIVE
explorer_1.set_state(VS.ACTIVE)
explorer_2.set_state(VS.ACTIVE)
explorer_3.set_state(VS.ACTIVE)

# <-- 3. DESCOMENTE ESTAS LINHAS
env.add_agent(rescuer_1)
env.add_agent(rescuer_2)
env.add_agent(rescuer_3)

# --- Defina o estado inicial dos socorristas como IDLE ---
# (O 'phy' só é criado *depois* do add_agent, por isso fazemos aqui)
# <-- 4. DESCOMENTE ESTAS LINHAS
rescuer_1.set_state(VS.IDLE)
rescuer_2.set_state(VS.IDLE)
rescuer_3.set_state(VS.IDLE)

# --- Inicie a simulação ---
print("Iniciando simulação...")
env.run()

print("--- Programa Finalizado ---")