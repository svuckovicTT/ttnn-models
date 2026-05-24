p = '/localdev/svuckovic/_workspace/repos/project-alchemy/ttnn-models/Qwen/Qwen3-4B/model/graph_0/main.py'
with open(p, 'r') as f:
    L = f.readlines()
N = L[:25142] + L[29701:]
with open(p, 'w') as f:
    f.writelines(N)
print(len(N))
