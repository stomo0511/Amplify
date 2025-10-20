from amplify import VariableGenerator
import japanmap as jm
    
colors = ["red", "green", "blue", "yellow"]
num_region = len(jm.pref_names) # = 48
num_colors = len(colors) # = 4
    
gen = VariableGenerator()
q = gen.array("Binary", shape=(num_region, num_colors)) # q[48][4]
q[:, num_colors - 1] = 1  # Domain-wall 用に最後の色を固定

# 制約条件
from amplify import sum as amplify_sum, one_hot, equal_to, domain_wall
    
# domain-wall 制約
reg_constraints = 0
for i in range(1, num_region):
    for c in range(num_colors-1):
        reg_constraints += equal_to(q[i, c] - q[i, c]*q[i, c+1], 0)

# 隣接する領域間の制約
adj_constraints = 0
for i in range(1, num_region):
    for j in jm.adjacent(i):  # j: 都道府県コード i の都道府県と隣接している都道府県コード
        if i < j:  # type: ignore
            for c in range(num_colors-1):
                adj_constraints += equal_to((q[i, c+1] - q[i, c]) * (q[j, c+1] - q[j, c]), 0)

    
model = reg_constraints + adj_constraints

# ソルバー実行
from amplify import FixstarsClient, solve
from datetime import timedelta
    
client = FixstarsClient()
client.parameters.timeout = timedelta(milliseconds=1000)  # タイムアウト
client.token = "AE/gd0y09RhHp0H0EcefLvtxYhbwl7Nk0US"  #20260113で有効
    
# 求解と結果の取得
result = solve(model, client)
if len(result) == 0:
    raise RuntimeError("Some constraints are unsatisfied.")

q_values = q.evaluate(result.best.values)

import numpy as np

color_indices = np.ndarray(num_region-1, dtype=int)
for i in range(1, num_region):
    for c in range(num_colors-1):
        if (q_values[i][c+1] - q_values[i][c]) == 1:
            color_indices[i-1] = c+1

color_map = {
    jm.pref_names[region_idx]: colors[color_idx]
    for region_idx, color_idx in enumerate(
        color_indices, start=1
    )  # region_idx は 1 スタートなことに注意
}

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = 6, 6
plt.imshow(jm.picture(color_map))
plt.show()