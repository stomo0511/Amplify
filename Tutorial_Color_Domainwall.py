from amplify import VariableGenerator
import japanmap as jm
    
colors = ["red", "green", "blue", "yellow"]
num_region = len(jm.pref_names) # = 48
num_colors = len(colors) # = 4
    
gen = VariableGenerator()
q = gen.array("Binary", shape=(num_region, num_colors))
# q[:, num_colors - 1] = 1  # Domain-wall 用に最後の色を固定

from amplify import sum as amplify_sum, one_hot, equal_to, domain_wall, greater_equal

right_constraints = amplify_sum(
    equal_to(q[i, num_colors - 1], 1)
    for i in range(1, num_region)
)
###########################################
# 各領域に対する制約: domain-wall
reg_constraints1 = amplify_sum(
    equal_to(q[i, c] - q[i, c]*q[i, c+1], 0)  for c in range(num_colors - 1) for i in range(1, num_region) 
)

###########################################
# wall は１つだけの制約
#   q[i,-1] = 0 を考慮している -> +q[i,0]
# reg_constraints2 = amplify_sum(
#     equal_to( q[i, 0] + amplify_sum(q[i, c+1] - q[i, c] for c in range(num_colors - 1)), 1)
#     for i in range(1, num_region)
# )
reg_constraints2 = amplify_sum(
    equal_to( q[i, num_colors-1], 1)
    for i in range(1, num_region)
)
# reg_constraints2 = amplify_sum(
#     greater_equal(q[i, c+1] - q[i,c], 0) for c in range(num_colors -1) for i in range(1, num_region)
# )


###########################################
# 隣接する領域間の制約
adj_constraints = amplify_sum(
    equal_to(
        q[i,0] * q[j,0] + 
        amplify_sum(
            (q[i, c+1] - q[i, c]) * (q[j, c+1] - q[j, c])
            for c in range(num_colors - 1)
        )
        , 0)
    for i in range(1, num_region)
    for j in jm.adjacent(i) # j: 都道府県コード i の都道府県と隣接している都道府県コード
    if i < j  # type: ignore
)

###########################################
model = reg_constraints1 + reg_constraints2 + adj_constraints

# 隣接リストの表示
for i in range(1, num_region):
    print(f"Adjacent of region {i}:", end="")
    for j in jm.adjacent(i):  # j: コード i の都道府県と隣接している都道府県コード
        if i < j:  # type: ignore
            print(f" {j}", end="")
    print()


# ソルバー実行
from amplify import FixstarsClient, solve
from datetime import timedelta
    
client = FixstarsClient()
client.parameters.timeout = timedelta(milliseconds=1000)  # タイムアウト
client.token = "AE/gd0y09RhHp0H0EcefLvtxYhbwl7Nk0US"  #20260113で有効
    
# 求解と結果の取得
result = solve(model, client, filter_solution=False)
if len(result) == 0:
    raise RuntimeError("Some constraints are unsatisfied.")

# 最適化目的関数の値の表示
print(f"objective = {result.best.objective}")

# 実行可能解の判定
if result.best.feasible:
    print("Solution is feasible.")
else:
    print("Solution is NOT feasible.")

q_val = q.evaluate(result.best.values)
print(q_val)

for i in range(1, num_region):
        for j in jm.adjacent(i):
            if i < j:  # type: ignore
                val = 0
                for c in range(num_colors - 1):
                    val += (q_val[i, c+1] - q_val[i, c]) * (q_val[j, c+1] - q_val[j, c])
                print(f"Adjacency constraint between region {i} and {j}: {val}")


import numpy as np

color_indices = np.ndarray(num_region-1, dtype=int)
for i in range(1, num_region):
    for c in range(num_colors-1):
        if (q_val[i][c+1] - q_val[i][c]) == 1:
            color_indices[i-1] = c+1
    # print(f"Region {i}: {q_val[i]} Color {color_indices[i-1]}")

color_map = {
    jm.pref_names[region_idx]: colors[color_idx]
    for region_idx, color_idx in enumerate(
        color_indices, start=1
    )  # region_idx は 1 スタートなことに注意
}
print(color_map)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = 6, 6
plt.imshow(jm.picture(color_map))
plt.show()