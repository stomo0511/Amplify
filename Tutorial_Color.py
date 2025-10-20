from amplify import VariableGenerator
import japanmap as jm
    
colors = ["red", "green", "blue", "yellow"]
num_colors = len(colors)
num_region = len(jm.pref_names)
    
gen = VariableGenerator()
q = gen.array("Binary", shape=(num_region, num_colors))

# 制約条件
from amplify import sum as amplify_sum, one_hot, equal_to
    
# 各領域に対する制約
reg_constraints = one_hot(q[1:], axis=1)  # ダミー都道府県は定式化には含めない
    
# 隣接する領域間の制約
adj_constraints = amplify_sum(
    equal_to(q[i, :] * q[j, :], 0, axis=())
    for i in range(1, num_region)
    for j in jm.adjacent(
        i
    )  # j: 都道府県コード i の都道府県と隣接している都道府県コード
    if i < j  # type: ignore
)
    
model = reg_constraints + adj_constraints

# ソルバー実行
from amplify import FixstarsClient, solve
from datetime import timedelta
    
client = FixstarsClient()
client.parameters.timeout = timedelta(milliseconds=5000)  # タイムアウト 5000 ms
client.token = "AE/gd0y09RhHp0H0EcefLvtxYhbwl7Nk0US"  #20260113で有効
    
# 求解と結果の取得
result = solve(model, client)
if len(result) == 0:
    raise RuntimeError("Some constraints are unsatisfied.")

q_values = q.evaluate(result.best.values)

import numpy as np

color_indices = (q_values[1:] @ np.arange(num_colors)).astype(
    int
)  # q_values の最初の行はダミー都道府県のものなので捨てる
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