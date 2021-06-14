from amplify import (
    BinaryPoly,
    BinaryQuadraticModel,
    sum_poly,
    gen_symbols,
    Solver,
    decode_solution,
)
from amplify.constraint import (
    equal_to, 
    penalty,
)
from amplify.client import FixstarsClient
import numpy as np
import japanmap as jm
import matplotlib.pyplot as plt

client = FixstarsClient()
client.token = "i5G6Ei3DKlGv2n6hsWBSBzWrmffLN4vn"  #20210011まで有効
client.parameters.timeout = 5000  # タイムアウト5秒

solver = Solver(client)

# 四色の定義
colors = ["red", "green", "blue", "yellow"]
num_colors = len(colors)
num_region = len(jm.pref_names) - 1  # 都道府県数を取得

# 都道府県数 x 色数 の変数を作成
q = gen_symbols(BinaryPoly, num_region, num_colors)

# 各領域に対する制約
# 一つの領域に一色のみ（one-hot）
# sum_{c=0}^{C-1} q_{i,c} = 1 for all i
reg_constraints = [
    equal_to(sum_poly([q[i][c] for c in range(num_colors)]), 1)
    for i in range(num_region)
]

# 隣接する領域間の制約
adj_constraints = [
    # 都道府県コードと配列インデックスは1ずれてるので注意
    penalty(q[i][c] * q[j - 1][c])
    for i in range(num_region)
    for j in jm.adjacent(i + 1)  # j: 隣接している都道府県コード
    if i + 1 < j
    for c in range(num_colors)
]

constraints = sum(reg_constraints) + sum(adj_constraints)

model = BinaryQuadraticModel(constraints)
result = solver.solve(model)
if len(result) == 0:
    raise RuntimeError("Any one of constraints is not satisfied.")

values = result[0].values
q_values = decode_solution(q, values, 1)

color_indices = np.where(np.array(q_values) == 1)[1]
color_map = {jm.pref_names[i + 1]: colors[color_indices[i]] for i in range(len(color_indices))}

plt.rcParams["figure.figsize"] = 6, 6
plt.imshow(jm.picture(color_map))
plt.show()