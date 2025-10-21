from amplify import VariableGenerator
import japanmap as jm
    
colors = ["red", "green", "blue", "yellow"]
num_region = len(jm.pref_names) # = 48
num_colors = len(colors) # = 4
    
gen = VariableGenerator()
q = gen.array("Binary", shape=(num_region, num_colors))
# q[:, num_colors - 1] = 1  # Domain-wall 用に最後の色を固定

from amplify import sum as amplify_sum, one_hot, equal_to, domain_wall, greater_equal

# right_constraints = amplify_sum(
#     equal_to(q[i, num_colors - 1], 1)
#     for i in range(1, num_region)
# )
###########################################
# 各領域に対する制約: domain-wall
reg_constraints1 = amplify_sum(
    equal_to(q[i, c] - q[i, c]*q[i, c+1], 0)  for c in range(num_colors - 1) for i in range(1, num_region)
)

###########################################
# wall は１つだけの制約
#   q[i,-1] = 0 を考慮している -> +q[i,0]
# reg_constraints2 = amplify_sum( # この制約は q[i, num_colors -1] = 1 と同値
#     equal_to( q[i, 0] + amplify_sum(q[i, c+1] - q[i, c] for c in range(num_colors - 1)), 1)
#     for i in range(1, num_region)
# )
reg_constraints2 = amplify_sum(
    equal_to( q[i, num_colors-1], 1)
    for i in range(1, num_region)
)

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
# アンカーの設定
def s(i, c):
    return q[i, 0] if c == 0 else (q[i, c] - q[i, c-1])

r = 2  # 例: 青森（jm は 1..47 が実県、0 はダミー）
anchor_hard = equal_to(s(r, 0), 1)   # q[r,0] == 1

###########################################
# model = reg_constraints1 + reg_constraints2 + adj_constraints
model = 60.0*reg_constraints1 + 20.0*reg_constraints2 + 2.0*adj_constraints + anchor_hard

# 隣接リストの表示
# for i in range(1, num_region):
#     print(f"Adjacent of region {i}:", end="")
#     for j in jm.adjacent(i):  # j: コード i の都道府県と隣接している都道府県コード
#         if i < j and j >= 1:  # type: ignore
#             print(f" {j}", end="")
#     print()


# ソルバー実行
from amplify import solve, FixstarsClient
# from amplify import solve, FujitsuDA4Client
# from amplify import solve, ToshibaSQBM2Client
from datetime import timedelta


client = FixstarsClient()
client.token = "AE/gd0y09RhHp0H0EcefLvtxYhbwl7Nk0US"  #20260113で有効
# client.token = "DA/ubdUmA6AskGYxBxdBkHgbMU2HJd4p6L5"  #20260129で有効
# client = ToshibaSQBM2Client()
# client.token = "SQBM+2/Xzu4OCAfFasnXkPVKmv7iPLuEtEWBsfd"  #20260129で有効

client.parameters.timeout = timedelta(milliseconds=2000)  # タイムアウト
    
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

import numpy as np

# 単調性チェック（違反があれば、その場で表示）
viol_monotone = []
for i in range(1, num_region):
    if np.any(q_val[i, :-1] > q_val[i, 1:]):   # b_c > b_{c+1} なら違反
        viol_monotone.append(i)
if viol_monotone:
    print("[WARN] Monotonicity violated at regions:", viol_monotone)

# 右端=1チェック
viol_right = [i for i in range(1, num_region) if q_val[i, num_colors-1] != 1]
if viol_right:
    print("[WARN] Right-end != 1 at regions:", viol_right)

# Δ を作る（0/1丸め後）
delta = q_val[:, 1:] - q_val[:, :-1]  # shape: (num_region, K-1)

# 隣接異色制約の検証（モデルと同じ式）
for i in range(1, num_region):
    for j in jm.adjacent(i):
        if i < j:
            val = int(q_val[i,0] * q_val[j,0] + np.dot(delta[i], delta[j]))
            if val != 0:
                print(f"[VIOL] Adj constraint i={i}, j={j} : {val}") 
# total = 0
# for i in range(1, num_region):
#         for j in jm.adjacent(i):
#             if i < j:  # type: ignore
#                 val = 0
#                 for c in range(num_colors - 1):
#                     val += q_val[i,0] * q_val[j,0] + (q_val[i, c+1] - q_val[i, c]) * (q_val[j, c+1] - q_val[j, c])
#                 print(f"Adjacency constraint between region {i} and {j}: {val}")
#                 total += val
# print(f"Total adjacency constraints value: {total}")

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