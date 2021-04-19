import numpy as np
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

# クライアント設定
client = FixstarsClient()
client.token = "jtSrmOum5m4eTuMEKDbrBekiOqa6nkCg"
client.parameters.timeout = 5000  # タイムアウト5秒
# client.parameters.outputs.duplicate = True  # 同じエネルギー値の解を列挙するオプション
# client.parameters.outputs.num_outputs = 0   # 見つかったすべての解を出力

# ソルバの生成
solver = Solver(client)

##################################################################################
# 定数、変数の宣言
# 研究室名
labs = ["Ando", "Toyoura", "Mao", "Iwanuma", "Go", "Takahashi", "Omata", "Ozawa", "Ohbuchi", "Watanabe", "Nabeshima", "Hattori", "Fukumoto", "Kinoshita", "Suzuki"]
nlab = len(labs)

# 研究室教員数
teachers = [1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1]
assert nlab == len(teachers), '研究室数と教員数配列の長さが違う'

# 研究室学生数
students = [4, 5, 4, 3, 3, 2, 4, 4, 4, 4, 5, 3, 4, 3, 3]
assert nlab == len(students), '研究室数と学生数配列の長さが違う'

# グループ名
grps = ["CS1", "CS2", "CS3", "CS4"]
ngrp = len(grps)   # グループ数

##################################################################################
# QUBO変数の生成: nlab x ngrp
q = gen_symbols(BinaryPoly, nlab, ngrp)

# 研究室の教員数
T = [
    sum_poly( [q[i][j] * teachers[i] for i in range(nlab)] ) for j in range(ngrp)
]

# 研究室の学生数
S = [
    sum_poly( [q[i][j] * students[i] for i in range(nlab)] ) for j in range(ngrp)
]

##################################################################################
# コスト関数：グループの学生数が等しいか？
cost = sum_poly(
    ngrp,
    lambda j: (S[j] - S[(j+1) % ngrp])**2
)

##################################################################################
# 行（研究室）に対する制約： one-hot制約（1つの研究室が属するグループは1つだけ）
row_constraints = [
    equal_to(sum_poly([q[i][j] for j in range(ngrp)]), 1) for i in range(nlab)
]

# 制約
constraints = sum(row_constraints)

# モデル
# model = cost + constraints
model = cost

# ソルバ起動
result = solver.solve(model)

if len(result) == 0:
    raise RuntimeError("Any one of constraints is not satisfied.")

energy = result[0].energy
values = result[0].values
q_values = decode_solution(q, values)
    
print(f"エネルギー: {energy}")

for j in range(ngrp):
    print(f"グループ {grps[j]} の教員数: {sum_poly([q_values[i][j] * teachers[i] for i in range(nlab)])}, 学生数: {sum_poly([q_values[i][j] * students[i] for i in range(nlab)])}")
print()
for j in range(ngrp):
    print(f"グループ {grps[j]} の教員: ", end="")
    for i in range(nlab):
        if (q_values[i][j] == 1):
            print(labs[i], ", ", end="")
    print()
print()
for i in range(nlab):
    print(f"{labs[i]} : {sum_poly([q_values[i][j] for j in range(ngrp)])}")
