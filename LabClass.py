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

##################################################################################
# クライアント設定
client = FixstarsClient()
client.token = "jtSrmOum5m4eTuMEKDbrBekiOqa6nkCg"
client.parameters.timeout = 5000  # タイムアウト5秒
client.parameters.outputs.duplicate = True  # 同じエネルギー値の解を列挙するオプション
client.parameters.outputs.num_outputs = 0   # 見つかったすべての解を出力

##################################################################################
# 定数、変数の宣言
# 研究室名
labs = ["Ando", "Toyoura", "Mao", "Iwanuma", "Go", "Takahashi", "Omata", "Ozawa", "Ohbuchi", "Watanabe", "Nabeshima", "Hattori", "Fukumoto", "Kinoshita", "Suzuki"]
nlab = len(labs)  # 研究室数

# 研究室教員数
nteachers = [1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1]
assert nlab == len(nteachers), "研究室数と教員数配列の長さが違う"

# 研究室学生数
nstudents = [4, 5, 4, 3, 3, 2, 4, 4, 4, 4, 5, 3, 4, 3, 3]
assert nlab == len(nstudents), "研究室数と学生数配列の長さが違う"

# グループ名
grps = ["CS1", "CS2", "CS3", "CS4"]
ngrp = len(grps)   # グループ数

##################################################################################
# QUBO変数の生成: nlab x ngrp
q = gen_symbols(BinaryPoly, nlab, ngrp)

# 幹事研究室の指定
# CS1: Ando, CS2: Kinoshita, CS3: Watanabe, CS4: Toyoura
q[labs.index("Ando")][grps.index("CS1")] = BinaryPoly(1)
q[labs.index("Kinoshita")][grps.index("CS2")] = BinaryPoly(1)
q[labs.index("Watanabe")][grps.index("CS3")] = BinaryPoly(1)
q[labs.index("Toyoura")][grps.index("CS4")] = BinaryPoly(1)

##################################################################################
# 研究室の教員数
T = [
    sum_poly( [q[i][j] * nteachers[i] for i in range(nlab)] ) for j in range(ngrp)
]

# 研究室の学生数
S = [
    sum_poly( [q[i][j] * nstudents[i] for i in range(nlab)] ) for j in range(ngrp)
]

# コスト関数：グループの学生数、教員数が等しいとき 0
cost = sum_poly(
    ngrp,
    lambda j: (S[j] - S[(j+1) % ngrp])**2
) + sum_poly(
    ngrp,
    lambda j: (T[j] - T[(j+1) % ngrp])**2   
)

##################################################################################
# 前年度と同じグループにならないようにする（ハード制約）
q2020 = gen_symbols(BinaryPoly, nlab, ngrp)

# ０で初期化
for i in range(nlab):
    for j in range(ngrp):
        q2020[i][j] = BinaryPoly(0)

# 2020年度のグループ分け
# CS1: Takahashi, Fukumoto, Toyoura
q2020[labs.index("Takahashi")][grps.index("CS1")] = BinaryPoly(1)
q2020[labs.index("Fukumoto")][grps.index("CS1")] = BinaryPoly(1)
q2020[labs.index("Toyoura")][grps.index("CS1")] = BinaryPoly(1)

# CS2: Go, Ozawa, Mao, Ohbushi
q2020[labs.index("Go")][grps.index("CS2")] = BinaryPoly(1)
q2020[labs.index("Ozawa")][grps.index("CS2")] = BinaryPoly(1)
q2020[labs.index("Mao")][grps.index("CS2")] = BinaryPoly(1)
q2020[labs.index("Ohbuchi")][grps.index("CS2")] = BinaryPoly(1)

# CS3: Omata, Ando, Hattori, Watanabe
q2020[labs.index("Omata")][grps.index("CS3")] = BinaryPoly(1)
q2020[labs.index("Ando")][grps.index("CS3")] = BinaryPoly(1)
q2020[labs.index("Hattori")][grps.index("CS3")] = BinaryPoly(1)
q2020[labs.index("Watanabe")][grps.index("CS3")] = BinaryPoly(1)

# CS4: Suzuki, Iwanuma, Kinoshita, Nabeshima
q2020[labs.index("Suzuki")][grps.index("CS3")] = BinaryPoly(1)
q2020[labs.index("Iwanuma")][grps.index("CS3")] = BinaryPoly(1)
q2020[labs.index("Kinoshita")][grps.index("CS3")] = BinaryPoly(1)
q2020[labs.index("Nabeshima")][grps.index("CS3")] = BinaryPoly(1)

##################################################################################
# 行（研究室）に対する制約： one-hot制約（1つの研究室が属するグループは1つだけ）
row_constraints = [
    equal_to(sum_poly([q[i][j] for j in range(ngrp)]), 1) for i in range(nlab)
]

##################################################################################
# 制約
constraints = sum(row_constraints)

# モデル
model = cost + 10*constraints

# ソルバの生成
solver = Solver(client)

##################################################################################
# ソルバ起動
# result = solver.solve(model)

# # 解が見つからないときのエラー出力
# if len(result) == 0:
#     raise RuntimeError("Any one of constraints is not satisfied.")

# energy = result[0].energy
# values = result[0].values
# q_values = decode_solution(q, values)

# ##################################################################################
# # 結果の表示   
# print(f"エネルギー: {energy}")

# for j in range(ngrp):
#     print(f"グループ {grps[j]} の教員数: {sum_poly([q_values[i][j] * nteachers[i] for i in range(nlab)])}, 学生数: {sum_poly([q_values[i][j] * nstudents[i] for i in range(nlab)])}")
# print()
# print("各グループの研究室の表示")
# for j in range(ngrp):
#     print(f"グループ {grps[j]} の教員: ", end="")
#     for i in range(nlab):
#         if (q_values[i][j] == 1):
#             print(labs[i], ", ", end="")
#     print()
# print()
# print("制約の確認（研究室が一度ずつ現れているか）")
# for i in range(nlab):
#     print(f"{labs[i]} : {sum_poly([q_values[i][j] for j in range(ngrp)])}")
