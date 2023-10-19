from amplify import (
    BinaryPoly,
    sum_poly,
    gen_symbols,
    Solver,
    decode_solution,
)
from amplify.constraint import (
    equal_to,
)
from amplify.client import FixstarsClient


##################################################################################
# クライアント設定
client = FixstarsClient()
client.token = "GsTUUgM3WjSpAzfeqHR4jWJyUWfzGZJG"  #20240111 まで有効
client.parameters.timeout = 10000  # タイムアウト10秒
client.parameters.outputs.duplicate = True  # 同じエネルギー値の解を列挙するオプション
client.parameters.outputs.num_outputs = 0   # 見つかったすべての解を出力

##################################################################################
# 定数、変数の宣言
#
# 研究室リスト
# 研究室名: [教員数, 研修1学生数, 研修2学生数]
# わざわざ辞書型としたのは、研究室名、教員数、学生数の組を入力し間違わないため
labsdict = {
    "Ando": [1, 4, 4],
    "Iwanuma": [2, 3, 3],
    "Ohbuchi": [2, 6, 6],
    "Ozawa": [2, 5, 5],
    "Omata": [1, 4, 5],  #
    "Mao": [2, 4, 4],
    "Kinoshita": [1, 4, 5],  #
    "Go": [1, 4, 4],
    "Suzuki": [1, 3, 4],  #
    "Takahashi": [1, 2, 2],
    "Toyoura": [1, 4, 4],
    "Nabeshima": [1, 3, 4],  #
    "Hattori": [1, 3, 3],
    "Fukumoto": [2, 5, 5],
    "Watanabe": [1, 2, 2],
}
nlabs = len(labsdict)  # 研究室数

labs = []
nteachers = []
nstudents1 = []
nstudents2 = []
for name, val in labsdict.items():
    labs.append(name)
    nteachers.append(val[0])
    nstudents1.append (val[1])
    nstudents2.append (val[2])

# グループ名
grps = ["CS1", "CS2", "CS3", "CS4"]
ngrps = len(grps)   # グループ数

##################################################################################
# QUBO変数の生成: nlab x ngrp
q = gen_symbols(BinaryPoly, nlabs, ngrps)

q[1][0] = 1
q[3][1] = 1
q[4][2] = 1
q[5][3] = 1

# グループ内の教員数
T = [sum_poly( [q[i][j] * nteachers[i] for i in range(nlabs)] ) for j in range(ngrps)]

# グループ内の学生数
S1 = [sum_poly( [q[i][j] * nstudents1[i] for i in range(nlabs)] ) for j in range(ngrps)]
S2 = [sum_poly( [q[i][j] * nstudents2[i] for i in range(nlabs)] ) for j in range(ngrps)]

##################################################################################
# コスト関数：各グループの学生数、教員数が等しいか？
cost = sum_poly(
    ngrps,
    lambda j: (S1[j] - S1[(j+1) % ngrps])**2
) + sum_poly(
    ngrps,
    lambda j: (S2[j] - S2[(j+1) % ngrps])**2
) + sum_poly(
    ngrps,
    lambda j: (T[j] - T[(j+1) % ngrps])**2   
)

##################################################################################
# 行（研究室）に対する制約： one-hot制約（1つの研究室が属するグループは1つだけ）
row_constraints = [
    equal_to(sum_poly([q[i][j] for j in range(ngrps)]), 1) for i in range(nlabs)
]

##################################################################################
# 制約
constraints = sum(row_constraints)

# モデル
model = cost + 5*constraints

# ソルバの生成
solver = Solver(client)
# solver.filter_solution = False  # 実行可能解以外をフィルタリングしない

##################################################################################
# ソルバ起動
result = solver.solve(model)

# 解が見つからないときのエラー出力
if len(result) == 0:
    raise RuntimeError("Any one of constraints is not satisfied.")

energy = result[0].energy
values = result[0].values
q_values = decode_solution(q, values)

##################################################################################
# 結果の表示   
print(f"エネルギー: {energy}")

for j in range(ngrps):
    print(f"グループ {grps[j]} の教員数: {sum([q_values[i][j] * nteachers[i] for i in range(nlabs)])}, 学生数1: {sum([q_values[i][j] * nstudents1[i] for i in range(nlabs)])}, 学生数2: {sum([q_values[i][j] * nstudents2[i] for i in range(nlabs)])}")
print()

print("各グループの研究室の表示")
for j in range(ngrps):
    print(f"グループ {grps[j]} の教員: ", end="")
    for i in range(nlabs):
        if (q_values[i][j] == 1):
            print(labs[i], ", ", end="")
    print()
print()

# print("制約の確認（研究室が一度ずつ現れているか）")
# for i in range(nlab):
#     print(f"{labs[i]} : {sum_poly([q_values[i][j] for j in range(ngrp)])}")
