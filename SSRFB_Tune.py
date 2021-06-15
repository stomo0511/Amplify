import csv
import pprint
# from amplify import (
#     BinaryPoly,
#     sum_poly,
#     gen_symbols,
#     Solver,
#     decode_solution,
# )
# from amplify.constraint import (
#     equal_to,
# )
# from amplify.client import FixstarsClient


##################################################################################
# クライアント設定
# client = FixstarsClient()
# client.token = "i5G6Ei3DKlGv2n6hsWBSBzWrmffLN4vn"  #20210911まで有効
# client.parameters.timeout = 5000  # タイムアウト5秒
# client.parameters.outputs.duplicate = True  # 同じエネルギー値の解を列挙するオプション
# client.parameters.outputs.num_outputs = 0   # 見つかったすべての解を出力

##################################################################################
# データファイルの読み込み
# (nb, ib, time)
with open('test.csv', 'r') as file: 
    reader = csv.reader(file)
    for row in reader:
        print(row)


# ##################################################################################
# # 定数、変数の宣言


# ##################################################################################
# # QUBO変数の生成: nlab x ngrp
# q = gen_symbols(BinaryPoly, nlab, ngrp)


# ##################################################################################
# # コスト関数：
# cost = sum_poly(
#     ngrp,
#     lambda j: (S[j] - S[(j+1) % ngrp])**2
# ) + sum_poly(
#     ngrp,
#     lambda j: (T[j] - T[(j+1) % ngrp])**2   
# )


# ##################################################################################
# # 制約関数：

# ##################################################################################
# # 行（研究室）に対する制約： one-hot制約（1つの研究室が属するグループは1つだけ）
# row_constraints = [
#     equal_to(sum_poly([q[i][j] for j in range(ngrp)]), 1) for i in range(nlab)
# ]

# ##################################################################################
# # 制約関数
# constraints = sum(row_constraints)

# # モデル
# model = cost + 5*constraints

# # ソルバの生成
# solver = Solver(client)
# # solver.filter_solution = False  # 実行可能解以外をフィルタリングしない

# ##################################################################################
# # ソルバ起動
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

# # print("制約の確認（研究室が一度ずつ現れているか）")
# # for i in range(nlab):
# #     print(f"{labs[i]} : {sum_poly([q_values[i][j] for j in range(ngrp)])}")
