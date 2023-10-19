import numpy as np
from amplify import (
    gen_symbols,
    BinaryPoly,
    Solver,
    decode_solution,
)
from amplify.client import FixstarsClient
from amplify.constraint import equal_to

####################################################
# 解きたい数独の問題
initial = np.array(
    [
        [6,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,6,3,2],
        [0,1,0,0,2,0,0,8,0],
        [0,0,0,7,0,2,0,0,0],
        [3,7,0,0,0,6,0,5,0],
        [1,0,5,0,0,0,8,0,0],
        [0,4,0,0,0,0,3,7,0],
        [5,0,0,0,6,0,0,0,9],
        [0,0,1,0,0,0,0,0,8],
    ]
)

####################################################
# 数独を出力する関数
def print_sudoku(sudoku):
    for i in range(9):
        line = ""
        if i==3 or i==6:
            print("---------------------")
        for j in range(9):
            if j==3 or j==6:
                line += "| "
            if sudoku[i][j]==0:
                line += "_ "
            else:
                line += str(sudoku[i][j]) + " "
        print(line)

####################################################
# 決定変数（バイナリ変数）の宣言
q = gen_symbols(BinaryPoly,9,9,9)

####################################################
# 初期値の代入
for i,j in zip(*initial.nonzero()):
    k = initial[i][j] -1
    q[i][j][k] = BinaryPoly(1)


####################################################
# 同じ行には1から9の各値が一つだけ入る（one-hot 制約）
row_constraints = [
    equal_to( sum( q[:][j][k] ), 1 ) for j in range(9) for k in range(9)
]

####################################################
# 同じ列には1から9の各値が一つだけ入る（one-hot 制約）
col_constraints = [
    equal_to( sum( q[i][:][k] ), 1 ) for i in range(9) for k in range(9)
]

####################################################
# 各マスには1から9のどれかが入る（one-hot 制約）
cell_constraints = [
    equal_to( sum( q[i][j][:] ), 1 ) for i in range(9) for j in range(9)
]

####################################################
# 各3x3マスには1から9の各値が一つだけ入る（one-hot 制約）
block_constraints = [
    equal_to(sum([q[i + m // 3][j + m % 3][k] for m in range(9)]), 1)
    for i in range(0, 9, 3)
    for j in range(0, 9, 3)
    for k in range(9)
]

####################################################
# イジングマシン設定
client = FixstarsClient()
client.token = "GsTUUgM3WjSpAzfeqHR4jWJyUWfzGZJG"  #20240111 まで有効
client.parameters.timeout = 1000  # タイムアウト1秒
client.parameters.outputs.duplicate = True  # 同じエネルギー値の解を列挙するオプション
client.parameters.outputs.num_outputs = 0   # 見つかったすべての解を出力

# 初期値の出力
print_sudoku(initial)

solver = Solver(client)
solver.filter_solution = False    # 制約を満たさない解をフィルタリングしない

# 目的関数
constraints = (
    sum(cell_constraints) +
    sum(row_constraints) +
    sum(col_constraints) +
    sum(block_constraints)
)

# 実行結果
results = solver.solve(constraints)

if len(results) > 0:
    print("satisfied")
    r = decode_solution(q,results[0].values)
    final = np.zeros(9*9, dtype=int).reshape(9,9)

    # print(r)
    for i in range(9):
        for j in range(9):
            final[i][j] = np.where(r[i][j] == 1)[0][0] + 1
    print_sudoku(final)
else:
    print("not satisfied")

