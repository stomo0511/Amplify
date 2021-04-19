from amplify import BinaryPoly, Solver, gen_symbols, decode_solution
from amplify.client import FixstarsClient

# クライアントの設定
client = FixstarsClient()               # Fixstars Optigan
client.parameters.timeout = 1000        # タイムアウト1秒   
client.token = "jtSrmOum5m4eTuMEKDbrBekiOqa6nkCg"
client.parameters.outputs.duplicate = True  # 同じエネルギー値の解を列挙するオプション（解が複数個あるため）

# 数の集合Aに対応する数のリスト
A = [2, 10, 3, 8, 5, 7, 9, 5, 3, 2]

# len(A): 変数の数
n = len(A)

# Binary 変数を生成
q = gen_symbols(BinaryPoly, n)

# 目的関数の構築
f = BinaryPoly()

for i in range(n):
    f += (2 * q[i] - 1) * A[i]

f = f ** 2

# ソルバーの構築
solver = Solver(client)                 # ソルバーに使用するクライアントを設定

# 問題を入力してマシンを実行
result = solver.solve(f)                # 問題を入力してマシンを実行

# 解が得られなかった場合、len(result) == 0
if len(result) == 0:
    raise RuntimeError("No solution was found")
    
partitions = set()

for sol in result:
    solution = decode_solution(q, sol.values)
    
    A0 = tuple(sorted([A[idx] for idx, val in enumerate(solution) if val != 1]))
    A1 = tuple(sorted([A[idx] for idx, val in enumerate(solution) if val == 1]))

    # 同じ分割がすでにリストに含まれていない場合
    if (A1, A0) not in partitions:
        partitions.add((A0, A1))

for p in partitions:
    print(f"sum A0 = {sum(p[0])}, sum A1 = {sum(p[1])}, partition: {p}")