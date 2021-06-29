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

import pandas as pd
import numpy as np

####################################################
# データファイルの読み込み
# (nb, ib, time)
data = pd.read_csv('M1_NoFlush.csv', skipinitialspace=True)

# pandas.Series -> list へ変換
nb = data.nb.values       # タイルサイズ
ib = data.ib.values       # 内部ブロック幅
time = data.time.values   # 実行時間

# 正規化速度 gflops
gflops = nb**3 / time / 10**9
data.insert(len(data.columns), 'gflops', gflops)

####################################################
# 定数設定
ndat = len(nb)     # データ数
ncan = 8           # パラメータペア数

####################################################
# クライアント設定
client = FixstarsClient()
client.token = "i5G6Ei3DKlGv2n6hsWBSBzWrmffLN4vn"  #20210911まで有効
client.parameters.timeout = 10000  # タイムアウト10秒
client.parameters.outputs.duplicate = True  # 同じエネルギー値の解を列挙するオプション
client.parameters.outputs.num_outputs = 0   # 見つかったすべての解を出力

####################################################
# バイナリ変数の生成
q = gen_symbols(BinaryPoly, ndat)

####################################################
# コスト関数：gflops 値の総和
cost = sum_poly( [-q[i] * gflops[i] for i in range(ndat)] )

####################################################
# 制約関数： "1"の変数の数 = ncan
const = equal_to( sum_poly( [q[i] for i in range(ndat)] ), ncan)

####################################################
# モデル
model = cost + 5*const

####################################################
# ソルバの生成、起動
solver = Solver(client)
result = solver.solve(model)

# 解が見つからないときのエラー出力
if len(result) == 0:
    raise RuntimeError("Any one of constraints is not satisfied.")

energy = result[0].energy
values = result[0].values
q_values = decode_solution(q, values)

####################################################
# 結果の表示
print(f"energy = {energy}")

for i in range(ndat):
    if (q_values[i] == 1):
        print(nb[i], ", ", end="")
print()
for i in range(ndat):
    if (q_values[i] == 1):
        print(gflops[i], ", ", end="")
print()
for i in range(ndat):
    if (q_values[i] == 1):
        print(ib[i], ", ", end="")
print()