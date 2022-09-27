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
data = pd.read_csv('Calc_MaxIB.csv', skipinitialspace=True)

nb = data.nb.values           # タイルサイズ
ib = data.ib.values           # 内部ブロック幅
gflops = data.gflops.values   # 正規化速度

####################################################
# 定数設定
ndat = len(nb)     # データ数
ncan = 8           # パラメータペア数

nb_min = min(nb)
nb_max = max(nb)
step = (nb_max - nb_min) / ncan

# 等間隔点
equivp = [nb_min + step*(i+1) for i in range(ncan)]

# 等間隔点と nb の距離
dist = [[ np.sqrt((equivp[i] - nb[j])**2) for j in range(ndat)] for i in range(ncan) ]

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
# コスト関数1：gflops 値の総和
total_g = sum_poly( q*gflops*(-1) )

####################################################
# コスト関数2：nb等間隔点からの距離
equiv_d = sum_poly( [dist[0][j] * q[j] for j in range(ndat)] ) + sum_poly( [dist[1][j] * q[j] for j in range(ndat)] ) + sum_poly( [dist[2][j] * q[j] for j in range(ndat)] )

####################################################
# 制約関数： "1"の変数の数 = ncan
const = equal_to( sum_poly(q), ncan )

####################################################
# モデル
# model = 10*total_g + equiv_d + 50*const
model = 10*total_g + 50*const

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