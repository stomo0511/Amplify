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
data = pd.read_csv("test_MaxIB.csv", skipinitialspace=True)

nb = data.nb.values            # タイルサイズ
ib = data.ib.values            # 内部ブロック幅
gflops = data.gflops.values    # 正規化速度



####################################################
# 定数設定
ndat = len(nb)     # データ数
ncan = 4           # パラメータペア数

nnb = nb / (max(nb) - min(nb)) # 正規化したnb

####################################################
# クライアント設定
client = FixstarsClient()
client.token = "i5G6Ei3DKlGv2n6hsWBSBzWrmffLN4vn"  #20210911まで有効
client.parameters.timeout = 5000  # タイムアウト5秒
client.parameters.outputs.duplicate = True  # 同じエネルギー値の解を列挙するオプション
client.parameters.outputs.num_outputs = 0   # 見つかったすべての解を出力

####################################################
# バイナリ変数の生成
q = gen_symbols(BinaryPoly, ndat)

####################################################
# 制約関数0： "1"の変数の数 = ncan
Const0 = equal_to( sum_poly(q), ncan )

####################################################
# コスト関数0：gflops 値の総和
Cost0 = sum_poly( q*gflops*(-1) )

####################################################
# コスト関数：隣の点との距離を最大にする
# two_d = - sum_poly([q[i]*q[j]*((nb[i] - nb[j]) / (nb_max - nb_min))**2 for i in range(ndat) for j in range(ndat)])

# for i in range(ndat-1):
#     print( i, ", ", i+1, ", ", nb[i] - nb[i+1])

####################################################
# モデル
model = Cost0 + 10*Const0

###################################################
# ソルバの生成、起動
solver = Solver(client)
solver.filter_solution = False   # 制約を満たさない解を許す
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
        print( "(", nb[i], ", ", ib[i], ") : ", "{:.3f}".format(gflops[i]) )

# 制約を満たした解か？
print(result[0].is_feasible)
if result[0].is_feasible == False:
    print(model.check_constraints(result[0].values))