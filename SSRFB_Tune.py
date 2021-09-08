from amplify import (
    BinaryPoly,
    sum_poly,
    gen_symbols,
    Solver,
    decode_solution,
)
from amplify.constraint import equal_to
from amplify.client import FixstarsClient

import pandas as pd
import numpy as np
import math

####################################################
# データファイルの読み込み
# (nb, ib, time)
data = pd.read_csv("Odys_MaxIB.csv", skipinitialspace=True)
# data = pd.read_csv("Odys_MaxIB2.csv", skipinitialspace=True)

nb = data.nb.values            # タイルサイズ
ib = data.ib.values            # 内部ブロック幅
gflops = data.gflops.values    # 正規化速度

ndat = len(nb)                 # データ数
ncan = 8                       # パラメータペア数

####################################################
# nb 等間隔点
nnb = nb / (max(nb) - min(nb)) # nb の正規化
step = (max(nnb) - min(nnb)) / ncan  # 間隔
equivp = [min(nnb) + step*(i+1) for i in range(ncan)]  # 等間隔点

# 等間隔点と nb の距離
dist = [[ ((equivp[i] - nnb[j])**2)**0.5 for j in range(ndat)] for i in range(ncan) ]

# 等間隔に収まる添字の数
c_nb = [0 for i in range(ncan)]

# min(nb) から min(nb)+step に収まる nb の個数
for i in range(ncan):
    low_nb = min(nnb) + i*step
    up_nb  = min(nnb) + (i+1)*step
    for j in range(ndat):
        if nnb[j] > low_nb and nnb[j] <= up_nb:
            c_nb[i] += 1
c_nb[0] += 1    # nnb = 下限 がカウントできてないため必要

####################################################
# クライアント設定
client = FixstarsClient()
client.token = "YPUHk3Oh0pIVYdwFB43uzcLFkEiq9zDf"  #20211207まで有効
client.parameters.timeout = 5000  # タイムアウト5秒
client.parameters.outputs.duplicate = True  # 同じエネルギー値の解を列挙するオプション
client.parameters.outputs.num_outputs = 0   # 見つかったすべての解を出力

####################################################
# バイナリ変数の生成
q = gen_symbols(BinaryPoly, ndat)

####################################################
# コスト関数0：gflops 値の総和
Cost0 = sum_poly( q*gflops*(-1) )

####################################################
# コスト関数1： nb 等間隔点からの距離が最小
Cost1 = sum_poly([ dist[i][j] * q[j] for i in range(ncan) for j in range(ndat) ])

####################################################
# コスト関数2： 二点間の距離が最大 -> 両端を選ぶ
Cost2 = - sum_poly( [q[i]*q[j] * math.fabs(nnb[i] - nnb[j]) for i in range(ndat) for j in range(ndat) if i != j] )

####################################################
# モデル
model = Cost0 + Cost2

####################################################
# 制約関数0： 等間隔内に一つの変数
st = 0
ed = c_nb[0]
for k in range(ncan):
    model += equal_to( sum_poly( [ 10*q[i] for i in list(range(st,ed)) ] ), 10)
    st += c_nb[k]
    ed += c_nb[(k+1) % ncan]

####################################################
# コスト関数3: 二点間の距離の最小値の和が最大 <- 未完
# Cost3 = - sum_poly( [q[i]*min([q[j]*math.fabs(nnb[i] - nnb[j]) for j in range(ncan) if j!=i])] for i in range(ncan) )
# print( min([[math.fabs(nnb[i] - nnb[j]) for j in range(ndat) if j!=i]]) for i in range(ndat) )
# for i in range(ndat):
#     print( min( [ math.fabs(nnb[i] - nnb[j]) for j in range(ndat) if j!=i] ) )

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