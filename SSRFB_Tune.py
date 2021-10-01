from amplify import (
    BinaryPoly,
    sum_poly,
    gen_symbols,
    Solver,
    decode_solution,
)
from amplify.constraint import equal_to
from amplify.client import FixstarsClient

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
import math

####################################################
# データファイルの読み込み
# (nb, ib, time)
# data = pd.read_csv("Odyssey_MaxIB.csv", skipinitialspace=True)
# data = pd.read_csv("Odyssey_MaxIB2.csv", skipinitialspace=True)
data = pd.read_csv("Odyssey_MaxIB3.csv", skipinitialspace=True)

nb = data.nb.values             # タイルサイズ
ib = data.ib.values             # 内部ブロック幅
gflops = data.gflops.values     # 正規化速度
ndat = len(nb)                  # データ数 (32〜512まで2刻み -> 241)

nnb = nb / (max(nb) - min(nb))  # タイルサイズ正規化

####################################################
# 候補ペア数 <- パラメータ
ncan = 4                        # パラメータペア数

####################################################
# nb 等間隔点
step = (max(nnb) - min(nnb)) / ncan  # 間隔
equivp = [ min(nnb) + step*(i+1) for i in range(ncan) ]  # 等間隔点（ min(nnb)は含まれない ）

# 等間隔点と nb の距離
# dist = [
    # [
        # ((equivp[i] - nnb[j])**2)**0.5 
        # for j in range(ndat)
    # ]
    # for i in range(ncan)
# ]

####################################################
# c_nb: 等間隔に収まる添字の数
c_nb = [0 for i in range(ncan)]

# low_nb から up_nb に収まる nnb の個数をカウント
for i in range(ncan):
    low_nb = min(nnb) + i*step
    up_nb  = min(nnb) + (i+1)*step
    for j in range(ndat):
        if nnb[j] > low_nb and nnb[j] <= up_nb:
            c_nb[i] += 1
c_nb[0] += 1    # nnb = 下限 がカウントできてないため必要

####################################################
# steepness （要素数は1つ少ない）
stpn = [ (gflops[i+1] - gflops[i]) for i in range(ndat-1) ]

################################################################
# convex hull
####################################################
# nb-gflops 平面上の凸包
points = np.array(
    [
        [ nnb[i], gflops[i] ] for i in range(ndat)
    ]
)

hull = ConvexHull(points)       # 凸包
points = hull.points            # 凸包内のすべての点
hpoints = points[hull.vertices] # 凸包の頂点
nhdat = len(hpoints)            # 凸包の頂点数

# 凸包の頂点を結ぶ点列
hp = np.vstack((hpoints, hpoints[0]))

####################################################
# 凸包の隣り合う２つの頂点を通る直線の傾きと切片
# (x1,y1), (x2,y2) を通る直線 y=ax+b
# x1: hp[j,0],   y1: hp[j,1]
# x2: hp[j+1,0], y2: hp[j+1,1]
# a: (y1 - y2) / (x1 - x2), b: (x1*y2 - x2*y1) / (x1 - x2)
SlopInt = np.array(
    [
        [
            (hp[j,1] - hp[j+1,1]) / (hp[j,0] - hp[j+1,0]), 
            (hp[j,0]*hp[j+1,1] - hp[j+1,0]*hp[j,1]) / (hp[j,0] - hp[j+1,0])
        ]
        for j in range(nhdat) 
    ]
)

####################################################
# MaxIBペアから凸包の辺に降ろした垂線の切片
# i:MaxIB, j:凸包の頂点（を結ぶ直線）
# (p,q) から y=ax+b に降ろした垂線 y = (-1/a)*x + beta
# p: points[i,0], q: points[i,1]
# a: SlopInt[j,0], b: SlopInt[j,1]
# beta: q + p/a
Beta = np.array(
    [
        [
            points[i,1] + points[i,0]/SlopInt[j,0]
            for i in range(ndat)
        ]
        for j in range(nhdat)
    ]
)

####################################################
# MaxIBペア (p,q) から凸包の辺に降ろした垂線の足 (s,t)
# [i,j], i:MaxIB, j:凸包の頂点（を結ぶ直線）
# a: SlopInt[j,0], b: SlopInt[j,1]
# beta: Beta[j,i]
Foot = np.array(
    [
        [
            [
                SlopInt[j,0]*(Beta[j,i] - SlopInt[j,1]) / (SlopInt[j,0]*SlopInt[j,0] + 1),
                (SlopInt[j,0]*SlopInt[j,0]*Beta[j,i] + SlopInt[j,1] ) / (SlopInt[j,0]*SlopInt[j,0] + 1)
            ]
            for i in range (ndat)
        ]
        for j in range(nhdat)
    ]
)

####################################################
# MaxIBペア (p,q) から垂線の足 (s,t) までの距離
# i: MaxIB, j: 足
# p: points[i,0], q: points[i,1]
# s: Foot[j,i,0], t Foot[j,i,1]: 
Dist = np.array(
    [
        [
            ( (points[i,0] - Foot[j,i,0])**2 + (points[i,1] - Foot[j,i,1])**2 )**0.5
            for i in range(ndat)
        ] 
        for j in range(nhdat)
    ]
)


# sys.exit()

####################################################
# バイナリ変数の生成
q = gen_symbols(BinaryPoly, ndat)

####################################################
# コスト関数0：gflops 値の総和
Cost0 = sum_poly( q*gflops*(-1) )

####################################################
# コスト関数1： nb 等間隔点からの距離が最小
# Cost1 = sum_poly([ dist[i][j] * q[j] for i in range(ncan) for j in range(ndat) ])

####################################################
# コスト関数2： 二点間の距離が最大 -> 両端を選ぶ
Cost2 = - sum_poly( [q[i]*q[j] * math.fabs(nnb[i] - nnb[j]) for i in range(ndat) for j in range(ndat) if i != j] )

####################################################
# コスト関数3: 二点間の距離の最小値の和が最大 <- 未完
# Cost3 = - sum_poly( [q[i]*min([q[j]*math.fabs(nnb[i] - nnb[j]) for j in range(ncan) if j!=i])] for i in range(ncan) )
# print( min([[math.fabs(nnb[i] - nnb[j]) for j in range(ndat) if j!=i]]) for i in range(ndat) )
# for i in range(ndat):
#     print( min( [ math.fabs(nnb[i] - nnb[j]) for j in range(ndat) if j!=i] ) )

####################################################
# コスト関数4: 凸包の線分までの距離が最小
Cost4 = sum_poly([Dist[j][i] * q[i] for j in range(nhdat) for i in range(ndat)])

####################################################
# モデル
# model = Cost0
# model = Cost0 + Cost2
model = Cost0 + Cost4

####################################################
# 制約関数0： 等間隔内に一つの変数
w = 10   # 制約の weight
st = 0
ed = c_nb[0]
for k in range(ncan):
    model += equal_to( sum_poly( [ w*q[i] for i in list(range(st,ed)) ] ), w)
    st += c_nb[k]
    ed += c_nb[(k+1) % ncan]

# model = Cost0 + Cost2 + w * Constraint

####################################################
# クライアント設定
client = FixstarsClient()
client.token = "YPUHk3Oh0pIVYdwFB43uzcLFkEiq9zDf"  #20211207まで有効
client.parameters.timeout = 10000  # タイムアウト10秒
client.parameters.outputs.duplicate = True  # 同じエネルギー値の解を列挙するオプション
client.parameters.outputs.num_outputs = 0   # 見つかったすべての解を出力

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

# 制約を満たした解か？
print(result[0].is_feasible)
if result[0].is_feasible == False:
    print(model.check_constraints(result[0].values))

Huris = np.array(
    [
        [ nb[i], ib[i], gflops[i] ] for i in range(ndat) if q_values[i] ==1
    ]
)

for i in range(len(Huris)):
    print( "(", Huris[i,0], ", ", Huris[i,1], ") : ", "{:.3f}".format(Huris[i,2]) )

####################################################
# 結果のプロット
points = np.array(
    [
        [ nb[i], gflops[i] ] for i in range(ndat)
    ]
)

hull = ConvexHull(points)       # 凸包
points = hull.points            # 凸包内のすべての点
hpoints = points[hull.vertices] # 凸包の頂点

# 凸包の頂点を結ぶ点列
hp = np.vstack((hpoints, hpoints[0]))
plt.plot(hp[:,0], hp[:,1])
plt.scatter(points[:,0], points[:,1], marker='.')
plt.scatter(hpoints[:,0], hpoints[:,1], s=40, c='blue', marker='^')
plt.scatter(Huris[:,0],Huris[:,2], s=30, c='red', marker='x')
plt.show()