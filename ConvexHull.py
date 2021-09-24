import math
from matplotlib import lines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from scipy.spatial import ConvexHull

# MaxIB 読み込み
data = pd.read_csv("Odyssey_MaxIB.csv", skipinitialspace=True)

nb = data.nb.values             # タイルサイズ
ib = data.ib.values             # 内部ブロック幅
gflops = data.gflops.values     # 正規化速度

ndat = len(nb)                  # データ数

nnb = nb / (max(nb) - min(nb))  # タイルサイズ正規化

# nb-gflops 平面
points = np.array([ [nnb[i], gflops[i]] for i in range(ndat) ])

hull = ConvexHull(points)       # 凸包
points = hull.points            # 凸包内のすべての点
hpoints = points[hull.vertices] # 凸包の頂点
nhdat = len(hpoints)            # 凸包の頂点数

# 凸包の頂点を結ぶ点列
hp = np.vstack((hpoints, hpoints[0]))

# 凸包の隣り合う２つの頂点を通る直線の傾きと切片
# (x1,y1), (x2,y2) を通る直線 y=ax+b
# x1: hp[j,0],   y1: hp[j,1]
# x2: hp[j+1,0], y2: hp[j+1,1]
# a: (y1 - y2) / (x1 - x2), b: (x1*y2 - x2*y1) / (x1 - x2)
SlopInt = np.array([ [ (hp[j,1] - hp[j+1,1]) / (hp[j,0] - hp[j+1,0]), (hp[j,0]*hp[j+1,1] - hp[j+1,0]*hp[j,1]) / (hp[j,0] - hp[j+1,0])] for j in range(nhdat)])

# MaxIBペアから凸包の辺に降ろした垂線の切片
# i:MaxIB, j:凸包の頂点（を結ぶ直線）
# (p,q) から y=ax+b に降ろした垂線 y = (-1/a)*x + beta
# p: points[i,0], q: points[i,1]
# a: SlopInt[j,0], b: SlopInt[j,1]
# beta: q + p/a
Beta = np.array([[ points[i,1] + points[i,0]/SlopInt[j,0] for i in range(ndat) ] for j in range(nhdat) ])

# MaxIBペア (p,q) から凸包の辺に降ろした垂線の足 (s,t)
# [i,j], i:MaxIB, j:凸包の頂点（を結ぶ直線）
# a: SlopInt[j,0], b: SlopInt[j,1]
# beta: Beta[j,i]
Foot = np.array([ [ [ SlopInt[j,0]*(Beta[j,i] - SlopInt[j,1]) / (SlopInt[j,0]*SlopInt[j,0] + 1), (SlopInt[j,0]*SlopInt[j,0]*Beta[j,i] + SlopInt[j,1] ) / (SlopInt[j,0]*SlopInt[j,0] + 1)] for i in range (ndat) ]for j in range(nhdat) ] )

# MaxIBペア (p,q) から垂線の足 (s,t) までの距離
# i: MaxIB, j: 足
# p: points[i,0], q: points[i,1]
# s: Foot[j,i,0], t Foot[j,i,1]: 
Dist = np.array([[ ((points[i,0] - Foot[j,i,0])**2 + (points[i,1] - Foot[j,i,1])**2)**0.5 for i in range(ndat)] for j in range(nhdat) ])
# print(Dist.shape)


plt.plot(hp[:,0], hp[:,1])
plt.scatter(points[:,0], points[:,1])
plt.show()