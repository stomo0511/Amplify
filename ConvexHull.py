import math
from matplotlib import lines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from scipy.spatial import ConvexHull

# MaxIB 読み込み
data = pd.read_csv("Odyssey_MaxIB.csv", skipinitialspace=True)

nb = data.nb.values            # タイルサイズ
ib = data.ib.values            # 内部ブロック幅
gflops = data.gflops.values    # 正規化速度

ndat = len(nb)                 # データ数

nnb = nb / (max(nb) - min(nb)) # 正規化

# nb-gflops 平面
points = [ [nnb[i], gflops[i]] for i in range(ndat) ]

# Convex-Hull
hull = ConvexHull(points)
points = hull.points
hull_points = points[hull.vertices]

nhdat = len(hull_points)       # 凸包の頂点数
 
hp = np.vstack((hull_points, hull_points[0]))

# 凸包の隣り合う２つの頂点を通る直線の傾きと切片
# (x1,y1), (x2,y2) を通る直線 y=ax+b
# x1: hp[j,0],   y1: hp[j,1]
# x2: hp[j+1,0], y2: hp[j+1,1]
# a: (y1 - y2) / (x1 - x2), b: (x1*y2 - x2*y1) / (x1 - x2)
SlopInt = [ [ (hp[j,1] - hp[j+1,1]) / (hp[j,0] - hp[j+1,0]), (hp[j,0]*hp[j+1,1] - hp[j+1,0]*hp[j,1]) / (hp[j,0] - hp[j+1,0])] for j in range(nhdat)]

# MaxIBペアから凸包の辺に降ろした垂線の傾きと切片
# [i,j], i:MaxIB, j:凸包
# (p,q) から y=ax+b に降ろした垂線 y = (-1/a)*x + beta
# p: point[i,0], q: point[i,1]
# a: SlopInt[j,0], b: SlopInt[j,1]
# beta: q + p/a
# P_SlopInt = [[ [ -1.0/SlopInt[j,0], points[i,1] + points[i,0]/SlopInt[j,0]] for i in range(ndat) ] for j in range(nhdat) ]
# for i in range(ndat):
    # for j in range(nhdat):
        # P_SlopInt[i,j] = 

# exit()

plt.plot(hp[:,0], hp[:,1])
plt.scatter(points[:,0], points[:,1])
plt.show()