import numpy as np
import matplotlib.pyplot as plt
from amplify import (
    BinaryPoly,
    BinaryQuadraticModel,
    sum_poly,
    gen_symbols,
    Solver,
    decode_solution,
)
from amplify.constraint import (
    equal_to,
    penalty,
)
from amplify.client import FixstarsClient

def gen_random_tsp(ncity: int):
    # 座標
    locations = np.random.uniform(size=(ncity, 2))

    # 距離行列
    all_diffs = np.expand_dims(locations, axis=1) - np.expand_dims(locations, axis=0)
    distances = np.sqrt(np.sum(all_diffs ** 2, axis=-1))

    return locations, distances

# 都市のプロット
def show_plot(locs: np.ndarray):
    plt.figure(figsize=(7, 7))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(*zip(*locations))
    plt.show()

# 都市と道順のプロット
def show_route(route: list, distances: np.ndarray, locations: np.ndarray):

    ncity = len(route)
    path_length = sum(
        [distances[route[i]][route[(i + 1) % ncity]] for i in range(ncity)]
    )

    x = [i[0] for i in locations]
    y = [i[1] for i in locations]
    plt.figure(figsize=(7, 7))
    plt.title(f"path length: {path_length}")
    plt.xlabel("x")
    plt.ylabel("y")

    for i in range(ncity):
        r = route[i]
        n = route[(i + 1) % ncity]
        plt.plot([x[r], x[n]], [y[r], y[n]], "b-")
    plt.plot(x, y, "ro")
    plt.show()

    return path_length

# 都市数
ncity = 32
locations, distances = gen_random_tsp(ncity)

# 変数の生成: ncity x nicity
q = gen_symbols(BinaryPoly, ncity, ncity)

# 回転対称性の除去（始点の固定）
q[0][0] = BinaryPoly(1)
for i in range(1, ncity):
    q[0][i] = BinaryPoly(0)
    q[i][0] = BinaryPoly(0)

# クライアント設定
client = FixstarsClient()
client.token = "jtSrmOum5m4eTuMEKDbrBekiOqa6nkCg"
client.parameters.timeout = 5000  # タイムアウト5秒

# ソルバの生成
solver = Solver(client)

##############################################################
# コスト関数（修正版）
# sum_{n=0}^{ncity-1} sum_{i=0}^{ncity-1} sum_{j=0}^{ncity-1} distance[i][j]*q[n][i]*q[n+1][j]
# (n+1) % ncity : n=ncity-1 の時に n+1 -> 0 とするため

# 各行の非ゼロ最小値をリストで取得
d_min = [d[np.nonzero(d)].min() for d in distances]

# コスト関数の係数を改変し定数項を加算
cost = sum_poly(
    ncity,
    lambda n: sum_poly(
        ncity,
        lambda i: sum_poly(
            ncity,
            lambda j: (distances[i][j] - d_min[i] if i != j else 0)
            * q[n][i]
            * q[(n + 1) % ncity][j],
        ),
    ),
) + sum(d_min)

# 各行の最小値を引いた上で全要素の最大値を取得
d_max_all = max(distances.max(axis=1) - d_min)

##############################################################
# 行に対する制約
# sum_{i=0}^{ncity-1} q[n][i] = 1 for all n
# リスト内包表記
row_constraints = [
    equal_to(sum_poly([q[n][i] for i in range(ncity)]), 1) for n in range(ncity)
]

# 列に対する制約
# sum_{n=0}^{ncity-1} q[n][i] = 1 for all i
# リスト内包表記
col_constraints = [
    equal_to(sum_poly([q[n][i] for n in range(ncity)]), 1) for i in range(ncity)
]

# 反転対称性の除去
# 順序に対する制約の追加
pem_constraint = [
    penalty(q[ncity - 1][i] * q[1][j])
    for i in range(ncity)
    for j in range(i + 1, ncity)
]

# 制約
constraints = sum(row_constraints) + sum(col_constraints) + sum(pem_constraint)

##############################################################
# ・制約条件の強さはコスト関数に対して十分大きな値にする
# ・制約の強さはできるだけ小さい方がよい解が出やすい
# ・TSPでは距離行列の最大値
constraints *= np.amax(distances)       # 制約条件の強さを設定

model = cost + constraints * d_max_all  # 論理模型オブジェクト

# ここまで定式化
##############################################################
# ソルバ起動
result = solver.solve(model)

if len(result) == 0:
    raise RuntimeError("Any one of constraints is not satisfied.")

energy, values = result[0].energy, result[0].values

# 結果のデコード
q_values = decode_solution(q, values, 1)

# 移動順の生成
route = np.where(np.array(q_values) == 1)[1]

# 都市、経路、経路長の表示
show_route(route, distances, locations)