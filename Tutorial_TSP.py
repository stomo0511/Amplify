import sys
import numpy as np
import matplotlib.pyplot as plt

# ランダムな配置の都市の生成
def gen_random_tsp(ncity: int):
    rng = np.random.default_rng()

    # 座標
    locations = rng.random(size=(ncity, 2))
    
    # 距離行列
    x = locations[:, 0]
    y = locations[:, 1]
    distances = np.sqrt(
        (x[:, np.newaxis] - x[np.newaxis, :]) ** 2
        + (y[:, np.newaxis] - y[np.newaxis, :]) ** 2
    )

    return locations, distances

# 都市のプロット
def show_plot(locations: np.ndarray):
    plt.figure(figsize=(7, 7))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(locations[:, 0], locations[:, 1])
    plt.show()

# 都市と道順のプロット
def show_route(NUM_CITIES, route: np.ndarray, distances: np.ndarray, locations: np.ndarray):
    path_length = sum([distances[route[i]][route[(i + 1) % NUM_CITIES]] for i in range(NUM_CITIES)])
    
    x = [i[0] for i in locations]
    y = [i[1] for i in locations]
    plt.figure(figsize=(7, 7))
    plt.title(f"path length: {path_length}")
    plt.xlabel("x")
    plt.ylabel("y")
    
    for i in range(NUM_CITIES):
        r = route[i]
        n = route[(i + 1) % NUM_CITIES]
        plt.plot([x[r], x[n]], [y[r], y[n]], "b-")
    plt.plot(x, y, "ro")
    plt.show()
    
    return path_length

from amplify import VariableGenerator
from amplify import sum as amplify_sum
from amplify import one_hot
from amplify import FixstarsClient, solve
from datetime import timedelta

###################################################################
if __name__ == "__main__":
    args = sys.argv
    
    if len(args) < 2:
        print("Usage: python3 Tutorial_TSP.py <Num of cities>")
        sys.exit(1)
    
    NUM_CITIES = int(args[1])
    locations, distances = gen_random_tsp(NUM_CITIES)

    # 都市のプロット
    # show_plot(locations)
    
    gen = VariableGenerator()
    q = gen.array("Binary", shape=(NUM_CITIES + 1, NUM_CITIES))
    q[NUM_CITIES, :] = q[0, :]  # 最初の都市と最後の都市は同じ。この固定はコスト関数、制約関数の定義より前に行う

    # コスト関数
    cost = amplify_sum(
        range(NUM_CITIES),
        lambda n: amplify_sum(
            range(NUM_CITIES),
            lambda i: amplify_sum(
                range(NUM_CITIES), lambda j: distances[i, j] * q[n, i] * q[n + 1, j]
            ),
        ),
    )
    
    # 制約関数
    # 最後の行を除いた q の各行のうち一つのみが 1 である制約
    row_constraints = one_hot(q[:-1], axis=1)
    
    # 最後の行を除いた q の各列のうち一つのみが 1 である制約
    col_constraints = one_hot(q[:-1], axis=0)
    
    constraints = row_constraints + col_constraints
    
    constraints *= np.amax(distances)  # 制約条件の強さを設定
    # model = objective + constraints
    model = cost + constraints
    
    client = FixstarsClient()
    client.token = "AE/ar62PjutSqmuoEa8bvfyrEmjE1rCpOqE"
    client.parameters.timeout = timedelta(milliseconds=1000)  # タイムアウト 1000 ミリ秒
    
    # ソルバーの設定と結果の取得
    result = solve(model, client)
    if len(result) == 0:
        raise RuntimeError("At least one of the constraints is not satisfied.")
    
    print(f"Path length: {result.best.objective}")
    
    q_values =q.evaluate(result.best.values)
    route = np.where(q_values[:-1] == 1)[1]
    
    # 都市と道順のプロット
    # show_route(NUM_CITIES, route, distances, locations)