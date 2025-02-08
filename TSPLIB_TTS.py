import sys
import re
import numpy as np
import matplotlib.pyplot as plt

# 都市のプロット
def show_plot(locations: np.ndarray):
    plt.figure(figsize=(7, 7))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(locations[:, 0], locations[:, 1])
    plt.show()

# 都市と道順のプロット
def show_route(ncity, route: np.ndarray, distances: np.ndarray, locations: np.ndarray):
    path_length = sum([distances[route[i]][route[(i + 1) % ncity]] for i in range(ncity)])
    
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

# TSPLIBファイルのロード関数
def load_tsp_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Extracting NODE_COORD_SECTION
    node_coord_section = re.search(r'NODE_COORD_SECTION\s+(.*?)\s+EOF', content, re.DOTALL)
    if not node_coord_section:
        raise ValueError("NODE_COORD_SECTION not found in the file")

    # Parse city positions
    node_lines = node_coord_section.group(1).strip().split('\n')
    locations = []

    for line in node_lines:
        parts = line.strip().split()
        city_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
        locations.append((x, y))

    ncity = len(locations)

    # Compute distance matrix (Euclidean distance)
    locations = np.array(locations)
    distances = np.zeros((ncity, ncity))

    for i in range(ncity):
        for j in range(ncity):
            if i != j:
                distances[i][j] = np.linalg.norm(locations[i] - locations[j])

    return ncity, distances, locations

from amplify import VariableGenerator
from amplify import sum as amplify_sum
from amplify import one_hot
from amplify import FixstarsClient, solve
from amplify import einsum, Poly
import pandas as pd
from datetime import timedelta
from tqdm import tqdm

###################################################################
if __name__ == "__main__":
    args = sys.argv
    
    if len(args) < 2:
        print("Usage: python3 TSPLIB.py <Data file name>")
        sys.exit(1)
    
    file_name = args[1]

    # TSPLIBファイルのロード
    ncity, distances, locations = load_tsp_file(file_name)
    print(f"Num. cities: {ncity}")

    # 都市のプロット
    # show_plot(locations)

    gen = VariableGenerator()
    q = gen.array("Binary", shape=(ncity + 1, ncity))
    q[ncity, :] = q[0, :]  # 最初の都市と最後の都市は同じ。この固定はコスト関数、制約関数の定義より前に行う

    # コスト関数
    cost = amplify_sum(
        range(ncity),
        lambda n: amplify_sum(
            range(ncity),
            lambda i: amplify_sum(
                range(ncity), lambda j: distances[i, j] * q[n, i] * q[n + 1, j]
            ),
        ),
    )
    
    # amplify einsum を使ったコスト関数の定義
    objective: Poly = einsum("ij,ni,nj->", distances, q[:-1], q[1:])  # type: ignore
    
    # 制約関数
    # 最後の行を除いた q の各行のうち一つのみが 1 である制約
    row_constraints = one_hot(q[:-1], axis=1)
    
    # 最後の行を除いた q の各列のうち一つのみが 1 である制約
    col_constraints = one_hot(q[:-1], axis=0)
    
    constraints = row_constraints + col_constraints
    
    constraints *= np.amax(distances)  # 制約条件の強さを設定
    model = objective + constraints
    # model = cost + constraints
    
    client = FixstarsClient()
    client.token = "AE/ar62PjutSqmuoEa8bvfyrEmjE1rCpOqE"
    client.parameters.timeout = timedelta(milliseconds=1000) # タイムアウトの設定
    client.parameters.outputs.sort = False  # ソルバーの出力をソートしない
    client.parameters.outputs.num_outputs = 0  # ソルバーの出力を全て表示


    # num_sample個のサンプルをとる
    num_sample = 10
    d = {"sample":[],"sampling_time":[],"energy":[]}

    for n in tqdm(range(num_sample)):
        result = solve(model, client, sort_solution=False)
        for t, s in zip(result.client_result.execution_time.time_stamps, result.solutions):
            if s.feasible:
                d["sample"].append(n)
                # print(f"n={n}")
                d["sampling_time"].append(t)
                # print(f"t={t}")
                d["energy"].append(s.objective)
                # print(f"e={s.objective}")

    # TTSを算出
    # threshold_energy = 426
    threshold_energy = 450

    def calc_TTS(d, tau):
        N = num_sample
        # tau ms 以内にthreshold_energy以下の解が得られたサンプル数
        n = (
            pd.DataFrame(d).query(f"sampling_time<={tau} & energy <= {threshold_energy}")
            ["sample"].unique().shape[0]
        )
        ps_tau = n/N
        TTS = tau * np.log(1-0.99)/np.log(1-ps_tau) if ps_tau != 0 else np.Inf
        return TTS

    # min_tau TTS(tau)を計算
    TTS_data = {"tau":[], "TTS":[]}
    for t in tqdm(range(1000)):
        TTS_data["TTS"].append(calc_TTS(d, t))
        TTS_data["tau"].append(t)
    print("tau:", np.argmin(TTS_data["TTS"]),"ms")
    print("TTS:", np.min(TTS_data["TTS"]),"ms")
