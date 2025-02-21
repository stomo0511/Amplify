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
from amplify import FixstarsClient
from amplify import one_hot, einsum
from amplify import solve
from amplify import Poly
from datetime import timedelta
import optuna

###################################################################
if __name__ == "__main__":
    args = sys.argv
    
    if len(args) < 2:
        print("Usage: python3 TSPLIB.py <Data file name>")
        sys.exit(1)
    
    file_name = args[1]

    #################################################################
    # TSPLIBファイルのロード
    ncity, distances, locations = load_tsp_file(file_name)
    print(f"Num. cities: {ncity}")

    # 都市のプロット
    # show_plot(locations)

    #################################################################
    # 決定変数の定義
    gen = VariableGenerator()
    q = gen.array("Binary", shape=(ncity + 1, ncity))
    q[ncity, :] = q[0, :]  # 最初の都市と最後の都市は同じ。この固定はコスト関数、制約関数の定義より前に行う

    #################################################################
    # amplify einsum を使ったコスト関数の定義
    # objective: Poly = einsum("ij,ni,nj->", distances, q[:-1], q[1:])  # type: ignore
    
    #################################################################
    # 制約関数
    # 最後の行を除いた q の各行のうち一つのみが 1 である制約
    # row_constraints = one_hot(q[:-1], axis=1)
    
    # 最後の行を除いた q の各列のうち一つのみが 1 である制約
    # col_constraints = one_hot(q[:-1], axis=0)
    
    # constraints = row_constraints + col_constraints
    
    # constraints *= np.amax(distances)  # 制約条件の強さを設定
    
    def objective(trial):
        objective: Poly = einsum("ij,ni,nj->", distances, q[:-1], q[1:])
        row_constraints = one_hot(q[:-1], axis=1)
        col_constraints = one_hot(q[:-1], axis=0)
        constraints = row_constraints + col_constraints
        
        alpha = trial.suggest_float("alpha", 0.0, 1.0)

        constraints *= np.amax(distances)*alpha

        model = objective + alpha*constraints
        
        # ソルバーの設定
        client = FixstarsClient()
        client.token = "AE/ar62PjutSqmuoEa8bvfyrEmjE1rCpOqE"
        client.parameters.timeout = timedelta(milliseconds=1000) # タイムアウトの設定

        result = solve(model, client)
        # if len(result) == 0:
        #     raise RuntimeError("At least one of the constraints is not satisfied.")
        
        if len(result) == 0:
            return np.amax(distances)*1000
        else:
            return result.best.objective
        
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    print(study.best_value)
    print(study.best_trial)
    
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    #################################################################
    # ソルバーの設定
    # client = FixstarsClient()
    # client.token = "AE/ar62PjutSqmuoEa8bvfyrEmjE1rCpOqE"
    # client.parameters.timeout = timedelta(milliseconds=1000) # タイムアウトの設定

    # 結果の取得
    # result = solve(model, client)
    # if len(result) == 0:
    #     raise RuntimeError("At least one of the constraints is not satisfied.")
    
    #################################################################
    # 結果の表示
    # print(f"Path length: {result.best.objective}")
    # print(result.client_result.execution_time.time_stamps)
    # print(result.solutions)
    
    #################################################################
    # 都市と道順のプロット
    # q_values =q.evaluate(result.best.values)
    # route = np.where(q_values[:-1] == 1)[1]    
    # show_route(ncity, route, distances, locations)
    
    