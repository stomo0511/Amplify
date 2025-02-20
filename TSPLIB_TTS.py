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

from amplify import (
    VariableGenerator,
    FixstarsClient,
    Model,
    Poly,
    sum as amplify_sum,
    solve,
    einsum,
    one_hot
)
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
from typing import TypedDict
import dataclasses

class FeasibleSamples(TypedDict):
    sample: list[int]
    sampling_time: list[int]
    energy: list[float]


@dataclasses.dataclass(frozen=True)
class TTSResult:
    feasible_sample_data: pd.DataFrame
    tts_par_tau: pd.DataFrame
    fs_rate: float
    tts: float  # ms
    tau: int  # ms


class TTSCalculator:
    def __init__(self, model: Model, ae_token: str, timeout: timedelta) -> None:
        self.model = model
        self.timeout = timeout
        self.__timeout_ms = self.__to_ms(timeout)
        self.client = FixstarsClient()
        self.__init_client(ae_token)

    def __init_client(self, ae_token: str) -> None:
        self.client.token = ae_token
        self.client.parameters.timeout = self.timeout
        self.client.parameters.outputs.sort = False
        self.client.parameters.outputs.num_outputs = 0

    def __to_ms(self, t: timedelta) -> int:
        return int(t.total_seconds() * 1000)

    def __calc_tts(
        self, df: pd.DataFrame, tau: int, threshold_energy: float, num_sample: int
    ) -> float:
        # tau ms 以内にthreshold_energy以下の解が得られたサンプル数
        filtered = df.query(f"sampling_time <= {tau} & energy <= {threshold_energy}")
        n = filtered["sample"].unique().shape[0]
        ps_tau = n / num_sample
        if ps_tau == 0.0:
            return np.inf
        else:
            return tau * np.log(1 - 0.99) / np.log(1 - ps_tau)

    def calc(self, threshold_energy: float, num_sample: int = 50) -> TTSResult:
        dic: FeasibleSamples = {"sample": [], "sampling_time": [], "energy": []}
        num_total = 0
        num_fs = 0
        for n in tqdm(range(num_sample), desc="Solving samples"):
            result = solve(self.model, self.client, sort_solution=False, filter_solution=False)
            if result.client_result is None:
                continue
            result.filter_solution = False
            num_total += len(result.solutions)
            for t, s in zip(
                result.client_result.execution_time.time_stamps, result.solutions
            ):
                if s.feasible:
                    num_fs += 1
                    dic["sample"].append(n)
                    dic["sampling_time"].append(self.__to_ms(t))
                    dic["energy"].append(s.objective)
        df = pd.DataFrame(dic)
        data = {"tau": [], "tts": []}
        for t in tqdm(range(self.__timeout_ms), desc="Calcilating TTS"):
            data["tts"].append(self.__calc_tts(df, t, threshold_energy, num_sample))
            data["tau"].append(t)
        res = TTSResult(
            feasible_sample_data=df,
            tts_par_tau=pd.DataFrame(data),
            fs_rate=num_fs / num_total,
            tts=np.min(data["tts"]),
            tau=int(np.argmin(data["tts"])),
        )
        return res

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
    
    num_sample = 10
    timeout = timedelta(milliseconds=1000)

    calculator = TTSCalculator(model, "AE/ar62PjutSqmuoEa8bvfyrEmjE1rCpOqE", timeout)
    res = calculator.calc(440, num_sample)

    print(res.feasible_sample_data)
    print(res.tts)
    print(res.tau)
