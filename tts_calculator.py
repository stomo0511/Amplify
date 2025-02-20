"""
## Requirements

requires-python = ">=3.12"

dependencies = [
    "amplify>=1.3.1",
    "numpy>=2.2.2",
    "pandas>=2.2.3",
    "tqdm>=4.67.1",
]

## Example

```py
def main() -> int:
    num_sample = 10
    timeout = timedelta(seconds=1)

    f = generate_random_matrix(64, 0)
    model, _ = qubo_teramoto_dw(f, 64, 32, 22_000_000, 7_500_000)

    calculator = TTSCalculator(model, AE_TOKEN, timeout)
    res = calculator.calc(39111120, num_sample)

    print(res.feasible_sample_data)
    print(res.tts)
    print(res.tau)
    return 0
```
"""

from datetime import timedelta
from typing import TypedDict
import dataclasses

from amplify import (
    FixstarsClient,
    Model,
    solve,
)
from tqdm import tqdm
import numpy as np
import pandas as pd


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
