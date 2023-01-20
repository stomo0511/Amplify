import pandas as pd
from amplify import BinaryPoly, gen_symbols, sum_poly

# 各店舗の要求人数情報の読み込み
dict_req = dict(location=["tenjin", "hakata"], employee=[2, 3])

# 各従業員の勤務希望情報の読み込み
dict_worker_loc = dict(
    worker_id=[0, 1, 2, 3, 4], tenjin=[2, 2, 1, 0, 1], hakata=[1, 1, 1, 1, 0]
)

# 要求人数、勤務希望のデータフレーム
df_req = pd.DataFrame.from_dict(dict_req, orient="index").T
# print("各店舗の従業員要求人数")

df_worker_loc = pd.DataFrame.from_dict(dict_worker_loc, orient="index").T

## 従業員id, 店舗id のセット
workers = df_worker_loc["worker_id"].values
locations = df_req["location"].values

## 各データ長を取得
num_workers = len(workers)
num_locations = len(locations)

## 店舗名 <-> 店舗id の変換辞書 <- *** 理解してない ***
idx2loc = dict((i, v) for i, v in enumerate(df_req["location"].values))
loc2idx = dict((v, i) for i, v in enumerate(df_req["location"].values))

# 従業員iが店舗lで勤務することを表す「決定変数」: location_variables[i][l]
location_variables = gen_symbols(BinaryPoly, num_workers, num_locations)

## 勤務不可能地域に関しては変数を定数化
for i in workers:
    for l in locations:
        worker_req = df_worker_loc.iloc[i][l]
        if worker_req == 0:
            # 勤務不可
            location_variables[i][loc2idx[l]] = BinaryPoly(0)

## 勤務不可能地域に関する変数が0に固定されている
# print(location_variables)

## 充足率の計算 <- *** sum_poly + lambda : やや難 ***
w = [
    (sum_poly(num_workers, lambda i: location_variables[i][l])) / df_req["employee"][l]
    for l in range(num_locations)
]

# 充足率の平均の最大化（最小化）
average_fill_rate_cost = -((sum_poly(w) / len(w)) ** 2)

# 充足率の分散の最小化
variance_fill_rate_cost = (
    sum_poly(len(w), lambda i: w[i] ** 2) / len(w) - (sum_poly(w) / num_locations) ** 2
)

# 従業員の希望度最大化（最小化）
location_cost = -sum_poly(
    num_workers,
    lambda i: sum_poly(
        num_locations,
        lambda l: df_worker_loc.loc[i][idx2loc[l]] * location_variables[i][l],
    ),
)