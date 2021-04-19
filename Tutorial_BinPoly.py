from amplify import BinaryPoly, gen_symbols, Solver, decode_solution
from amplify.client import FixstarsClient

# クライアントの設定
client = FixstarsClient()               # Fixstars Optigan
client.parameters.timeout = 1000        # タイムアウト1秒   
client.token = "jtSrmOum5m4eTuMEKDbrBekiOqa6nkCg"
# client.parameters.outputs.duplicate = True  # 同じエネルギー値の解を列挙するオプション（解が複数個ある場合）

# 4要素の1次元配列のバイナリ変数
q_1d = gen_symbols(BinaryPoly, 4)

# 3x2 の 2次元配列型のバイナリ変数
q_2d = gen_symbols(BinaryPoly, 3, 2)

# 変数インデックスが10から始まる 3x2 の 2次元配列型のバイナリ変数
q_2d_2 = gen_symbols(BinaryPoly, 10, (3, 2))


print(f"q_1d = {q_1d}")
print(f"q_2d = {q_2d}")
print(f"q_2d_2 = {q_2d_2}")

q = gen_symbols(BinaryPoly,4)

# q_0 * q_1 + q_2
f0 = q[0] * q[1] + q[2]

# q_1 + q_3 + 1
f1 = q[1] + q[3] + 1

# (q_0 * q_1 + q_2) + (q_1 + q_3 + 1)
f2 = f0 + f1

# (q_1 + q_3 + 1) * (q_1 + q_3 + 1)
f3 = f1 ** 2

print(f"f0 = {f0}")
print(f"f1 = {f1}")
print(f"f2 = {f2}")
print(f"f3 = {f3}")

s1 = gen_symbols(BinaryPoly, 4)
s2 = gen_symbols(BinaryPoly, 4)

# インデックスをずらさないと、同一の変数が定義されてしまう
print(s1, s2)

# s1 の分だけインデックスをずらして変数生成
s3 = gen_symbols(BinaryPoly, len(s1), (4, ))

# 異なる変数が定義できる
print(s1, s3) 