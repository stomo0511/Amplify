from amplify import(
    gen_symbols,
    BinaryPoly,
    sum_poly, pair_sum, product
) 

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

##########################
# 例 1
# バイナリ変数を1次元配列形式に8個生成
q = gen_symbols(BinaryPoly, 8)

# 二値変数や多項式のリストを指定すると、その総和を計算
f0 = sum_poly(q)

print(f"f0 = {f0}")

##########################
# 例 2
# バイナリ変数を3個生成
q = gen_symbols(BinaryPoly, 3)

# インデックスを受け取る関数とインデックスの上限値を指定して、総和を取ることも可能
f1 = sum_poly(3, lambda i:
        sum_poly(3, lambda j: q[i] * q[j]))

print(f"f1 = {f1}")

##########################
# 例 3
# 2x2のバイナリ変数を生成
q = gen_symbols(BinaryPoly, 2, 2)

# 2乗と四則演算を含む数式の2重和
f2 = sum_poly(2, lambda i: (
        sum_poly(2, lambda j: q[i][j]) - 1) ** 2)

print(f"f2 = {f2}")

##########################
# 例 4
# バイナリ変数を3個生成
q = gen_symbols(BinaryPoly, 3)

f3 = pair_sum(q)

print(f"f3 = {f3}")

##########################
# 例 5
# バイナリ変数を3個生成
q = gen_symbols(BinaryPoly, 3)

f4 = product(q)

print(f"f4 = {f4}")

##########################
# コンストラクタを用いた二値変数多項式の構築
# q_0
f0 = BinaryPoly({(0): 1})

# 2 * q_0 * q_1 + 1
f1 = BinaryPoly({(0, 1): 2, (): 1})


print(f"f0 = {f0}")
print(f"f1 = {f1}")