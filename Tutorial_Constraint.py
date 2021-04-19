from amplify import BinaryPoly, Solver, sum_poly, gen_symbols, decode_solution
from amplify.constraint import clamp, equal_to, greater_equal, less_equal, penalty
from amplify.client import FixstarsClient

# クライアントの設定
client = FixstarsClient()               # Fixstars Optigan
client.parameters.timeout = 1000        # タイムアウト1秒   
client.token = "jtSrmOum5m4eTuMEKDbrBekiOqa6nkCg"
client.parameters.outputs.duplicate = True  # 同じエネルギー値の解を列挙するオプション
client.parameters.outputs.num_outputs = 0   # 見つかったすべての解を出力

# ソルバーの構築
solver = Solver(client)                 # ソルバーに使用するクライアントを設定

# バイナリ変数：要素数2
q = gen_symbols(BinaryPoly, 2)

###########################################################
# NAND制約を与える多項式
g_NAND = q[0] * q[1]

# NAND制約をペナルティ制約条件に変換
p_NAND = penalty(g_NAND)

print("NAND")
print(f"p_NAND = {p_NAND}")

# 制約条件を満たす解を求める
result = solver.solve(p_NAND)
for sol in result:
    energy = sol.energy
    values = sol.values
    
    print(f"energy = {energy}, {q} = {decode_solution(q, values)}")

###########################################################
# OR制約を与える多項式
g_OR = q[0] * q[1] - q[0] - q[1] + 1
p_OR = penalty(g_OR)

print("OR")
print(f"p_OR = {p_OR}")

# 制約条件を満たす解を求める
result = solver.solve(p_OR)
for sol in result:
    energy = sol.energy
    values = sol.values
    
    print(f"energy = {energy}, {q} = {decode_solution(q, values)}")

###########################################################
# 等式制約：one-hot制約
# バイナリ変数：要素数3
q = gen_symbols(BinaryPoly, 3)
g = (sum_poly(q) - 1) ** 2  # one-hot 制約に対応したペナルティ関数

print("One-Hot")
print(f"g = {g}")

# 問題を解いて結果を表示
result = solver.solve(g)
for sol in result:
    energy = sol.energy
    values = sol.values
    
    print(f"energy = {energy}, {q} = {decode_solution(q, values)}")

###########################################################
# 等式制約：q0 * q1 + q2 = 1
g = equal_to(q[0] * q[1] + q[2], 1)  # 等式制約

print("Equal")
print(f"g = {g}")

# 問題を解いて結果を表示
result = solver.solve(g)
for sol in result:
    energy = sol.energy
    values = sol.values
    
    print(f"energy = {energy}, {q} = {decode_solution(q, values)}")

###########################################################
# 不等式制約 less_equal：q0 + q1 + q2 <= 1
g = less_equal(sum_poly(q), 1)  # 不等式制約

print("Inequality")
print(f"g = {g}")

# 問題を解いて結果を表示
result = solver.solve(g)
for sol in result:
    energy = sol.energy
    values = sol.values
    
    print(f"energy = {energy}, {q} = {decode_solution(q, values)}")

###########################################################
# 不等式制約 greater_equal：q0 + q1 + q2 >= 2
g = greater_equal(sum_poly(q), 2)  # 不等式制約

print("Inequality")
print(f"g = {g}")

# 問題を解いて結果を表示
result = solver.solve(g)
for sol in result:
    energy = sol.energy
    values = sol.values
    
    print(f"energy = {energy}, {q} = {decode_solution(q, values)}")

###########################################################
# 不等式制約 clamp：1 <= q0 + q1 + q2 <= 2
g = clamp(sum_poly(q), 1, 2)  # 不等式制約

print("Clamp")
print(f"g = {g}")

# 問題を解いて結果を表示
result = solver.solve(g)
for sol in result:
    energy = sol.energy
    values = sol.values
    
    print(f"energy = {energy}, {q} = {decode_solution(q, values)}")

###########################################################
# 複数の制約
# バイナリ変数：要素数2
q = gen_symbols(BinaryPoly, 2)

g1 = penalty(q[0])
g2 = penalty(q[1])

print("Penalties")
print(f"g1 + g2 : {g1 + g2}")

