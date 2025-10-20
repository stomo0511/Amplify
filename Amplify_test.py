# import random
# from amplify import (
#     BinaryPoly,
#     BinaryQuadraticModel,
#     # sum_poly,
#     # gen_symbols,
#     # Solver,
#     # decode_solution,
# )
# from amplify.constraint import (
#     equal_to,
#     penalty,
# )
# from amplify.client import FixstarsClient

# print(cost)
# print(constraints)
# print(type(cost))
# print(type(constraints))

# f = gen_symbols(BinaryPoly, 5)
# print(f)
# for i in range(5):
#     f[i] = 1

# print(sum(f))

# 0, 1 の乱数列の生成
# print([random.randint(0,1) for i in range(10)])

# f = BinaryPoly({(0, 1, 2, 3): 2, (0,3): -2, (0, 1): 1, (0): 1, (1): 1, (2): 1}, -1)
# model = BinaryQuadraticModel(f)

# print(model.input_poly)
# print(model.logical_poly)

from amplify import VariableGenerator, solve_qubo
import numpy as np

# QUBO行列の生成（例：5000変数のランダムなQUBO）
n = 5000
Q = np.random.randn(n, n)
Q = (Q + Q.T) / 2  # 対称化

# 解く
result = solve_qubo(Q, num_repeats=10)  # num_repeats で高速反復制御

# 解の取得
x = result.values
energy = result.energy

print(f"Energy: {energy}")
