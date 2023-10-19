from amplify import BinarySymbolGenerator, Solver, decode_solution
from amplify.client.ocean import DWaveSamplerClient
 
client = DWaveSamplerClient()
client.token = "DWAVE/OtxGdrPUxBzmu9VhLmA5unth8btkM2Qp"
# 実行するソルバを指定 (使用可能ソルバは solver_names で取得)
client.solver = "Advantage_system4.1"
client.parameters.num_reads = 100
 
gen = BinarySymbolGenerator()
q = gen.array(3)
f = 2 * q[0] * q[1] - q[0] - q[2] + 1
 
solver = Solver(client)
 
result = solver.solve(f)

energy = result[0].energy
values = result[0].values
q_values = decode_solution(q, values)

print('Energy: ', energy)
print('q[0] = ', q_values[0])
print('q[1] = ', q_values[1])
print('q[2] = ', q_values[2])