import tvm
from tvm import te

n = 1024
k = te.reduce_axis((0, n), name='k')

A = te.placeholder((n, n), name='A')
B = te.placeholder((n, n), name='B')

D = te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='D')
E = te.compute((n, n), lambda i, j: D[i, j] + B[i, j], name='E')
F = te.compute((n,), lambda i: te.sum(E[i, k], axis=k), name='F')

s = te.create_schedule(F.op)

print(tvm.lower(s, [A, B, E], simple_mode=True))
print("---------cutting line---------")

g = s.create_group(outputs = E, inputs = [A, B], include_inputs=True)
g.compute_at(s[F], F.op.reduce_axis[0])

print(tvm.lower(s, [A, B, E], simple_mode=True))