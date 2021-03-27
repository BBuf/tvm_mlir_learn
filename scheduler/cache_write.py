import tvm
from tvm import te

n = 1024
dtype = "float32"
A = te.placeholder((n, n), dtype=dtype, name='A')
k = te.reduce_axis((0, n), name='k')
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

BW = s.cache_write(B, "local")

print(tvm.lower(s, [A, B], simple_mode=True))