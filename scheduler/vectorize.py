import tvm
import numpy
import timeit
from tvm import te

M = 1024
N = 1024
A = te.placeholder((M, N), name='A')
B = te.placeholder((M, N), name='B')
C = te.compute(
           (M, N),
           lambda x, y: A[x, y] + B[x, y],
           name='C')

s = te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].vectorize(yi)

print(tvm.lower(s, [A, B, C], simple_mode=True))