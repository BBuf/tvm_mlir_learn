import tvm
from tvm import te

M = 1024
K = 1024
N = 1024

# Algorithm
k = te.reduce_axis((0, K), 'k')
A = te.placeholder((M, K), name='A')
B = te.placeholder((K, N), name='B')
C = te.compute(
           (M, N),
           lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
           name='C')

# Default schedule
s = te.create_schedule(C.op)
ir_m = tvm.lower(s, [A, B, C], simple_mode=True,name='mmult')
rt_m = tvm.build(ir_m, [A, B, C], target='c', name='mmult')

# print tir
print("tir:\n", ir_m.astext(show_meta_data=False))
# print source code
print("source code:\n",rt_m.get_source())