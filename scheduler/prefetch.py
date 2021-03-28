# prefetch利用数据的空间局部性，用于使得前一个iter的计算与后一个iter的
# 访存overlap起来，以提高访存和计算的并行度，减少耗时。本质上是软件
# 流水线的概念，不是硬件prefetch。
import tvm
from tvm import te

n = 1024
dtype = "float32"
k = te.reduce_axis((0, n), name='k')
A = te.placeholder((n, n), dtype=dtype, name='A')
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].prefetch(A, s[B].op.reduce_axis[0], 1)
print(tvm.lower(s, [A, B], simple_mode=True))

# primfn(A_1: handle, B_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {B: Buffer(B_2: Pointer(float32), float32, [1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, B_1: B} {
#   for (i: int32, 0, 1024) {
#     B_2[i] = 0f32
#     for (k: int32, 0, 1024) {
#       B_2[i] = ((float32*)B_2[i] + (float32*)A_2[((i*1024) + k)])
#     }
#   }
# }


# ---------cutting line---------
# primfn(A_1: handle, B_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {B: Buffer(B_2: Pointer(float32), float32, [1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, B_1: B} {
#   for (i: int32, 0, 1024) {
#     B_2[i] = 0f32
#     for (k: int32, 0, 1024) {
#       for (prefetch.A.1: int32, 0, 1) {
#         for (prefetch.A.0: int32, 0, 1) {
#           @tir.prefetch(@tir.address_of((float32*)A_2[(((k*1024) + i) + 1024)], dtype=handle), 0, 3, 1, dtype=float32)
#         }
#       }
#       B_2[i] = ((float32*)B_2[i] + (float32*)A_2[((i*1024) + k)])
#     }
#   }
# }