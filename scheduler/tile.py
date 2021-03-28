# tile将stage的两个维度按照各自的factor拆分，并以固定顺序依次返回两个outer
# 和两个inner的iter，从而增加循环层数，形成更小的计算任务。事实上，tile是
# 可以由split和reorder来实现的，tile是矩阵乘法和卷积计算的重要schedule。
import tvm
from tvm import te

n = 1024
A = te.placeholder((n, n), name='A')
B = te.placeholder((n, n), name='B')
K = te.reduce_axis((0, n), name='K')
C = te.compute((n, n), lambda i, j: te.sum(A[i, K] * B[K, j], axis=K), name='C')

s = te.create_schedule(C.op)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)

print(tvm.lower(s, [A, B, C], simple_mode=True))

# primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
#              B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, B_1: B, C_1: C} {
#   for (i: int32, 0, 1024) {
#     for (j: int32, 0, 1024) {
#       C_2[((i*1024) + j)] = 0f32
#       for (K: int32, 0, 1024) {
#         C_2[((i*1024) + j)] = ((float32*)C_2[((i*1024) + j)] + ((float32*)A_2[((i*1024) + K)]*(float32*)B_2[((K*1024) + j)]))
#       }
#     }
#   }
# }


# ---------cutting line---------
# primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
#              B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, B_1: B, C_1: C} {
#   for (i.outer: int32, 0, 32) {
#     for (j.outer: int32, 0, 32) {
#       for (i.inner: int32, 0, 32) {
#         for (j.inner: int32, 0, 32) {
#           C_2[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] = 0f32
#           for (K: int32, 0, 1024) {
#             C_2[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] = ((float32*)C_2[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] + ((float32*)A_2[(((i.outer*32768) + (i.inner*1024)) + K)]*(float32*)B_2[(((K*1024) + (j.outer*32)) + j.inner)]))
#           }
#         }
#       }
#     }
#   }
# }