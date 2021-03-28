# unroll是一种常见的循环优化方法，减分支预测失败减少，
# 如果循环体内语句没有数据相关，增加了并发执行的机会，
# 也有利于指令流水线的调度https://en.wikipedia.org/wiki/Loop_unrolling。
import tvm
from tvm import te

n = 1024
A = te.placeholder((n, n), name='A')
B = te.placeholder((n, n), name='B')
C = te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='C')

s = te.create_schedule(C.op)

xo, xi = s[C].split(s[C].op.axis[0], factor=4)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].unroll(xi)

print(tvm.lower(s, [A, B, C], simple_mode=True))

# primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
#              B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, B_1: B, C_1: C} {
#   for (i.outer: int32, 0, 256) {
#     for (i.inner: int32, 0, 4) {
#       for (j: int32, 0, 1024) {
#         C_2[(((i.outer*4096) + (i.inner*1024)) + j)] = ((float32*)A_2[(((i.outer*4096) + (i.inner*1024)) + j)] + (float32*)B_2[(((i.outer*4096) + (i.inner*1024)) + j)])
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
#   for (i.outer: int32, 0, 256) {
#     for (j: int32, 0, 1024) {
#       C_2[((i.outer*4096) + j)] = ((float32*)A_2[((i.outer*4096) + j)] + (float32*)B_2[((i.outer*4096) + j)])
#     }
#     for (j_1: int32, 0, 1024) {
#       C_2[(((i.outer*4096) + j_1) + 1024)] = ((float32*)A_2[(((i.outer*4096) + j_1) + 1024)] + (float32*)B_2[(((i.outer*4096) + j_1) + 1024)])
#     }
#     for (j_2: int32, 0, 1024) {
#       C_2[(((i.outer*4096) + j_2) + 2048)] = ((float32*)A_2[(((i.outer*4096) + j_2) + 2048)] + (float32*)B_2[(((i.outer*4096) + j_2) + 2048)])
#     }
#     for (j_3: int32, 0, 1024) {
#       C_2[(((i.outer*4096) + j_3) + 3072)] = ((float32*)A_2[(((i.outer*4096) + j_3) + 3072)] + (float32*)B_2[(((i.outer*4096) + j_3) + 3072)])
#     }
#   }