# set_scope指定stage计算结果所在的存储层次，为tensor选择最优的存储位置，
# 适用于设置线程间的共享内存。事实上，set_scope是cache_read的子操作。
import tvm
from tvm import te

n = 1024
dtype = "float32"
A = te.placeholder((n, n), dtype=dtype, name='A')
k = te.reduce_axis((0, n), name='k')
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name='B')
C = te.compute((n,), lambda i: B[i] + 10, name='C')

s = te.create_schedule(C.op)

print(tvm.lower(s, [A, C], simple_mode=True))
print("---------cutting line---------")

s[B].set_scope('shared')

print(tvm.lower(s, [A, C], simple_mode=True))


# primfn(A_1: handle, C_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, C_1: C} {
#   attr [B: Pointer(float32)] "storage_scope" = "global";
#   allocate(B, float32, [1024]) {
#     for (i: int32, 0, 1024) {
#       B[i] = 0f32
#       for (k: int32, 0, 1024) {
#         B[i] = ((float32*)B[i] + (float32*)A_2[((i*1024) + k)])
#       }
#     }
#     for (i_1: int32, 0, 1024) {
#       C_2[i_1] = ((float32*)B[i_1] + 10f32)
#     }
#   }
# }


# ---------cutting line---------
# primfn(A_1: handle, C_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, C_1: C} {
#   attr [B: Pointer(float32)] "storage_scope" = "shared";
#   allocate(B, float32, [1024]) {
#     for (i: int32, 0, 1024) {
#       B[i] = 0f32
#       for (k: int32, 0, 1024) {
#         B[i] = ((float32*)B[i] + (float32*)A_2[((i*1024) + k)])
#       }
#     }
#     for (i_1: int32, 0, 1024) {
#       C_2[i_1] = ((float32*)B[i_1] + 10f32)
#     }
#   }
# }