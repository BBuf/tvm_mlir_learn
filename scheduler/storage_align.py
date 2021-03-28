# storage_align把stage对应的存储空间以factor为单位、以offset为偏置重新对齐，
# 以避免GPU共享访问时的bank conflict，关于bank conflict可以参考https://devblogs.nvidia.com/using-shared-memory-cuda-cc/。
import tvm
from tvm import te

n = 1024
factor =100
offset =8
dtype = "float32"
A = te.placeholder((n, n), dtype=dtype, name='A')
k = te.reduce_axis((0, n), name='k')
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name='B')

s = te.create_schedule(B.op)
AA = s.cache_read(A, "shared", [B])

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[AA].storage_align(AA.op.axis[0], factor, offset)

print(tvm.lower(s, [A, B], simple_mode=True))

# primfn(A_1: handle, B_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {B: Buffer(B_2: Pointer(float32), float32, [1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, B_1: B} {
#   attr [A.shared: Pointer(float32)] "storage_scope" = "shared";
#   allocate(A.shared, float32, [1048576]) {
#     for (ax0: int32, 0, 1024) {
#       for (ax1: int32, 0, 1024) {
#         A.shared[((ax0*1024) + ax1)] = (float32*)A_2[((ax0*1024) + ax1)]
#       }
#     }
#     for (i: int32, 0, 1024) {
#       B_2[i] = 0f32
#       for (k: int32, 0, 1024) {
#         B_2[i] = ((float32*)B_2[i] + (float32*)A.shared[((i*1024) + k)])
#       }
#     }
#   }
# }


# ---------cutting line---------
# primfn(A_1: handle, B_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {B: Buffer(B_2: Pointer(float32), float32, [1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, B_1: B} {
#   attr [A.shared: Pointer(float32)] "storage_scope" = "shared";
#   allocate(A.shared, float32, [1134592]) {
#     for (ax0: int32, 0, 1024) {
#       for (ax1: int32, 0, 1024) {
#         A.shared[((ax0*1108) + ax1)] = (float32*)A_2[((ax0*1024) + ax1)]
#       }
#     }
#     for (i: int32, 0, 1024) {
#       B_2[i] = 0f32
#       for (k: int32, 0, 1024) {
#         B_2[i] = ((float32*)B_2[i] + (float32*)A.shared[((i*1108) + k)])
#       }
#     }
#   }
# }