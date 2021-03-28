# cache_write和cache_read对应，是先在shared memory中存放计算结果，最后将结果
# 写回到global memory。当然在真实的场景中，我们往往是会将结果先放着register中，最后写回。
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
#   attr [B.local: Pointer(float32)] "storage_scope" = "local";
#   allocate(B.local, float32, [1024]) {
#     for (i.c: int32, 0, 1024) {
#       B.local[i.c] = 0f32
#       for (k: int32, 0, 1024) {
#         B.local[i.c] = ((float32*)B.local[i.c] + (float32*)A_2[((i.c*1024) + k)])
#       }
#     }
#     for (i: int32, 0, 1024) {
#       B_2[i] = (float32*)B.local[i]
#     }
#   }
# }