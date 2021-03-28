# rfactor对原tensor在axis方向以factor_axis为间隔做reduction操作。
import tvm
from tvm import te

n = 1024
k = te.reduce_axis((0, n), name='k')

A = te.placeholder((n,), name='A')
B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)
ko, ki = s[B].split(s[B].op.reduce_axis[0], 32)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

BR = s.rfactor(B, ki)

print(tvm.lower(s, [A, B], simple_mode=True))

# primfn(A_1: handle, B_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {B: Buffer(B_2: Pointer(float32), float32, [1], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024], [])}
#   buffer_map = {A_1: A, B_1: B} {
#   B_2[0] = 0f32
#   for (k.outer: int32, 0, 32) {
#     for (k.inner: int32, 0, 32) {
#       B_2[0] = ((float32*)B_2[0] + (float32*)A_2[((k.outer*32) + k.inner)])
#     }
#   }
# }


# ---------cutting line---------
# primfn(A_1: handle, B_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {B: Buffer(B_2: Pointer(float32), float32, [1], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024], [])}
#   buffer_map = {A_1: A, B_1: B} {
#   attr [B.rf: Pointer(float32)] "storage_scope" = "global";
#   allocate(B.rf, float32, [32]) {
#     for (k.inner: int32, 0, 32) {
#       B.rf[k.inner] = 0f32
#       for (k.outer: int32, 0, 32) {
#         B.rf[k.inner] = ((float32*)B.rf[k.inner] + (float32*)A_2[((k.outer*32) + k.inner)])
#       }
#     }
#     B_2[0] = 0f32
#     for (k.inner.v: int32, 0, 32) {
#       B_2[0] = ((float32*)B_2[0] + (float32*)B.rf[k.inner.v])
#     }
#   }
# }