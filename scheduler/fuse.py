# fuse用于融合两个iter，将两层循环合并到一层，其返回值为iter类型，可以多次合并。
import tvm
from tvm import te

n = 1024
A = te.placeholder((n,), name='A')
k = te.reduce_axis((0, n), name='k')

B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].fuse(ko, ki)

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
#   B_2[0] = 0f32
#   for (k.outer.k.inner.fused: int32, 0, 1024) {
#     B_2[0] = ((float32*)B_2[0] + (float32*)A_2[k.outer.k.inner.fused])
#   }
# }