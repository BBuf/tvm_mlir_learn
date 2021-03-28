# split是fuse的反操作，把iter以factor为间隔分离成outer与inner两层迭代，增加循环层数，
# 用于将循环操作分割为更小的子任务。事实上，以CUDA为例，gridDim和blockDim都可以最多
# 是三维，所以通过split可以产生新的维度用于绑定到grid和block上[3]。
import tvm
from tvm import te

n = 1024
A = te.placeholder((n,), name='A')
k = te.reduce_axis((0, n), name='k')

B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)

print(tvm.lower(s, [A, B], simple_mode=True))

# primfn(A_1: handle, B_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {B: Buffer(B_2: Pointer(float32), float32, [1], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024], [])}
#   buffer_map = {A_1: A, B_1: B} {
#   B_2[0] = 0f32
#   for (k: int32, 0, 1024) {
#     B_2[0] = ((float32*)B_2[0] + (float32*)A_2[k])
#   }
# }


# ---------cutting line---------
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