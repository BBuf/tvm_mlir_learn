# parallel将指定iter的for循环替换为parallel操作，从而在GPU以外的CPU等设备上实现并行。
import tvm
from tvm import te
n = 1024
m = 1024

A = te.placeholder((n, m), name='A')
l = te.reduce_axis((0, m), name = 'l')

B = te.compute((n,), lambda i: te.sum(A[i, l], axis=l), name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].parallel(B.op.reduce_axis[0])
print(tvm.lower(s, [A, B], simple_mode=True))

# primfn(A_1: handle, B_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {B: Buffer(B_2: Pointer(float32), float32, [1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, B_1: B} {
#   for (i: int32, 0, 1024) {
#     B_2[i] = 0f32
#     for (l: int32, 0, 1024) {
#       B_2[i] = ((float32*)B_2[i] + (float32*)A_2[((i*1024) + l)])
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
#     for (l: int32, 0, 1024) "parallel" {
#       B_2[i] = ((float32*)B_2[i] + (float32*)A_2[((i*1024) + l)])
#     }
#   }
# }