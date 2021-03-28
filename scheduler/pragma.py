# pragma用于添加编译注释，使编译器遵循pragma的要求，实现unroll,
#  vectorize等调度功能。事实上一个新的优化规则，都可以看做
# 是一种gragma，也被称作directive[https://en.wikipedia.org/wiki/Directive_(programming)]。
import tvm
from tvm import te

n = 1024
m = 1024
A = te.placeholder((n, m), name='A')
k = te.reduce_axis((0, n), name='k')
l = te.reduce_axis((0, m), name = 'l')

B = te.compute((n,), lambda i: te.sum(A[i, l], axis=l), name='B')

s = te.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=4)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].pragma(ki, "unroll")

print(tvm.lower(s, [A, B], simple_mode=True))

# primfn(A_1: handle, B_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {B: Buffer(B_2: Pointer(float32), float32, [1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, B_1: B} {
#   for (i: int32, 0, 1024) {
#     B_2[i] = 0f32
#     for (l.outer: int32, 0, 256) {
#       for (l.inner: int32, 0, 4) {
#         B_2[i] = ((float32*)B_2[i] + (float32*)A_2[(((i*1024) + (l.outer*4)) + l.inner)])
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
#   for (i: int32, 0, 1024) {
#     B_2[i] = 0f32
#     for (l.outer: int32, 0, 256) {
#       B_2[i] = ((float32*)B_2[i] + (float32*)A_2[((i*1024) + (l.outer*4))])
#       B_2[i] = ((float32*)B_2[i] + (float32*)A_2[(((i*1024) + (l.outer*4)) + 1)])
#       B_2[i] = ((float32*)B_2[i] + (float32*)A_2[(((i*1024) + (l.outer*4)) + 2)])
#       B_2[i] = ((float32*)B_2[i] + (float32*)A_2[(((i*1024) + (l.outer*4)) + 3)])
#     }
#   }
# }