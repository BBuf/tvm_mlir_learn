# vectorize把iter方向上的循环迭代替换成ramp，从而通过SIMD指令实现数据的批量计算，
# 并且只有在数据size为常数、且分割的iter为2的幂（即满足SIMD的计算数量）时才会发
# 生替换，否则vectorize没有效果，是SIMD计算设备的常用schedule。
import tvm
import numpy
import timeit
from tvm import te

M = 1024
N = 1024
A = te.placeholder((M, N), name='A')
B = te.placeholder((M, N), name='B')
C = te.compute(
           (M, N),
           lambda x, y: A[x, y] + B[x, y],
           name='C')

s = te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].vectorize(yi)

print(tvm.lower(s, [A, B, C], simple_mode=True))

# primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
#              B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, B_1: B, C_1: C} {
#   for (x.outer: int32, 0, 32) {
#     for (y.outer: int32, 0, 32) {
#       for (x.inner: int32, 0, 32) {
#         for (y.inner: int32, 0, 32) {
#           C_2[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)] = ((float32*)A_2[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)] + (float32*)B_2[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)])
#         }
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
#   for (x.outer: int32, 0, 32) {
#     for (y.outer: int32, 0, 32) {
#       for (x.inner: int32, 0, 32) {
#         C_2[ramp((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)), 1, 32)] = ((float32x32*)A_2[ramp((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)), 1, 32)] + (float32x32*)B_2[ramp((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)), 1, 32)])
#       }
#     }
#   }
# }