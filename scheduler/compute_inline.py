# compute_inline把独立的计算操作转化成内联函数形式，在使用到原
# 计算结果时再调用内联函数完成运算，通过compute_inline来减少一个stage。
import tvm
from tvm import te

n = 1024
k = 3
pad = 2
A = te.placeholder((n, n), name='A')
W = te.placeholder((k, k), name='W')
m = (n - k + 2 * pad) + 1
Apad = te.compute((n + 2 * pad, n + 2 * pad),
                lambda yy, xx: te.if_then_else(
                    te.all(yy >= pad, yy < pad + n, xx >= pad, xx < pad + n), 
                    A[yy - pad, xx - pad], 0.0),
                    name='Apad')

ry = te.reduce_axis((0, k), name='ry')
rx = te.reduce_axis((0, k), name='rx')

B = te.compute((m, m),
                lambda yy, xx: 
                    te.sum(Apad[yy + ry, xx + rx] * W[ry, rx],
                    axis=[ry, rx]),
                    name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, W, B], simple_mode=True))
print("---------cutting line---------")

s[Apad].compute_inline()

print(tvm.lower(s, [A, W, B], simple_mode=True))
exit(0)

# primfn(A_1: handle, W_1: handle, B_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {B: Buffer(B_2: Pointer(float32), float32, [1026, 1026], []),
#              W: Buffer(W_2: Pointer(float32), float32, [3, 3], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, W_1: W, B_1: B} {
#   attr [Apad: Pointer(float32)] "storage_scope" = "global";
#   allocate(Apad, float32, [1056784]) {
#     for (yy: int32, 0, 1028) {
#       for (xx: int32, 0, 1028) {
#         Apad[((yy*1028) + xx)] = @tir.if_then_else(((((2 <= yy) && (yy < 1026)) && (2 <= xx)) && (xx < 1026)), (float32*)A_2[(((yy*1024) + xx) - 2050)], 0f32, dtype=float32)
#       }
#     }
#     for (yy_1: int32, 0, 1026) {
#       for (xx_1: int32, 0, 1026) {
#         B_2[((yy_1*1026) + xx_1)] = 0f32
#         for (ry: int32, 0, 3) {
#           for (rx: int32, 0, 3) {
#             B_2[((yy_1*1026) + xx_1)] = ((float32*)B_2[((yy_1*1026) + xx_1)] + ((float32*)Apad[((((yy_1*1028) + (ry*1028)) + xx_1) + rx)]*(float32*)W_2[((ry*3) + rx)]))
#           }
#         }
#       }
#     }
#   }
# }


# ---------cutting line---------
# primfn(A_1: handle, W_1: handle, B_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {B: Buffer(B_2: Pointer(float32), float32, [1026, 1026], []),
#              W: Buffer(W_2: Pointer(float32), float32, [3, 3], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, W_1: W, B_1: B} {
#   for (yy: int32, 0, 1026) {
#     for (xx: int32, 0, 1026) {
#       B_2[((yy*1026) + xx)] = 0f32
#       for (ry: int32, 0, 3) {
#         for (rx: int32, 0, 3) {
#           B_2[((yy*1026) + xx)] = ((float32*)B_2[((yy*1026) + xx)] + (@tir.if_then_else(((((2 <= (yy + ry)) && ((yy + ry) < 1026)) && (2 <= (xx + rx))) && ((xx + rx) < 1026)), (float32*)A_2[(((((yy*1024) + (ry*1024)) + xx) + rx) - 2050)], 0f32, dtype=float32)*(float32*)W_2[((ry*3) + rx)]))
#         }
#       }
#     }
#   }
# }