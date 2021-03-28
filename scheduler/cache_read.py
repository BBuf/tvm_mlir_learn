# cache_read将tensor读入指定存储层次scope的cache，这个设计的意义在于显
# 式利用现有计算设备的on-chip memory hierarchy。这个例子中，会先将A的数
# 据load到shared memory中，然后计算B。在这里，我们需要引入一个stage的概念，
# 一个op对应一个stage，也就是通过cache_read会新增一个stage。

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

AA = s.cache_read(A, "shared", [B])

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