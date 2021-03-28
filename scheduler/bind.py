# bind将iter绑定到block或thread的index上，从而把循环的任务分
# 配到线程，实现并行化计算，这是针对CUDA后端最核心的部分。

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

s[B].bind(ko, te.thread_axis("blockIdx.x"))
s[B].bind(ki, te.thread_axis("threadIdx.x"))

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
#   attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 32;
#   attr [reduce_temp0: Pointer(float32)] "storage_scope" = "local";
#   allocate(reduce_temp0, float32, [1]);
#   attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32 {
#     attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
#     @tir.tvm_thread_allreduce(1u32, (float32*)A_2[((blockIdx.x*32) + threadIdx.x)], True, reduce_temp0, blockIdx.x, threadIdx.x, dtype=handle)
#     B_2[0] = (float32*)reduce_temp0[0]
#   }
# }