# compute_at将当前的stage附着到目标stage的指定iter方向上，
# 同时与目标stage采用相同的并行方式，在其内部完成当前stage
# 的计算。往往compute_at会与cache_read和cache_write一起使用。
import tvm
from tvm import te

n = 1024
A = te.placeholder((n,), name='A')
k = te.reduce_axis((0, n), 'k')
B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)
ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)
BF = s.rfactor(B, ki)

tx = te.thread_axis("threadIdx.x")
s[B].bind(s[B].op.reduce_axis[0], tx)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[BF].compute_at(s[B], s[B].op.reduce_axis[0])

print(tvm.lower(s, [A, B], simple_mode=True))

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
#     attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
#     attr [reduce_temp0: Pointer(float32)] "storage_scope" = "local";
#     allocate(reduce_temp0, float32, [1]) {
#       attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
#       @tir.tvm_thread_allreduce(1u32, (float32*)B.rf[threadIdx.x], True, reduce_temp0, threadIdx.x, dtype=handle)
#       B_2[0] = (float32*)reduce_temp0[0]
#     }
#   }
# }


# ---------cutting line---------
# primfn(A_1: handle, B_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {B: Buffer(B_2: Pointer(float32), float32, [1], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024], [])}
#   buffer_map = {A_1: A, B_1: B} {
#   attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
#   attr [B.rf: Pointer(float32)] "storage_scope" = "local";
#   allocate(B.rf, float32, [1]);
#   attr [reduce_temp0: Pointer(float32)] "storage_scope" = "local";
#   allocate(reduce_temp0, float32, [1]) {
#     B.rf[0] = 0f32
#     for (k.outer: int32, 0, 32) {
#       B.rf[0] = ((float32*)B.rf[0] + (float32*)A_2[((k.outer*32) + threadIdx.x)])
#     }
#     attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
#     @tir.tvm_thread_allreduce(1u32, (float32*)B.rf[0], True, reduce_temp0, threadIdx.x, dtype=handle)
#     B_2[0] = (float32*)reduce_temp0[0]
#   }
# }