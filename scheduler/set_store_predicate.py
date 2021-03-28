# set_store_predicate设置了store的条件，适用于在多线程调度中预防写操作之间的冲突。
import tvm
from tvm import te

n = 1024
A = te.placeholder((n,), name='A')
k = te.reduce_axis((0, n), 'k')
B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
BF = s.rfactor(B, ki)
tx = te.thread_axis("threadIdx.x")
s[BF].compute_at(s[B], s[B].op.reduce_axis[0])

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].set_store_predicate(tx.var.equal(0))

print(tvm.lower(s, [A, B], simple_mode=True))

# primfn(A_1: handle, B_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {B: Buffer(B_2: Pointer(float32), float32, [1], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024], [])}
#   buffer_map = {A_1: A, B_1: B} {
#   attr [B.rf: Pointer(float32)] "storage_scope" = "global";
#   allocate(B.rf, float32, [1]) {
#     B_2[0] = 0f32
#     for (k.inner.v: int32, 0, 16) {
#       B.rf[0] = 0f32
#       for (k.outer: int32, 0, 64) {
#         B.rf[0] = ((float32*)B.rf[0] + (float32*)A_2[((k.outer*16) + k.inner.v)])
#       }
#       B_2[0] = ((float32*)B_2[0] + (float32*)B.rf[0])
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
#   allocate(B.rf, float32, [1]) {
#     B_2[0] = 0f32
#     for (k.inner.v: int32, 0, 16) {
#       B.rf[0] = 0f32
#       for (k.outer: int32, 0, 64) {
#         B.rf[0] = ((float32*)B.rf[0] + (float32*)A_2[((k.outer*16) + k.inner.v)])
#       }
#       if (threadIdx.x: int32 == 0) {
#         B_2[0] = ((float32*)B_2[0] + (float32*)B.rf[0])
#       }
#     }
#   }
# }
