# create_group对从inputs到outputs的所有stage创建group，group本质上是一个虚拟stage，
# 可以通过操作这个虚拟stage来一起操作这个group里的所有stage。本例中，通过compute_at
# 使这个group中的D和E，一起附着到指定操作中。
import tvm
from tvm import te

n = 1024
k = te.reduce_axis((0, n), name='k')

A = te.placeholder((n, n), name='A')
B = te.placeholder((n, n), name='B')

D = te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='D')
E = te.compute((n, n), lambda i, j: D[i, j] + B[i, j], name='E')
F = te.compute((n,), lambda i: te.sum(E[i, k], axis=k), name='F')

s = te.create_schedule(F.op)

print(tvm.lower(s, [A, B, E], simple_mode=True))
print("---------cutting line---------")

g = s.create_group(outputs = E, inputs = [A, B], include_inputs=True)
g.compute_at(s[F], F.op.reduce_axis[0])

print(tvm.lower(s, [A, B, E], simple_mode=True))

# primfn(A_1: handle, B_1: handle, E_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {E: Buffer(E_2: Pointer(float32), float32, [1024, 1024], []),
#              B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, B_1: B, E_1: E} {
#   attr [D: Pointer(float32)] "storage_scope" = "global";
#   allocate(D, float32, [1048576]);
#   attr [F: Pointer(float32)] "storage_scope" = "global";
#   allocate(F, float32, [1024]) {
#     for (i: int32, 0, 1024) {
#       for (j: int32, 0, 1024) {
#         D[((i*1024) + j)] = ((float32*)A_2[((i*1024) + j)] + (float32*)B_2[((i*1024) + j)])
#       }
#     }
#     for (i_1: int32, 0, 1024) {
#       for (j_1: int32, 0, 1024) {
#         E_2[((i_1*1024) + j_1)] = ((float32*)D[((i_1*1024) + j_1)] + (float32*)B_2[((i_1*1024) + j_1)])
#       }
#     }
#     for (i_2: int32, 0, 1024) {
#       F[i_2] = 0f32
#       for (k: int32, 0, 1024) {
#         F[i_2] = ((float32*)F[i_2] + (float32*)E_2[((i_2*1024) + k)])
#       }
#     }
#   }
# }


# ---------cutting line---------
# primfn(A_1: handle, B_1: handle, E_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {E: Buffer(E_2: Pointer(float32), float32, [1024, 1024], []),
#              B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
#   buffer_map = {A_1: A, B_1: B, E_1: E} {
#   attr [F: Pointer(float32)] "storage_scope" = "global";
#   allocate(F, float32, [1024]);
#   attr [D: Pointer(float32)] "storage_scope" = "global";
#   allocate(D, float32, [1]);
#   for (i: int32, 0, 1024) {
#     F[i] = 0f32
#     for (k: int32, 0, 1024) {
#       D[0] = ((float32*)A_2[((i*1024) + k)] + (float32*)B_2[((i*1024) + k)])
#       E_2[((i*1024) + k)] = ((float32*)D[0] + (float32*)B_2[((i*1024) + k)])
#       F[i] = ((float32*)F[i] + (float32*)E_2[((i*1024) + k)])
#     }
#   }
# }
