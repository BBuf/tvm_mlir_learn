# tensorize将计算作为整体，编译为一个tensor_intrin函数中。这是因为很多计算属于常用计算，
# 针对这些计算已经有了很好的built-in的schedule，通过tensorize可以直接调用这些内置的in
# trinsic，其实这也就是intrinsic在计算机科学中的本意[https://en.wikipedia.org/wiki/Intrinsic_function]。

import tvm
from tvm import te

N, M, L = 1024, 512, 64
A = te.placeholder((N, L), name='A')
B = te.placeholder((M, L), name='B')
k = te.reduce_axis((0, L), name='k')
C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[j, k], axis=k), name='C')
s = te.create_schedule(C.op)

def intrin_gemv(m, l):
    a = te.placeholder((l,), name='a')
    b = te.placeholder((m, l), name='b')
    k = te.reduce_axis((0, l), name='k')
    c =  te.compute((m,), lambda i: te.sum(a[k] * b[i, k], axis=k), name='c')
    Abuf = tvm.tir.decl_buffer(a.shape, a.dtype, name='A', offset_factor=1, strides=[1])
    Bbuf = tvm.tir.decl_buffer(b.shape, b.dtype, name='B', offset_factor=1, strides=[te.var("s1"), 1])
    Cbuf = tvm.tir.decl_buffer(c.shape, c.dtype, name='C', offset_factor=1, strides=[1])
    
    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(tvm.tir.call_extern("int32", "gemv_update", cc.access_ptr("w"), aa.access_ptr("r"), bb.access_ptr("r"), m, l, bb.strides[0]))
        return ib.get()
    
    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Abuf, b: Bbuf, c: Cbuf})

factor = 16
x, y = C.op.axis
z, = C.op.reduce_axis
yo, yi = s[C].split(y, factor=factor)
s[C].reorder(x, yo, yi, z)

gemv = intrin_gemv(factor, L)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].tensorize(yi, gemv)

print(tvm.lower(s, [A, B, C], simple_mode=True))


# primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 512], []),
#              B: Buffer(B_2: Pointer(float32), float32, [512, 64], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 64], [])}
#   buffer_map = {A_1: A, B_1: B, C_1: C} {
#   for (i: int32, 0, 1024) {
#     for (j.outer: int32, 0, 32) {
#       for (j.inner: int32, 0, 16) {
#         C_2[(((i*512) + (j.outer*16)) + j.inner)] = 0f32
#         for (k: int32, 0, 64) {
#           C_2[(((i*512) + (j.outer*16)) + j.inner)] = ((float32*)C_2[(((i*512) + (j.outer*16)) + j.inner)] + ((float32*)A_2[((i*64) + k)]*(float32*)B_2[(((j.outer*1024) + (j.inner*64)) + k)]))
#         }
#       }
#     }
#   }
# }


# ---------cutting line---------
# primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
#   attr = {"global_symbol": "main", "tir.noalias": True}
#   buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 512], []),
#              B: Buffer(B_2: Pointer(float32), float32, [512, 64], []),
#              A: Buffer(A_2: Pointer(float32), float32, [1024, 64], [])}
#   buffer_map = {A_1: A, B_1: B, C_1: C} {
#   for (i: int32, 0, 1024) {
#     for (j.outer: int32, 0, 32) {
#       @tir.call_extern("gemv_update", @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), C_2, ((i*512) + (j.outer*16)), 16, 2, dtype=handle), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), A_2, (i*64), 64, 1, dtype=handle), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), B_2, (j.outer*1024), 1024, 1, dtype=handle), 16, 64, 64, dtype=int32)
#     }
#   }
# }