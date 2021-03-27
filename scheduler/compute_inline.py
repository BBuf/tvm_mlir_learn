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