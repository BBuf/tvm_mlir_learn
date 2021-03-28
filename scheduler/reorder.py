# reorder用于重置循环iter的内外顺序，根据局部性原理，最大化利用cache中的现有数据，
# 减少反复载入载出的情况。注意，这里到底怎样的顺序是最优化的是一个很有趣的问题。
# 以矩阵乘法为例，M, N, K三维，往往是将K放在最外层可以最大程度利用局部性。这个具体例子，具体探究。
import tvm
from tvm import te

n = 1024
A = te.placeholder((n, n), name='A')
B = te.placeholder((n,n), name='B')
C = te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='C')

s = te.create_schedule(C.op)

xo, xi = s[C].split(s[C].op.axis[0], factor=32)
yo, yi = s[C].split(s[C].op.axis[1], factor=32)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].reorder(xo, yo, yi, xi)

print(tvm.lower(s, [A, B, C], simple_mode=True))