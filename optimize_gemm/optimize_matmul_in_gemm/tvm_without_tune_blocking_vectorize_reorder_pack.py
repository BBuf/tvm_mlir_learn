# 21.227 GFLOPS

import os
import tvm
import tvm.testing
from tvm import te
import numpy
import timeit

os.environ['TVM_NUM_THREADS']=str(1)

# The size of the matrix
# (M, K) x (K, N)
M = 1024
K = 1024
N = 1024
GFLOPS = 2 * M * K * N * 1e-9

target = "llvm -mcpu=core-avx2"
dev = tvm.device(target, 0)

# 计算C(M, N) = A(M, K) x B(K, N)
def matmul(M, N, K, dtype):
    # Algorithm
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    bn = 32
    kfactor = 4

    packedB = te.compute(
    (N / bn, K, bn), lambda bigN, k, littleN: B[k, bigN * bn + littleN], name="packedB"
    )
    C = te.compute(
        (M, N),
        lambda m, n: te.sum(A[m, k] * packedB[n // bn, k, tvm.tir.indexmod(n, bn)], axis=k),
        name="C",
    )

    s = te.create_schedule(C.op)

    mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (kaxis,) = s[C].op.reduce_axis
    ko, ki = s[C].split(kaxis, factor=kfactor)

    s[C].reorder(mo, no, ko, mi, ki, ni)
    s[C].vectorize(ni)

    bigN, _, littleN = s[packedB].op.axis
    s[packedB].vectorize(littleN)
    s[packedB].parallel(bigN)
    return s, [A, B, C]

# 检查矩阵乘法结果是否正确，并返回乘法函数
def get_matmul_func(sch, args, dtype):
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

    answer = numpy.dot(a.numpy(), b.numpy())

    opt_level = 3
    with tvm.transform.PassContext(opt_level=opt_level):
        func = tvm.build(sch, args, target=target, name="mmult")
    assert func

    print(tvm.lower(sch, args, simple_mode=True))
    # print(func.get_source("asm"))

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)
    
    return func

def benchmark(matmul_func, dtype):
    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

    np_repeat = 500
    np_runing_time = timeit.timeit(
        setup="import numpy\n"
        "M = " + str(M) + "\n"
        "K = " + str(K) + "\n"
        "N = " + str(N) + "\n"
        'dtype = "' + str(dtype) + '"\n'
        "a = numpy.random.rand(M, K).astype(dtype)\n"
        "b = numpy.random.rand(K, N).astype(dtype)\n",
        stmt="answer = numpy.dot(a, b)",
        number=np_repeat,
    )
    # print("Numpy running time: %f" % (np_runing_time / np_repeat))

    answer = numpy.dot(a.numpy(), b.numpy())

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    matmul_func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    evaluator = matmul_func.time_evaluator(matmul_func.entry_name, dev, number=500)

    tvm_time = evaluator(a, b, c).mean
    print("TVM without tune: %f" % tvm_time)
    tvm_glops = (GFLOPS / tvm_time)
    print(f'TVM Without Tune GFLOPS: {tvm_glops}')

def main(argv):
    dtype = "float32"
    sch, args = matmul(M, N, K, dtype)
    func = get_matmul_func(sch, args, dtype)
    benchmark(func, dtype)
    # print(tvm.lower(sch, args, simple_mode=True))

import sys
if __name__ == '__main__':
    main(sys.argv)
