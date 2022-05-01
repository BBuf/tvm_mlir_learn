
import tvm
import tvm.testing
from tvm import te, auto_scheduler
import numpy
import timeit
import os

# The size of the matrix
# (M, K) x (K, N)
M = 1024
K = 1024
N = 1024

target = "llvm -mcpu=core-avx2"
dev = tvm.device(target, 0)

EVAL_REPEAT_TIME = 500

# 参考reference [4] 定义矩阵乘法运算
# 计算C(M, N) = A(M, K) x B(K, N)
@auto_scheduler.register_workload
def matmul(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )

    return [A, B, C]

def autotune(M, N, K, dtype, target_name, log_file):
    print(target_name)
    target = tvm.target.Target(target_name)
    task = tvm.auto_scheduler.SearchTask(func=matmul, args=(M, N, K, dtype), target=target)

    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    tune_option = None
    measure_ctx = None
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=6000,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )

    task.tune(tune_option)
    sch, args = task.apply_best(log_file)

# 检查矩阵乘法结果是否正确，并返回乘法函数
def get_matmul_func(M, N, K, dtype, target_name, log_file):
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

    answer = numpy.dot(a.numpy(), b.numpy())
    
    target = tvm.target.Target(target_name)
    task = tvm.auto_scheduler.SearchTask(func=matmul, args=(M, N, K, dtype), target=target)
    sch, args = task.apply_best(log_file)
    func = tvm.build(sch, args, target=target, name="matmul")
    assert func

    # print(tvm.lower(sch, args, simple_mode=True))
    # print(func.get_source("asm"))
    # func.export_library("tvm_autoscheduler.so")

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)
    
    return func

def benchmark(matmul_func, dtype):
    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

    np_repeat = EVAL_REPEAT_TIME
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
    print("Numpy running time: %f" % (np_runing_time / np_repeat))

    answer = numpy.dot(a.numpy(), b.numpy())

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    matmul_func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    evaluator = matmul_func.time_evaluator(matmul_func.entry_name, dev, number=EVAL_REPEAT_TIME)
    print("TVM autoscheduler tuned: %f" % evaluator(a, b, c).mean)

def main(argv):       
    if (len(argv) > 1 and argv[1] == 'float32'):
        dtype = "float32"
        log_file = "matmul_autoscheduler_32.json"
    else:
        dtype = "float64"
        log_file = "matmul_autoscheduler_64.json"
    
    if (len(argv) == 3 and argv[2] == 'tune'):
        autotune(M, N, K, dtype, target, log_file)
    
    func = get_matmul_func(M, N, K, dtype, target, log_file)
    benchmark(func, dtype)

import sys
if __name__ == '__main__':
    main(sys.argv)

