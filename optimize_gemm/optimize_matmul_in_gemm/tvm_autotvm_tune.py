import tvm
import tvm.testing
from tvm import te
import numpy
import timeit

from tvm import te, autotvm, auto_scheduler
import os
import sys
import logging

os.environ['TVM_NUM_THREADS']=str(1)

# The size of the matrix
# (M, K) x (K, N)
M = 1024
K = 1024
N = 1024
GFLOPS = 2 * M * K * N * 1e-9

target = "llvm -mcpu=core-avx2"
dev = tvm.device(target, 0)

EVAL_REPEAT_TIME = 500

# 计算C(M, N) = A(M, K) x B(K, N)
@autotvm.template("tutorial/matmul")
def matmul(M, N, K, dtype):
    # Algorithm
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    
    cfg = autotvm.get_config()
    cfg.define_split("tile_x", M, num_outputs=2)
    cfg.define_split("tile_y", N, num_outputs=2)
    cfg.define_split("tile_k", K, num_outputs=2)

    bn = cfg["tile_y"].size[-1]
    packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    C = te.compute((M, N),
                    lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
                    name = 'C')
    s = te.create_schedule(C.op)

    CC = s.cache_write(C, "global")

    mo, mi = cfg["tile_x"].apply(s, C, C.op.axis[0])
    no, ni = cfg["tile_y"].apply(s, C, C.op.axis[1])
    s[C].reorder(mo, no, mi, ni)

    s[CC].compute_at(s[C], no)

    mc, nc = s[CC].op.axis
    (kaxis,) = s[CC].op.reduce_axis
    ko, ki = cfg["tile_k"].apply(s, CC, kaxis)
    
    cfg.define_reorder("reorder", [mc, ki, nc], "all")
    cfg["reorder"].apply(s, CC, [mc, ki, nc])
    cfg.define_annotate('ann', [mc, ki, nc], policy='try_unroll_vec')
    cfg['ann'].apply(s, CC, [mc, ki, nc])

    # parallel
    s[C].parallel(mo)
    s[C].unroll(mi)
    s[C].vectorize(ni)

    bigN, _, littleN = s[packedB].op.axis
    s[packedB].vectorize(littleN)
    s[packedB].parallel(bigN)

    return s, [A, B, C]

def tune_matmul(dtype, log_file):
    if (dtype == 'float32'):
        log_tmp_file = "matmul_autotvm_32.log.tmp"
    else:
        log_tmp_file = "matmul_autotvm_64.log.tmp"

    task = autotvm.task.create("tutorial/matmul", args=(N, K, M, dtype), target=target)
    print(task.config_space)

    # logging config (for printing tuning log to the screen)
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))


    measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=5))

    # Begin tuning with RandomTuner, log records to file `matmul_autotvm.log`
    # You can use alternatives like XGBTuner.

    # begin tuning, log records to file `matmul_autotvm.log`
    # tuner = autotvm.tuner.GridSearchTuner(task)
    tuner = autotvm.tuner.XGBTuner(task)
    n_trial = 6000
    early_stopping = 800
    if os.path.exists(log_tmp_file):
        os.remove(log_tmp_file)
    tuner.tune(n_trial=n_trial,
            early_stopping=early_stopping,        
            measure_option=measure_option,
            callbacks=[autotvm.callback.progress_bar(n_trial),
                        autotvm.callback.log_to_file(log_tmp_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(log_tmp_file, log_file)


# 检查矩阵乘法结果是否正确，并返回乘法函数
def get_matmul_func(dtype, log_file):
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

    answer = numpy.dot(a.numpy(), b.numpy())

    with autotvm.apply_history_best(log_file):
        with tvm.target.Target(target):
            s, arg_bufs = matmul(N, K, M, dtype)
            func = tvm.build(s, arg_bufs, target=target, name="matmul")
    assert func

    # print(tvm.lower(s, arg_bufs, simple_mode=True))
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
    tvm_time = evaluator(a, b, c).mean
    print("TVM autotvm tuned: %f" % tvm_time)
    tvm_glops = (GFLOPS / tvm_time)
    print(f'TVM autotvm tuned GFLOPS: {tvm_glops}')

def main(argv):
    if (len(argv) > 1 and argv[1] == 'float32'):
        dtype = "float32"
        log_file = "matmul_autotvm_32.log"
    else:
        dtype = "float64"
        log_file = "matmul_autotvm_64.log"

    if (len(argv) == 3 and argv[2] == 'tune'):
        tune_matmul(dtype, log_file)
        
    func = get_matmul_func(dtype, log_file)
    benchmark(func, dtype)

import sys
if __name__ == '__main__':
    main(sys.argv)
