import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import dense

# build model
m = 24
n = 24
k = 64

@auto_scheduler.register_workload
def gemm():
    input = te.placeholder((m, k), name="input")
    weight = te.placeholder((n, k), name="weight")
    output = topi.nn.dense(input, weight)
    return [input, weight, output]

target = tvm.target.Target("llvm")


task = auto_scheduler.SearchTask(
    func=gemm, target=target
)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

log_file = "gemm.json"
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run auto-tuning (search)
task.tune(tune_option)
# Apply the best schedule
sch, args = task.apply_best(log_file)

# Kill the measurement process
del measure_ctx

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))


func = tvm.build(sch, args, target)

# Check correctness
input_np = np.random.uniform(size=(m, k)).astype(np.float32)
weight_np = np.random.uniform(size=(n, k)).astype(np.float32)
out_np = np.matmul(input_np, weight_np)

dev = tvm.cpu()
input_tvm = tvm.nd.array(input_np, device=dev)
weight_tvm = tvm.nd.array(weight_np, device=dev)
out_tvm = tvm.nd.empty(out_np.shape, device=dev)

func(input_tvm, weight_tvm, out_tvm)

# Check results
np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

# Evaluate execution time
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(input_tvm, weight_tvm, out_tvm).results) * 1000)
)


