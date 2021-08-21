import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor

# build model
m = 24
n = 24
k = 64

x = relay.var("x", relay.TensorType((m, k), dtype='float32'))
y = relay.var("y", relay.TensorType((n, k), dtype='float32'))
z = relay.nn.dense(x, y)

net = relay.Function([x, y], z)

# build and lowering
module = tvm.IRModule.from_expr(net)
lib = relay.build(module, "llvm")


dev = tvm.cpu(0)
input1 = tvm.nd.array(np.random.uniform(size=[m, k]).astype('float32'), dev)
input2 = tvm.nd.array(np.random.uniform(size=[n, k]).astype('float32'), dev)
m = graph_executor.GraphModule(lib["default"](dev))
# set inputs
m.set_input("x", input1)
m.set_input("y", input2)
# execute
m.run()
# get outputs
tvm_output = m.get_output(0)

print(net.body)
# free_var %x: Tensor[(2), float32];
# nn.softmax(%x)
print(module)
# def @main(%x: Tensor[(2), float32]) {
#   nn.softmax(%x)
# }
