import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor

# build model
n = 2
x = relay.var("x", shape=(n,), dtype='float32')
y = relay.nn.softmax(x)
net = relay.Function([x], y)

# build and lowering
module = tvm.IRModule.from_expr(net)
lib = relay.build(module, "llvm")


dev = tvm.cpu(0)
input = tvm.nd.array(np.random.uniform(size=[n]).astype('float32'), dev)
m = graph_executor.GraphModule(lib["default"](dev))
# set inputs
m.set_input("x", input)
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
