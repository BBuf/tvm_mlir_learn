# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _tutorial-tensor-expr-get-started:

Get Started with Tensor Expression
==================================
**Author**: `Tianqi Chen <https://tqchen.github.io>`_

This is an introductory tutorial to the Tensor expression language in TVM.
TVM uses a domain specific tensor expression for efficient kernel construction.

In this tutorial, we will demonstrate the basic workflow to use
the tensor expression language.
"""
from __future__ import absolute_import, print_function

import tvm
import tvm.testing
from tvm import te
import numpy as np

# 全局环境定义

tgt_host = "llvm"
# 如果启用了GPU，则将其更改为相应的GPU，例如：cuda、opencl、rocm
tgt = "cuda"

######################################################################
# Vector Add Example
# ------------------
# 在本教程中，我们将使用向量加法示例演示工作流。
#

######################################################################
# 作为第一步，我们需要描述我们的计算。TVM采用张量语义，每个中间结果表示为一个多维数组。用户需要描述生成张量的计算规则。
# 我们首先定义一个符号变量n来表示形状。然后我们定义两个占位符张量，A和B，具有给定的形状（n，）
# 然后我们用一个计算函数来描述结果张量C。计算函数采用张量的形式，以及描述张量每个位置的计算规则的lambda函数。
# 在这个阶段没有计算发生，因为我们只是声明应该如何进行计算。
#
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))

######################################################################
# 调度计算
# 虽然上面的几行描述了计算规则，但是我们可以用很多方法来计算C，因为C可以在轴上用数据并行的方式来计算。TVM要求用户提供一个称为schedule的计算描述。
# schedule是程序中变换计算循环的一组集合。在我们构造了schedule之后，默认情况下，schedule以串行方式按行的主要顺序计算C。
#
# .. code-block:: c
#
#   for (int i = 0; i < n; ++i) {
#     C[i] = A[i] + B[i];
#   }
#
s = te.create_schedule(C.op)

######################################################################
# 我们调用`te.create_schedule`来创建scheduler，然后使用split构造来拆分C的第一个轴，
# 这将把原来的一个迭代轴拆分成两个迭代轴的乘积
#
# .. code-block:: c
#
#   for (int bx = 0; bx < ceil(n / 64); ++bx) {
#     for (int tx = 0; tx < 64; ++tx) {
#       int i = bx * 64 + tx;
#       if (i < n) {
#         C[i] = A[i] + B[i];
#       }
#     }
#   }
#
bx, tx = s[C].split(C.op.axis[0], factor=64)

######################################################################
# 最后，我们将迭代轴bx和tx绑定到GPU计算grid中的线程。这些是特定于GPU的构造，允许我们生成在GPU上运行的代码。
#
if tgt == "cuda" or tgt == "rocm" or tgt.startswith("opencl"):
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))

######################################################################
# Compilation
# 上面我们已经完成了指定scheduler，接下来我们就可以将上面的所有代码编译成一个TVM的函数了。
# 默认情况下，TVM会将其编译成一个类型擦除函数，可以直接从Python端调用。下面我们使用`tvm,build`来创建一个编译函数，
# 编译函数接收scheduler，函数签名（包含输入输出）以及我们需要编译到的目标语言。编译`fadd`的结果是一个GPU设备函数
# （如果涉及GPU）以及一个调用GPU函数的host端包装器。`fadd`是生成的主机包装函数，它在内部包含对生成的设备函数的引用。
#
fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

######################################################################
# 编译后的TVM函数公开了一个简洁的C API，可以被任何语言调用。TVM在python中提供了一个最小
# 的array API来帮助快速测试和原型开发。阵列API基于DLPack标准。这个array API基
# 于https://github.com/dmlc/dlpack 标准。要运行这个函数，首先需要创建一个GPU context，
# 然后使用`tvm.nd.array`将数据拷贝到GPU，再使用我们编译好的函数`fadd`来执行计算，最后
# `asnumpy()`将GPU端的array拷贝回CPU使用numpy进行计算，最后比较两者的差距。这部分的代码如下：
#
ctx = tvm.context(tgt, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# Inspect the Generated Code
# --------------------------
# You can inspect the generated code in TVM. The result of tvm.build
# is a TVM Module. fadd is the host module that contains the host wrapper,
# it also contains a device module for the CUDA (GPU) function.
#
# The following code fetches the device module and prints the content code.
#

if tgt == "cuda" or tgt == "rocm" or tgt.startswith("opencl"):
    dev_module = fadd.imported_modules[0]
    print("-----GPU code-----")
    print(dev_module.get_source())
else:
    print(fadd.get_source())

######################################################################
# .. note:: Code Specialization
#
#   As you may have noticed, the declarations of A, B and C all
#   take the same shape argument, n. TVM will take advantage of this
#   to pass only a single shape argument to the kernel, as you will find in
#   the printed device code. This is one form of specialization.
#
#   On the host side, TVM will automatically generate check code
#   that checks the constraints in the parameters. So if you pass
#   arrays with different shapes into fadd, an error will be raised.
#
#   We can do more specializations. For example, we can write
#   :code:`n = tvm.runtime.convert(1024)` instead of :code:`n = te.var("n")`,
#   in the computation declaration. The generated function will
#   only take vectors with length 1024.
#

######################################################################
# Save Compiled Module
# --------------------
# Besides runtime compilation, we can save the compiled modules into
# a file and load them back later. This is called ahead of time compilation.
#
# The following code first performs the following steps:
#
# - It saves the compiled host module into an object file.
# - Then it saves the device module into a ptx file.
# - cc.create_shared calls a compiler (gcc) to create a shared library
#
from tvm.contrib import cc
from tvm.contrib import utils

temp = utils.tempdir()
fadd.save(temp.relpath("myadd.o"))
if tgt == "cuda":
    fadd.imported_modules[0].save(temp.relpath("myadd.ptx"))
if tgt == "rocm":
    fadd.imported_modules[0].save(temp.relpath("myadd.hsaco"))
if tgt.startswith("opencl"):
    fadd.imported_modules[0].save(temp.relpath("myadd.cl"))
cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])
print(temp.listdir())

######################################################################
# .. note:: Module Storage Format
#
#   The CPU (host) module is directly saved as a shared library (.so).
#   There can be multiple customized formats of the device code.
#   In our example, the device code is stored in ptx, as well as a meta
#   data json file. They can be loaded and linked separately via import.
#

######################################################################
# Load Compiled Module
# --------------------
# We can load the compiled module from the file system and run the code.
# The following code loads the host and device module separately and
# re-links them together. We can verify that the newly loaded function works.
#
fadd1 = tvm.runtime.load_module(temp.relpath("myadd.so"))
if tgt == "cuda":
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.ptx"))
    fadd1.import_module(fadd1_dev)

if tgt == "rocm":
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.hsaco"))
    fadd1.import_module(fadd1_dev)

if tgt.startswith("opencl"):
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.cl"))
    fadd1.import_module(fadd1_dev)

fadd1(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# Pack Everything into One Library
# --------------------------------
# In the above example, we store the device and host code separately.
# TVM also supports export everything as one shared library.
# Under the hood, we pack the device modules into binary blobs and link
# them together with the host code.
# Currently we support packing of Metal, OpenCL and CUDA modules.
#
fadd.export_library(temp.relpath("myadd_pack.so"))
fadd2 = tvm.runtime.load_module(temp.relpath("myadd_pack.so"))
fadd2(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# .. note:: Runtime API and Thread-Safety
#
#   The compiled modules of TVM do not depend on the TVM compiler.
#   Instead, they only depend on a minimum runtime library.
#   The TVM runtime library wraps the device drivers and provides
#   thread-safe and device agnostic calls into the compiled functions.
#
#   This means that you can call the compiled TVM functions from any thread,
#   on any GPUs.
#

######################################################################
# Generate OpenCL Code
# --------------------
# TVM provides code generation features into multiple backends,
# we can also generate OpenCL code or LLVM code that runs on CPU backends.
#
# The following code blocks generate OpenCL code, creates array on an OpenCL
# device, and verifies the correctness of the code.
#
if tgt.startswith("opencl"):
    fadd_cl = tvm.build(s, [A, B, C], tgt, name="myadd")
    print("------opencl code------")
    print(fadd_cl.imported_modules[0].get_source())
    ctx = tvm.cl(0)
    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
    fadd_cl(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

######################################################################
# Summary
# -------
# This tutorial provides a walk through of TVM workflow using
# a vector add example. The general workflow is
#
# - Describe your computation via a series of operations.
# - Describe how we want to compute use schedule primitives.
# - Compile to the target function we want.
# - Optionally, save the function to be loaded later.
#
# You are more than welcome to checkout other examples and
# tutorials to learn more about the supported operations, scheduling primitives
# and other features in TVM.
#
