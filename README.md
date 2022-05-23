# tvm_mlir_learn

## preoject introduction

- `scheduler` TVM 中 scheduler 详细举例，这里将 https://zhuanlan.zhihu.com/p/94846767 这篇文章的例子用TVM 0.8.0.dev 重写。
- `dataflow_controlflow` 数据流和控制流的区别示例，这里是Pytorch为例子。
- `ansor` Ansor这篇OSDI论文的翻译以及基于Ansor做一些实验。
- `relay` TVM 中一些 Relay 相关的示例，比如如何自定义 Pass，如何在 Jetson Nano 中运行DarkNet的YOLO模型等。 
- `codegen` TVM 中 Codegen 相关示例，基于张量表达式和Relay IR。
- `torchscript` Pytorch的TorchScript的用法。
- compile_tvm_in_docker.md 。在Docker中编译TVM。
- `tvm_pytorch_resnet18_inference.py` 使用 TVM 在 X86 CPU 上运行 Pytorch 的 ResNet18 模型。
- `tvm_onnx_resnet18_inference.py` TVM 加载 ResNet18 的 ONNX 模型进行推理。
- `pytorch_resnet18_export_onnx.py` Pytorch 导出 ResNet18 的 ONNX 模型示例。
- `optimize_gemm` 让深度学习编译器来指导我们写代码，以GEMM为例。

## video collection

- [What Is MLIR && What Is TVM？](https://mp.weixin.qq.com/s/Xj2iW9tFUGidlzLqEzoixQ)
- [TVM Conf 2020 - An Introduction to TVM Part1](https://mp.weixin.qq.com/s/NaMxlNzPrRlBYJfJ7ivjuw)
- [TVM Conf 2020 - An Introduction to TVM Part2](https://mp.weixin.qq.com/s/KAG0DjnhQcGEJa-hRFiBfg)
- [Torch MLIR公开会议翻译视频（自制中英双字完整版）](https://mp.weixin.qq.com/s/d0jJFYdUncvNstefvvm-6w)
- [TVM命令行驱动程序 视频教程](https://mp.weixin.qq.com/s/XWKsQ7dPKv8IPhhPAoiVQQ)
- [基于 MLIR 完成对 GEMM 的编译优化 中英视频上，中部分](https://mp.weixin.qq.com/s/9wyM3hKsJA0YxFsms1Rpuw)

## learning note

- [白杨：TVM源语-Compute篇](https://mp.weixin.qq.com/s/ohWy5yBrsKpzApfjQLXWJg)
- [MLSys 15-884: Course Introduction](https://mp.weixin.qq.com/s/79lzlCHAxQEE0EQcxL07XQ)

-------------------------------------------------------------------------------------------

- [【社区实践】为 TVM 新增 OneFlow 前端](https://mp.weixin.qq.com/s/mwIc9DZo4r7YgYsPus-2tA)
- [【TVM 三代优化巡礼】在X86上将普通的矩阵乘法算子提速90倍](https://mp.weixin.qq.com/s/d8v9Q3EAkv8TknP5Hh7N7A)
- [【论文解读】基于MLIR生成矩阵乘法的高性能GPU代码，性能持平cuBLAS](https://mp.weixin.qq.com/s/gbpqYwPbtHp1RIYPD_ZlCg)
- [【从零开始学深度学习编译器】二十，MLIR的Pattern Rewrite机制](https://mp.weixin.qq.com/s/7QwJvTZ9Z2KbUwxqvQHC2g)
- [【从零开始学深度学习编译器】十九，MLIR的Pass机制实践](https://mp.weixin.qq.com/s/qmFpGtH0oB_ml0LQGPUqPA)
- [MLIR：摩尔定律终结的编译器基础结构 论文解读](https://mp.weixin.qq.com/s/SLzMKYugrkhQifqahfdVNw)
- [【从零开始学深度学习编译器】十八，MLIR中的Interfaces](https://mp.weixin.qq.com/s/yD-b75p1An4YTpfoIgB8mQ)
- [【用沐神的方法阅读PyTorch FX论文】](https://mp.weixin.qq.com/s/JENCa_GNGPHhOspGb79ugA)
- [【以OneFlow为例探索MLIR的实际开发流程】](https://mp.weixin.qq.com/s/eUIm4QZbKU69B9_h3f109A)
- [【从零开始学深度学习编译器】十七，MLIR ODS要点总结下篇](https://mp.weixin.qq.com/s/TsaMULNUXIVlUPnVs2WexA)
- [【从零开始学深度学习编译器】十六，MLIR ODS要点总结上篇](https://mp.weixin.qq.com/s/SFHWUm63BqsD9SWwuW83mA)
- [【从零开始学深度学习编译器】十五，MLIR Toy Tutorials学习笔记之Lowering到LLVM IR](https://mp.weixin.qq.com/s/ve2l3luRzIeDwG4PHjhDlQ)
- [【从零开始学深度学习编译器】十四，MLIR Toy Tutorials学习笔记之部分Lowering](https://mp.weixin.qq.com/s/3hAf7zxEKwRvnVAKhziTmA)
- [【从零开始学深度学习编译器】十三，如何在MLIR里面写Pass？](https://mp.weixin.qq.com/s/3N9DK7aQtjoLgs-s0lP-jg)
- [【从零开始学深度学习编译器】十二，MLIR Toy Tutorials学习笔记一](https://mp.weixin.qq.com/s/jMHesvKmAUU5dYH0WznulA)
- [【从零开始学深度学习编译器】十一，初识MLIR](https://mp.weixin.qq.com/s/4pD00N9HnPiIYUOGSnSuIw)
- [可以让深度学习编译器来指导算子优化吗](https://mp.weixin.qq.com/s/goAtJKe6p0e3pbp5vcQWfA)
- [【从零开始学深度学习编译器】十，TVM的整体把握](https://mp.weixin.qq.com/s/9nnrXhzP_gqFEPuIMdEE5w)
- [Ansor论文阅读笔记&&论文翻译](https://mp.weixin.qq.com/s/OJCHzh4opNN2Mnomz_6L9Q)
- [【从零开始学深度学习编译器】九，TVM的CodeGen流程](https://mp.weixin.qq.com/s/n7-ZTzCwFOvHrrzg4gFXQQ)
- [【从零开始学深度学习编译器】番外二，在Jetson Nano上玩TVM](https://mp.weixin.qq.com/s/7Wvv4VOPdj6N_CEg8bJFXw)
- [【从零开始学深度学习编译器】八，TVM的算符融合以及如何使用TVM Pass Infra自定义Pass](https://mp.weixin.qq.com/s/QphPwnRE5uANJk2qiqlI6w)
- [【从零开始学深度学习编译器】七，万字长文入门TVM Pass](https://mp.weixin.qq.com/s/IMm1nurpoESFRLxHcEYxcQ)
- [【从零开始学深度学习编译器】六，TVM的编译流程详解](https://mp.weixin.qq.com/s/CZzC5klWoFftUlOKkpvEZg)
- [【从零开始学深度学习编译器】五，TVM Relay以及Pass简介](https://mp.weixin.qq.com/s/5JAWE9RTTXwDJR5HqlsCzA)
- [【从零开始学深度学习编译器】番外一，Data Flow和Control Flow](https://mp.weixin.qq.com/s/Kt4xDLo-NRui8Whl0DqcSA)
- [【从零开始学深度学习编译器】四，解析TVM算子](https://mp.weixin.qq.com/s/1YlTSUArDIzY-9zeUAIfhQ)
- [【从零开始学TVM】三，基于ONNX模型结构了解TVM的前端](https://mp.weixin.qq.com/s/KFxd3zf76EP3DFcCAPZjvQ)
- [【从零开始学深度学习编译器】二，TVM中的scheduler](https://mp.weixin.qq.com/s/fPpqKL3uaaJ5QlNS79DZ5Q)
- [【从零开始学深度学习编译器】一，深度学习编译器及TVM 介绍](https://mp.weixin.qq.com/s/sZLWjYebbHjCgQ6XAZCiOw)

