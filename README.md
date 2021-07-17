# tvm_learn

## preoject introduction

- scheduler tvm schedule 详细举例，这里将 https://zhuanlan.zhihu.com/p/94846767 这篇文章的例子用TVM 0.8.0.dev 重写。
- dataflow_controlflow 数据流和控制流的区别示例。
- relay TVM 中一些 Relay 相关的示例，比如如何自定义 Pass，如何在 Jetson Nano 中运行DarkNet的YOLO模型等。 
- `tvm_pytorch_resnet18_inference.py` 使用 TVM 在 X86 CPU 上运行 Pytorch 的 ResNet18 模型。
- `pytorch_resnet18_export_onnx.py` Pytorch 导出 ResNet18 的 ONNX 模型示例。
- `tvm_onnx_resnet18_inference.py` TVM 加载 ResNet18 的 ONNX 模型进行推理。

## learning note

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
