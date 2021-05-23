# tvm_learn

- scheduler tvm schedule详细举例，这里将https://zhuanlan.zhihu.com/p/94846767 这篇文章的例子用TVM 0.8.0.dev 重写。
- `tvm_pytorch_resnet18_inference.py` 使用TVM在X86 CPU上运行Pytorch的ResNet18模型。
- `pytorch_resnet18_export_onnx.py` Pytorch导出ResNet18的ONNX模型示例。
- `tvm_onnx_resnet18_inference.py` TVM加载ResNet18的ONNX模型进行推理。
