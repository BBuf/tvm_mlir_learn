# tvm_mlir_learn

> 我也维护了一个cuda学习仓库，有需要的小伙伴可以点一点star：https://github.com/BBuf/how-to-optim-algorithm-in-cuda

## 项目结构介绍

- `scheduler` TVM 中 scheduler 详细举例，这里将 https://zhuanlan.zhihu.com/p/94846767 这篇文章的例子用TVM 0.8.0.dev 重写。
- `dataflow_controlflow` 数据流和控制流的区别示例，这里是Pytorch为例子。
- `paper_reading` 编译器方面的一些论文阅读，如 PET / Ansor/ MLIR 等。
- `relay` TVM 中一些 Relay 相关的示例，比如如何自定义 Pass，如何在 Jetson Nano 中运行DarkNet的YOLO模型等。 
- `codegen` TVM 中 Codegen 相关示例，基于张量表达式和Relay IR。
- `torchscript` Pytorch的TorchScript的用法。
- compile_tvm_in_docker.md 。在Docker中编译TVM。
- `tvm_pytorch_resnet18_inference.py` 使用 TVM 在 X86 CPU 上运行 Pytorch 的 ResNet18 模型。
- `tvm_onnx_resnet18_inferentaicce.py` TVM 加载 ResNet18 的 ONNX 模型进行推理。
- `pytorch_resnet18_export_onnx.py` Pytorch 导出 ResNet18 的 ONNX 模型示例。
- `optimize_gemm` 让深度学习编译器来指导我们写代码，以GEMM为例。

# AI编译器/LLVM相关学习资料整理

## 视频收集

### GiantPandaCV 翻译的视频

- [What Is MLIR && What Is TVM？](https://mp.weixin.qq.com/s/Xj2iW9tFUGidlzLqEzoixQ)
- [TVM Conf 2020 - An Introduction to TVM Part1](https://mp.weixin.qq.com/s/NaMxlNzPrRlBYJfJ7ivjuw)
- [TVM Conf 2020 - An Introduction to TVM Part2](https://mp.weixin.qq.com/s/KAG0DjnhQcGEJa-hRFiBfg)
- [Torch MLIR公开会议翻译视频（自制中英双字完整版）](https://mp.weixin.qq.com/s/d0jJFYdUncvNstefvvm-6w)
- [TVM命令行驱动程序 视频教程](https://mp.weixin.qq.com/s/XWKsQ7dPKv8IPhhPAoiVQQ)
- [基于 MLIR 完成对 GEMM 的编译优化 中英视频上，中部分](https://mp.weixin.qq.com/s/9wyM3hKsJA0YxFsms1Rpuw)
- [TVM TensorIR 视频讲解（熟肉）](https://mp.weixin.qq.com/s/MkUAuQlhZAF25wXlgtHD2Q)
- [What Is LLVM?](https://www.bilibili.com/video/BV1yY411K7YA?spm_id_from=333.999.0.0&vd_source=347c9d161e405bfb1666662e320106d3)
- [How To Install LLVM？](https://www.bilibili.com/video/BV1ka411s7rS/?vd_source=347c9d161e405bfb1666662e320106d3)
- [Running the LLVM Tools](https://www.bilibili.com/video/BV1m3411M7zv/)
- [LLVM IR介绍](https://www.bilibili.com/video/BV1K34y1W7dn?spm_id_from=333.999.0.0)

> LLVM 系列视频对应的源码在：https://github.com/lac-dcc/llvm-course

### 国内其它up主的编译器视频（包含LLVM/MLIR/TVM）

#### LLVM相关视频

- [LLVM设计架构](https://www.bilibili.com/video/BV1CG4y1V7Dn/?spm_id_from=333.788&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [LLVM IR详解](https://www.bilibili.com/video/BV1PP411u7NR/?spm_id_from=333.788&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [LLVM前端和优化层](https://www.bilibili.com/video/BV1vd4y1t7vS/?spm_id_from=333.788&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [LLVM后端和代码生成](https://www.bilibili.com/video/BV1cd4y1b7ho/?spm_id_from=333.788&vd_source=4dffb0fbabed4311f4318e8c6d253a10)

LLVM相关的视频比较少，youtube上比较多，上面 GiantPandaCV 翻译的几期 LLVM 入门视频也是来源于 youtube，大家可以自行查找学习。

#### MLIR相关视频

- [人工智能编译器MLIR-官方入门教程讲解](https://www.bilibili.com/video/BV1Hd4y1U7mb/?vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [MLIR Toy Tutorial概述](https://www.bilibili.com/video/BV1s7411K7rR/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [MLIR & python binding简介](https://www.bilibili.com/video/BV1s7411K7fp/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [[MLIR] 使用MLIR完成一个端到端的编译流程](https://www.bilibili.com/video/BV1Wp4y1z72d/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解(一)：AI编译器是啥？](https://www.bilibili.com/video/BV1yP4y1d7gz/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解(二)：TPU-MLIR简介](https://www.bilibili.com/video/BV19d4y1B7eR/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（三）：MLIR语法介绍（上）](https://www.bilibili.com/video/BV1CP411n7fj/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（四）：MLIR语法介绍（中）](https://www.bilibili.com/video/BV1Gt4y1F7mt/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（五）：MLIR语法介绍（下）](https://www.bilibili.com/video/BV1UN4y1w72r/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（六）：前端转换](https://www.bilibili.com/video/BV1yv4y1S7WT/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（七）：MLIR- Dialect Conversion](https://www.bilibili.com/video/BV1UG411c7nm/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（八）：Lowering in TPU-MLIR](https://www.bilibili.com/video/BV1gg411z7mC/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（九）：量化概述](https://www.bilibili.com/video/BV1d8411j7t4/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（十）：量化推导](https://www.bilibili.com/video/BV1SW4y1H7Uu/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（十一）：量化校准](https://www.bilibili.com/video/BV1qK411R75k/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（十二）：量化感知训练](https://www.bilibili.com/video/BV12g411J7WQ/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（十三）：精度验证](https://www.bilibili.com/video/BV14e4y1M79d/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（十四）：Pattern Rewriting](https://www.bilibili.com/video/BV1R44y1d7xv/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（十五）：模型适配](https://www.bilibili.com/video/BV1mM411y7Ep/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR系列讲解（十六）：图优化](https://www.bilibili.com/video/BV1AR4y1U7D6/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [ep17 | TPU-MLIR Introduction ：To ONNX Format](https://www.bilibili.com/video/BV1FD4y1H7pT/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [Ep18 TPU Memory](https://www.bilibili.com/video/BV1T24y1G7pu/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [Ep19 TPU Memory (2)](https://www.bilibili.com/video/BV1VY4y1y7ET/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [Ep20 Add a New Operator](https://www.bilibili.com/video/BV1tL411r71p/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [Ep21 fuse prepocess 【AI编译器】](https://www.bilibili.com/video/BV1ao4y1H7m8/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [ep1｜TPU-MLIR Introduction AI Compiler](https://www.bilibili.com/video/BV1V24y1h7J1/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR Ep2 TPU-MLIR Overview](https://www.bilibili.com/video/BV1cR4y1z7Rb/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TPU-MLIR Ep3 MLIR Brief Intro](https://www.bilibili.com/video/BV1b14y1c7jN/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [AI框架源码走读：tpu-mlir（一）](https://zhuanlan.zhihu.com/p/613328745)
- [AI框架源码走读：tpu-mlir（二）](https://zhuanlan.zhihu.com/p/615180103)
- [AI框架源码走读：tpu-mlir（三）](https://zhuanlan.zhihu.com/p/618707936)
- [TPU-MLIR线上分享会（一）：论文讲解](https://www.bilibili.com/video/BV1My4y1o73Q/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [MegCC 用模型编译的方式实现超轻量端上高性能推理](https://www.zhihu.com/zvideo/1579066161320120320)
- [动态shape深度学习编译器论文分享：DISC](https://www.bilibili.com/video/BV16R4y1U7J5/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)

#### TVM相关视频

- [陈天奇 机器学习课程](https://mlc.ai/zh/chapter_introduction/index.html)
- [AI-Compiler科普——TVM的使用讲解](https://www.bilibili.com/video/BV1MK4y1u7nF/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TVM流程梳理](https://www.bilibili.com/video/BV123411r7o8/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TVM-Realy流程梳理](https://www.bilibili.com/video/BV1uP4y1W7fc/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [AI编译器后端优化介绍](https://www.bilibili.com/video/BV17D4y177bP/?spm_id_from=333.788&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [算子的计算和调度](https://www.bilibili.com/video/BV1K84y1x7Be/?spm_id_from=333.788&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [算子优化的手工方式](https://www.bilibili.com/video/BV1ZA411X7WZ/?spm_id_from=333.788&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [算子循环优化](https://www.bilibili.com/video/BV1r14y1w7hG/?spm_id_from=333.788&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [指令和存储优化](https://www.bilibili.com/video/BV11d4y1a7J6/?spm_id_from=333.788&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [Auto Tuning原理](https://www.bilibili.com/video/BV1uA411D7JF/?spm_id_from=333.788&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TVM简介](https://www.bilibili.com/video/BV14N4y1c7zq/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TVM自动调度算法AutoTVM](https://www.bilibili.com/video/BV1114y1e7FK/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [ANSOR：为深度学习生成高性能张量程序](https://www.bilibili.com/video/BV1m14y1Y7LN/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TVM 编译流程与中间表示分析（一）](https://www.bilibili.com/video/BV1v3411U7fM/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)
- [TVM 编译流程与中间表示分析（二）](https://www.bilibili.com/video/BV1624y1v7wx/?spm_id_from=333.999.0.0&vd_source=4dffb0fbabed4311f4318e8c6d253a10)

## GiantPandaCV原创的学习笔记

- [TVM 学习指南（个人版）](https://mp.weixin.qq.com/s/NM5yvxW2JSbR06RmrR3ubw)
- [白杨：TVM源语-Compute篇](https://mp.weixin.qq.com/s/ohWy5yBrsKpzApfjQLXWJg)
- [MLSys 15-884: Course Introduction](https://mp.weixin.qq.com/s/79lzlCHAxQEE0EQcxL07XQ)
- [OSDI 2021 PET 论文解读（代码生成相关工作）](https://zhuanlan.zhihu.com/p/533807811)
- [Buddy-MLIR 项目详解（入门 MLIR 极佳选择）](https://mp.weixin.qq.com/s/uE5VhU_s3NgndPk2X6zbAA)
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

## 其它博客和网站精选（TVM&MLIR&LLVM 相关）

### LLVM精选

- [LLVM Tutorial](https://llvm.org/docs/tutorial/index.html)
- [miniSysY 编译实验课程，学习LLVM的中文入门资料](https://buaa-se-compiling.github.io/miniSysY-tutorial/)
- [中科院 LLVM每日谈专栏](https://zhuanlan.zhihu.com/llvm-clang)
- [使用LLVM实现一门语言（一）Lexer](https://zhuanlan.zhihu.com/p/334730846)
- [使用LLVM实现一门语言（二）Parser](https://zhuanlan.zhihu.com/p/334739920)
- [使用LLVM实现一门语言（三）Code Generation to LLVM IR](https://zhuanlan.zhihu.com/p/334756681)
- [使用LLVM实现一门语言（四）Optimizer](https://zhuanlan.zhihu.com/p/334791822)
- [使用LLVM实现一门语言（五）Adding a JIT Compiler](https://zhuanlan.zhihu.com/p/334797700)
- [使用LLVM实现一门语言（六）SSA](https://zhuanlan.zhihu.com/p/335303123)
- [使用LLVM实现一门语言（七）Control Flow](https://zhuanlan.zhihu.com/p/335344134)
- [使用LLVM实现一门语言（八）User-defined Operators](https://zhuanlan.zhihu.com/p/336243654)
- [使用LLVM实现一门语言（九）Mutable Variables](https://zhuanlan.zhihu.com/p/336929719)

### TVM精选

- [深度学习编译器 TVM 代码串讲](https://zhuanlan.zhihu.com/p/446976730)
- [TVM编译流程与中间表示分析](https://zhuanlan.zhihu.com/p/596526031)
- [TVM Overview](https://chhzh123.github.io/blogs/2020-03-26-tvm-flow/)
- [TVM - Relay IR计算图可视化](https://chhzh123.github.io/blogs/2020-03-25-relay-ir-viz/)
- [TVM - 代码生成流程](https://chhzh123.github.io/blogs/2020-03-26-tvm-flow/)
- [TVM/VTA代码生成流程](https://krantz-xrf.github.io/2019/10/24/tvm-workflow.html)
- [tvm算子优化schedule（一）--CPU篇](https://zhuanlan.zhihu.com/p/403163009)
- [tvm算子优化schedule（二）--GPU篇](https://zhuanlan.zhihu.com/p/403370698)
- [TVM Runtime System 概述](https://zhuanlan.zhihu.com/p/504066888)
- [TVM PackedFunc实现机制](https://hjchen2.github.io/2020/01/10/TVM-PackedFunc%E5%AE%9E%E7%8E%B0%E6%9C%BA%E5%88%B6/)
- [深入理解TVM：Python/C++互调（上）](https://zhuanlan.zhihu.com/p/363991566)
- [Round-tripping objects through the FFI](https://discuss.tvm.apache.org/t/round-tripping-objects-through-the-ffi/8440)
- [TVM 自底向上（一）：基本框架和概念](https://zhuanlan.zhihu.com/p/532873577)
- [TVM 自底向上（二）：TIR 的概念和编译原理](https://zhuanlan.zhihu.com/p/533161438)
- [TVM 自底向上（三）：TE 的概念和编译原理](https://zhuanlan.zhihu.com/p/534313816)
- [TVM 自底向上（四）：TE/TIR Schedule 的原理](https://zhuanlan.zhihu.com/p/534062007)
- [深入理解TVM专栏，主要是对部分codebase的解读](https://www.zhihu.com/column/c_1394234963567394816)
- [tvm schedule详细举例](https://zhuanlan.zhihu.com/p/94846767)
- [TVM - 代码生成流程](https://chhzh123.github.io/blogs/2020-03-26-tvm-flow/)
- [Relax: TVM 的下一代图层级 IR](https://zhuanlan.zhihu.com/p/523395133)
- [TVM之Tensor数据结构解读](https://zhuanlan.zhihu.com/p/341257418)
- [TVM之设计模式解读（一）--visitor模式](https://zhuanlan.zhihu.com/p/341334406)
- [TVM之设计模式解读（二）--责任链模式](https://zhuanlan.zhihu.com/p/342108378)
- [TVM之TIR相关数据结构](https://zhuanlan.zhihu.com/p/343654464)
- [TVM之设计模式解读（三）-单例模式,模板方法模式](https://zhuanlan.zhihu.com/p/342238892)
- [TVM之tir 转换成llvm ir](https://zhuanlan.zhihu.com/p/344553283)
- [TVM之graph_runtime](https://zhuanlan.zhihu.com/p/345085746)
- [TVM之relay.build流程解读](https://zhuanlan.zhihu.com/p/348696198)
- [TVM学习（一）](https://zhuanlan.zhihu.com/p/333706468)
- [TVM学习（二）：算符融合](https://zhuanlan.zhihu.com/p/337824083)
- [TVM学习（三）编译流程](https://zhuanlan.zhihu.com/p/338550499)
- [TVM学习（四）codegen](https://zhuanlan.zhihu.com/p/339566528)
- [TVM学习（五）schedule](https://zhuanlan.zhihu.com/p/341498731)
- [TVM学习（六）细读前端](https://zhuanlan.zhihu.com/p/346514871)
- [TVM学习（七）算子](https://zhuanlan.zhihu.com/p/351403985)
- [TVM学习（八）pass总结](https://zhuanlan.zhihu.com/p/358437531)
- [TVM学习（九）codegen中的内存申请](https://zhuanlan.zhihu.com/p/363721019)
- [TVM学习（十）从relay到TOPI](https://zhuanlan.zhihu.com/p/374516615)
- [TVM TensorIR 浅析](https://zhuanlan.zhihu.com/p/451854416)
- [TVM图编译器NNVM简单探究](https://zhuanlan.zhihu.com/p/90528541)
- [TVM图编译器Relay简单探究](https://zhuanlan.zhihu.com/p/91283238)
- [基于TensorIR生成mma指令并实现16x16x4矩阵乘](https://zhuanlan.zhihu.com/p/455166274)
- [基于TVM的PTX Tensor Core汇编代码生成](https://zhuanlan.zhihu.com/p/456242751)
- [一个tvm(te)实现的cutlass efficient gemm](https://zhuanlan.zhihu.com/p/560729749)
- [TIR Script CUTLASS Efficient Gemm](https://zhuanlan.zhihu.com/p/562360659)
- [TVM系列「一」TVM概览](https://zhuanlan.zhihu.com/p/381324332)
- [TVM系列「二」TVM学习资源](https://zhuanlan.zhihu.com/p/381330616)
- [TVM系列「三」TVM官方文档的结构](https://zhuanlan.zhihu.com/p/381331888)
- [TVM系列「四」TVM的使用：compute+schedule双剑合璧](https://zhuanlan.zhihu.com/p/381333188)
- [TVM系列「五」TVM整体架构及其代码生成](https://zhuanlan.zhihu.com/p/381691430)
- [TVM系列「六」Relay IR与Relay Pass](https://zhuanlan.zhihu.com/p/390087648)
- [TVM系列「七」AutoTVM（AutoTune）](https://zhuanlan.zhihu.com/p/392015642)
- [TVM系列「八」AutoScheduler「Ansor」](https://zhuanlan.zhihu.com/p/394765523)

### MLIR精选

- [机器学习编译器代码生成相关 MLIR Dialect](https://www.lei.chat/zh/posts/mlir-codegen-dialects-for-machine-learning-compilers/)
- [编译器与中间表示: LLVM IR, SPIR-V, 以及 MLIR](https://www.lei.chat/zh/posts/compilers-and-irs-llvm-ir-spirv-and-mlir/)
- [MLIR Vector Dialect 以及 Patterns](https://www.lei.chat/zh/posts/mlir-vector-dialect-and-patterns/)
- [MLIR Linalg Dialect 以及 Patterns](https://www.lei.chat/zh/posts/mlir-linalg-dialect-and-patterns/)
- [向外借力：Pluto助力MLIR编译器的多面体优化](https://mp.weixin.qq.com/s/n33DyOeTjA93HavZBZb94g)
- [IREE编译流程解析](https://hjchen2.github.io/2023/01/04/IREE%E7%BC%96%E8%AF%91%E6%B5%81%E7%A8%8B/)
- [IREE编译流程解析(一)](https://hjchen2.github.io/2023/01/04/IREE%E7%BC%96%E8%AF%91%E6%B5%81%E7%A8%8B1/)
- [IREE编译流程解析(二)](https://hjchen2.github.io/2023/01/04/IREE%E7%BC%96%E8%AF%91%E6%B5%81%E7%A8%8B2/)
- [IREE编译流程解析(三)](https://hjchen2.github.io/2023/01/04/IREE%E7%BC%96%E8%AF%91%E6%B5%81%E7%A8%8B3/)
- [IREE编译流程解析(四)](https://hjchen2.github.io/2023/01/04/IREE%E7%BC%96%E8%AF%91%E6%B5%81%E7%A8%8B4/)
- [IREE编译流程解析(五)](https://hjchen2.github.io/2023/02/13/IREE%E7%BC%96%E8%AF%91%E6%B5%81%E7%A8%8B5/)
- [IREE编译流程解析(六)](https://hjchen2.github.io/2023/02/24/IREE%E7%BC%96%E8%AF%91%E6%B5%81%E7%A8%8B6/)
- [megcc 开箱评测](https://zhuanlan.zhihu.com/p/605385779)
- [阿里 BladeDISC 深度学习编译器正式开源](https://zhuanlan.zhihu.com/p/462641670)
- [全面支持 PyTorch 2.0：BladeDISC 5月~11月新功能发布](https://zhuanlan.zhihu.com/p/590314270)
- [【GTC 22】通过 PAI-Blade 更方便、更鲁棒地使用 TensorRT](https://zhuanlan.zhihu.com/p/490295901)

### 其它编译器&&论文阅读

开拓眼界...

- [Glenside : 如何自动发现im2col布局转换?](https://zhuanlan.zhihu.com/p/456616977)
- [基于Halide自动生成Kernel Fusion & Tiling](https://zhuanlan.zhihu.com/p/489888931)
- [AKG: 使用post-tiling fusion策略完成无副作用的内存优化](https://zhuanlan.zhihu.com/p/535606722)
- [[教程翻译] Polyhedral Tutorials](https://zhuanlan.zhihu.com/p/553703704)
- [带宽受限下的DSA后端优化](https://zhuanlan.zhihu.com/p/585176512)
- [Equality Saturation优化在AI编译器中遇到的挑战](https://zhuanlan.zhihu.com/p/605459519)
- [DSA后端Compute Schedule与Buffer Schedule](https://zhuanlan.zhihu.com/p/609483844)
- [ASPLOS，我的初体验](https://zhuanlan.zhihu.com/p/113340891)
- [读You and Your Research笔记](https://zhuanlan.zhihu.com/p/114014432)
- [[阅读笔记] AStitch @ASPLOS 2022](https://zhuanlan.zhihu.com/p/477984880)
- [[阅读笔记] RAKE @ASPLOS 2022](https://zhuanlan.zhihu.com/p/511381790)
- [[阅读笔记] NASA @ISCA 2021](https://zhuanlan.zhihu.com/p/513464183)
- [[阅读笔记] BOLT @MLSys 2022](https://zhuanlan.zhihu.com/p/514032549)
- [[阅读笔记] Alpa/Parax @OSDI 2022](https://zhuanlan.zhihu.com/p/521211578)
- [[阅读笔记] SIMD^2 ISCA 2022](https://zhuanlan.zhihu.com/p/528108829)
- [AMOS ISCA 2022](https://zhuanlan.zhihu.com/p/530626092)
- [[阅读笔记] PCCS MICRO 2021](https://zhuanlan.zhihu.com/p/586308472)
- [[阅读笔记] Planaria@MICRO 2020](https://zhuanlan.zhihu.com/p/589773030)
- [Chimera HPCA 2023](https://zhuanlan.zhihu.com/p/612913262)
- [在MacBook Pro 2019上优化GEMM](https://zhuanlan.zhihu.com/p/468304964)
- [OSDI '20 | RAMMER (NNFusion) 如何进一步压榨加速器性能](https://zhuanlan.zhihu.com/p/275837455)
- [算子调度优化论文分享：Rammer](https://zhuanlan.zhihu.com/p/616050345)

## 系统性的专栏或者网站

- [陈天奇 MLC课程](https://mlc.ai/zh/index.html)
- [深度学习编译器学习笔记和实践体会](https://zhuanlan.zhihu.com/c_1169609848697663488)
- [蓝色的味道](https://zhuanlan.zhihu.com/frozengene)
- [TVM官方专栏](https://zhuanlan.zhihu.com/tvmai)
- [Apache TVM 中文站](https://tvm.hyper.ai/)
- [深度学习编译器学习笔记和实践体会](https://www.zhihu.com/column/c_1169609848697663488)

## 工具介绍

- [FFI Navigator: 跨语言调用跳转IDE插件](https://zhuanlan.zhihu.com/p/103426525)
- [如何Debug TVM的源码](https://zhuanlan.zhihu.com/p/481972756)

