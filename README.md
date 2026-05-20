# tvm_mlir_learn

Learning notes and experiments for deep learning compilers.

This repository collects examples around TVM, MLIR, LLVM, TorchScript, Relay, code generation, scheduling, and compiler-guided kernel optimization. It is kept as a public learning archive for AI compiler systems.

## Contents

- `scheduler/`: TVM scheduler examples and scheduling experiments.
- `dataflow_controlflow/`: small examples comparing data flow and control flow concepts.
- `paper_reading/`: notes for compiler and ML systems papers such as PET, Ansor, and MLIR-related work.
- `relay/`: Relay examples, custom pass experiments, and model deployment demos.
- `codegen/`: TVM code generation examples based on tensor expressions and Relay IR.
- `torchscript/`: TorchScript usage examples.
- `optimize_gemm/`: GEMM optimization experiments guided by compiler ideas.
- `compile_tvm_in_docker.md`: TVM build notes in Docker.

## Related Repositories

- CUDA and GPU optimization: https://github.com/BBuf/how-to-optim-algorithm-in-cuda
- Deep learning framework notes: https://github.com/BBuf/how-to-learn-deep-learning-framework

## Status

Legacy learning archive. I may still reference this repository, but new public-facing documentation will use English entry points.
