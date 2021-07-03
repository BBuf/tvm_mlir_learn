# Jetbot(Jetson Nano) Example(Origin Code forked from https://github.com/irvingzhang0512/tvm_tests)

## 1. Overview
+ Environments: Server and Jetbot(Jetson Nano).
+ Target: use TVM-DarkNet to detect videos on Jetbot.
+ Features:
  + [x] Server rpc auto-tuning.
  + [x] Server cross compilation
  + [x] Server libs/graphs/params exports.
  + [x] Jetbot videos detection. 
+ For more information, please check my [notes](https://zhuanlan.zhihu.com/p/95742125)


## 2. Scripts
+ `server_rpc_tune.py`: use Server CPU and RPC to auto-tune Jetbot.
+ `server_cross_compile.py`: cross compile and export libs/graphs/params on Server.
+ `jetson_detect_video.py`: run detection on Jetbot.
+ `deploy_model_on_jetsonnano.py`: run resnet18 on Jetbot. (Local Run, Not RPC)

## 3. requirements
```
mxnet
```