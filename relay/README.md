# TVM Realy Learn

## 1. Overview
Learn TVM Relay, include how to use relay build a network, introduction of tvm pass infra, how to run a dl network in jetsonnano, 
how to autotune(by AutoTVM) conv2d op in target(2080Ti, JetsonNano) and so on.


## 2. Scripts
+ `simplenet.py`: how to use relay build a conv+bn+relu simplenet.
+ `tvm_relay_cuda.py`: how to autotune(by AutoTVM) conv2d op in target(2080Ti, JetsonNano).
+ `use_pass_infra.py`: introdction of tvm pass infra and how to define a pass in python level.
+ `jetsonnao`: how to use tvm relay in jetsonnano.

## 3. requirements

```sh
tvm == 0.8.0.dev
psutil 
xgboost 
tornado 
cloudpickle
```
