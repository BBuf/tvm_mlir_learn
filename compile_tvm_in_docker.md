```
git clone --recursive https://github.com/apache/tvm tvm
```



```
nvidia-docker run --rm -v /home/zhangxiaoyu/OneFlowWork/tvm/:/home/tvm_learn -it tvmai/demo-gpu bash
```



```
mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make -j4
```



```
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```



```
export TVM_HOME=/home/tvm_learn/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

