## cpufp

This is a cpu tool for testing the floating-points peak performance. Now it supports linux and x86-64 platform. It can automatically recognize the x86 instruction sets and select the proper set to do test.

build:  
sh build.sh

test:  
./cpufp num_threads

clean:  
sh clean.sh

[UPDATE] 2019-10-09  
Add support of avx512f and avx512_vnni instructions.

Example on Intel i7 1065G7(SunnyCove, 4 cores, 8 threads, base@1.3GHz, turbo@3.9GHz):

$ ./cpufp 1  
Thread(s): 1  
avx512_vnni int8 perf: 484.2888 gops.  
avx512f fp32 perf: 121.2331 gflops.  
avx512f fp64 perf: 60.6260 gflops.  
fma fp32 perf: 123.6942 gflops.  
fma fp64 perf: 61.8030 gflops.  
avx fp32 perf: 58.7558 gflops.  
avx fp64 perf: 26.9633 gflops.  
sse fp32 perf: 31.0415 gflops.  
sse fp64 perf: 15.5088 gflops.  
$ ./cpufp 4  
Thread(s): 4  
avx512_vnni int8 perf: 1783.9178 gops.  
avx512f fp32 perf: 446.2898 gflops.  
avx512f fp64 perf: 223.2302 gflops.  
fma fp32 perf: 444.4888 gflops.  
fma fp64 perf: 222.1889 gflops.  
avx fp32 perf: 204.9414 gflops.  
avx fp64 perf: 89.2331 gflops.  
sse fp32 perf: 111.3561 gflops.  
sse fp64 perf: 55.7209 gflops.  

The next version may support ARMv7 and ARMv8 architectures.  
Welcome for bug reporting.
