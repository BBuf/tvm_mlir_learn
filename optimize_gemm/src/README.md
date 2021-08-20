# X86 gemm optimize src

- 编译：`make` 
- 运行 `./unit_test` 即可获得当前方法的gflops
- 当前测试的CPU型号是：`64  Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz`

|文件名|优化方法|gFLOPs|线程数|
|--|--|--|--|
|MMult1.h|无任何优化|2.42gflops|1|
|MMult2.h|一次计算4个元素|1.5gflops|1|
|MMult_1x4_3.h|一次计算4个元素|1.4gflops|1|
|MMult_1x4_4.h|一次计算4个元素|1.4gflops|1|
|MMult_1x4_5.h|一次计算4个元素(将4个循环合并为1个)|1.5gflops|1|
|MMult_1x4_6.h|一次计算4个元素(我们在寄存器中累加C的元素，并对a的元素使用寄存器)|1.6gflops|1|
|MMult_1x4_7.h|在MMult_1x4_6的基础上用指针来寻址B中的元素|5.0gflops|1|
|MMult_1x4_8.h|在MMult_1x4_7的基础上循环展开四个（展开因子的相对任意选择）|5gflops|1|
|MMult_1x4_9.h|在MMult_1x4_8的基础上使用间接寻址的方法|5gflops|1|
|MMult_4x4_3.h|一次计算C中的4x4小块|1.4gflops|1|
|MMult_4x4_4.h|一次计算C中的4x4小块|1.4gflops|1|
|MMult_4x4_5.h|一次计算C中的4x4小块,将16个循环合并一个|1.5gflops|1|
|MMult_4x4_6.h|一次计算C中的4x4小块(我们在寄存器中累加C的元素，并对a的元素使用寄存器)|8.2gflops|1|
|MMult_4x4_7.h|在MMult_4x4_6的基础上用指针来寻址B中的元素|8.4gflops|1|
|MMult_4x4_8.h|使用更多的寄存器|7.7gflops|1|
|MMult_4x4_10.h|SSE指令集优化|8.5gflops|1|
|MMult_4x4_11.h|SSE指令集优化, 并且为了保持较小问题规模所获得的性能，我们分块矩阵C（以及相应的A和B） |8.5gflops|1|
|MMult_4x4_13.h|SSE指令集优化, 对矩阵A和B进行Pack，这样就可以连续访问内存|33.0gflops|1|
