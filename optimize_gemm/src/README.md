# armv7a gemm optimize src

- 编译：`make` 
- 在编译之后运行时只需要新增一个重定向命令，即可获得记录了矩阵大小和GFlops的txt文件，例： ./unit_test >> now.txt， 注意now.txt需要自己先创建，并保证它有可写的权限。
- 当前测试的CPU型号是：Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz 12 Cores。

