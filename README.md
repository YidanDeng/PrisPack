# Preconditioned iterative solver package for large sparse linear systems (PrisPack)

## 准备工作 

* 工具：

  * 主要开发工具：CLion、CMake
  * 文本编辑：Typora
  * 虚拟机：Linux操作系统（Ubuntu 20.04 - WSL 子系统)

* 远程协作：用于统一管理代码的提交与合并。

  * 平台：gitee（码云）、github

* 编译模式：	![image-20240905162203888](./markdown_pictures/README/image-20240905162203888.png)

  * 命令行编译指令：

    * Debug模式：

      ```shell
      rm -rf ./cmake-build-debug/*;
      cmake -S . -G "Unix Makefiles" -D CMAKE_C_COMPILER="/usr/bin/gcc" -D CMAKE_CXX_COMPILER="/usr/bin/g++" -D CMAKE_BUILD_TYPE=Debug -B cmake-build-debug && cmake --build cmake-build-debug -- -j 12;
      # 如果需要编译CUDA代码，则加上：-D CMAKE_CUDA_COMPILER="/usr/local/cuda-12.3/bin/nvcc"
      ```

    * Release模式：

      ```shell
      rm -rf ./cmake-build-release/*;
      cmake -S . -G "Unix Makefiles" -D CMAKE_C_COMPILER="/usr/bin/gcc" -D CMAKE_CXX_COMPILER="/usr/bin/g++" -D CMAKE_BUILD_TYPE=Release -B cmake-build-release && cmake --build cmake-build-release -- -j 12;
      # 如果需要编译CUDA代码，则加上：-D CMAKE_CUDA_COMPILER="/usr/local/cuda-12.3/bin/nvcc"
      ```

  * Debug模式：项目不采用任何编译器级别的优化手段，主要用于debug（例如：gdb断点调试）

  * Release模式：采用最高级别的代码优化，主要用于运行代码、测试代码效果（例如：指令排序、向量化、快速数学优化等）

## 项目文件结构及说明

说明：以下加粗字体表示文件夹，不加粗表示文件（注意文件后缀，```.cpp/.h/.hpp```是c++文件，```.cu/.cuh```是CUDA文件）。本项目可以通过CMakeList文件自动检测当前主机是否有cuda组件，如果没有cuda则不编译cu文件。

* **config**：用于放置一些顶层配置文件。
  * config.h:  放置一些类型别名设置、全局参数配置等；
  * CUDAheaders.cuh: 放置CUDA头文件；
  * debug.h:  放置错误处理函数或debug相关的宏；
  * headers.h: 放置c++要用到的标准库头文件。
* **external_package**：用于存放一些第三方库文件和第三方库对应的CMake配置文件。
  * **cmakeFind**：放置第三方库的CMake配置文件（一般文件名为```FindXXX.cmake```），目前已经有SuiteSparse的AMD库（用于矩阵重排）、OpenBLAS库（用于快速稠密矩阵向量计算、稠密矩阵数值计算等）、METIS（用于图划分矩阵重排）、MKL库（用于对比底层代码效率）；
  * **METIS**：图划分计算库，**需要另外编译**。
  * **OpenBLAS**：稠密矩阵向量计算库，**需要另外编译**。
  * **SuiteSparse**：线性代数计算库，**需要另外编译**。
* **include / source**:  ```.h/.hpp/.cuh```被放在include目录中，```.cpp/.cu```被放在source目录中。
  * **CUDA**：主要是针对GPU加速的程序文件。
    * **BLAS**： 用于存放一些由CUSPARSE库和CUBLAS库编写的代码。
      * CUBLAStools.cuh / CUBLAStools.cu：单GPU上CUBLAS相关操作函数；
      * CUSPARSEtools.cuh / CUSPARSEtools.cu：单GPU上的CUSPARSE相关操作函数；
    * cuKernel.cuh：用于实现一些自定义核函数；
    * cuThrust.cuh：用于封装Thrust库提供的向量方法；
    * StreamController.cuh / StreamController.cu：用来分配、释放和统一管理GPU上的流。
    * EventController.cuh /EventController.cu：用来分配、释放和统一管理GPU上的事件。
  * **VectorClass**
    * BaseVector.h：一个抽象类，不可实例化，主要作为公共接口来调具体子类对象的方法。
    * VectorTools.h：用于存放一些利用多态性质实现的Vector方法。
    * AlignedVector.h / AlignedVector.cpp：BaseVector的子类，底层采用对齐内存实现。
    * DenseVector.h / DenseVector.cpp：BaseVector的子类，底层采用一般内存实现，默认申请的内存被初始化为0。
    * PageLockedVector.cuh / PageLockedVector.cu：BaseVector的子类，底层采用锁页内存实现，主要在GPU与CPU之间需要异步传输数据时使用。
    * DeviceVector.cuh / DeviceVector.cu：一个抽象类，不可实例化，用于管理GPU上的向量。
    * SyncDeviceVector.cuh / SyncDeviceVector.cu：基于同步模式申请的GPU上的向量。
    * AutoAllocateVector.h/ AutoAllocateVector.cpp：一个用于分配和管理CPU上向量的工具，可作为向量使用。
  * **MatrixClass**：主要存放和矩阵有关的数据结构和方法。
    * BaseMatrix.h：**一个抽象类，不可实例化**，主要作为公共接口来调具体子类对象的方法；
    * COOMatrix.h / COOMatrix.cpp：BaseMatrix的子类，用于创建COO格式的矩阵；
    * CSRMatrix.h / CSRMatrix.cpp：BaseMatrix的子类，实现基于稀疏压缩行存储的矩阵类；
    * DenseMatrix.h / DenseMatrix.cpp：BaseMatrix的子类，实现基于稠密格式的矩阵类（数据结构参照BLAS标准设计）；
    * DeviceCSRMatrix.cuh / DeviceCSRMatrix.cu：用于分配、释放和操作GPU上的CSR矩阵；
    * ModelProblem.h / ModelProblem.cpp：用于实现一些模型问题的离散化结果，例如泊松方程离散化、拉普拉斯方程离散化等；
    * MatrixTools.h：一些基于矩阵的辅助函数（例如：从COO转换为CSR）
  * **Precondition**:主要存放相关预条件方法。
    * BasePreconditon.h：**一个抽象类，不可实例化**，主要作为公共接口来调具体子类对象的方法；
    * IncompleteCholesky.h / IncompleteCholesky.cpp：IC预条件。
    * IncompleteLU.h / IncompleteLU.cpp：ILU预条件。
    * IncompleteLDLT.h / IncompleteLDLT.cpp：ILDLT预条件。
    * AMSEDPrecondition.h / AMSEDPrecondition.cpp：AMSED预条件和MSLR预条件，用于求解对称问题。
    * GMSLRPrecondition.h / GMSLRPrecondition.cpp：GMSLR预条件，用于求解不对称问题。
  * **utils**：一些具体的工具函数和方法。
    * **ExternalTools**：
      * MatrixReorderTools.h：基于Suitesparse和RCM库实现AMD和RCM矩阵重排序方法。
      * MetisTools.h：基于Metis库封装矩阵重排序方法。
    * **MemoryTools**：用来管理不同类型的内存。
      * DeviceMemoryController.cuh：管理GPU上的内存。
      * HostMemoryController.hpp：管理CPU上的内存。
      * SharedPtrTools.h：智能指针工具，可以创建一维或二维共享型实例化对象。
      * UniquePtrTools.h：智能指针工具，可以创建一维或二维独占型实例化对象。
    * **TimerTools**:
      * GPUtimer.hpp：GPU上的计时函数，统计GPU上的代码执行时间。
      * CPUtimer.cuh：CPU上的计时函数，统计CPU上的代码执行时间。
    * **TestTools**：
      * checkTools.hpp：一些用于检查代码执行结果是否准确的函数。
      * generateTools.hpp：一些用于生成测试数据的函数。
      * ReadMtxTools.h / ReadMtxTools.cpp：仿照mmio编写的CPP风格MTX读数据工具。
      * WriteMtxTools.h / WriteMtxTools.cpp：仿照mmio编写的CPP风格MTX写数据工具。
    * BaseUtils.h：用来存放一些不好归类但有用的数据结构和方法。
    * ErrorHandler.h：一些异常处理工具。
  * **test**：主要存放一些测试文件，所有测试文件必须以```test```开头，后缀是具体要测试的模块，名称应直观易懂。
    * **CUDA**：用于放一些CUDA测试文件。
      * testCublas.cu：测试cuBLAS库。
      * testCusparse.cu：测试cuSPARSE库。
    * testUtils.cpp：测试一些工具函数和工具类。
    * testVector.cpp：测试向量类。
    * testMatrix.cpp：测试矩阵类。
    * testMatrixOrdering.cpp：测试矩阵重排函数。
    * testMetis.cpp：测试Metis库。
    * testMKL.cpp：测试MKL库。
    * testOpenBLAS.cpp：测试OpenBLAS库。
    * testPrecond.cpp：测试预条件类。
    * testSolver.cpp：测试求解器类。
    * testAMSEDSolver.cpp：测试AMSED预条件。
* main.cpp：一个示例运行文件。
* CMakeLists.txt：CMake的配置文件，用来管理整个项目的编译工作。
* .gitignore：用来屏蔽一些不参与git管理的文件路径，如果有不需要上传至git仓库的文件或文件夹，必须在这个文件中添加相关文件或文件夹的路径。