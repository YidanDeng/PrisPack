cmake_minimum_required(VERSION 3.16)
project(PrisPack)

set(CMAKE_CXX_STANDARD 17)

#[[========================== C/C++编译选项配置 ==========================]]
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external_packages/cmakeFind/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external_packages/cmakeFind/FindSuiteSparse")

# 核心编译器参数配置
set(USE_VECTORIZE ON)  # 设置一个bool类型的变量，用来控制是否开启编译器自动向量化
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU") # using GCC
    #[[ GCC各类优化选项的解释：
        -fstrict-aliasing：启用严格别名规则检查，用于提高编译器优化的可能性。
        -ftree-vectorize: 开启向量化。
        -fno-tree-vectorize: 关闭向量化。
        -fopt-info-HOST-optimized：在编译过程中输出有关矢量化优化的信息。
        -fopenmp-simd：启用OpenMP指令的SIMD优化。
        -march=native -mtune=native：使用本地CPU架构进行优化，以获得最佳性能。
        -ffast-math：启用快速数学优化，如不对NaN和无穷大进行检查等。
     ]]
    if (USE_VECTORIZE)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fstrict-aliasing -fopenmp-simd -march=native -mtune=native -ffast-math -ftree-vectorize -fopt-info-vec-optimized")
    else ()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fstrict-aliasing -march=native -mtune=native -ffast-math -fno-tree-vectorize -fopt-info-vec-optimized")
    endif ()
    if ("${CMAKE_CXX_COMPILER_VERSION}" VERSION_GREATER "7.4.0")    # 根据当前编译器版本开启对512位向量的支持
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mprefer-vector-width=512")
    endif ("${CMAKE_CXX_COMPILER_VERSION}" VERSION_GREATER "7.4.0")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") # using Clang
    #[[
        -Rpass=loop-vectorize           :   标识成功矢量化的循环。
        -Rpass-missed=loop-vectorize    :   标识矢量化失败的循环，并指示是否指定了矢量化。
        -Rpass-analysis=loop-vectorize  :   标识导致矢量化失败的语句。如果另外提供了-fsave优化记录，则可能会列出导致矢量化失败的多种原因（这种行为在将来可能会发生变化）。
    ]]
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fstrict-aliasing -fvectorize -march=native -mtune=native -ffast-math -Rpass-analysis=loop-vectorize")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC") # using MSVC
    # 启用最高级别的优化
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Ox")
    # string(REPLACE "/Od" "/Ox" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}") # 如果编译器告警，把上面的注释掉，替换成现在这句
    # 启用向量化
    if (USE_VECTORIZE)
        string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
        string(REPLACE "/RTC1" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
        add_compile_options(/arch:AVX2)     # VS 2019貌似不支持512位向量
        add_compile_options(/fp:fast)
        # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
        add_compile_options(/GL)    # 开启这个就默认开启LTCG
        add_link_options(/INCREMENTAL:NO)   # INCREMENTAL是LTCG默认的选项，和程序不兼容，所以取消
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Qvec-report:2")
    endif ()
endif ()
# 传递给子目录
set(GLOBAL_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE INTERNAL "")
set(GLOBAL_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" CACHE INTERNAL "")

# OpenMP 配置
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OpenMP found.")
    add_definitions(-DOPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# OpenBLAS 库
find_package(OpenBLAS REQUIRED)
if (OpenBLAS_FOUND)
    include_directories(${OpenBLAS_INCLUDE_DIR})
    message(STATUS "Finding OpenBLAS library and header... done.")
endif ()

# MKL库
# 查找MKL包
find_package(MKL REQUIRED)
if (MKL_FOUND)
    # 包含头文件
    include_directories(${MKL_INCLUDE_DIR})
    # 添加编译选项
    add_definitions(-DMKL_FOUND)
endif (MKL_FOUND)


# Metis库
find_package(METIS REQUIRED)
if (METIS_FOUND)
    # 包含 Metis 的头文件目录
    include_directories(${METIS_INCLUDE_DIRS})
    message(STATUS "Finding Metis library and header... done.")
endif ()

# SuiteSparse库
#SuiteSparseConfig
find_package(SuiteSparse_config REQUIRED)
if (SUITESPARSE_CONFIG_FOUND)
    include_directories(${SUITESPARSE_CONFIG_INCLUDE_DIR})
    message(STATUS "Finding SuiteSparse_config library and header... done.")
endif ()

# AMD库(SuiteSparse)
find_package(AMD REQUIRED)
if (AMD_FOUND)
    message(STATUS "Finding AMD functions ... done.")
    # 设置try_compile的输出目录，通常在二进制目录下创建一个子目录
    set(TEST_COMPILE_DIR ${CMAKE_BINARY_DIR}/TestAMD)
    file(MAKE_DIRECTORY ${TEST_COMPILE_DIR})
    # 尝试编译测试代码
    try_compile(
            HAVE_AMD_FUNC
            ${TEST_COMPILE_DIR}            # 编译输出目录
            ${CMAKE_SOURCE_DIR}/external_packages/cmakeFind/FindSuiteSparse/SuiteSparseTestCompile.cpp # 测试源文件
            COMPILE_DEFINITIONS "-DHAVE_AMD_FUNC" # 需要的编译定义
            LINK_LIBRARIES ${AMD_LIBRARY}   # 链接的库
    )
    if (HAVE_AMD_FUNC)
        include_directories(${AMD_INCLUDE_DIR})
        message(STATUS "Trying AMD files -- works")
    else ()
        message(STATUS "Trying AMD files -- failed!")
    endif ()
else ()
    message(STATUS "AMD functions was not found.")
endif ()


# RCM库
add_subdirectory(external_packages/RCM)

#[[========================== CUDA编译选项配置 ==========================]]
#[[
    如果linux系统下编译时报找不到CUDA编译器相关的提示，请在编译时为cmake添加以下选项（对应路径填实际CUDA toolkit安装路径）：
    -G "Unix Makefiles"
    -D CMAKE_CUDA_COMPILER="/usr/local/cuda-12.3/bin/nvcc"
    -D CMAKE_C_COMPILER="/usr/bin/gcc"
    -D CMAKE_CXX_COMPILER="/usr/bin/g++"
]]
# 检查是否设置了CMAKE_CUDA_COMPILER
if (NOT CMAKE_CUDA_COMPILER)
    message(STATUS "CMAKE_CUDA_COMPILER not set. Skipping CUDA support.")
else ()
    if (WIN32)
        # 在Windows系统上执行的语句
        message("This is a Windows system. Setting CUDA options...")
        find_package(CUDA REQUIRED)
    elseif (UNIX)
        # 在UNIX系统上执行的语句
        enable_language(CUDA)   #激活CUDA语言支持
        message("This is a UNIX-like system. Setting CUDA options...")
        find_package(CUDAToolkit REQUIRED) # 如果不显式设置CMAKE_CUDA_ARCHITECTURES变量，CMake会自动检测并选择当前系统支持的CUDA架构
    endif ()
endif ()

if (CUDAToolkit_FOUND OR CUDA_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES OFF) # 自动检测并适应当前 GPU 架构
    set(CMAKE_CUDA_STANDARD 17)
    # 获取nvcc编译器的路径
    get_filename_component(CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_ROOT_DIR ${CUDA_BIN_DIR}/.. ABSOLUTE)
    # 假设CUDA头文件位于CUDA Toolkit根目录下的include文件夹
    set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include")
    # 设置CUDA库路径
    set(CUDA_LIBRARY_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    # 将CUDA头文件路径添加到项目的包含目录列表中
    include_directories(${CUDA_INCLUDE_DIRS})
    message("Finding CUDA toolkits -- done.")
    message(STATUS "CUDA headers path: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA compiler path: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA library path: ${CUDA_LIBRARY_DIRS}")
    message(STATUS "CUDA found. Compiling with CUDA support.")
    add_definitions(-DCUDA_ENABLED)
    # sets CMAKE_{C,CXX}_FLAGS from CUDA compile flags. Includes DEBUG and RELEASE
    set(CUDA_PROPAGATE_HOST_FLAGS ON) # default is on
    set(CUDA_SEPARABLE_COMPILATION ON) # default is off
    #    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 ${ARCH_FLAGS})
    # 根据编译类型设置不同的优化选项
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O0 -g -G)
    elseif (CMAKE_BUILD_TYPE MATCHES Release)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 # 编译优化
                -Xcompiler "-mavx2"         # 传递给底层编译器的额外标志
                -Xcompiler "-march=native"  # 针对本机架构优化
                --use_fast_math
                -ftz=true   # flush-to-zero，把特别小的数直接当0算，提升计算速度
        )
    elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo) # 结合了发布版本的性能优化和调试信息
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O2 -g --generate-line-info)
    endif ()
    #    include_directories(${CUDA_INCLUDE_DIRS})
else ()
    message(STATUS "CUDA not found. Compiling without CUDA support.")
endif ()


#[[========================== 可执行文件配置 ==========================]]
file(GLOB_RECURSE CPPUtils ${CMAKE_SOURCE_DIR}/source/utils/*.cpp)
message("Including util files: ${CPPUtils}")
file(GLOB CPPMatrixClass ${CMAKE_SOURCE_DIR}/source/MatrixClass/*.cpp)
message("Including Matrix files: ${CPPMatrixClass}")
file(GLOB CPPVectorClass ${CMAKE_SOURCE_DIR}/source/VectorClass/*.cpp)
message("Including Vector files: ${CPPVectorClass}")
file(GLOB CPPPrecond ${CMAKE_SOURCE_DIR}/source/Preconditioner/*.cpp)
message("Including Precondition files: ${CPPPrecond}")
file(GLOB CPPSolver ${CMAKE_SOURCE_DIR}/source/Solver/*.cpp)
message("Including Solver files: ${CPPSolver}")

if (CUDAToolkit_FOUND OR CUDA_FOUND)
    #[[ CUDA 文件]]
    message("Execute both CUDA files and CPP files...")
    file(GLOB_RECURSE CUDAFiles ${CMAKE_SOURCE_DIR}/source/MatrixClass/*.cu
            ${CMAKE_SOURCE_DIR}/source/VectorClass/*.cu
            ${CMAKE_SOURCE_DIR}/source/Preconditioner/*.cu
            ${CMAKE_SOURCE_DIR}/source/CUDA/*.cu)
    message("Including CUDA files: ${CUDAFiles}")
    add_executable(testAMSEDSolver ${CMAKE_SOURCE_DIR}/source/test/testAMSEDSolver.cpp ${CPPVectorClass} ${CPPMatrixClass}
            ${CPPUtils} ${CPPPrecond} ${CPPSolver} ${CUDAFiles})
    add_executable(testPrecond ${CMAKE_SOURCE_DIR}/source/test/testPrecond.cpp ${CPPVectorClass} ${CPPMatrixClass}
            ${CPPUtils} ${CPPPrecond} ${CUDAFiles})

    # 基本测试文件
    add_executable(testCusparse ${CMAKE_SOURCE_DIR}/source/test/CUDA/testCusparse.cu ${CPPVectorClass} ${CPPUtils}
            ${CPPMatrixClass} ${CUDAFiles})
    add_executable(testCublas ${CMAKE_SOURCE_DIR}/source/test/CUDA/testCublas.cu ${CPPVectorClass} ${CPPUtils}
            ${CPPMatrixClass} ${CUDAFiles})

    #查找并包含CUDA库（例如CuBLAS和CuSPARSE）
    find_library(CUBLAS_LIB cublas PATHS ${CUDA_LIBRARY_DIRS})
    find_library(CUSPARSE_LIB cusparse PATHS ${CUDA_LIBRARY_DIRS})
    message(STATUS "CuBLAS lib path: ${CUBLAS_LIB}")
    message(STATUS "CuSPARSE lib path: ${CUSPARSE_LIB}")

    target_link_libraries(testAMSEDSolver ${CUBLAS_LIB} ${CUSPARSE_LIB})
    target_link_libraries(testPrecond ${CUBLAS_LIB} ${CUSPARSE_LIB})
    target_link_libraries(testCusparse ${CUBLAS_LIB} ${CUSPARSE_LIB})
    target_link_libraries(testCublas ${CUBLAS_LIB} ${CUSPARSE_LIB})


    #[[========================== 链接库配置 ==========================]]
    # 将RCM库链接到指定对象
    target_link_libraries(testAMSEDSolver RCMTools)
    target_link_libraries(testPrecond RCMTools)

    if (OpenBLAS_FOUND)
        target_link_libraries(testAMSEDSolver ${OpenBLAS_LIBRARIES})
        target_link_libraries(testPrecond ${OpenBLAS_LIBRARIES})
        message("Linking OpenBLAS: ${OpenBLAS_LIBRARIES}")
    endif ()

    if (METIS_FOUND)
        target_link_libraries(testAMSEDSolver ${METIS_LIBRARY} ${GKLIB_LIBRARY})
        target_link_libraries(testPrecond ${METIS_LIBRARY} ${GKLIB_LIBRARY})
        message("Linking metis and gklib: ${METIS_LIBRARY} ${GKLIB_LIBRARY}")
    endif ()

    if (AMD_FOUND)
        target_link_libraries(testAMSEDSolver ${AMD_LIBRARY} ${SUITESPARSE_CONFIG_LIBRARIES})
        target_link_libraries(testPrecond ${AMD_LIBRARY} ${SUITESPARSE_CONFIG_LIBRARIES})
    endif ()

else ()
    message("Execute only CPP files...")
    # 其他测试文件
    add_executable(testMatrix ${CMAKE_SOURCE_DIR}/source/test/testMatrix.cpp ${CPPVectorClass} ${CPPMatrixClass} ${CPPUtils})
    add_executable(testVector ${CMAKE_SOURCE_DIR}/source/test/testVector.cpp ${CPPVectorClass} ${CPPMatrixClass} ${CPPUtils})
    add_executable(testUtils ${CMAKE_SOURCE_DIR}/source/test/testUtils.cpp ${CPPVectorClass} ${CPPMatrixClass} ${CPPUtils})
    add_executable(testPrecond ${CMAKE_SOURCE_DIR}/source/test/testPrecond.cpp ${CPPVectorClass} ${CPPMatrixClass}
            ${CPPUtils} ${CPPPrecond})
    add_executable(testSolver ${CMAKE_SOURCE_DIR}/source/test/testSolver.cpp ${CPPVectorClass} ${CPPMatrixClass}
            ${CPPUtils} ${CPPPrecond} ${CPPSolver})
    add_executable(testMatrixOrdering ${CMAKE_SOURCE_DIR}/source/test/testMatrixOrdering.cpp ${CPPVectorClass} ${CPPMatrixClass}
            ${CPPUtils} ${CPPPrecond} ${CPPSolver})

    add_executable(testMKL ${CMAKE_SOURCE_DIR}/source/test/testMKL.cpp ${CPPVectorClass} ${CPPMatrixClass}
            ${CPPUtils} ${CPPPrecond} ${CPPSolver})

    add_executable(testAMSEDSolver ${CMAKE_SOURCE_DIR}/source/test/testAMSEDSolver.cpp ${CPPVectorClass} ${CPPMatrixClass} ${CPPUtils} ${CPPPrecond} ${CPPSolver})
    add_executable(${PROJECT_NAME} main.cpp)

    #[[========================== 链接库配置 ==========================]]
    # 将RCM库链接到指定对象
    target_link_libraries(testMatrixOrdering RCMTools)
    target_link_libraries(testPrecond RCMTools)
    target_link_libraries(testSolver RCMTools)
    target_link_libraries(testAMSEDSolver RCMTools)
    target_link_libraries(testMKL RCMTools)

    if (OpenBLAS_FOUND)
        add_executable(testOpenBLAS ${CMAKE_SOURCE_DIR}/source/test/testOpenBLAS.cpp ${CPPVectorClass} ${CPPMatrixClass}
                ${CPPUtils})
        target_link_libraries(testSolver ${OpenBLAS_LIBRARIES})
        target_link_libraries(testOpenBLAS ${OpenBLAS_LIBRARIES})
        target_link_libraries(testMatrixOrdering ${OpenBLAS_LIBRARIES})
        target_link_libraries(testMKL ${OpenBLAS_LIBRARIES})
        target_link_libraries(testPrecond ${OpenBLAS_LIBRARIES})
        target_link_libraries(testAMSEDSolver ${OpenBLAS_LIBRARIES})
        message("Linking OpenBLAS: ${OpenBLAS_LIBRARIES}")
    endif ()

    if (METIS_FOUND)
        add_executable(testMetis ${CMAKE_SOURCE_DIR}/source/test/testMetis.cpp)
        target_link_libraries(testMetis ${METIS_LIBRARY} ${GKLIB_LIBRARY})
        target_link_libraries(testSolver ${METIS_LIBRARY} ${GKLIB_LIBRARY})
        target_link_libraries(testPrecond ${METIS_LIBRARY} ${GKLIB_LIBRARY})
        target_link_libraries(testMatrixOrdering ${METIS_LIBRARY} ${GKLIB_LIBRARY})
        target_link_libraries(testMKL ${METIS_LIBRARY} ${GKLIB_LIBRARY})
        target_link_libraries(testAMSEDSolver ${METIS_LIBRARY} ${GKLIB_LIBRARY})
        message("Linking metis and gklib: ${METIS_LIBRARY} ${GKLIB_LIBRARY}")
    endif ()

    if (AMD_FOUND)
        target_link_libraries(testMatrixOrdering ${AMD_LIBRARY} ${SUITESPARSE_CONFIG_LIBRARIES})
        target_link_libraries(testMKL ${AMD_LIBRARY} ${SUITESPARSE_CONFIG_LIBRARIES})
        target_link_libraries(testPrecond ${AMD_LIBRARY} ${SUITESPARSE_CONFIG_LIBRARIES})
        target_link_libraries(testSolver ${AMD_LIBRARY} ${SUITESPARSE_CONFIG_LIBRARIES})
        target_link_libraries(testAMSEDSolver ${AMD_LIBRARY} ${SUITESPARSE_CONFIG_LIBRARIES})
    endif ()

    if(MKL_FOUND)
        target_link_libraries(testMKL -m64 ${MKL_LIBRARIES})
    endif ()
endif ()


