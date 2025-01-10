# - Try to find Intel MKL
# Once done, this will define
#
#  MKL_FOUND - system has Intel MKL
#  MKL_INCLUDE_DIR - the MKL include directory
#  MKL_LIBRARIES - link these to use MKL
#  MKL_DEFINITIONS - compiler switches required for using MKL

if (MKL_INCLUDE_DIR)
    # Already in cache, be silent
    set (MKL_FIND_QUIETLY TRUE)
endif (MKL_INCLUDE_DIR)

# MKL include目录，使用默认路径即可
find_path(MKL_INCLUDE_DIR
        NAMES mkl.h
        HINTS
        /opt/intel/oneapi/mkl/2025.0/include
        /opt/intel/oneapi/mkl/latest/include
        ${CMAKE_SOURCE_DIR}/external/mkl/include
        PATH_SUFFIXES mkl
)

# MKL库文件目录，使用默认路径即可
set(MKL_LIB_PATHS
        /opt/intel/oneapi/mkl/2025.0/lib
        /opt/intel/oneapi/mkl/latest/lib
        ${CMAKE_SOURCE_DIR}/external/mkl/lib
)

# 查找具体的MKL库
find_library(MKL_INTEL_LP64_LIB
        NAMES mkl_intel_lp64
        HINTS ${MKL_LIB_PATHS}
)

find_library(MKL_GNU_THREAD_LIB
        NAMES mkl_gnu_thread
        HINTS ${MKL_LIB_PATHS}
)

find_library(MKL_CORE_LIB
        NAMES mkl_core
        HINTS ${MKL_LIB_PATHS}
)

find_library(MKL_RT_LIB
        NAMES mkl_rt
        HINTS ${MKL_LIB_PATHS}
)

# 组合库
set(MKL_LIBRARIES
        ${MKL_INTEL_LP64_LIB}
        ${MKL_GNU_THREAD_LIB}
        ${MKL_CORE_LIB}
        ${MKL_RT_LIB}
        pthread
        m
        dl
)

# 处理QUIETLY和REQUIRED参数
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG
        MKL_INCLUDE_DIR
        MKL_INTEL_LP64_LIB
        MKL_GNU_THREAD_LIB
        MKL_CORE_LIB
)

# 如果找到MKL
if (MKL_FOUND)
    message(STATUS "MKL found.")
    message(STATUS "MKL include path: ${MKL_INCLUDE_DIR}")
    message(STATUS "MKL libraries: ${MKL_LIBRARIES}")

    # 设置编译宏定义（可选）
    set(MKL_DEFINITIONS "-DMKL_ILP64")
else (MKL_FOUND)
    if (MKL_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find MKL")
    endif (MKL_FIND_REQUIRED)
endif (MKL_FOUND)

# 避免在后续查找中重复搜索
mark_as_advanced(
        MKL_INCLUDE_DIR
        MKL_INTEL_LP64_LIB
        MKL_GNU_THREAD_LIB
        MKL_CORE_LIB
)