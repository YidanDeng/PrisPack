# RCM/CMakeLists.txt
# 引用根目录传递的全局优化标志
set(CMAKE_CXX_FLAGS_RELEASE "${GLOBAL_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${GLOBAL_CXX_FLAGS_DEBUG}")

# 添加包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 创建 RCM 库
add_library(RCMTools SHARED ${CMAKE_CURRENT_SOURCE_DIR}/lib/rcm.cpp)
#add_library(RCMTools SHARED lib/rcm.cpp)

# 设置输出目录
# 根据编译模式设置不同的后缀
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(RCMTools PROPERTIES
            OUTPUT_NAME "RCMTools_debug"
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug
    )
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    set_target_properties(RCMTools PROPERTIES
            OUTPUT_NAME "RCMTools_release"
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/release
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/release
    )
endif ()


# 公开包含目录给其他目标使用
target_include_directories(RCMTools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


