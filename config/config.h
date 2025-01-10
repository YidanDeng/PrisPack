#ifndef PMSLS_DEV_CONFIG_H
#define PMSLS_DEV_CONFIG_H

extern "C++" {
// 类型规约
typedef short INT16;
typedef int INT32;
typedef long long int INT64;
typedef unsigned int UINT32;
typedef unsigned short int UINT16;
typedef unsigned char UINT8;
typedef float FLOAT32;
typedef double FLOAT64;

// 全局定义
#define ALIGNED_BYTES 64        // 设置缓存行大小（对齐字节数）
// 宏定义检查是否接近零
#define IS_FLOAT_NEAR_ZERO(x) (fabsf(x) <= FLT_EPSILON * 10)
#define IS_DOUBLE_NEAR_ZERO(x) (fabs(x) <= DBL_EPSILON * 10)
#define IS_LONG_DOUBLE_NEAR_ZERO(x) (fabsl(x) <= LDBL_EPSILON * 10)
#define PRINT_MAX_LENGTH 30
// 起始下标设定
#define USE_ZERO_BASED_INDEX 0
#define USE_ONE_BASED_INDEX 1
#define MAX_ITER_NUM_SOLVER 500 //求解器的最大迭代次数

// openmp
#define THREAD_NUM 4
#define MAX_THREAD_NUM 4

/* CUDA 配置 */
#define MIN_REDUCE_SYNC_SIZE    warpSize
#define MAX_BLOCK_SIZE          256
#define MAX_SEGMENT_SIZE        64
#define MAX_STREAM_SIZE         16
#define DEFAULT_GPU             0

#define DEVICE_PTR(DATA_TYPE)       thrust::device_ptr<DATA_TYPE>
#define DEVICE_VECTOR(DATA_TYPE)    DEVICE::DeviceVector<DATA_TYPE>
#define EXTRACT_DEV_PTR(DEVICE_PTR)     thrust::raw_pointer_cast(DEVICE_PTR)
#define PACK_RAW_PTR(DEVICE_PTR)    thrust::device_pointer_cast(DEVICE_PTR)

/* MTX配置 */
enum MTX_DATA_TYPE {
    MTX_DATA_REAL, ///< 实数域矩阵
    MTX_DATA_COMPLEX, ///< 复数域矩阵
    MTX_DATA_PATTERN, ///< 整数构成的矩阵
    MTX_DATA_INTEGER ///< 模式矩阵（只有元素的位置，没有具体的值）
};

typedef enum MTX_DATA_TYPE mtx_data_type_t;

enum MTX_STORAGE_SCHEME {
    MTX_STORAGE_GENERAL, ///< 一般矩阵
    MTX_STORAGE_HERMITIAN, ///< Hermitian矩阵
    MTX_STORAGE_SYMMETRIC, ///< 对称矩阵
    MTX_STORAGE_SKEW ///< 反对称矩阵
};

typedef enum MTX_STORAGE_SCHEME mtx_storage_scheme_t;

enum MTX_STRUCTURE_TYPE {
    MTX_STRUCTURE_COO,
    MTX_STRUCTURE_ARRAY
};

typedef enum MTX_STRUCTURE_TYPE mtx_structure_type_t;
}


#endif //PMSLS_DEV_CONFIG_H
