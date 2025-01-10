#ifndef CUDATOOLS_CUDATOOLS_DEVICE_CU
#define CUDATOOLS_CUDATOOLS_DEVICE_CU

/**
 * @author: 邓轶丹
 * @date: 2023/11/22
 * @details: 基于CUDA的内核（仅在GPU上调用）代码
 */

#include "../utils/MemoryTools/DeviceMemoryController.cuh"


namespace DEVICE {
    template<typename ValType, typename CoefType1, typename CoefType2>
    __global__ void
    vecAddDevice(const CoefType1 &alpha, const ValType *a, const CoefType2 &beta, const ValType *b,
                 ValType *c, const int &data_length) {
        unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_idx < data_length)
            c[global_idx] = alpha * a[global_idx] + beta * b[global_idx];
    }


    /* 块内约简求和
     * 参考：《并行计算与高性能计算》P369 */
    template<typename ValType>
    __device__ void vecReductionWithinBlock(ValType *data) {
        const unsigned int thread_idx = threadIdx.x;
        const unsigned int block_size = blockDim.x;

        /* 对于块内线程号大于warpSize的线程，使用成对约简将其约简到线程号小于warpSize的那些线程上 */
        for (size_t offset = block_size >> 1; offset > MIN_REDUCE_SYNC_SIZE; offset >>= 1) {
            if (thread_idx < offset) {
                data[thread_idx] = data[thread_idx] + data[thread_idx + offset];
            }
            __syncthreads();        // 阻塞块直至块内的线程全都执行到这一行，但块间线程不能同步（含写入操作必须执行这一步）
        }
        /* 对于前warpSize个线程，直接约简 */
        if (thread_idx < MIN_REDUCE_SYNC_SIZE) {
            for (int offset = MIN_REDUCE_SYNC_SIZE; offset > 1; offset >>= 1) {
                data[thread_idx] = data[thread_idx] + data[thread_idx + offset];
                __syncthreads();
            }
            if (thread_idx == 0) data[thread_idx] += data[thread_idx + 1];   // 约简到只剩两个值，则直接相加
        }
    }

    template<typename ValType>
    __device__ void vecReductionWithinBlock_Kahan(ValType *data) {
        const unsigned int thread_idx = threadIdx.x;
        const unsigned int block_size = blockDim.x;

        /* 对于块内线程号大于warpSize的线程，使用成对约简将其约简到线程号小于warpSize的那些线程上 */
        ValType correction{0}, corrected_val, temp_sum;
        for (size_t offset = block_size >> 1; offset > MIN_REDUCE_SYNC_SIZE; offset >>= 1) {
            if (thread_idx >= offset && correction != 0) {
                data[thread_idx] -= correction;
                __syncthreads();
                correction = 0;
            }
            if (thread_idx < offset) {
                corrected_val = data[thread_idx + offset];
                temp_sum = data[thread_idx] + corrected_val;
                correction = (temp_sum - data[thread_idx]) - corrected_val;
                data[thread_idx] = temp_sum;
                __syncthreads();        // 阻塞块直至块内的线程全都执行到这一行，但块间线程不能同步（含写入操作必须执行这一步）
            }
        }
        if (thread_idx >= MIN_REDUCE_SYNC_SIZE && correction != 0) {
            data[thread_idx] -= correction;
            __syncthreads();
            correction = 0;
        }
        /* 对于前warpSize个线程，直接约简 */
        if (thread_idx < MIN_REDUCE_SYNC_SIZE) {
            for (int offset = MIN_REDUCE_SYNC_SIZE; offset > 1; offset >>= 1) {
                if (thread_idx >= offset && correction != 0) {
                    data[thread_idx] -= correction;
                    __syncthreads();
                    correction = 0;
                }
                corrected_val = data[thread_idx + offset];
                temp_sum = data[thread_idx] + corrected_val;
                correction = (temp_sum - data[thread_idx]) - corrected_val;
                data[thread_idx] = temp_sum;
                __syncthreads();
            }
        }
        if (thread_idx == 1 && correction != 0) data[thread_idx] -= correction;
        __syncthreads();
        if (thread_idx == 0) data[thread_idx] += data[thread_idx + 1];   // 约简到只剩两个值，则直接相加
    }


    /* 第一阶段：将线程块内的结果相加，并将求和后的值写入辅助数组 */
    /** @brief 用于求GPU上的归约求和。 */
    template<typename ValType>
    __global__ void vecReduceSumDevice1(
            const size_t data_length,            // 数组总长
            const ValType *data_ptr,        // 数组头指针
            ValType *reduced_ptr   // 用来存储所有块的块内约简结果
    ) {
        // 动态共享内存，主要用于共享内存的大小在编译时不能确定的情况，在运行该函数时必须指定<块个数，块大小，每块要分配的共享内存字节数>
        extern __shared__ ValType temp_vecReduceSum1[];

        const unsigned int thread_idx = threadIdx.x;
        const unsigned int block_idx = blockIdx.x;
        const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

        temp_vecReduceSum1[thread_idx] = 0.0;
        if (global_idx < data_length) {      // 把待计算的值拷贝到本地
            temp_vecReduceSum1[thread_idx] = data_ptr[global_idx];
        }

        __syncthreads();

        vecReductionWithinBlock(temp_vecReduceSum1);

        //  把本地结果写入到临时数组中
        if (thread_idx == 0) {
            reduced_ptr[block_idx] = temp_vecReduceSum1[0];
        }
    }


    /* 第二阶段：将第一阶段计算的块间结果求和相加（此时计算过程只使用一个线程块） */
    template<typename ValType>
    __global__ void vecReduceSumDevice2(
            const size_t data_length,                  ///< 第一阶段求出的临时结果数组长度；
            ValType *auxiliary_ptr                     ///< 第一阶段用到的临时数组。
    ) {
        extern __shared__ ValType temp_vecReduceSum2[];        // 还是动态共享内存
        const unsigned int thread_idx = threadIdx.x;
        const unsigned int block_size = blockDim.x;

        unsigned int global_idx = thread_idx;      // 数据的实际索引值

        temp_vecReduceSum2[thread_idx] = 0.0;

        // 把第一阶段的结果读入到本地
        if (thread_idx < data_length) temp_vecReduceSum2[thread_idx] = auxiliary_ptr[global_idx];

        // 对于数组超出线程数量的部分，先将其约简到对应线程上
        for (global_idx += block_size; global_idx < data_length; global_idx += block_size) {
            temp_vecReduceSum2[thread_idx] += auxiliary_ptr[global_idx];
        }

        __syncthreads();        // 写入值的操作一定要有这个步骤


        vecReductionWithinBlock(temp_vecReduceSum2);       // 调用通用的约简函数求最终值

        if (thread_idx == 0) {
            auxiliary_ptr[0] = temp_vecReduceSum2[0];     // 求和结果存在结果数组的第一个位置上，由0号线程写入最终结果变量
        }
    }

    /** @brief 用来求GPU上的Kahan求和。 */
    template<typename ValType>
    __global__ void vecReduceKahanDevice1(const size_t data_length,           ///< 数组总长；
                                          const ValType *data_ptr,            ///< 数组头指针；
                                          ValType *reduced_ptr                ///< 用来存储所有块的块内约简结果。
    ) {
        // 动态共享内存，主要用于共享内存的大小在编译时不能确定的情况，在运行该函数时必须指定<块个数，块大小，每块要分配的共享内存字节数>
        extern __shared__ ValType temp_vecReduceKahan1[];
        const unsigned int thread_idx = threadIdx.x;
        const unsigned int block_idx = blockIdx.x;
        const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

        temp_vecReduceKahan1[thread_idx] = 0.0;
        // 把待计算的值拷贝到本地
        if (global_idx < data_length) temp_vecReduceKahan1[thread_idx] = data_ptr[global_idx];
        __syncthreads();
        vecReductionWithinBlock_Kahan(temp_vecReduceKahan1);

        //  把本地结果写入到临时数组中
        if (thread_idx == 0) reduced_ptr[block_idx] = temp_vecReduceKahan1[0];
    }


    template<typename ValType>
    __global__ void vecReduceKahanDevice2(
            const size_t data_length,                 ///< 第一阶段求出的临时结果数组长度；
            ValType *auxiliary_ptr                    ///< 第一阶段用到的临时数组。
    ) {
        extern __shared__ ValType temp_vecReduceKahan2[];        // 还是动态共享内存
        const unsigned int thread_idx = threadIdx.x;
        const unsigned int block_size = blockDim.x;

        unsigned int global_idx = thread_idx;      // 数据的实际索引值

        temp_vecReduceKahan2[thread_idx] = 0.0;

        // 把第一阶段的结果读入到本地
        if (thread_idx < data_length) temp_vecReduceKahan2[thread_idx] = auxiliary_ptr[global_idx];

        // 对于数组超出线程数量的部分，先将其约简到对应线程上
        for (global_idx += block_size; global_idx < data_length; global_idx += block_size) {
            temp_vecReduceKahan2[thread_idx] += auxiliary_ptr[global_idx];
        }
        __syncthreads();        // 写入值的操作一定要有这个步骤

        vecReductionWithinBlock_Kahan(temp_vecReduceKahan2);       // 调用通用的约简函数求最终值

        if (thread_idx == 0) {
            auxiliary_ptr[0] = temp_vecReduceKahan2[0];     // 求和结果存在spad数组的第一个位置上，由0号线程写入最终结果变量
        }
    }


    /** @brief 对于分好段的数据，进行并行归约，每个线程负责segment_size长度的数据。 */
    template<typename ValType>
    __global__ void vecReduceSegmentedDevice(
            ValType *segmented_data,                ///< 已经分好段的数据，每段等长，段长度为参数segment_size；
            ValType *res,                           ///< 对每个分段并行求和，其结果存在这个数组中（需要提前开辟好对应的空间）；
            const size_t data_length,               ///< 原始数组长度；
            int segment_size                        ///< 分段长度，必须是2的n次幂。
    ) {
        const unsigned int thread_idx = threadIdx.x;
        const unsigned int block_idx = blockIdx.x;
        const unsigned int block_size = blockDim.x;
        const unsigned int global_idx = block_idx * block_size + thread_idx;
//        extern __shared__ ValType temp_vecReduceSegmented[];
//
//        temp_vecReduceSegmented[thread_idx] = 0;
//        if (global_idx < data_length) temp_vecReduceSegmented[thread_idx] = segmented_data[global_idx];
//        __syncthreads();

        const unsigned int offset = thread_idx % segment_size;
        const unsigned int final_idx = global_idx / segment_size + offset;
        if (offset == 0 && global_idx < data_length) {
            ValType temp_segSum = segmented_data[global_idx];
            for (int i = 1; i < segment_size; ++i) {
//                temp_vecReduceSegmented[thread_idx] += temp_vecReduceSegmented[thread_idx + i];
                temp_segSum += segmented_data[global_idx + i];
            }
//            __syncthreads();
//            res[final_idx] = temp_vecReduceSegmented[thread_idx];
            res[final_idx] = temp_segSum;
            __syncthreads();
        }
    }

// 以下这段代码可能在多个cu文件中编译时报“multi definition”的错误，初步怀疑是没有用模板
//    /** @brief 生成映射数组，用来将原始向量投射到分段后的新位置上 */
//    __global__ void
//    vecGenSegMapDevice(const UINT32 *old_offset, const UINT32 *new_offset, UINT32 *new_map, int row_num) {
//        const UINT32 thread_idx = threadIdx.x;
//        const UINT32 block_idx = blockIdx.x;
//        const UINT32 block_size = blockDim.x;
//        const UINT32 global_idx = block_idx * block_size + thread_idx;
//
//        UINT32 old_lower, old_upper, new_lower;
//        if (global_idx < row_num) {
//            old_lower = old_offset[global_idx];
//            old_upper = old_offset[global_idx + 1];
//            new_lower = new_offset[global_idx];
//            for (unsigned int i = 0; i < old_upper - old_lower; ++i) {
//                new_map[old_lower + i] = new_lower + i;
//            }
//            __syncthreads();
//        }
//    }


    /** @brief 对于每段长度不同的数据，进行并行分段归约，每个线程负责row_offset[global_idx + 1] - row_offset[global_idx]长度的数据。
     * @details 这个内核将使用row_num个线程，不需要额外分配空间。 */
    template<typename ValType>
    __global__ void vecReduceForDiffLenDevice(
            ValType *data,                          ///< 原始数据；
            ValType *res,                           ///< 对每个分段并行求和，其结果存在这个数组中（需要提前开辟好对应的空间）；
            const UINT32 *row_offset,               ///< 存储每一段待归约数据长度；
            const UINT32 row_num                    ///< 原始矩阵行数。
    ) {
        const unsigned int thread_idx = threadIdx.x;
        const unsigned int block_idx = blockIdx.x;
        const unsigned int block_size = blockDim.x;
        const unsigned int global_idx = block_idx * block_size + thread_idx;
        if (global_idx < row_num) {
            const UINT32 lower = row_offset[global_idx], upper = row_offset[global_idx + 1];
            ValType res_each_row = 0;
            for (UINT32 i = lower; i < upper; ++i) {
                res_each_row += data[i];
            }
            res[global_idx] = res_each_row;
            __syncthreads();
        }
    }

}
#endif //CUDATOOLS_CUDATOOLS_DEVICE_CU