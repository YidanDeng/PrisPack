/*
 * @author  邓轶丹
 * @date    2023/11/30
 * @details GPU上的内存管理函数
 */

#ifndef CUDATOOLS_DEVICEMEMORYCONTROLLER_CUH
#define CUDATOOLS_DEVICEMEMORYCONTROLLER_CUH

#include "../../../config/CUDAheaders.cuh"
#include "../ErrorHandler.h"
#include "HostMemoryController.hpp"

enum allocType {
    DoNotAlloc = 0,
    AllocHost = 1,
    AllocDevice = 2
};

typedef enum allocType allocType_t;

namespace DEVICE {

    /** @brief 创建GPU上内存 */
    template<typename T>
    inline cudaError_t allocMemoryDevice(T *&dev_ptr, size_t byte_size) {
        T *new_dev_ptr = nullptr;
        cudaError_t status = cudaMalloc(&new_dev_ptr, byte_size);
        if (new_dev_ptr != nullptr) dev_ptr = new_dev_ptr;
        return status;
    }

    /** @brief 创建CPU上的锁页内存 */
    template<typename T>
    inline cudaError_t allocMemoryHost(T *&host_ptr, size_t byte_size) {
        return cudaMallocHost(&host_ptr, byte_size);
    }

    /** @brief 释放CPU上的锁页内存 */
    template<typename T>
    inline cudaError_t freeMemoryHost(T *&host_ptr) {
        cudaError_t status = cudaFreeHost(host_ptr);
        host_ptr = nullptr;
        return status;
    }


    /** @brief 释放当前一维数组，并且重置指针为空指针 */
    template<typename T>
    inline cudaError_t freeAndResetDevicePointer(T *&dev_ptr) {
        cudaError_t status = cudaFree(dev_ptr);
        dev_ptr = nullptr;
        return status;
    }

    /** @brief 实现批量释放任意数量的有限个一维数组 */
    template<typename ...T>
    inline void freeGPUmemoryGroup(T *&...device_ptr) {
        INT32 status_list[] = {(freeAndResetDevicePointer(device_ptr))...};
        UINT32 args_length = sizeof...(device_ptr);
        for (UINT32 i = 0; i < args_length; ++i) {
            THROW_EXCEPTION(status_list[i] != cudaSuccess,
                            THROW_BAD_ALLOC("cudaFree failed at pointer " + std::to_string(i)))
        }
    }

    /** @brief 实现从CPU到GPU的数据拷贝 */
    template<typename T>
    inline void copyFromHostToDevice(
            const T *host_ptr,        ///< 主机上的内存指针
            T *&dev_ptr,        ///< 设备上的内存指针
            size_t byte_size,      ///< 内存空间大小（单位：字节）
            allocType_t alloc_dev_memory     ///< 是否需要分配设备上的内存，如果需要则传入“AllocDevice”
    ) {
        if (alloc_dev_memory == AllocDevice) {        // 这里必须注意dev_ptr是不是一个可以被赋值的指针，否则会出错
            INT32 status = allocMemoryDevice(dev_ptr, byte_size);
            if (status != memoryOptionSuccess) {
                std::cerr << "[ERROR] Allocation for host HOST failed!" << std::endl;
                exit(-1);
            }
        }
        cudaMemcpy(dev_ptr, host_ptr, byte_size, cudaMemcpyHostToDevice);
    }

    /** @brief 实现在GPU内部发生的数据拷贝 */
    template<typename T>
    inline void copyFromDeviceToDevice(
            const T *dev_src,             ///< GPU上的源内存指针
            T *&dev_dst,            ///< GPU上的目标内存指针
            size_t byte_size,          ///< 内存大小（单位：字节）
            allocType_t alloc_dst_memory     ///< 是否需要给目标内存分配空间，如果需要则传入“AllocDevice”
    ) {
        if (alloc_dst_memory == AllocDevice) {
            INT32 status = allocMemoryDevice(dev_dst, byte_size);
            if (status != cudaSuccess) {
                std::cerr << "[ERROR] Allocation for host HOST failed!" << std::endl;
                exit(-1);
            }
        }
        cudaMemcpy(dev_dst, dev_src, byte_size, cudaMemcpyDeviceToDevice);
    }

    /** @brief 实现从GPU到CPU的数据拷贝 */
    template<typename T>
    inline void copyFromDeviceToHost(
            const T *dev_ptr,         ///< GPU上的内存指针
            T *&host_ptr,       ///< CPU上的内存指针
            size_t byte_size,      ///< 数据大小（单位：字节）
            allocType_t alloc_host_memory     ///< 是否需要对目标指针分配空间，如果需要给host_ptr分配内存，传入“AllocHost”
    ) {
        if (alloc_host_memory == AllocHost) {
            INT32 status = HOST::allocAlignedMemory(host_ptr, byte_size);
            if (status != memoryOptionSuccess) {
                std::cerr << "[ERROR] Allocation for host HOST failed!" << std::endl;
                exit(-1);
            }
        }
        cudaMemcpy(host_ptr, dev_ptr, byte_size, cudaMemcpyDeviceToHost);
    }

    /** @brief 实现在CPU内部发生的数据拷贝，这个操作主要由GPU发起，注意和直接使用memcpy的区别 */
    template<typename T>
    inline void copyFromHostToHost(
            const T *src_ptr,         ///< CPU上的源指针
            T *&dst_ptr,        ///< CPU上的目的指针
            size_t byte_size,      ///< 内存大小（单位：字节）
            allocType_t alloc_dst_memory     ///< 是否需要对目标指针分配空间，如果需要给host_ptr分配内存，传入“AllocHost”
    ) {
        if (alloc_dst_memory == AllocHost) {
            INT32 status = HOST::allocAlignedMemory(dst_ptr, byte_size);
            if (status != memoryOptionSuccess) {
                std::cerr << "[ERROR] Allocation for host HOST failed!" << std::endl;
                exit(-1);
            }
        }
        cudaMemcpy(dst_ptr, src_ptr, byte_size, cudaMemcpyHostToHost);
    }
}


#endif //CUDATOOLS_DEVICEMEMORYCONTROLLER_CUH
