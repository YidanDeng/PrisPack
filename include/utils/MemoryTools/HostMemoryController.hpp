/*
 * @author  邓轶丹
 * @date    2023/11/27
 * @details CPU上的内存管理函数
 */

#ifndef CUDATOOLS_ALIGNEDMEMCOTROLLER_H
#define CUDATOOLS_ALIGNEDMEMCOTROLLER_H
extern "C++" {
#include "../../../config/headers.h"


enum memoryError {
    memoryOptionSuccess = 0,
    memoryAllocationFailed = 1,
    memoryFreeFailed = 2
};

typedef enum memoryError memoryError_t;


namespace HOST {
    static inline void currentMemoryUsage(long long int &total) {
        struct rusage r_usage{}; // 定义一个 rusage 结构体变量，用于存储资源使用情况
        getrusage(RUSAGE_SELF, &r_usage); // 获取当前进程的资源使用情况
        total = r_usage.ru_maxrss; // 获取最大驻留集大小（以KB为单位）
    }


    template<typename T>
    memoryError_t allocUnalignedMemory(T *&unaligned_memory, const size_t & alloc_bytes) {
        T *new_ptr = nullptr;
        new_ptr = (T *) malloc(alloc_bytes);
        if (new_ptr) {
            unaligned_memory = new_ptr;
            return memoryOptionSuccess;
        } else {
#ifndef NDEBUG
            SHOW_ERROR("HOST allocation for unaligned HOST failed!")
#endif
            return memoryAllocationFailed;
        }
    }

    template<typename T>
    memoryError_t reallocUnalignedMemory(T *&unaligned_memory, const size_t & alloc_bytes) {
        T *new_ptr = nullptr;
        new_ptr = (T *) realloc(unaligned_memory, alloc_bytes);
        if (new_ptr) {
            unaligned_memory = new_ptr;
            return memoryOptionSuccess;
        } else {
#ifndef NDEBUG
            SHOW_ERROR("HOST re-allocation for unaligned HOST failed!")
#endif
            return memoryAllocationFailed;
        }
    }


    template<typename T>
    memoryError_t allocAlignedMemory(T *&aligned_memory, const size_t & alloc_bytes) {
        // 调整对齐内存到指定的对齐数
        aligned_memory = nullptr;
        INT32 alignment = ALIGNED_BYTES;
#ifdef _WIN32 // Windows系统，不论32位还是64位都会定义
        T *new_ptr = nullptr;
        new_ptr = (T *) _aligned_malloc(alloc_bytes, alignment);
        if (!new_ptr) {
            std::cerr << "[ERROR] allocating aligned HOST failed!" << std::endl;
            return memoryAllocationFailed;
        }
        aligned_memory = new_ptr;
#else  // Linux系统
        void *new_mem = nullptr;
        INT32 status = posix_memalign(&new_mem, alignment, alloc_bytes);
        if (status != 0) return memoryAllocationFailed;
        aligned_memory = (T *) new_mem;
#endif
        return memoryOptionSuccess;
    }

    template<typename T>
    memoryError_t reallocAlignedMemory(T *&aligned_memory, const size_t & old_bytes, const size_t & new_bytes) {
        // 调整对齐内存到指定的对齐数
#ifdef _WIN32 // Windows系统，不论32位还是64位都会定义
        T *new_ptr = nullptr;
        new_ptr = (T *) _aligned_realloc(aligned_memory, new_bytes, ALIGNED_BYTES);
        if (!new_ptr) {
            std::cerr << "[ERROR] re-allocating aligned HOST failed!" << std::endl;
            return memoryAllocationFailed;
        }
        aligned_memory = new_ptr;
#else  // Linux系统
        void *new_mem = nullptr;
        INT32 status = posix_memalign(&new_mem, ALIGNED_BYTES, new_bytes);
        if (status != 0) return memoryAllocationFailed;
        size_t cpy_bytes = new_bytes <= old_bytes ? new_bytes : old_bytes;
        void *copy_mem = memcpy(new_mem, aligned_memory, cpy_bytes);
        if (copy_mem != new_mem) {
#ifndef NDEBUG
            SHOW_ERROR("The HOST copy func failed!")
#endif
            exit(EXIT_FAILURE);
        }
        free(aligned_memory);
        aligned_memory = (T *) new_mem;
#endif
        return memoryOptionSuccess;
    }


    template<typename T>
    memoryError_t freeAlignedMemory(T *&aligned_mem) {
        // 释放对齐内存
        try {
#ifdef _WIN32
            _aligned_free(aligned_mem);
#else
            free(aligned_mem);
#endif
            aligned_mem = nullptr;
        } catch (std::exception &error) {
            std::cerr << YELLOW << __FILE__ << ", " << __func__ << ", " << __LINE__ << ": " << L_RED
                      << "[ERROR] Release aligned HOST failed! error type: " << error.what() << COLOR_NONE << std::endl;
            return memoryFreeFailed;
        }
        return memoryOptionSuccess;
    }

    template<typename T>
    memoryError_t freeUnalignedMemory(T *&unaligned_memory) {
        try {
            free(unaligned_memory);
            unaligned_memory = nullptr;
        } catch (std::exception &error) {
            std::cerr << YELLOW << __FILE__ << ", " << __func__ << ", " << __LINE__ << ": " << L_RED
                      << "[ERROR] Release unaligned HOST failed! error type: " << error.what() << COLOR_NONE
                      << std::endl;
            return memoryFreeFailed;
        }
        return memoryOptionSuccess;
    }


    template<typename ...T>
    void freeAlignedMemoryGroup(T *&...host_ptr) {
        INT32 args_length = sizeof...(host_ptr);
        INT32 status_list[] = {(freeAlignedMemory(host_ptr))...};
        for (INT32 i = 0; i < args_length; ++i) {
            if (status_list[i] != memoryOptionSuccess) {
                std::cerr << YELLOW << __FILE__ << ", " << __func__ << ", " << __LINE__ << ": " << L_RED
                          << "[ERROR] Host-HOST Free failed!" << COLOR_NONE << std::endl;
                exit(-1);
            }
        }
    }


    template<typename ...T>
    void freeUnalignedMemoryGroup(T *&...host_ptr) {
        INT32 args_length = sizeof...(host_ptr);
        INT32 status_list[] = {(freeUnalignedMemory(host_ptr))...};
        for (INT32 i = 0; i < args_length; ++i) {
            if (status_list[i] != memoryOptionSuccess) {
                std::cerr << YELLOW << __FILE__ << ", " << __func__ << ", " << __LINE__ << ": " << L_RED
                          << "[ERROR] Host-HOST Free failed!" << COLOR_NONE << std::endl;
                exit(-1);
            }
        }
    }


    template<typename T>
    inline memoryError_t allocMemory(T *&host_ptr, const size_t & alloc_bytes) {
        return alloc_bytes <= ALIGNED_BYTES ? allocUnalignedMemory(host_ptr, alloc_bytes) :
               allocAlignedMemory(host_ptr, alloc_bytes);
    }

    template<typename T>
    inline memoryError_t reallocMemory(T *&host_ptr, const size_t & old_bytes, const size_t & new_bytes) {
        return new_bytes <= ALIGNED_BYTES ? reallocUnalignedMemory(host_ptr, new_bytes) :
               reallocAlignedMemory(host_ptr, old_bytes, new_bytes);
    }

    template<typename T>
    inline memoryError_t freeMemory(T *&host_ptr, const size_t & alloc_bytes) {
        return alloc_bytes <= ALIGNED_BYTES ? freeUnalignedMemory(host_ptr) : freeAlignedMemory(host_ptr);
    }
}




}


#endif //CUDATOOLS_ALIGNEDMEMCOTROLLER_H
