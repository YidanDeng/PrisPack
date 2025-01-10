
#include "../../include/VectorClass/AlignedVector.h"

namespace HOST {
    template<typename ValType>
    AlignedVector<ValType>::AlignedVector(const UINT32 &length) {
        BaseVector<ValType>::m_memoryType = memoryAligned;
        if (length > 0) {
            BaseVector<ValType>::m_length = length;
            BaseVector<ValType>::m_byteSize = length * sizeof(ValType);
            memoryError_t status = allocMemory(BaseVector<ValType>::m_valuesPtr, BaseVector<ValType>::m_byteSize);
            if (status == memoryOptionSuccess) {
#ifndef NINFO
                SHOW_INFO("Allocation constructor executes successfully.")
#endif
            } else {
#ifndef NDEBUG
                SHOW_ERROR("Allocation constructor failed!")
#endif
                exit(memoryAllocationFailed);
            }
        }
    }

    template<typename ValType>
    AlignedVector<ValType>::AlignedVector(const AlignedVector<ValType> &pre)
            : HostVector<ValType>(pre) {
        BaseVector<ValType>::m_memoryType = memoryAligned;
#ifndef NINFO
        SHOW_INFO("Copy constructor for Aligned Vector begin!")
#endif
    }

    template<typename ValType>
    AlignedVector<ValType>::AlignedVector(AlignedVector<ValType> &&pre) noexcept {
#ifndef NINFO
        SHOW_INFO("Move constructor for Aligned Vector begin!")
#endif
        BaseVector<ValType>::m_memoryType = memoryAligned;
        BaseVector<ValType>::m_length = pre.m_length;
        BaseVector<ValType>::m_byteSize = pre.m_byteSize;
        BaseVector<ValType>::m_valuesPtr = pre.m_valuesPtr;
        pre.m_length = 0;
        pre.m_byteSize = 0;
        pre.m_valuesPtr = nullptr;
    }

    template<typename ValType>
    AlignedVector<ValType>::~AlignedVector() {
        int status = freeMemory(BaseVector<ValType>::m_valuesPtr, BaseVector<ValType>::m_byteSize);
        if (status == memoryOptionSuccess) {
#ifndef NINFO
            SHOW_INFO("Auto-free func executes successfully.")
#endif
        } else {
#ifndef NDEBUG
            SHOW_ERROR("Auto-free failed!")
#endif
            exit(memoryAllocationFailed);
        }
    }

    template<typename ValType>
    AlignedVector<ValType> &AlignedVector<ValType>::operator=(const AlignedVector<ValType> &pre) {
#ifndef NINFO
        SHOW_INFO("Copy assignment for Dense Vector begin!")
#endif
        if (&pre == this)
            return *this;
        resize(pre.m_length, RESERVE_DATA);
        HostVector<ValType>::copy(pre);
        return *this;
    }

    template<typename ValType>
    AlignedVector<ValType> &AlignedVector<ValType>::operator=(AlignedVector<ValType> &&pre) noexcept {
#ifndef NINFO
        SHOW_INFO("Move assignment for Dense Vector begin!")
#endif
        if (&pre == this)
            return *this;
        BaseVector<ValType>::m_length = pre.m_length;
        BaseVector<ValType>::m_byteSize = pre.m_byteSize;
        BaseVector<ValType>::m_valuesPtr = pre.m_valuesPtr;
        pre.m_length = 0;
        pre.m_byteSize = 0;
        pre.m_valuesPtr = nullptr;
        return *this;
    }

    template<typename ValType>
    void AlignedVector<ValType>::clear() {
        memoryError_t status = freeMemory(BaseVector<ValType>::m_valuesPtr, BaseVector<ValType>::m_byteSize);
        if (status == memoryOptionSuccess) {
#ifndef NINFO
            SHOW_INFO("free-func executes successfully.")
#endif
        } else {
#ifndef NDEBUG
            SHOW_ERROR("free-func failed!")
#endif
            exit(memoryAllocationFailed);
        }
        BaseVector<ValType>::m_valuesPtr = nullptr;
        BaseVector<ValType>::m_length = 0;
        BaseVector<ValType>::m_byteSize = 0;
    }

    template<typename ValType>
    void AlignedVector<ValType>::resize(const UINT32 &newLen, UINT8 needReserve) {
#ifndef NINFO
        SHOW_INFO("Re-allocating HOST...")
#endif
        if (BaseVector<ValType>::m_valuesPtr && newLen == BaseVector<ValType>::m_length && needReserve) return;
        INT32 status = memoryOptionSuccess;
        size_t alloc_bytes = newLen * sizeof(ValType);
//        SHOW_INFO("resize bytes: " << alloc_bytes)
        if (BaseVector<ValType>::m_valuesPtr == nullptr) {
            status |= allocMemory(BaseVector<ValType>::m_valuesPtr, alloc_bytes);
        } else {
            if (!needReserve) {
                status |= freeMemory(BaseVector<ValType>::m_valuesPtr, BaseVector<ValType>::m_byteSize);
                status |= allocMemory(BaseVector<ValType>::m_valuesPtr, alloc_bytes);
            } else {
                status |= reallocMemory(BaseVector<ValType>::m_valuesPtr, BaseVector<ValType>::m_byteSize, alloc_bytes);
            }
        }
        if (status == memoryOptionSuccess) {
#ifndef NINFO
            SHOW_INFO("Re-alloc-func executes successfully.")
#endif
        } else {
#ifndef NDEBUG
            SHOW_ERROR("Re-alloc-func failed!")
#endif
            exit(memoryAllocationFailed);
        }
        BaseVector<ValType>::m_length = newLen;
        BaseVector<ValType>::m_byteSize = alloc_bytes;
    }
} // HOST