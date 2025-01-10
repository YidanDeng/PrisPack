/*
 * @author  邓轶丹
 * @date    2024/4/25
 */
#include "../../include/VectorClass/PageLockedVector.cuh"

namespace HOST {
    template<typename ValType>
    PageLockedVector<ValType>::PageLockedVector(const UINT32 &len) {
        BaseVector<ValType>::m_memoryType = memoryPageLocked;
        if (len > 0) {
            BaseVector<ValType>::m_length = len;
            BaseVector<ValType>::m_byteSize = len * sizeof(ValType);
            CHECK_CUDA(DEVICE::allocMemoryHost(BaseVector<ValType>::m_valuesPtr, BaseVector<ValType>::m_byteSize))
        }
    }

    template<typename ValType>
    PageLockedVector<ValType>::PageLockedVector(const PageLockedVector<ValType> &pre_vec) {
        BaseVector<ValType>::m_memoryType = memoryPageLocked;
        CHECK_CUDA(DEVICE::allocMemoryHost(BaseVector<ValType>::m_valuesPtr, pre_vec.m_byteSize))
        BaseVector<ValType>::m_byteSize = pre_vec.m_byteSize;
        BaseVector<ValType>::m_length = pre_vec.m_length;
        DEVICE::copyFromHostToHost(pre_vec.m_valuesPtr, BaseVector<ValType>::m_valuesPtr,
                           BaseVector<ValType>::m_byteSize, DoNotAlloc);
    }

    template<typename ValType>
    PageLockedVector<ValType>::PageLockedVector(PageLockedVector<ValType> &&pre) noexcept {
        BaseVector<ValType>::m_memoryType = memoryPageLocked;
        BaseVector<ValType>::m_valuesPtr = pre.m_valuesPtr;
        BaseVector<ValType>::m_length = pre.m_length;
        BaseVector<ValType>::m_byteSize = pre.m_byteSize;
        pre.m_valuesPtr = nullptr;
        pre.m_byteSize = 0;
        pre.m_length = 0;
    }

    template<typename ValType>
    PageLockedVector<ValType>::~PageLockedVector() {
        CHECK_CUDA(DEVICE::freeMemoryHost(BaseVector<ValType>::m_valuesPtr))
#ifndef NINFO
        SHOW_INFO("Destructor for page-locked memory finish!")
#endif
    }

    template<typename ValType>
    PageLockedVector<ValType> &PageLockedVector<ValType>::operator=(PageLockedVector<ValType> &&pre_vec) noexcept {
        if (&pre_vec == this)
            return *this;
        if (BaseVector<ValType>::m_valuesPtr) clear();
        BaseVector<ValType>::m_valuesPtr = pre_vec.m_valuesPtr;
        BaseVector<ValType>::m_byteSize = pre_vec.m_byteSize;
        BaseVector<ValType>::m_length = pre_vec.m_length;
        pre_vec.m_valuesPtr = nullptr;
        pre_vec.m_byteSize = 0;
        pre_vec.m_length = 0;
        return *this;
    }

    template<typename ValType>
    PageLockedVector<ValType> &PageLockedVector<ValType>::operator=(const PageLockedVector<ValType> &pre_vec) {
        if (&pre_vec == this)
            return *this;
        resize(pre_vec.m_length, RESERVE_DATA);
        DEVICE::copyFromHostToHost(pre_vec.m_valuesPtr, BaseVector<ValType>::m_valuesPtr, pre_vec.m_byteSize, DoNotAlloc);
        return *this;
    }

    template<typename ValType>
    void PageLockedVector<ValType>::clear() {
        CHECK_CUDA(DEVICE::freeMemoryHost(BaseVector<ValType>::m_valuesPtr))
        BaseVector<ValType>::m_valuesPtr = nullptr;
        BaseVector<ValType>::m_byteSize = 0;
        BaseVector<ValType>::m_length = 0;
    }

    template<typename ValType>
    void PageLockedVector<ValType>::resize(const UINT32 &newLen, UINT8 needReserve) {
        if (BaseVector<ValType>::m_valuesPtr && newLen == BaseVector<ValType>::m_length &&
            needReserve)
            return;
        UINT32 byteSize = newLen * sizeof(ValType);
        INT32 status = cudaSuccess;
        if (BaseVector<ValType>::m_valuesPtr == nullptr) {
            status |= DEVICE::allocMemoryHost(BaseVector<ValType>::m_valuesPtr, byteSize);
        } else {
            if (!needReserve) {
                status |= DEVICE::freeMemoryHost(BaseVector<ValType>::m_valuesPtr);
                status |= DEVICE::allocMemoryHost(BaseVector<ValType>::m_valuesPtr, byteSize);
            } else {
                HOST::DenseVector<ValType> temp(newLen);
                temp.copy(*this);
                status |= DEVICE::freeMemoryHost(BaseVector<ValType>::m_valuesPtr);
                status |= DEVICE::allocMemoryHost(BaseVector<ValType>::m_valuesPtr, byteSize);
                HostVector<ValType>::copy(temp);
            }
        }
        if (status != cudaSuccess) {
#ifndef NDEBUG
            SHOW_ERROR("Resize memory failed!")
#endif
            exit(memoryAllocationFailed);
        }
        BaseVector<ValType>::m_length = newLen;
        BaseVector<ValType>::m_byteSize = newLen * sizeof(ValType);
    }
} // DEVICE