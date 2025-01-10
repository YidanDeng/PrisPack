#include "../../include/VectorClass/DenseVector.h"

namespace HOST {
    template<typename ValType>
    DenseVector<ValType>::DenseVector(const UINT32 &len) {
        BaseVector<ValType>::m_memoryType = memoryBase;
        if (len > 0) {
#ifndef NINFO
            SHOW_INFO("Constructor with parameters for Dense Vector begin!")
#endif
            BaseVector<ValType>::m_byteSize = len * sizeof(ValType);
            BaseVector<ValType>::m_valuesPtr = (ValType *) calloc(BaseVector<ValType>::m_byteSize, 1);
            BaseVector<ValType>::m_length = len;
            BAD_ALLOC_CHECK(BaseVector<ValType>::m_valuesPtr, DEBUG_MESSEGE_OPTION,
                            "[ERROR] Initialize dense vector failed!");
        }
    }

    template<typename ValType>
    DenseVector<ValType>::DenseVector(const DenseVector<ValType> &pre_vec)
            : HostVector<ValType>(pre_vec) {
        BaseVector<ValType>::m_memoryType = memoryBase;
#ifndef NINFO
        SHOW_INFO("Copy constructor for Dense Vector begin!")
#endif
    }


    template<typename ValType>
    DenseVector<ValType>::DenseVector(DenseVector<ValType> &&pre) noexcept {
#ifndef NINFO
        SHOW_INFO("Move constructor for Dense Vector begin!")
#endif
        BaseVector<ValType>::m_memoryType = memoryBase;
        BaseVector<ValType>::m_length = pre.m_length;
        BaseVector<ValType>::m_byteSize = pre.m_byteSize;
        BaseVector<ValType>::m_valuesPtr = pre.m_valuesPtr;
        pre.m_length = 0;
        pre.m_byteSize = 0;
        pre.m_valuesPtr = nullptr;
    }

    template<typename ValType>
    DenseVector<ValType> &DenseVector<ValType>::operator=(const DenseVector<ValType> &pre_vec) {
#ifndef NINFO
        SHOW_INFO("Copy assignment for Dense Vector begin!")
#endif
        if (this == &pre_vec)// 保证自我复制安全，相当于一个default操作
            return *this;
        resize(pre_vec.m_length, RESERVE_DATA);
        HostVector<ValType>::copy(pre_vec);
        return *this;
    }

    template<typename ValType>
    DenseVector<ValType> &DenseVector<ValType>::operator=(DenseVector<ValType> &&pre_vec) noexcept {
#ifndef NINFO
        SHOW_INFO("Move assignment for Dense Vector begin!")
#endif
        if (this == &pre_vec)// 保证自我复制安全，相当于一个default操作
            return *this;
        HostVector<ValType>::move(pre_vec);
        return *this;
    }


    template<typename ValType>
    DenseVector<ValType>::~DenseVector() {
        free(BaseVector<ValType>::m_valuesPtr);
#ifndef NINFO
        SHOW_INFO("Destructor for Dense Vector finished!")
#endif
    }


    template<typename ValType>
    void DenseVector<ValType>::resize(const UINT32 &newLen, UINT8 needReserve) {
        if (BaseVector<ValType>::m_valuesPtr && newLen == BaseVector<ValType>::m_length && needReserve) return;
        size_t newByteSize = newLen * sizeof(ValType);
        INT32 status = memoryOptionSuccess;
        if (BaseVector<ValType>::m_valuesPtr == nullptr) {
            BaseVector<ValType>::m_valuesPtr = (ValType *) calloc(newByteSize, 1);
            if (BaseVector<ValType>::m_valuesPtr == nullptr) status |= memoryAllocationFailed;
        } else {
            if (!needReserve) {
                free(BaseVector<ValType>::m_valuesPtr);
                BaseVector<ValType>::m_valuesPtr = nullptr;
                BaseVector<ValType>::m_valuesPtr = (ValType *) calloc(newByteSize, 1);
                if (BaseVector<ValType>::m_valuesPtr == nullptr) status |= memoryAllocationFailed;
            } else {
                status |= reallocUnalignedMemory(BaseVector<ValType>::m_valuesPtr, newByteSize);
            }
        }
#ifndef NDEBUG
        if (status != memoryOptionSuccess) {
            SHOW_ERROR("Resize memory failed!")
            exit(memoryAllocationFailed);
        }
#endif
        if (needReserve && newLen > BaseVector<ValType>::m_length) {
            std::fill_n(this->m_valuesPtr + BaseVector<ValType>::m_length, newLen - BaseVector<ValType>::m_length, 0);
        }
        BaseVector<ValType>::m_length = newLen;
        BaseVector<ValType>::m_byteSize = newByteSize;
    }

    template<typename ValType>
    void DenseVector<ValType>::clear() {
        free(BaseVector<ValType>::m_valuesPtr);
        BaseVector<ValType>::m_valuesPtr = nullptr;
        BaseVector<ValType>::m_byteSize = 0;
        BaseVector<ValType>::m_length = 0;
    }

}// namespace HOST
