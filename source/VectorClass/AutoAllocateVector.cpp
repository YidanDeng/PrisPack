/*
 * @author  邓轶丹
 * @date    2024/5/15
 */

#include "../../include/VectorClass/AutoAllocateVector.h"

namespace HOST {
    /** @brief 用来借助智能指针实例化vector对象，这样无需手动delete
     * @attention 该函数仅作为当前类的一个功能函数，不建议放在外部调用 */
    template<typename ValType>
    inline void initializeVector(std::unique_ptr<HostVector<ValType>> &vecPtr, const UINT32 &length,
                                 const memoryType_t &memoryType) {
        if (vecPtr) vecPtr.reset(nullptr);
        if (memoryType == memoryBase) {
            vecPtr = std::make_unique<DenseVector<ValType>>(length);
        } else if (memoryType == memoryAligned) {
            vecPtr = std::make_unique<AlignedVector<ValType>>(length);
        }
#ifdef CUDA_ENABLED     // 该宏在CmakeList中定义，由编译器自动检测当前主机是否支持CUDA架构
        else if (memoryType == memoryPageLocked) {
            vecPtr = std::make_unique<HOST::PageLockedVector<ValType>>(length);
        }
#endif
        else {
#ifndef NWARN
            SHOW_WARN("You are trying to use an unrecognizable vector type! The vector type was reset to default type.")
#endif
            vecPtr = std::make_unique<DenseVector<ValType>>(length);
        }
    }

    template<typename ValType>
    AutoAllocateVector<ValType>::AutoAllocateVector() {
        initializeVector(m_vecPtr, 0, memoryBase);
    }

    template<typename ValType>
    AutoAllocateVector<ValType>::AutoAllocateVector(const UINT32 &length, const memoryType_t &memoryTypeHost) {
        initializeVector(m_vecPtr, length, memoryTypeHost);
    }

    template<typename ValType>
    AutoAllocateVector<ValType>::AutoAllocateVector(const AutoAllocateVector<ValType> &pre_memory) {
        initializeVector(m_vecPtr, pre_memory.m_vecPtr->getLength(), pre_memory.m_vecPtr->getMemoryType());
        m_vecPtr->copy(*pre_memory.m_vecPtr);
    }

    template<typename ValType>
    AutoAllocateVector<ValType>::AutoAllocateVector(AutoAllocateVector<ValType> &&pre_memory) noexcept {
        m_vecPtr = std::move(pre_memory.m_vecPtr);
        // 由于移动后原对象变为空指针，所以必须重新生成一个空对象，避免后续指针调用出现异常
        initializeVector(pre_memory.m_vecPtr, 0, m_vecPtr->getMemoryType());
    }

    template<typename ValType>
    AutoAllocateVector<ValType> &AutoAllocateVector<ValType>::operator=(const AutoAllocateVector<ValType> &pre_memory) {
        if (&pre_memory == this)
            return *this;
        if (!m_vecPtr)
            initializeVector(m_vecPtr, pre_memory.m_vecPtr->getLength(), pre_memory.m_vecPtr->getMemoryType());
        m_vecPtr->resize(pre_memory.m_vecPtr->getLength(), RESERVE_NO_DATA);
        m_vecPtr->copy(*pre_memory.m_vecPtr);
        return *this;       // 少加了这个，在O3级别下会异常退出
    }

    template<typename ValType>
    AutoAllocateVector<ValType> &AutoAllocateVector<ValType>::operator=(AutoAllocateVector<ValType> &&pre_memory)
    noexcept {
        if (&pre_memory == this)
            return *this;
        m_vecPtr.reset(nullptr);
        m_vecPtr = std::move(pre_memory.m_vecPtr);
        // 由于移动后原对象变为空指针，所以必须重新生成一个空对象，避免后续指针调用出现异常
        initializeVector(pre_memory.m_vecPtr, 0, m_vecPtr->getMemoryType());
        return *this;
    }

    template<typename ValType>
    void AutoAllocateVector<ValType>::reset(const UINT32 &length, const memoryType_t &memoryTypeHost) {
#ifndef NWARN
        if (length == m_vecPtr->getLength() && memoryTypeHost == m_vecPtr->getMemoryType())
            SHOW_WARN("The new length and memory type are both equal to those of previous vector. "
                      "The original memory and values will be cleared and reset.")
#endif
        initializeVector(m_vecPtr, length, memoryTypeHost);
    }
} // HOST