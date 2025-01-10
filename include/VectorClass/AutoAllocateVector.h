/*
 * @author  邓轶丹
 * @date    2024/5/15
 * @details 基于智能指针实现向量的动态管理工具
 */

#ifndef PMSLS_NEW_AUTOALLOCATEVECTOR_H
#define PMSLS_NEW_AUTOALLOCATEVECTOR_H

#include "AlignedVector.h"
#include "DenseVector.h"

#ifdef CUDA_ENABLED

#include "PageLockedVector.cuh"

#endif

namespace HOST {
    /** @brief 用来动态管理智能指针与分配对应的向量对象*/
    template<typename ValType>
    class AutoAllocateVector {
    private:
        // 通过unique指针指向特定的对象，可在对象生存期结束时自动析构，所以无需手动释放
        std::unique_ptr<HostVector<ValType> > m_vecPtr;

    public:
        AutoAllocateVector();

        ~AutoAllocateVector() = default;

        AutoAllocateVector(const UINT32 &length, const memoryType_t &memoryTypeHost);

        AutoAllocateVector(const AutoAllocateVector<ValType> &pre_memory);

        AutoAllocateVector(AutoAllocateVector<ValType> &&pre_memory) noexcept;

        AutoAllocateVector &operator=(const AutoAllocateVector<ValType> &pre_memory);

        AutoAllocateVector &operator=(AutoAllocateVector<ValType> &&pre_memory) noexcept;

        /** @brief 重载[]，使对应对象可以如同数组一样使用 */
        inline ValType &operator[](UINT32 idx) {
#ifndef NDEBUG
            THROW_EXCEPTION(!m_vecPtr || idx >= m_vecPtr->getLength(), THROW_RANGE_ERROR("The index out of range!"))
#endif
            return (*m_vecPtr)[idx];
        }

        /** @brief 另一种形式重载[]，用于只读操作 */
        inline const ValType &operator[](UINT32 idx) const {
#ifndef NDEBUG
            THROW_EXCEPTION(!m_vecPtr || idx >= m_vecPtr->getLength(), THROW_RANGE_ERROR("The index out of range!"))
#endif
            return (*m_vecPtr)[idx];
        }

        inline std::unique_ptr<HostVector<ValType> > &operator->() {
            return m_vecPtr;
        }

        inline const std::unique_ptr<HostVector<ValType> > &operator->() const {
            return m_vecPtr;
        }

        /** @brief 重载“*”，使对应对象解引用 */
        inline HostVector<ValType> &operator*() {
            return *m_vecPtr;
        }

        /** @brief 另一种形式重载“*”，使对应对象为只读对象 */
        inline const HostVector<ValType> &operator*() const {
            return *m_vecPtr;
        }

        inline void setValue(UINT32 idx, ValType val) {
#ifndef NDEBUG
            THROW_EXCEPTION(!m_vecPtr || idx >= m_vecPtr->getLength(), THROW_OUT_OF_RANGE("The index out of range!"))
#endif
            (*m_vecPtr)[idx] = val;
        }


        inline ValType getValue(UINT32 idx) {
#ifndef NDEBUG
            THROW_EXCEPTION(!m_vecPtr || idx >= m_vecPtr->getLength(), THROW_OUT_OF_RANGE("The index out of range!"))
#endif
            return (*m_vecPtr)[idx];
        }
        /** @brief 获取当前向量的长度 */
        inline UINT32 getLength() const {
            return m_vecPtr->getLength();
        }

        /** @brief 获取当前向量的字节大小 */
        inline size_t getByteSize() const {
            return m_vecPtr->getByteSize();
        }

        /** @brief 获取当前内存空间的首地址（裸指针） */
        inline ValType *getRawValPtr() const {
            return m_vecPtr->getRawValPtr();
        }

        /** @brief 获取当前内存类型。 */
        inline memoryType_t getMemoryType() const {
            return m_vecPtr->getMemoryType();
        }

        /** @brief 获取当前向量所在的设备，-1表示在CPU上，正整数或0表示在GPU上 */
        inline INT32 getLocation() const {
            return m_vecPtr->getLocation();
        }

        /** @brief 用于调整向量空间的大小
         * @param [in] newLength: 需要的向量大小
         * @param [in] needReserve: 是否保留原始向量中的值（0：不保留；1：保留；使用时建议传对应的宏“RESERVE_NO_DATA”或“RESERVE_DATA”*/
        inline void resize(const UINT32 &newLength, const UINT8 &needReserve) {
            m_vecPtr->resize(newLength, needReserve);
        }

        /** @brief 函数转发，针对多个函数重载的情况 */
        template<typename... ArgType>
        inline void copy(ArgType &&... Args) {
            m_vecPtr->copy(std::forward<ArgType>(Args)...);
        }

        /** @brief 函数转发，针对多个函数重载的情况 */
        template<typename... ArgType>
        inline void add(ArgType &&... Args) {
            m_vecPtr->add(std::forward<ArgType>(Args)...);
        }

        /** @brief 用于重新调整向量长度或存储空间类型 */
        void reset(const UINT32 &length, const memoryType_t &memoryTypeHost);

        inline void printVector(const char *message) const {
            m_vecPtr->printVector(message);
        }

        inline void clear() {
            m_vecPtr->clear();
        }
    };


    template class AutoAllocateVector<INT32>;
    template class AutoAllocateVector<UINT32>;
    template class AutoAllocateVector<FLOAT32>;
    template class AutoAllocateVector<FLOAT64>;
} // HOST

#endif //PMSLS_NEW_AUTOALLOCATEVECTOR_H
