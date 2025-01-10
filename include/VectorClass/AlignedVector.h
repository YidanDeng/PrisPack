/*
 * @author  邓轶丹
 * @date    2024/4/23
 * @details CPU上的内存对齐向量类，适用于多线程环境
 */

#ifndef PMSLS_TEMP_ALIGNEDVECTOR_H
#define PMSLS_TEMP_ALIGNEDVECTOR_H

#include "BaseVector.h"
#include "../utils/BaseUtils.h"
#include "../utils/MemoryTools/HostMemoryController.hpp"

namespace HOST {
    template<typename ValType>
    class AlignedVector : public HostVector<ValType> {
    public:
        AlignedVector() {
            BaseVector<ValType>::m_memoryType = memoryAligned;
        }

        /** @brief 采用对齐内存的稠密向量构造函数，向量元素无默认值。
         * @param [in] length: 需要申请的向量长度，后续可调用resize函数调整长度。*/
        explicit AlignedVector(const UINT32 &length);

        AlignedVector(const AlignedVector<ValType> &pre);

        AlignedVector(AlignedVector<ValType> &&pre) noexcept;

        ~AlignedVector() override;

        AlignedVector &operator=(AlignedVector<ValType> &&pre) noexcept;

        AlignedVector &operator=(const AlignedVector<ValType> &pre);


        // 其他函数
        /** @brief 调整当前向量的长度
         * @param [in] newLen: 调整后的新长度
         * @param [in] needReserve: 是否需要保留原先存储的值 */
        void resize(const UINT32 &newLen, UINT8 needReserve) override;

        void clear() override;
    };


    template
    class AlignedVector<INT32>;

    template
    class AlignedVector<UINT32>;

    template
    class AlignedVector<FLOAT32>;

    template
    class AlignedVector<FLOAT64>;

//    template
//    class AlignedVector<std::complex<FLOAT32>>;
//
//    template
//    class AlignedVector<std::complex<FLOAT64>>;


} // HOST

#endif //PMSLS_TEMP_ALIGNEDVECTOR_H
