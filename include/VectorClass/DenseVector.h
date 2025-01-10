/*
 * @author  邓轶丹
 * @date    2024/4/23
 * @details CPU上的一般向量类，在申请内存时自动清零，即所有向量元素均置为0
 */

#ifndef GENERAL_VECTOR_H
#define GENERAL_VECTOR_H

#include "BaseVector.h"
#include "../utils/BaseUtils.h"
#include "../utils/MemoryTools/HostMemoryController.hpp"


namespace HOST {
    template<typename ValType>
    class DenseVector final : public HostVector<ValType> {
    public:
        // 子类构造函数和析构函数
        /** @brief 子类无参构造函数，生成空向量 */
        DenseVector() {
            BaseVector<ValType>::m_memoryType = memoryBase;
        }

        /** @brief 稠密向量构造函数，默认生成全0向量。
        * @param [in] len: 预计元素个数 */
        explicit DenseVector(const UINT32 &len);

        /** @brief 子类拷贝构造函数。 */
        DenseVector(const DenseVector<ValType> &pre_vec);

        /** @brief 子类移动构造函数。 */
        DenseVector(DenseVector<ValType> &&pre) noexcept;


        /** @brief 子类析构函数。 */
        ~DenseVector() override;


        // 其他操作函数
        /** @brief 调整当前向量的长度
         * @param [in] newLen: 调整后的新长度
         * @param [in] needReserve: 是否需要保留原先存储的值 */
        void resize(const UINT32 &newLen, UINT8 needReserve) override;

        void clear() override;

        /* 运算符重载*/
        /** @brief 重载GeneralVector中的"="。 */
        DenseVector<ValType> &operator=(const DenseVector<ValType> &pre_vec);

        /** @brief 重载GeneralVector中的"="。（移动拷贝） */
        DenseVector &operator=(DenseVector<ValType> &&pre_vec) noexcept;
    };


    template
    class DenseVector<INT32>;

    template
    class DenseVector<UINT32>;

    template
    class DenseVector<FLOAT32>;

    template
    class DenseVector<FLOAT64>;

//    template
//    class DenseVector<std::complex<FLOAT32>>;
//
//    template
//    class DenseVector<std::complex<FLOAT64>>;


}
#endif


