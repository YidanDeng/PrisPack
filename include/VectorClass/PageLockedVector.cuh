/*
 * @author  邓轶丹
 * @date    2024/4/25
 * @details 通过CUDA实现的CPU上锁页内存向量类，适用于多GPU异步场景
 */

#ifndef PMSLS_NEW_PAGELOCKEDVECTOR_CUH
#define PMSLS_NEW_PAGELOCKEDVECTOR_CUH

#include "DenseVector.h"
#include "../../config/CUDAheaders.cuh"
#include "../../include/utils/MemoryTools/DeviceMemoryController.cuh"

namespace HOST {
    template<typename ValType>
    class PageLockedVector : public HostVector<ValType> {
    public:
        // 子类构造函数和析构函数
        /** @brief 子类无参构造函数，生成空向量 */
        PageLockedVector() {
            BaseVector<ValType>::m_memoryType = memoryPageLocked;
        }

        /** @brief 采用锁页内存的稠密向量构造函数，向量元素无默认值。
        * @param [in] len: 预计元素个数，后期可调用resize函数调整大小。 */
        explicit PageLockedVector(const UINT32 &len);

        /** @brief 子类拷贝构造函数。 */
        PageLockedVector(const PageLockedVector<ValType> &pre_vec);

        /** @brief 子类移动构造函数。 */
        PageLockedVector(PageLockedVector<ValType> &&pre) noexcept;

        /** @brief 子类析构函数。 */
        ~PageLockedVector() override;


        // 其他操作函数
        /** @brief 调整当前向量的长度
         * @param [in] newLen: 调整后的新长度
         * @param [in] needReserve: 是否需要保留原先存储的值 */
        void resize(const UINT32 &newLen, UINT8 needReserve) override;

        void clear() override;

        /* 运算符重载*/
        /** @brief 重载GeneralVector中的"="。 */
        PageLockedVector<ValType> &operator=(const PageLockedVector<ValType> &pre_vec);

        /** @brief 重载GeneralVector中的"="。（移动拷贝） */
        PageLockedVector &operator=(PageLockedVector<ValType> &&pre_vec) noexcept;
    };


    template
    class PageLockedVector<INT32>;

    template
    class PageLockedVector<UINT32>;

    template
    class PageLockedVector<FLOAT32>;

    template
    class PageLockedVector<FLOAT64>;

} // DEVICE

#endif //PMSLS_NEW_PAGELOCKEDVECTOR_CUH
