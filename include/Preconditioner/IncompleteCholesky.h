/*
 * @author  刘玉琴、邓轶丹
 * @date    2024/6/10
 * @details IC分解预条件
 */

#ifndef PMSLS_NEW_INCOMPLETECHOLESKY_H
#define PMSLS_NEW_INCOMPLETECHOLESKY_H

#include "BasePreconditon.h"
#include "../utils/MemoryTools/SharedPtrTools.h"
#include "../utils/TimerTools/CPUtimer.hpp"
#include "../MatrixClass/CSRMatrix.h"

namespace HOST {
    /** @brief IC分解
     * @details 根据模板类型实现相同精度+混合精度计算（矩阵分解用低精度，回带求解时结果向量用高精度存储）
     * @note: 相关求解接口在父类中定义，这里并不对外开放 */
    template<typename LowPrecisionType, typename HighPrecisionType>
    class IncompleteCholesky : public TriangularPrecondition<HighPrecisionType> {
    private:
        std::shared_ptr<CSRMatrix<HighPrecisionType> > m_matA; ///< 原始矩阵
        SharedObject<CSRMatrix<LowPrecisionType> > m_matTransL; ///< IC分解后得到的上三角部分
        FLOAT64 m_threshold{0}; ///< 人为设定的阈值用于删除元素，小于(discardThr * 列向量的范数)的非零元会被删除

        /** @brief 计算预条件矩阵下三角部分对应的稀疏三角方程组，即y = L^{-1} z */
        void MSolveLowerUsePtr(HighPrecisionType *vec) override;

        /* 对于IC，这个方法不用执行 */
        void MSolveDiagonalUsePtr(HighPrecisionType *vec) override {}

        /** @brief 计算预条件矩阵上三角部分对应的稀疏三角方程组，即z = L^{-T} x */
        void MSolveUpperUsePtr(HighPrecisionType *vec) override;

    public:
        IncompleteCholesky() = default;

        /** @brief IC构造函数
         * @param [in] matA:    待分解的系数矩阵（智能指针）
         * @param [in] threshold: drop阈值
         * @attention 由于矩阵A是外部声明并初始化，这里转存它的底层智能指针，并且必须是shared类型 */
        IncompleteCholesky(const std::shared_ptr<CSRMatrix<HighPrecisionType> > &matA, const FLOAT64 &threshold);

        IncompleteCholesky(const IncompleteCholesky<LowPrecisionType, HighPrecisionType> &pre);

        IncompleteCholesky(IncompleteCholesky<LowPrecisionType, HighPrecisionType> &&pre) noexcept;

        IncompleteCholesky &operator=(const IncompleteCholesky<LowPrecisionType, HighPrecisionType> &pre);

        IncompleteCholesky &operator=(IncompleteCholesky<LowPrecisionType, HighPrecisionType> &&pre) noexcept;

        ~IncompleteCholesky() override = default;

        void setup() override;

        /** @brief 获取分解后的上三角部分（只读，不允许在原位上修改） */
        inline const SharedObject<CSRMatrix<LowPrecisionType> > &getTransL() const {
            return m_matTransL;
        }
    };

    template class IncompleteCholesky<FLOAT32, FLOAT32>;
    template class IncompleteCholesky<FLOAT64, FLOAT64>;
    template class IncompleteCholesky<FLOAT32, FLOAT64>;
} // HOST


#endif //PMSLS_NEW_INCOMPLETECHOLESKY_H
