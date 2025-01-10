/*
 * @author  刘玉琴、邓轶丹
 * @date    2024/11/13
 * @details ILDLT分解预条件
 */

#ifndef INCOMPLETELDLT_H
#define INCOMPLETELDLT_H

#include "BasePreconditon.h"
#include "../utils/MemoryTools/SharedPtrTools.h"
#include "../MatrixClass/CSRMatrix.h"


namespace HOST {
    /** @brief ILDLT分解
     * @details 根据模板类型实现相同精度+混合精度计算（矩阵分解用低精度，回带求解时结果向量用高精度存储）
     * @note: 相关求解接口在父类中定义，这里并不对外开放 */
    template<typename LowPrecisionType, typename HighPrecisionType>
    class IncompleteLDLT : public TriangularPrecondition<HighPrecisionType> {
    private:
        std::shared_ptr<CSRMatrix<HighPrecisionType> > m_matA; ///< 原始矩阵（转存变量，底层仅转存智能指针）
        SharedObject<CSRMatrix<LowPrecisionType> > m_matTransL; ///< ILDLT分解后的L^T矩阵（上三角阵，不含对角元），对角线单独存储
        SharedObject<AutoAllocateVector<HighPrecisionType> > m_diagVals; ///< U矩阵对角元
        HighPrecisionType m_tolerance{1e-4}; ///< drop阈值
    protected:
        void MSolveUpperUsePtr(HighPrecisionType *vec) override;

        void MSolveDiagonalUsePtr(HighPrecisionType *vec) override;

        void MSolveLowerUsePtr(HighPrecisionType *vec) override;

    public:
        IncompleteLDLT() = default;

        /** @brief ILDLT分解，类似于IC分解，将矩阵A分解为L * D * L^{T}的形式，避免了对角元为负无法进行IC分解的问题
         * @param [in] matA 待分解矩阵（shared指针）
         * @param [in] tolerance 阈值参数，类比于IC，对小于阈值的非零元进行drop */
        IncompleteLDLT(const std::shared_ptr<CSRMatrix<HighPrecisionType> > &matA, HighPrecisionType tolerance);

        IncompleteLDLT(const IncompleteLDLT &other)
            : TriangularPrecondition<HighPrecisionType>(other),
              m_matA(other.m_matA),
              m_matTransL(other.m_matTransL),
              m_diagVals(other.m_diagVals),
              m_tolerance(other.m_tolerance) {
        }

        IncompleteLDLT(IncompleteLDLT &&other) noexcept
            : TriangularPrecondition<HighPrecisionType>(std::move(other)),
              m_matA(std::move(other.m_matA)),
              m_matTransL(std::move(other.m_matTransL)),
              m_diagVals(std::move(other.m_diagVals)),
              m_tolerance(other.m_tolerance) {
        }

        IncompleteLDLT & operator=(const IncompleteLDLT &other) {
            if (this == &other)
                return *this;
            TriangularPrecondition<HighPrecisionType>::operator =(other);
            m_matA = other.m_matA;
            m_matTransL = other.m_matTransL;
            m_diagVals = other.m_diagVals;
            m_tolerance = other.m_tolerance;
            return *this;
        }

        IncompleteLDLT & operator=(IncompleteLDLT &&other) noexcept {
            if (this == &other)
                return *this;
            TriangularPrecondition<HighPrecisionType>::operator =(std::move(other));
            m_matA = std::move(other.m_matA);
            m_matTransL = std::move(other.m_matTransL);
            m_diagVals = std::move(other.m_diagVals);
            m_tolerance = other.m_tolerance;
            return *this;
        }

        void setup() override;

        ~IncompleteLDLT() override = default;

    };

    template class IncompleteLDLT<FLOAT32, FLOAT32>;
    template class IncompleteLDLT<FLOAT64, FLOAT64>;
    template class IncompleteLDLT<FLOAT32, FLOAT64>;
}


#endif //INCOMPLETELDLT_H
