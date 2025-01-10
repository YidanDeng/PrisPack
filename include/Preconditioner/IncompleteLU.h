/*
 * @author  刘玉琴、邓轶丹
 * @date    2024/6/12
 * @details ILU分解预条件
 */

#ifndef PMSLS_NEW_INCOMPLETELU_H
#define PMSLS_NEW_INCOMPLETELU_H

#include "BasePreconditon.h"
#include "../utils/MemoryTools/SharedPtrTools.h"
#include "../MatrixClass/CSRMatrix.h"
#include "../utils/TimerTools/CPUtimer.hpp"

namespace HOST {
    /** @brief ILUT分解
     * @details 根据模板类型实现相同精度+混合精度计算（矩阵分解用低精度，回带求解时结果向量用高精度存储）
     * @note: 相关求解接口在父类中定义，这里并不对外开放 */
    template<typename LowPrecisionType, typename HighPrecisionType>
    class IncompleteLU : public TriangularPrecondition<HighPrecisionType> {
    private:
        std::shared_ptr<CSRMatrix<HighPrecisionType> > m_matA; ///< 原始矩阵（转存变量，底层仅转存智能指针）
        SharedObject<CSRMatrix<LowPrecisionType> > m_L; ///< ILU分解后的L矩阵（不含对角元），注：L矩阵对角元全为1
        SharedObject<CSRMatrix<LowPrecisionType> > m_U; ///< ILU分解后的U矩阵（不含对角元），对角线单独存储
        SharedObject<AutoAllocateVector<LowPrecisionType> > m_diagVals; ///< U矩阵对角元
        UINT32 m_lfill{0}; ///< 填充参数，L的每一列和U的每一列最多有lfill个元素(不考虑对角线元素)
        HighPrecisionType m_tolerance{0}; ///< drop阈值

    protected:
        /** @brief 计算预条件矩阵上三角部分对应的稀疏三角方程组，即z = L^{-T} x */
        void MSolveUpperUsePtr(HighPrecisionType *vec) override;


        void MSolveDiagonalUsePtr(HighPrecisionType *vec) override {
        }

        /** @brief 计算预条件矩阵下三角部分对应的稀疏三角方程组，即y = L^{-1} z */
        void MSolveLowerUsePtr(HighPrecisionType *vec) override;

    public:
        IncompleteLU() = default;

        /** @brief ILUT分解    双drop策略：  1.L,U中小于【m_tol*norm(行向量)】的元素被删除
         *                                  2.只保留L和U每一列中最大的lfill个元素
         * @param [in] matA 待分解矩阵
         * @param [in] lfill 填充参数，L的每一列和U的每一列最多有lfill个元素(不考虑对角线元素)
         * @param [in] tolerance 删除（drop）元素的阈值 */
        IncompleteLU(const std::shared_ptr<CSRMatrix<HighPrecisionType> > &matA, UINT32 lfill,
                     HighPrecisionType tolerance);

        ~IncompleteLU() override = default;

        IncompleteLU(const IncompleteLU &other)
            : TriangularPrecondition<HighPrecisionType>(other),
              m_matA(other.m_matA),
              m_L(other.m_L),
              m_U(other.m_U),
              m_diagVals(other.m_diagVals),
              m_lfill(other.m_lfill),
              m_tolerance(other.m_tolerance) {
        }

        IncompleteLU(IncompleteLU &&other) noexcept
            : TriangularPrecondition<HighPrecisionType>(std::move(other)),
              m_matA(std::move(other.m_matA)),
              m_L(std::move(other.m_L)),
              m_U(std::move(other.m_U)),
              m_diagVals(std::move(other.m_diagVals)),
              m_lfill(other.m_lfill),
              m_tolerance(other.m_tolerance) {
        }

        IncompleteLU & operator=(const IncompleteLU &other) {
            if (this == &other)
                return *this;
            TriangularPrecondition<HighPrecisionType>::operator =(other);
            m_matA = other.m_matA;
            m_L = other.m_L;
            m_U = other.m_U;
            m_diagVals = other.m_diagVals;
            m_lfill = other.m_lfill;
            m_tolerance = other.m_tolerance;
            return *this;
        }

        IncompleteLU & operator=(IncompleteLU &&other) noexcept {
            if (this == &other)
                return *this;
            TriangularPrecondition<HighPrecisionType>::operator =(std::move(other));
            m_matA = std::move(other.m_matA);
            m_L = std::move(other.m_L);
            m_U = std::move(other.m_U);
            m_diagVals = std::move(other.m_diagVals);
            m_lfill = other.m_lfill;
            m_tolerance = other.m_tolerance;
            return *this;
        }

        void setup() override;

        inline const SharedObject<CSRMatrix<LowPrecisionType> > &getL() const {
            return m_L;
        }

        inline const SharedObject<CSRMatrix<LowPrecisionType> > &getU() const {
            return m_U;
        }

        inline const SharedObject<AutoAllocateVector<LowPrecisionType> > &getDiagVals() {
            return m_diagVals;
        }
    };


    template class IncompleteLU<FLOAT32, FLOAT32>;
    template class IncompleteLU<FLOAT64, FLOAT64>;
    template class IncompleteLU<FLOAT32, FLOAT64>;
} // HOST

#endif //PMSLS_NEW_INCOMPLETELU_H
