/*
 * @author  袁心磊
 * @date    2024/6/15
 * @details 实现GMRES求解器，标准GMRES、PGMRES
 */

#ifndef PMSLS_NEW_GMRES_H
#define PMSLS_NEW_GMRES_H

#include "BaseSolver.h"
#include "../Preconditioner/BasePreconditon.h"
#include "../utils/TimerTools/CPUtimer.hpp"


namespace HOST {
    enum GMRESType {
        NormalGMRES,
        FlexibleGMRES,
    };

    typedef enum GMRESType GMRESType_t;

    template<typename ValType>
    class GMRES : public BaseSolver<ValType> {
    private:
        UINT32 m_n;
        FLOAT64 m_tol;
        INT32 m_maxIter;
        bool m_IsConverged;
        INT32 m_numIter;
        FLOAT64 m_finalError;
        bool m_restart;//是否重启
        UINT32 m_m;//重启前迭代几次
        // BasePrecondition<ValType>*
        std::shared_ptr<BasePrecondition<ValType>> m_precondPtr;        // 默认置为nullptr
        GMRESType_t m_GMRESType;

    public:
        GMRES(const UINT32 &problemSize);

        GMRES(const UINT32 &problemSize, const std::shared_ptr<BasePrecondition<ValType>> &precond,
              const GMRESType_t &gmresType);

        GMRES(const UINT32 &problemSize, const std::shared_ptr<BasePrecondition<ValType>> &precond,
              const FLOAT64 &tolerance, const GMRESType_t &gmresType);

        ~GMRES() override = default;

        void setRestart(bool restart, UINT32 m);

        void givensTransform(HOST::DenseMatrix<ValType> &H, HostVector<ValType> &c, HostVector<ValType> &s,
                             HostVector<ValType> &ksi, const UINT32 &j);

        void retrospectiveSolver(HOST::DenseMatrix<ValType> &H, HostVector<ValType> &ksi, const UINT32 &dim);

        void solve(const HOST::CSRMatrix<ValType> &matA, const HostVector<ValType> &b, HostVector<ValType> &x) override;

        inline INT32 getNumIter() {
            return m_numIter;
        }

        inline FLOAT64 getError() {
            return m_finalError;
        }

        inline bool getConvergence() {
            return m_IsConverged;
        }

        inline const PreconditionType_t &getPrecondtionType() const {
            return m_precondPtr->getPreconditionType();
        }
    };

    template
    class GMRES<FLOAT32>;

    template
    class GMRES<FLOAT64>;

} // HOST

#endif //PMSLS_NEW_GMRES_H
