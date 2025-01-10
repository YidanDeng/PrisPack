/*
 * @author  袁心磊
 * @date    2024/6/20
 * @details 实现CG求解器，标准CG、PCG
 */

#ifndef PMSLS_NEW_CG_H
#define PMSLS_NEW_CG_H
#include "BaseSolver.h"
#include "../Preconditioner/BasePreconditon.h"
#include "../utils/TimerTools/CPUtimer.hpp"

namespace HOST {
    enum CGType {
        NormalCG,
        FlexibleCG,
    };
    typedef enum CGType CGType_t;

    template<typename ValType>
    class CG : public BaseSolver<ValType>{

    private:
        UINT32 m_n;                 ///< Size of problem
        FLOAT64 m_tol;              ///<  Tolerence
        UINT32 m_maxIter;
        bool m_IsConverged;
        UINT32 m_numIter;
        FLOAT64 m_finalError;
        // BasePrecondition<ValType>*
        std::shared_ptr<BasePrecondition<ValType>> m_precondPtr;        // 默认置为nullptr
        CGType_t m_CGType;

    public:
        CG(const UINT32 &problemSize, const FLOAT64 &tol, const UINT32 &maxIter);

        CG(const UINT32 &problemSize, const std::shared_ptr<BasePrecondition<ValType>> &precond,
           const CGType_t &CGType, const FLOAT64 &tol, const UINT32 &maxIter);

        ~CG() override = default;

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
    class CG<FLOAT32>;

    template
    class CG<FLOAT64>;

} // HOST

#endif //PMSLS_NEW_CG_H
