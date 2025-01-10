/*
 * @author  袁心磊
 * @date    2024/6/20
 * @details 实现CG求解器，标准CG、PCG
 */

#include "../../include/Solver/CG.h"

namespace HOST {
    template<typename ValType>
    CG<ValType>::CG(const UINT32 &problemSize, const FLOAT64 &tol, const UINT32 &maxIter) {
        m_n = problemSize;
        m_tol = tol;
        m_maxIter = maxIter;
        m_IsConverged = false;
        m_numIter = 0;
        m_finalError = (FLOAT64) (-1);
    }

    template<typename ValType>
    CG<ValType>::CG(const UINT32 &problemSize, const std::shared_ptr<BasePrecondition<ValType> > &precond,
                    const CGType_t &CGType, const FLOAT64 &tol, const UINT32 &maxIter) {
        m_n = problemSize;
        m_tol = tol;
        m_maxIter = maxIter;
        m_IsConverged = false;
        m_numIter = 0;
        m_finalError = (FLOAT64) (-1);
        m_precondPtr = precond;
        m_CGType = CGType;
    }

    template<typename ValType>
    void CG<ValType>::solve(const CSRMatrix<ValType> &matA, const HostVector<ValType> &b, HostVector<ValType> &x) {
        m_IsConverged = false;
        m_numIter = 0;
        m_finalError = (FLOAT64) (-1);
        AutoAllocateVector<ValType> r(m_n, memoryBase), v(m_n, memoryBase);
        AutoAllocateVector<ValType> w(m_n, memoryBase);
        AutoAllocateVector<ValType> d(m_n, memoryBase);
        ValType value1, value2;
        if (m_precondPtr.get() != nullptr) m_precondPtr->setup();
        UINT32 m = m_maxIter;
        FLOAT64 norm_b = b.norm_2(), norm;
        matA.MatPVec(x, *r);
        w->add(1, b, -1, *r);
        r = w; ///<r0
        if (m_precondPtr.get()) m_precondPtr->MInvSolve(*w);
        d = w; ///< d0,w0
        matA.MatPVec(*d, *v);
        value1 = d->innerProduct(*r);
        value2 = d->innerProduct(*v);
        value1 /= value2;
        x.add(1, x, value1, *d); ///<x1
        r.add(1, *r, -value1, *v); ///<r1

        for (UINT32 i = 1; i < m_maxIter; ++i) {
            norm = r->norm_2() / norm_b;
            if (norm < m_tol) {
                m_IsConverged = true;
                m = i;
                break;
            }
            w = r;
            if (m_CGType == FlexibleCG && m_precondPtr.get()) {
                m_precondPtr->MInvSolve(*w);
            }
            matA.MatPVec(*d, *v); ///<A*d0 = v
            value1 = w->innerProduct(*v); ///< w1*v
            value2 = d->innerProduct(*v); ///< d0*v
            value1 = value1 / value2;
            d.add(1, *w, -value1, *d);

            value1 = d->innerProduct(*r);
            matA.MatPVec(*d, *v);
            value2 = d->innerProduct(*v);
            value1 = value1 / value2;
            x.add(1, x, value1, *d);
            r.add(1, *r, -value1, *v);
        }
        this->m_numIter = m + 1;
        this->m_finalError = norm;
    }
} // HOST
