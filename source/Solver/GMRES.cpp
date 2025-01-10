/*
 * @author  袁心磊
 * @date    2024/6/15
 * @details 实现GMRES求解器，标准GMRES、PGMRES
 */

#include "../../include/Solver/GMRES.h"

namespace HOST {

    template<typename ValType>
    GMRES<ValType>::GMRES(const UINT32 &problemSize) {
        this->m_n = problemSize;
        this->m_tol = 1e-6;
        this->m_maxIter = MAX_ITER_NUM_SOLVER;
        this->m_IsConverged = false;
        this->m_numIter = -1;
        this->m_finalError = (FLOAT64) (-1);

        m_m = problemSize;
        m_restart = false;
    }

    template<typename ValType>
    GMRES<ValType>::GMRES(const UINT32 &problemSize, const std::shared_ptr<BasePrecondition<ValType>> &precond,
                          const GMRESType_t &gmresType) {
        this->m_n = problemSize;
        this->m_precondPtr = precond;
        this->m_GMRESType = gmresType;
        this->m_tol = 1e-7;
        this->m_maxIter = MAX_ITER_NUM_SOLVER;
        this->m_IsConverged = false;
        this->m_numIter = -1;
        this->m_finalError = (FLOAT64) (-1);

        this->m_m = problemSize;
        m_restart = false;

    }

    template<typename ValType>
    GMRES<ValType>::GMRES(const UINT32 &problemSize, const std::shared_ptr<BasePrecondition<ValType>> &precond,
                          const FLOAT64 &tolerance, const GMRESType_t &gmresType) {
        this->m_n = problemSize;
        this->m_precondPtr = precond;
        this->m_GMRESType = gmresType;
        this->m_tol = tolerance;
        this->m_maxIter = MAX_ITER_NUM_SOLVER;
        this->m_IsConverged = false;
        this->m_numIter = -1;
        this->m_finalError = (FLOAT64) (-1);

        this->m_m = problemSize;
        m_restart = false;

    }


    template<typename ValType>
    void GMRES<ValType>::setRestart(bool restart, UINT32 m) {
        m_m = m;
        m_restart = restart;
        this->m_maxIter = this->m_maxIter / m;
        if (m >= this->m_n) {
            m_m = this->m_n - 1;
        }
        if (!restart) {
            m_m = this->m_n - 1;
        }
    }

    template<typename ValType>
    void GMRES<ValType>::givensTransform(DenseMatrix<ValType> &H, HostVector<ValType> &c, HostVector<ValType> &s,
                                         HostVector<ValType> &ksi, const UINT32 &j) {
        ValType val0 = 0.0, val1 = 0.0, val2 = 0.0, tao = 0.0;
        for (UINT32 i = 0; i < j; ++i) {
            H.getValue(i, j, val1);
            H.getValue(i + 1, j, val2);
            val0 = c.getValue(i) * val2 - s.getValue(i) * val1;
            H.setValue(i + 1, j, val0);
            val0 = c.getValue(i) * val1 + s.getValue(i) * val2;
            H.setValue(i, j, val0);
        }
        H.getValue(j, j, val1);
        H.getValue(j + 1, j, val2);
        if (fabs(val1) > fabs(val2)) {
            tao = val2 / val1;
            c.setValue(j, 1 / sqrt(fabs((ValType) 1 + tao * tao)));
            s.setValue(j, c.getValue(j) * tao);
        } else {
            tao = val1 / val2;
            s.setValue(j, 1 / sqrt(fabs((ValType) 1 + tao * tao)));
            c.setValue(j, s.getValue(j) * tao);
        }
        val0 = c.getValue(j) * val1 + s.getValue(j) * val2;
        H.setValue(j, j, val0);
        val0 = 0.0;
        H.setValue(j + 1, j, val0);
        ksi.setValue(j + 1, -s.getValue(j) * ksi.getValue(j));
        ksi.setValue(j, c.getValue(j) * ksi.getValue(j));
    }

    template<typename ValType>
    void GMRES<ValType>::retrospectiveSolver(DenseMatrix<ValType> &H, HostVector<ValType> &ksi, const UINT32 &dim) {
        ValType val0 = 0.0, val1 = 0.0;
        for (auto i = (int) dim - 1; i >= 0; --i) {
            for (int j = i + 1; j < dim; ++j) {
                val0 = ksi.getValue(i);
                H.getValue(i, j, val1);
                val0 -= val1 * ksi.getValue(j);
                ksi.setValue(i, val0);
            }
            val1 = ksi.getValue(i);
            H.getValue(i, i, val0);
            val1 /= val0;
            ksi.setValue(i, val1);
        }
        ksi.setValue(dim, 0);
    }

    template<typename ValType>
    void
    GMRES<ValType>::solve(const HOST::CSRMatrix<ValType> &matA, const HostVector<ValType> &b, HostVector<ValType> &x) {
        this->m_IsConverged = false;
        this->m_numIter = -1;
        this->m_finalError = (FLOAT64) (-1);
        UINT32 m = this->m_m;
        UINT32 n = m_n;
        ValType beta = 0.0, val0 = 0.0;
        FLOAT64 rNorm = 0.0, relative_error = 0.0;
        FLOAT64 bNorm = std::sqrt(b.sumKahan(0,n,[](ValType x){return x*x;}));
        HOST::AutoAllocateVector<ValType> vj(n, memoryBase), wj(n, memoryBase), r(n, memoryBase);
        HOST::AutoAllocateVector<ValType> ksi(m + 1, memoryBase), s(m + 1, memoryBase), c(m + 1, memoryBase);
        HOST::DenseMatrix<ValType> V(DenseMatColumnFirst, n, m + 1, memoryBase), H(DenseMatColumnFirst, m + 1, m,
                                                                                   memoryBase);
//        CPU_TIMER_FUNC()
//        CPU_TIMER_BEGIN()
        if (m_precondPtr) m_precondPtr->setup();
//        CPU_TIMER_END()
//        std::cout << " --- setup executes: " << CPU_EXEC_TIME() << std::endl;

//        CPU_TIMER_BEGIN()
        SharedObject<HOST::DenseMatrix<ValType>> Z;
        if (m_GMRESType == FlexibleGMRES) {
            Z.construct(DenseMatColumnFirst, n, m + 1, memoryBase);
        }

        matA.MatPVec(x, *r);
        vj.add(1, b, -1, *r);
        beta = std::sqrt(vj->sumKahan(0,n,[](ValType x){return x*x;}));
        //更新xk和残差
        INT32 k = -1;
//        std::cout << "[INFO] b norm: " << bNorm << std::endl;
//        std::cout << "[INFO] tol: " << m_tol << std::endl;
        while (!(this->m_IsConverged) && k < this->m_maxIter) {
            ++k;
            //Householder变换构造标准正交基
            vj->scale(1 / beta);
            V.setValsByCol(0, *vj);
            ksi->fillVector(0, this->m_m, 0);
            ksi->setValue(0, beta);
            for (UINT32 j = 0; j < m; ++j) {
                // 需要调预条件方法
                if (m_GMRESType == FlexibleGMRES) {
#ifndef NDEBUG
                    THROW_EXCEPTION(!m_precondPtr,
                                    THROW_INVALID_ARGUMENT("FlexibleGMRES must use precondtioner!"))
#endif
                    m_precondPtr->MInvSolve(*vj);
                    Z->setValsByCol(j, *vj);
                }
                matA.MatPVec(*vj, *wj);
                //Arnoldi正交分解
                for (UINT32 i = 0; i < j + 1; ++i) {
                    V.getValsByCol(i, *vj);
                    val0 = wj->innerProduct(*vj);
                    H.setValue(i, j, val0);
                    vj->scale(val0);
                    wj.add(-1, *vj);
                }
                val0 = std::sqrt(wj->sumKahan(0,n,[](ValType x){return x*x;}));
                H.setValue(j + 1, j, val0);
                if (fabs(val0) == 0) {
                    m = j;
                    break;
                }
                vj = wj;
                vj->scale(1 / val0);
                V.setValsByCol(j + 1, *vj);
                //将Givens变换Gi作用于矩阵H(i+1，i)的最后一列
                givensTransform(H, *c, *s, *ksi, j);
                rNorm = fabs(ksi.getValue(j + 1));
                relative_error = rNorm / bNorm;
                if (relative_error < this->m_tol) {
//                    std::cout << "[INFO] curr rnorm: " << rNorm << std::endl;
                    m = j;
                    break;
                }
            }
            //求解最小二乘问题，转化为求解上三角R_m y=g_m
            retrospectiveSolver(H, *ksi, m);
            ksi.setValue(m, 0);
            if (m_GMRESType == FlexibleGMRES) {
                Z->MatVec(*ksi, *wj);
            } else if (m_GMRESType == NormalGMRES) {
                V.MatVec(*ksi, *wj);
            }
            x.add(1, *wj);

            //更新停止条件
            this->m_IsConverged = (relative_error < this->m_tol);
            matA.MatPVec(x, *r);
            vj->add(1, b, -1, *r);
            beta = (ValType) rNorm;
        }
//        std::cout << "[INFO] final err: " << relative_error << std::endl;
        this->m_finalError = relative_error;
        this->m_numIter = k * this->m_m + m;
//        CPU_TIMER_END()
//        std::cout << " --- core executes: " << CPU_EXEC_TIME() << std::endl;
    }


} // HOST