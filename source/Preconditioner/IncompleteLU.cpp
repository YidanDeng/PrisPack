/*
* @author  刘玉琴、邓轶丹
 * @date    2024/6/12
 * @details ILU分解预条件
 */

#include "../../include/Preconditioner/IncompleteLU.h"
#include "../../include/VectorClass/VectorTools.h"


namespace HOST {
    template<typename LowPrecisionType, typename HighPrecisionType>
    IncompleteLU<LowPrecisionType, HighPrecisionType>::IncompleteLU(
        const std::shared_ptr<CSRMatrix<HighPrecisionType> > &matA, UINT32 lfill, HighPrecisionType tolerance) {
        this->m_precondType = PreconditionILUT;
        m_matA = matA; // 浅拷贝
        this->m_ArowNum = m_matA->getRowNum();
        this->m_AcolNum = m_matA->getColNum();
        this->m_Annz = m_matA->getNNZnum(0, this->m_ArowNum - 1);
        m_lfill = lfill;
        m_tolerance = tolerance;
        UINT32 guessNNZnum = this->m_ArowNum * lfill;
#ifdef OPENMP_FOUND
#pragma omp critical
        {
            m_L.construct(this->m_ArowNum, this->m_ArowNum, guessNNZnum, matA->getMemoryType());
            m_U.construct(this->m_ArowNum, this->m_ArowNum, guessNNZnum, matA->getMemoryType());
            m_diagVals.construct(this->m_ArowNum, matA->getMemoryType());
        }
#else
        m_L.construct(this->m_ArowNum, this->m_ArowNum, guessNNZnum, matA->getMemoryType());
        m_U.construct(this->m_ArowNum, this->m_ArowNum, guessNNZnum, matA->getMemoryType());
        m_diagVals.construct(this->m_ArowNum, matA->getMemoryType());
#endif
    }


    template<typename LowPrecisionType, typename HighPrecisionType>
    void IncompleteLU<LowPrecisionType, HighPrecisionType>::setup() {
#ifndef NDEBUG
        THROW_EXCEPTION(m_matA.get() == nullptr, THROW_INVALID_ARGUMENT("The original matrix A was not initialized!"))
        THROW_EXCEPTION(this->m_ArowNum == 0 || this->m_Annz == 0 || m_matA.get() == nullptr
                        || !m_L.get() || !m_U.get() || !m_diagVals.get(),
                        THROW_INVALID_ARGUMENT("The original data was not initialized!"))
#endif
        // 如果三角分解已经求解过，则什么都不做
        if (this->m_isReady == PRECONDITION_READY) return;
        UINT32 n = m_matA->getRowNum();
        INT32 len, lenu, lenl, nnzL, nnzU;
        INT32 i;
        UINT32 j, nzcount, col, k, k1, k2;
        INT32 jpos, jrow, upos;
        HighPrecisionType tnorm, tolnorm;
        LowPrecisionType fact, lxu;
        LowPrecisionType t;
        UINT32 *mat_i, *mat_j, *L_i, *U_i, *ja;
        HighPrecisionType *mat_value;
        LowPrecisionType *D, *ma;
        mat_i = m_matA->getRowOffsetPtr(0);
        mat_j = m_matA->getColIndicesPtr(0);
        mat_value = m_matA->getCSRValuesPtr(0);
        L_i = m_L->getRowOffsetPtr(0);
        L_i[0] = 0;
        auto L_j = m_L->getColIndicesPtr(0);
        auto L_a = m_L->getCSRValuesPtr(0);
        nnzL = 0;
        D = &(*m_diagVals)[0];
        U_i = m_U->getRowOffsetPtr(0);
        U_i[0] = 0;
        auto U_j = m_U->getColIndicesPtr(0);
        auto U_a = m_U->getCSRValuesPtr(0);
        nnzU = 0;

        AutoAllocateVector<INT32> iw(n, memoryBase);
        AutoAllocateVector<INT32> jbuf(n, memoryBase);
        AutoAllocateVector<LowPrecisionType> wn(n, memoryBase);
        AutoAllocateVector<LowPrecisionType> w(n, memoryBase);

        for (i = 0; i < n; i++) iw[i] = -1;
        //主循环
        for (i = 0; i < n; i++) {
            UINT32 begin = mat_i[i];
            UINT32 end = mat_i[i + 1];
            nzcount = end - begin;
            tnorm = 0;
            for (j = begin; j < end; j++) {
                tnorm += fabs(mat_value[j]); //tnorm 为行和
            }
            THROW_EXCEPTION(tnorm < 1e-15, THROW_LOGIC_ERROR("ILUT: zero row encountered!"))
            tnorm /= (HighPrecisionType) nzcount; //tnorm 为1范数
            tolnorm = m_tolerance * tnorm; //阈值

            /* 存储矩阵mat第i行的L部分和U部分 */
            lenu = 0;
            lenl = 0;
            jbuf[i] = i;
            w[i] = 0;
            iw[i] = i;
            for (j = begin; j < end; j++) {
                col = mat_j[j];
                t = mat_value[j];
                if (col < i) {
                    iw[col] = lenl; //iw为非零元指示器，值为0，1，2，3...(<i)（更新L部分）
                    jbuf[lenl] = (INT32) col; //指向iw，记录非这行对角线左边零元的列
                    w[lenl] = t; //和jbuf对应，记录非零元值
                    lenl++; //lenl为对角i左边非零元个数
                } else if (col == i) {
                    w[i] = t; //对角线元素
                } else {
                    lenu++; //lenu为右边非零元个数
                    jpos = i + lenu; //值为i+1,i+2,i+3...用来记录非零元在jbuf和w中的下标
                    iw[col] = jpos; //非零元指示器（U部分）
                    jbuf[jpos] = (INT32) col; //记录上三角部分（这行对角线右边部分）非零元的列索引
                    w[jpos] = t; //记录上三角部分（这行对角线右边部分）非零元的值
                }
            }

            j = -1;
            len = 0;
            /* 消元 */
            while (++j < lenl) {
                /*----------------------------------------------------------------------------
                 *  select the smallest column index among jbuf[k], k = j+1, ..., lenl
                 *--------------------------------------------------------------------------*/
                jrow = jbuf[j]; //jrow为最小列索引
                jpos = (INT32) j; //jpos为对应的在数组中的位置（下标）
                /* determine smallest column index  选出最小的列索引 */
                for (k = j + 1; k < lenl; k++) {
                    if (jbuf[k] < jrow) {
                        //当前jrow不是最小列索引
                        jrow = jbuf[k]; //找到最小列索引
                        jpos = (INT32) k; //对应的下标
                    }
                }
                if (jpos != j) {
                    //当前jbuf[j]不是最小列索引，则交换位置，使jbuf[j]存的永远是最小列索引
                    col = jbuf[j];
                    jbuf[j] = jbuf[jpos];
                    jbuf[jpos] = (INT32) col;
                    iw[jrow] = (INT32) j; //交换相应的iw，非零元指示器
                    iw[col] = jpos;
                    t = w[j]; //交换w
                    w[j] = w[jpos];
                    w[jpos] = t;
                }

                /* get the multiplier */
                fact = w[j] * D[jrow]; //wk=wk/a(k,k)
                w[j] = fact;
                /* zero out element in row by resetting iw(n+jrow) to -1 */
                iw[jrow] = -1;

                /* combine current row and row jrow */
                k1 = U_i[jrow];
                k2 = U_i[jrow + 1];
                for (k = k1; k < k2; k++) {
                    col = (INT32) U_j[k];
                    jpos = iw[col];
                    lxu = -fact * U_a[k]; //-wk*Uk
                    /* if fill-in element is small then disregard */
                    //满足drop规则：小于阈值且是fill-in，所以后来的jpos==-1（填充fill-in）一定是大于阈值的
                    if (fabs(lxu) < tolnorm && jpos == -1) continue;

                    if (col < i) {
                        /* dealing with lower part */
                        if (jpos == -1) {
                            //jpos为-1说明原矩阵没有这个非零元，则它是fill-in
                            /* this is a fill-in element ，将其信息进行更新，即jubf,iw,w，lenl*/
                            jbuf[lenl] = (INT32) col;
                            iw[col] = lenl;
                            w[lenl] = lxu;
                            lenl++;
                        } else {
                            w[jpos] += lxu; //w=w-wk*Uk
                        }
                    } else {
                        /* dealing with upper part */
                        if (jpos == -1 && fabs(lxu) > tolnorm) {
                            /* this is a fill-in element
                             * jpos为-1说明原矩阵没有这个非零元，则它是fill-in*/
                            lenu++;
                            upos = i + lenu;
                            jbuf[upos] = (INT32) col;
                            iw[col] = upos;
                            w[upos] = lxu;
                        } else {
                            w[jpos] += lxu; //更新w
                        }
                    }
                }
            }
            /* restore iw */
            iw[i] = -1;
            for (j = 0; j < lenu; j++) {
                iw[jbuf[i + j + 1]] = -1;
            }

            /*---------- case when diagonal is zero */
            THROW_EXCEPTION(fabs(w[i]) < 1e-15,
                            THROW_LOGIC_ERROR("ILUT: zero diagonal encountered! Zero diagonal row index is: "
                                + std::to_string(i)))

            /*-----------Update diagonal */
            D[i] = (LowPrecisionType) 1 / w[i];

            /* update L-matrix */
            //    len = min( lenl, lfil );
            len = lenl < m_lfill ? lenl : m_lfill;
            for (j = 0; j < lenl; j++) {
                wn[j] = fabs(w[j]);
                iw[j] = (INT32) j;
            }
            qsplit(&wn[0], &iw[0], lenl, len); //只保留前p大个元素，p为人为设定的lfill和非零元个数lenl中较小值
            nnzL = (INT32) L_i[i] + len;
            L_i[i + 1] = nnzL;
            if (nnzL > m_L->getValNum()) {
#ifdef OPENMP_FOUND
#pragma omp critical
#endif
                m_L->resize(this->m_ArowNum, this->m_ArowNum, m_L->getValNum() * 2, RESERVE_DATA);
                L_i = m_L->getRowOffsetPtr(0);
                L_j = m_L->getColIndicesPtr(0);
                L_a = m_L->getCSRValuesPtr(0);
            }
            ja = L_j + L_i[i]; //ja指针指向L_j的第i行开始的位置，用于后续填入列索引
            ma = L_a + L_i[i]; //ma指针指向L_a的第i行开始的位置，用于后续填入值
            for (j = 0; j < len; j++) {
                jpos = iw[j];
                ja[j] = jbuf[jpos];
                ma[j] = w[jpos];
            }
            for (j = 0; j < lenl; j++) iw[j] = -1;

            /* update U-matrix */
            //    len = min( lenu, lfil );
            len = lenu < m_lfill ? lenu : m_lfill;
            for (j = 0; j < lenu; j++) {
                wn[j] = fabs(w[i + j + 1]);
                iw[j] = i + (INT32) j + 1;
            }
            qsplit(&wn[0], &iw[0], lenu, len); //wn前p（len）大个元素，以及对iw也进行了一样的排序
            nnzU = (INT32) U_i[i] + len;
            U_i[i + 1] = nnzU;
            if (nnzU > m_U->getValNum()) {
#ifdef OPENMP_FOUND
#pragma omp critical
#endif
                m_U->resize(this->m_ArowNum, this->m_ArowNum, m_U->getValNum() * 2, RESERVE_DATA);
                U_i = m_U->getRowOffsetPtr(0);
                U_j = m_U->getColIndicesPtr(0);
                U_a = m_U->getCSRValuesPtr(0);
            }
            ja = U_j + U_i[i];
            ma = U_a + U_i[i];
            for (j = 0; j < len; j++) {
                jpos = iw[j];
                ja[j] = jbuf[jpos];
                ma[j] = w[jpos];
            }
            for (j = 0; j < lenu; j++) {
                iw[j] = -1;
            }
        }
        this->m_Mnnz = nnzL + nnzU + this->m_ArowNum;
        this->m_isReady = PRECONDITION_READY;
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void IncompleteLU<LowPrecisionType, HighPrecisionType>::MSolveUpperUsePtr(HighPrecisionType *vec) {
        UINT32 i, j, k1, k2;
        UINT32 *U_i = m_U->getRowOffsetPtr(0);
        UINT32 *U_j = m_U->getColIndicesPtr(0);
        LowPrecisionType *U_data = m_U->getCSRValuesPtr(0);
        for (i = this->m_ArowNum; i > 0; i--) {
            k1 = U_i[i - 1];
            k2 = U_i[i];
            for (j = k1; j < k2; j++) {
                vec[i - 1] -= U_data[j] * vec[U_j[j]];
            }
            vec[i - 1] *= (*m_diagVals)[i - 1];
        }
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void IncompleteLU<LowPrecisionType, HighPrecisionType>::MSolveLowerUsePtr(HighPrecisionType *vec) {
        UINT32 i, j, k1, k2;
        UINT32 *L_i = m_L->getRowOffsetPtr(0);
        UINT32 *L_j = m_L->getColIndicesPtr(0);
        LowPrecisionType *L_data = m_L->getCSRValuesPtr(0);
        for (i = 0; i < this->m_ArowNum; i++) {
            k1 = L_i[i];
            k2 = L_i[i + 1];
            for (j = k1; j < k2; j++) {
                vec[i] -= L_data[j] * vec[L_j[j]];
            }
        }
    }
} // HOST
