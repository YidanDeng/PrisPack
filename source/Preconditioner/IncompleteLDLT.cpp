/*
 * @author  刘玉琴、邓轶丹
 * @date    2024/11/13
 * @details ILDLT分解预条件
 */

#include "../../../include/Preconditioner/IncompleteLDLT.h"

namespace HOST {
    template<typename LowPrecisionType, typename HighPrecisionType>
    IncompleteLDLT<LowPrecisionType, HighPrecisionType>::IncompleteLDLT(
            const std::shared_ptr<CSRMatrix<HighPrecisionType> > &matA, HighPrecisionType tolerance) {
        this->m_precondType = PreconditionILDLT;
        m_matA = matA; // 浅拷贝
        this->m_ArowNum = matA->getRowNum();
        this->m_AcolNum = matA->getColNum();
#ifndef NDEBUG
        if (this->m_ArowNum != this->m_AcolNum) {
            SHOW_ERROR("The number of rows and columns must be same!");
        }
#endif
        this->m_Annz = matA->getNNZnum(0, this->m_ArowNum - 1);
        m_tolerance = tolerance;
        UINT32 currNNZnum = this->m_ArowNum > 0 ? this->m_Annz : 0;
        UINT32 guessNNZnum = currNNZnum * 2;
#ifdef OPENMP_FOUND
#pragma omp critical
        {
            // 多线程环境下，内存分配不是线程安全的，所以要使用临界区
            m_matTransL.construct(this->m_ArowNum, this->m_ArowNum, guessNNZnum, matA->getMemoryType());
            m_diagVals.construct(this->m_ArowNum, matA->getMemoryType());
        }
#else
        m_matTransL.construct(this->m_ArowNum, this->m_ArowNum, guessNNZnum, matA->getMemoryType());
        m_diagVals.construct(this->m_ArowNum, matA->getMemoryType());
#endif
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void IncompleteLDLT<LowPrecisionType, HighPrecisionType>::setup() {
#ifndef NDEBUG
        THROW_EXCEPTION(m_matA.get() == nullptr, THROW_INVALID_ARGUMENT("The original matrix A was not initialized!"))
        THROW_EXCEPTION(this->m_ArowNum == 0 || this->m_Annz == 0 || m_matA.get() == nullptr
                        || m_matTransL.get() == nullptr || m_diagVals.get() == nullptr,
                        THROW_INVALID_ARGUMENT("The original data was not initialized!"))
#endif
        if (this->m_isReady == PRECONDITION_READY) return;
        INT32 numNz;
        UINT32 ii, jj, kk, idx, n_indices, idx_start, k_next, first;
        HighPrecisionType colThr, A_ij, diagElement, d_kk, L_ij, L_jk, L_ik;
        //某一列的临时数组
        UINT32 colNum = m_matA->getColNum();
        std::unique_ptr<UINT8[]> vUsedCol = std::make_unique<UINT8[]>(colNum);
        AutoAllocateVector<UINT32> vIdxCol(colNum, memoryBase);
        AutoAllocateVector<HighPrecisionType> vValCol(colNum, memoryBase);
        AutoAllocateVector<UINT32> vBuffer(colNum, memoryBase);
        //某一行的临时数组
        AutoAllocateVector<UINT32> vNonZeroRow(colNum, memoryBase);
        AutoAllocateVector<UINT32> vNextIdxRow(colNum, memoryBase);
        // 初始化标记变量
        for (UINT32 i = 0; i < colNum; i++) {
            vUsedCol[i] = 0;
            vNonZeroRow[i] = UINT32_MAX;
        }
        // 获取A的三个成员变量的实际内存指针，用于读取
        const UINT32 *vIndicesPtrA = m_matA->getRowOffsetPtr(0);
        const UINT32 *vIndicesA = m_matA->getColIndicesPtr(0);
        const HighPrecisionType *vDataA = m_matA->getCSRValuesPtr(0);
        // L的三个成员变量，用于写入(暂时取出，后面如果需要resize矩阵则需要重新获取指针，因为resize可能导致地址发生变化)
        UINT32 *vIndicesPtr = m_matTransL->getRowOffsetPtr(0);
        UINT32 *vIndices = m_matTransL->getColIndicesPtr(0);
        LowPrecisionType *vData = m_matTransL->getCSRValuesPtr(0);

        numNz = 0; // 当前transL中的非零元个数
        vIndicesPtr[0] = 0;
        //遍历矩阵L的每一列
        for (jj = 0; jj < colNum; jj++) {
            //矩阵A的第j列（第j行）信息
            n_indices = 0;
            colThr = 0.0;
            for (idx = vIndicesPtrA[jj]; idx < vIndicesPtrA[jj + 1]; idx++) {
                ii = vIndicesA[idx];
                A_ij = vDataA[idx];
                // 考虑A的下三角部分
                if (ii >= jj) {
                    colThr += fabs(A_ij);
                    // 当前列还未处理的元素
                    if (!vUsedCol[ii]) {
                        vValCol[ii] = 0.0;
                        vUsedCol[ii] = 1;
                        vIdxCol[n_indices] = ii;
                        n_indices += 1;
                    }
                    vValCol[ii] += A_ij;
                }
            }
            colThr *= m_tolerance;
            // Compute new values for column j using nonzero values L_jk of row j
            kk = vNonZeroRow[jj];
            while (kk != UINT32_MAX) {
                idx_start = vNextIdxRow[kk];
                L_jk = vData[idx_start];
                d_kk = (*m_diagVals)[kk];
                // 非零元L_ik  Using nonzero values L_ik of column k
                for (idx = idx_start; idx < vIndicesPtr[kk + 1]; idx++) {
                    ii = vIndices[idx];
                    L_ik = vData[idx];
                    // Activate column element if it is not in use yet
                    if (!vUsedCol[ii]) {
                        vUsedCol[ii] = 1;
                        vValCol[ii] = 0.0;
                        vIdxCol[n_indices] = ii;
                        n_indices += 1;
                    }
                    vValCol[ii] -= L_ik * L_jk * d_kk;
                }
                // 第k列的下一个非零元 Advance to next non-zero element in column k if it exists
                idx_start += 1;
                k_next = vNonZeroRow[kk];
                // Update start of next row
                if (idx_start < vIndicesPtr[kk + 1]) {
                    vNextIdxRow[kk] = idx_start;
                    ii = vIndices[idx_start];
                    vNonZeroRow[kk] = vNonZeroRow[ii];
                    vNonZeroRow[ii] = kk;
                }
                kk = k_next;
            }
            diagElement = vValCol[jj];
            // If pivot value is zero
            THROW_EXCEPTION(fabs(diagElement) < 1e-15,
                            THROW_LOGIC_ERROR("ILDLT: zero diagonal encountered! Zero diagonal row index is: "
                                              + std::to_string(jj)))

            (*m_diagVals)[jj] = diagElement;
            // Write diagonal element into matrix L
            vNextIdxRow[jj] = numNz;

            // Output indices must be sorted
            ArrayMergeSort(&vIdxCol[0], n_indices, &vBuffer[0]);

            // Write column j into matrix L
            first = 1;
            for (idx = 0; idx < n_indices; idx++) {
                ii = vIdxCol[idx];
                if (ii != jj) {
                    L_ij = vValCol[ii];
                    // Drop small values
                    if (ii != jj && fabs(L_ij) >= colThr) {
                        // Next row starts here
                        if (first) {
                            first = 0;
                            vNonZeroRow[jj] = vNonZeroRow[ii];
                            vNonZeroRow[ii] = jj;
                        }
                        // Write element L_ij into L
                        L_ij = L_ij / diagElement;
                        vData[numNz] = L_ij;
                        vIndices[numNz] = ii;
                        numNz++;
                        if (numNz >= m_matTransL->getValNum()) {
                            // 如果当前非零元个数超出预先分配的内存大小，则直接扩大到原来的2倍
#ifdef OPENMP_FOUND
                            // 多线程环境下，内存分配不是线程安全的，所以要使用临界区
#pragma omp critical
#endif
                            m_matTransL->resize(m_matA->getRowNum(), m_matA->getRowNum(), 2 * numNz, RESERVE_DATA);
                            vIndicesPtr = m_matTransL->getRowOffsetPtr(0);
                            vIndices = m_matTransL->getColIndicesPtr(0);
                            vData = m_matTransL->getCSRValuesPtr(0);
                        }
                    }
                }
                // Clear column information
                vUsedCol[ii] = 0;
            }
            // Keep track of number of elements per column
            vIndicesPtr[jj + 1] = numNz;
        }
#ifndef NDEBUG
        THROW_EXCEPTION(m_matTransL->getRowOffsetPtr(0)[m_matTransL->getColNum()] != numNz,
                        THROW_LOGIC_ERROR("The IC decomposition was incorrect!"))
#endif
        this->m_isReady = PRECONDITION_READY;
        this->m_Mnnz = numNz * 2 + this->m_ArowNum;
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void IncompleteLDLT<LowPrecisionType, HighPrecisionType>::MSolveUpperUsePtr(HighPrecisionType *vec) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_isReady == PRECONDITION_NOT_READY, THROW_LOGIC_ERROR("The setup func was not ready!"))
#endif
        UINT32 i, k, k1, k2;
        const UINT32 *rowOffset = m_matTransL->getRowOffsetPtr(0);
        const UINT32 *colIndices = m_matTransL->getColIndicesPtr(0);
        const LowPrecisionType *value = m_matTransL->getCSRValuesPtr(0);
        for (i = m_matTransL->getRowNum() - 1; i + 1 > 0; --i) {
            k1 = rowOffset[i];
            k2 = rowOffset[i + 1];
            for (k = k1; k < k2; k++) {
                vec[i] -= value[k] * vec[colIndices[k]];
            }
        }
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void IncompleteLDLT<LowPrecisionType, HighPrecisionType>::MSolveDiagonalUsePtr(HighPrecisionType *vec) {
        for (UINT32 i = 0; i < this->m_ArowNum; ++i) {
#ifndef NDEBUG
            THROW_EXCEPTION(fabs((*m_diagVals)[i]) < 1e-15, THROW_LOGIC_ERROR("The diagonal value is not positive!"))
#endif
            vec[i] /= (*m_diagVals)[i];
        }
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void IncompleteLDLT<LowPrecisionType, HighPrecisionType>::MSolveLowerUsePtr(HighPrecisionType *vec) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_isReady == PRECONDITION_NOT_READY, THROW_LOGIC_ERROR("The setup func was not ready!"))
#endif
        UINT32 n, i, k, k1, k2;
        n = m_matTransL->getRowNum();
        const UINT32 *U_i = m_matTransL->getRowOffsetPtr(0);
        const UINT32 *U_j = m_matTransL->getColIndicesPtr(0);
        const LowPrecisionType *U_data = m_matTransL->getCSRValuesPtr(0);
        for (i = 0; i < n; i++) {
            k1 = U_i[i];
            k2 = U_i[i + 1];
            for (k = k1; k < k2; ++k) {
                vec[U_j[k]] -= U_data[k] * vec[i];
            }
        }
    }
}
