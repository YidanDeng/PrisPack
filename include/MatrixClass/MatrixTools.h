/*
 * @author  邓轶丹
 * @date    2024/5/12
 * @details 各种矩阵操作函数
 */

#ifndef PMSLS_NEW_MATRIXTOOL_H
#define PMSLS_NEW_MATRIXTOOL_H

#include "CSRMatrix.h"
#include "COOMatrix.h"


namespace HOST {
    /* ================================= 各种类型矩阵之间的转换函数 ================================= */
    template<typename ValType>
    void transCOO2CSR(COOMatrix<ValType> &inMat, CSRMatrix<ValType> &outMat) {
        UINT32 row, col, nnz, i, j;
        ValType val;
        row = inMat.getRowNum();
        col = inMat.getColNum();
        nnz = inMat.getNNZnum();
        outMat.resize(row, col, nnz, RESERVE_NO_DATA);
        const UINT32 *a_i = inMat.getRowIndicesPtr();
        const UINT32 *a_j = inMat.getColIndicesPtr();
        const ValType *a_data = inMat.getCOOValuesPtr();

        UINT32 *A_i = outMat.getRowOffsetPtr(0);
        UINT32 *A_j = outMat.getColIndicesPtr(0);
        ValType *A_data = outMat.getCSRValuesPtr(0);
#ifndef NDEBUG
        // 因为debug模式把未成功填充的行都默认填充为UINT32_MAX，所以下面就会在计算rowOffset时出错
        std::fill_n(A_i, row + 1, 0);
#endif
        // 计算CSR的rowOffset
        for (i = 0; i < nnz; i++) {
            A_i[a_i[i] + 1]++;
        }
        for (i = 0; i < row; i++) {
            A_i[i + 1] += A_i[i];
        }
        for (i = 0; i < nnz; i++) {
            j = a_i[i];
            val = a_data[i];
            A_data[A_i[j]] = val;
            A_j[A_i[j]++] = a_j[i];
        }
        for (i = row; i > 0; i--) {
            A_i[i] = A_i[i - 1];
        }
        A_i[0] = 0;
        // 由于写入非零元时，其列索引不一定有序排列，所以需要重新调整非零元排序，使其变为标准CSR格式
        UINT32 *sortListPtr1 = A_j;
        ValType *sortListPtr2 = A_data;
        UINT32 currNNZ;
        for (UINT32 rowIdx = 0; rowIdx < row; ++rowIdx) {
            currNNZ = A_i[rowIdx + 1] - A_i[rowIdx];
            sortVectorPair(sortListPtr1, sortListPtr2, currNNZ);
            sortListPtr1 += currNNZ;
            sortListPtr2 += currNNZ;
        }
    }
} // HOST


#endif //PMSLS_NEW_MATRIXTOOL_H
