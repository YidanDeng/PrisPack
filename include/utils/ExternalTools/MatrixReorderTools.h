/*
 * @author  邓轶丹
 * @date    2024/11/18
 * @details 矩阵重排序接口，核心通过第三方库实现，目前已有AMD(Approximate Minimum Degree)排序和RCM(Reverse Cuthill-McKee)排序
 */

#ifndef MATRIX_REORDER_TOOLS_H
#define MATRIX_REORDER_TOOLS_H

#include "../../MatrixClass/CSRMatrix.h"
#include "amd.h"
#include "rcm.hpp"

enum MatrixReorderOption {
    MatrixReorderNo,
    MatrixReorderAMD,
    MatrixReorderRCM
};

typedef enum MatrixReorderOption MatrixReorderOption_t;

namespace HOST {
    /** @brief 近似最小度排序
     * @note 输入本应是CSC格式的矩阵，但我们主要使用对称变换（即原始矩阵必须是结构对称的），所以这里仍旧使用CSR矩阵
     * @param [in] mat : 待排序矩阵（稀疏对称矩阵，至少是结构对称）
     * @param [in, out] perm 排序后的映射数组，记录排序后的编号对应的原始编号（数组下标：现节点编号，数组元素：现节点对应的原编号） */
    template<typename ValType>
    void amdReorderCSR(const CSRMatrix<ValType> &mat, HostVector<UINT32> &perm) {
#ifndef NDEBUG
        THROW_EXCEPTION(mat.getRowNum()!= mat.getColNum(), THROW_LOGIC_ERROR("The matrix must be square matrix!"))
        THROW_EXCEPTION(mat.getRowNum() >= INT_MAX,
                        THROW_OUT_OF_RANGE("The row dim of CSR mat is too large! It causes overflow of int type!"))
#endif
        INT32 status;
        INT32 n = (INT32) mat.getRowNum();
        if (perm.getLength() != n) perm.resize(n, RESERVE_NO_DATA);
        const INT32 *Ai, *Aj;
        INT32 *p;
        Ai = (const INT32 *) mat.getRowOffsetPtr(0);
        Aj = (const INT32 *) mat.getColIndicesPtr(0);
        p = (INT32 *) &perm[0];
        // 编号本身是0-base，所以无需转换编号起始范围
        status = amd_order(n, Ai, Aj, p, nullptr, nullptr);
        THROW_EXCEPTION(status != AMD_OK && status != AMD_OK_BUT_JUMBLED, THROW_LOGIC_ERROR("AMD-api failed!"))
    }


    /** @brief 逆Cuthill-McKee排序
     * @param [in] mat : 待排序矩阵（稀疏对称矩阵，至少是结构对称）
     * @param [in, out] perm : 排序后的映射数组，记录排序后的编号对应的原始编号（数组下标：现节点编号，数组元素：现节点对应的原编号）*/
    template<typename ValType>
    void rcmReorderCSR(const CSRMatrix<ValType> &mat, HostVector<UINT32> &perm) {
#ifndef NDEBUG
        THROW_EXCEPTION(mat.getRowNum()!= mat.getColNum(), THROW_LOGIC_ERROR("The matrix must be square matrix!"))
        THROW_EXCEPTION(mat.getRowNum() >= INT_MAX,
                        THROW_OUT_OF_RANGE("The row dim of CSR mat is too large! It causes overflow of int type!"))
#endif
        // 将原CSR矩阵转换为1-base形式
        INT32 *rowOffsetPtr = (INT32 *) mat.getRowOffsetPtr(0);
        INT32 *colIdxPtr = (INT32 *) mat.getColIndicesPtr(0);
        UINT32 nnzNum = mat.getNNZnum(0, mat.getRowNum() - 1), dim = mat.getRowNum();
        if (perm.getLength() != dim) perm.resize(dim, RESERVE_NO_DATA);
#ifndef NDEBUG
        THROW_EXCEPTION(nnzNum >= INT_MAX || dim >= INT_MAX,
                        THROW_OUT_OF_RANGE("The none-zero element number or row dim is too large! Overflow occurred!"))
#endif

        for (UINT32 i = 0; i <= dim; ++i) {
            rowOffsetPtr[i] += 1;
        }
        for (UINT32 i = 0; i < nnzNum; ++i) {
            colIdxPtr[i] += 1;
        }
        genrcm(mat.getRowNum(), nnzNum, rowOffsetPtr, colIdxPtr, (INT32 *) &perm[0]);
        // 排序完再还原为0-base
        for (UINT32 i = 0; i <= dim; ++i) {
            rowOffsetPtr[i] -= 1;
        }
        for (UINT32 i = 0; i < nnzNum; ++i) {
            colIdxPtr[i] -= 1;
        }
        for (UINT32 i = 0; i < dim; ++i) {
            perm[i] -= 1;
        }
    }


    /** @brief 用来对已经执行过图划分排序的子区域内点重新进行排序，以减少矩阵分解时的填充
     * @param [in] originMat    :   原矩阵（大矩阵）
     * @param [in] originPerm   :   原大矩阵对应的完整perm数组（主要是图划分直接得到的perm）
     * @param [in, out] subPermStartShift     : 待排序的子区域在原perm数组中的起始下标
     * @param [in, out] subPermLength         : 子区域集合大小 */
    template<typename ValType>
    void reorderSubMatrixCSR(MatrixReorderOption_t reorderOption, const CSRMatrix<ValType> &originMat,
                             HostVector<UINT32> &originPerm, UINT32 subPermStartShift, UINT32 subPermLength) {
        if (reorderOption == MatrixReorderNo) return;
        CSRMatrix<ValType> subMat;
        AutoAllocateVector<UINT32> copyPerm(subPermLength, memoryBase), subPerm(0, memoryBase);
        // 存储原来的全局结点编号
        copyPerm.copy(originPerm, subPermStartShift, 0, subPermLength);
        // 获得子矩阵
        originMat.getSubMatrix(*copyPerm, *copyPerm, subMat);
        // 对子矩阵进行排序，得到的是局部结点编号的映射关系（下标：现局部节点编号；值：原局部节点编号）
        if (reorderOption == MatrixReorderAMD) amdReorderCSR(subMat, *subPerm);
        else if (reorderOption == MatrixReorderRCM) rcmReorderCSR(subMat, *subPerm);
#ifndef NDEBUG
        else {
            TRY_CATCH(THROW_LOGIC_ERROR("Wrong type of matrix ordering!"))
        }
#endif
        // 重排当前子区域
        for (UINT32 i = 0; i < subPermLength; ++i) {
            originPerm[i + subPermStartShift] = copyPerm[subPerm[i]];
        }
    }
}


#endif //MATRIX_REORDER_TOOLS_H
