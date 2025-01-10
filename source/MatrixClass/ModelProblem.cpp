/*
 * @author  邓轶丹
 * @date    2024/5/8
 * @details 实现模型问题相关操作函数
 */
#include "../../include/MatrixClass/ModelProblem.h"

namespace HOST {
    template<typename ValType>
    void generatePoissonCSR(CSRMatrix<ValType> &mat, const UINT32 &diag_block_size) {
        UINT32 final_dim = diag_block_size * diag_block_size;
        UINT32 nnz = (diag_block_size + 2 * (diag_block_size - 1)) * diag_block_size +
                     (diag_block_size - 1) * diag_block_size * 2; // 其余矩阵块（相当于若干个单位阵）的非0元个数
        mat.resize(final_dim, final_dim, nnz, RESERVE_NO_DATA);
        // 用智能指针管理临时Vector对象，用move函数转移CSR中的各私有成员变量再写入值
        std::unique_ptr<HostVector<UINT32>> temp_offset;
        std::unique_ptr<HostVector<UINT32>> temp_colInd;
        std::unique_ptr<HostVector<ValType>> temp_val;
        memoryType_t memoryType = mat.getMemoryType();
        HOST::initializeVector(temp_offset, 0, memoryType);
        HOST::initializeVector(temp_colInd, 0, memoryType);
        HOST::initializeVector(temp_val, 0, memoryType);
        mat.moveRowOffsetTo(*temp_offset);
        mat.moveColIndicesTo(*temp_colInd);
        mat.moveValuesTo(*temp_val);

        /* 对CSR格式矩阵进行赋值。*/
        INT32 recent_nnz = 0; // 记录当前非零元个数
        for (UINT32 k = 0; k < diag_block_size; ++k) {
            // 按行填充每个块
            for (UINT32 i = k * diag_block_size; i < (k + 1) * diag_block_size; ++i) {
                /* 开始填充每一行的数据 */
                (*temp_offset)[i] = recent_nnz;
                if (k != 0) { // 填充对角块下方的-I（即单位阵取负号）
                    (*temp_colInd)[recent_nnz] = i - diag_block_size;
                    (*temp_val)[recent_nnz] = -1;
                    recent_nnz++;
                }
                for (INT32 j = (INT32) i - 1; j <= (INT32) i + 1; ++j) {
                    if (j == (INT32) i) { // 修改对角块的对角元（即整个矩阵的对角元）
                        (*temp_colInd)[recent_nnz] = j;
                        (*temp_val)[recent_nnz] = 4;
                        recent_nnz++;
                    } else {
                        if (j >= (INT32) (k * diag_block_size) &&
                            j < (INT32) ((k + 1) * diag_block_size)) {
                            // j的范围不能超出当前块的范围（即当前块在原始矩阵中的列标范围）
                            (*temp_colInd)[recent_nnz] = j;
                            (*temp_val)[recent_nnz] = -1;
                            recent_nnz++;
                        }
                    }
                }
                if (k != diag_block_size - 1) { // 填充对角块上方的-I（即单位阵取负号）
                    (*temp_colInd)[recent_nnz] = i + diag_block_size;
                    (*temp_val)[recent_nnz] = -1;
                    recent_nnz++;
                }
            }
        }
        (*temp_offset)[final_dim] = recent_nnz;
        mat.moveRowOffsetFrom(*temp_offset);
        mat.moveColIndicesFrom(*temp_colInd);
        mat.moveValuesFrom(*temp_val);
    }


    template<typename ValType>
    void
    generateLaplacianCOO(COOMatrix<ValType> &mat, const INT32 &nx, const INT32 &ny, const INT32 &nz,
                         const ValType &alphax, const ValType &alphay, const ValType &alphaz, const ValType &shift) {
        INT32 n = nx * ny * nz;
        INT32 i, j, k, ii, jj;
        ValType v, vxp, vxn, vyp, vyn, vzp, vzn, vd;
        //2D问题 or 3D问题
        INT32 numones = 0;
        if (nx == 1) numones++;//如果nx为1，说明该维没有进行剖分
        if (ny == 1) numones++;
        if (nz == 1) numones++;

        bool is2D = numones > 0;
        if (numones == 2) {
            SHOW_WARN("generating a 1D Laplacian problem")
        }
        v = -1.0;
        vxp = v - alphax;
        vxn = v + alphax;
        vyp = v - alphay;
        vyn = v + alphay;
        vzp = v - alphaz;
        vzn = v + alphaz;
        vd = is2D ? 4.0 : 6.0;
        vd -= shift;
        if (is2D) {
            mat.resize(n, n, n * 5, RESERVE_NO_DATA);
        } else {
            mat.resize(n, n, n * 7, RESERVE_NO_DATA);
        }
        for (ii = 0; ii < n; ii++) {
            k = ii / (nx * ny);
            i = (ii - k * nx * ny) / nx;
            j = ii - k * nx * ny - i * nx;
            if (k > 0) {
                jj = ii - nx * ny;
                mat.pushBack(ii, jj, vzn);
            }
            if (k < nz - 1) {
                jj = ii + nx * ny;
                mat.pushBack(ii, jj, vzp);
            }

            if (i > 0) {
                jj = ii - nx;
                mat.pushBack(ii, jj, vyn);
            }

            if (i < ny - 1) {
                jj = ii + nx;
                mat.pushBack(ii, jj, vyp);
            }

            if (j > 0) {
                jj = ii - 1;
                mat.pushBack(ii, jj, vxn);
            }

            if (j < nx - 1) {
                jj = ii + 1;
                mat.pushBack(ii, jj, vxp);
            }
            mat.pushBack(ii, ii, vd);
        }
    }

} // HOST


namespace HOST {
    template void generatePoissonCSR(CSRMatrix<FLOAT32> &mat, const UINT32 &diag_block_size);

    template void generatePoissonCSR(CSRMatrix<FLOAT64> &mat, const UINT32 &diag_block_size);


    template
    void generateLaplacianCOO(COOMatrix<FLOAT32> &mat, const INT32 &nx, const INT32 &ny, const INT32 &nz,
                              const FLOAT32 &alphax, const FLOAT32 &alphay, const FLOAT32 &alphaz,
                              const FLOAT32 &shift);

    template
    void generateLaplacianCOO(COOMatrix<FLOAT64> &mat, const INT32 &nx, const INT32 &ny, const INT32 &nz,
                              const FLOAT64 &alphax, const FLOAT64 &alphay, const FLOAT64 &alphaz,
                              const FLOAT64 &shift);
}

