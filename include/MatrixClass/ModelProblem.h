/*
 * @author  邓轶丹
 * @date    2024/5/8
 * @details 模型问题函数
 */
#ifndef PMSLS_NEW_MODELPROBLEM_H
#define PMSLS_NEW_MODELPROBLEM_H

#include "CSRMatrix.h"
#include "COOMatrix.h"
#include "MatrixTools.h"
#include "../VectorClass/VectorTools.h"

namespace HOST {
    /** @brief 生成测试矩阵（非奇异系数矩阵）
     * @details 该矩阵均分为diag_block_size大小的块，实际矩阵规模为块大小的平方倍，由于矩阵非奇异，则AX=b的b无论如何构造一定有唯一解。
     * @param [in] diag_block_size: 指定的对角块的大小。
     */
    template<typename ValType>
    void generatePoissonCSR(CSRMatrix<ValType> &mat, const UINT32 &diag_block_size);

    /**
   * @brief 生成laplacian矩阵(2D问题5点差分、3D问题7点差分)
   * @param nx x维度的size
   * @param ny y维度的size
   * @param nz z维度的size
   * @param alphax x维度上的alpha值，用于构造非对称矩阵，对称设为0.0
   * @param alphay y维度上的alpha值，用于构造非对称矩阵，对称设为0.0
   * @param alphaz z维度上的alpha值，用于构造非对称矩阵，对称设为0.0
   * @param shift 对角线偏移系数 */
    template<typename ValType>
    void generateLaplacianCOO(COOMatrix<ValType> &mat, const INT32 &nx, const INT32 &ny, const INT32 &nz,
                              const ValType &alphax, const ValType &alphay, const ValType &alphaz,
                              const ValType &shift);

    /**
   * @brief 生成laplacian矩阵(2D问题5点差分、3D问题7点差分)
   * @param nx x维度的size
   * @param ny y维度的size
   * @param nz z维度的size
   * @param alphax x维度上的alpha值，用于构造非对称矩阵，对称设为0.0
   * @param alphay y维度上的alpha值，用于构造非对称矩阵，对称设为0.0
   * @param alphaz z维度上的alpha值，用于构造非对称矩阵，对称设为0.0
   * @param shift 对角线偏移系数（是否不定，非0表示非正定） */
    template<typename ValType>
    inline void
    generateLaplacianCSR(CSRMatrix<ValType> &mat, const INT32 &nx, const INT32 &ny, const INT32 &nz,
                         const ValType &alphax, const ValType &alphay, const ValType &alphaz, const ValType &shift) {
        COOMatrix<ValType> cooMat;
        generateLaplacianCOO(cooMat, nx, ny, nz, alphax, alphay, alphaz, shift);
        transCOO2CSR(cooMat, mat);
    }

} // HOST



#endif //PMSLS_NEW_MODELPROBLEM_H
