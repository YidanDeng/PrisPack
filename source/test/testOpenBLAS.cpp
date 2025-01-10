/*
 * @author  邓轶丹
 * @date    2024/6/28
 * @details 测试OpenBLAS（LAPACK）
 */

#include "../../../config/config.h"
#include "lapacke.h"
#include "../../../include/MatrixClass/DenseMatrix.h"

int main() {
    char JOBZ = 'V';
    char RANGE = 'I';
    int n = 4;
    double D[4] = {2, 3, 2, 3};     // 对角线元素
    double E[3] = {1, 1, 1};        // 次对角线元素
    double vl{0.0}, vu{0.0};
    int il = 2;
    int iu = 4;
    double tmp = LAPACKE_dlamch('S');
    double abstol = 2 * tmp;
    int m = 3;
    double w[n];
    HOST::DenseMatrix<FLOAT64> z(HOST::DenseMatColumnFirst, 4, m, memoryBase);
    int ldz = 4;
    int ifail[n];
    int info;
    // 求对称三对角矩阵特征值和特征向量
    info = LAPACKE_dstevx(LAPACK_COL_MAJOR, JOBZ, RANGE, n, D, E, vl, vu, il, iu, abstol, &m, w, z.getMatValPtr(), ldz,
                          ifail);
    std::cout << info << std::endl;
    std::cout << m << std::endl;
    //输出选定特征值
    std::cout << "[INFO] eigen values: ";
    for (UINT32 i = 0; i < m; i++) {
        std::cout << " " << w[i];
    }
    std::cout << std::endl;
    //输出对应特征向量
    z.printMatrix("eigen vector");
    HOST::AutoAllocateVector<FLOAT64> a(4, memoryBase);
    for (UINT32 i = 0; i < 4; i++) {
        a[i] = z.getMatValPtr()[4 + i];
    }
    a.printVector("");
    std::cout << a->norm_2() << std::endl;
    //补充：通过和matlab中eig的结果对比，是正确的
    return 0;
}

