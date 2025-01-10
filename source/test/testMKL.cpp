/*
 * @author  邓轶丹
 * @date    2024/12/31
 * @details 测试MKL
 */
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <mkl.h>

#include "../../include/MatrixClass/CSRMatrix.h"
#include "../../include/MatrixClass/DenseMatrix.h"
#include "../../include/MatrixClass/ModelProblem.h"
#include "../../include/utils/TestTools/checkTools.hpp"
#include "../../include/utils/TestTools/generateTools.hpp"
#include "../../include/utils/TimerTools/CPUtimer.hpp"

#define HIGH_PRECISION_TYPE double
#define LOW_PRECISION_TYPE float


class MatrixMultiplier {
private:
    int m, n, k; // A(m,k), B(k,n), C(m,n)
    std::vector<double> A, B, C;

    // 随机初始化矩阵
    void initializeMatrices() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        A.resize(m * k);
        B.resize(k * n);
        C.resize(m * n, 0.0);

        for (int i = 0; i < m * k; ++i) A[i] = dis(gen);
        for (int i = 0; i < k * n; ++i) B[i] = dis(gen);
    }

public:
    MatrixMultiplier(int m, int k, int n) : m(m), k(k), n(n) {
        initializeMatrices();
    }

    // 使用MKL进行矩阵乘法
    void multiply() {
        // 设置线程数为系统最大可用线程
        mkl_set_num_threads(24);

        // 使用最优参数的cblas_dgemm调用
        cblas_dgemm(
            CblasRowMajor, // 行主序矩阵
            CblasNoTrans, // A不转置
            CblasNoTrans, // B不转置
            m, // 输出矩阵行数
            n, // 输出矩阵列数
            k, // A的列数/B的行数
            1.0, // 标量alpha
            A.data(), // A矩阵指针
            k, // A的列数(leading dimension)
            B.data(), // B矩阵指针
            n, // B的列数(leading dimension)
            0.0, // 标量beta
            C.data(), // C矩阵指针
            n // C的列数(leading dimension)
        );
    }

    // 性能测试
    void benchmarkMultiply(int iterations = 10) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            multiply();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        std::cout << "矩阵大小: " << m << "x" << k << " * " << k << "x" << n << std::endl;
        std::cout << "平均执行时间: " << diff.count() / iterations * 1000 << " ms" << std::endl;
        std::cout << "使用线程数: " << mkl_get_max_threads() << std::endl;
    }
};

void test_gemm() {
    /* 对于大型的稠密矩阵，提升较大 */
    INT32 rowNumA = 80000, colNumA = 32, colNumB = 160;
    HOST::DenseMatrix<HIGH_PRECISION_TYPE> hostMatA(HOST::DenseMatColumnFirst, rowNumA, colNumA, memoryAligned);
    HOST::DenseMatrix<HIGH_PRECISION_TYPE> hostMatB(HOST::DenseMatColumnFirst, colNumA, colNumB, memoryAligned);
    HOST::DenseMatrix<HIGH_PRECISION_TYPE> hostMatC(HOST::DenseMatColumnFirst, rowNumA, colNumB, memoryAligned);
    HOST::DenseMatrix<HIGH_PRECISION_TYPE> hostMatC_copy(HOST::DenseMatColumnFirst, rowNumA, colNumB, memoryAligned);
    HIGH_PRECISION_TYPE* hostMatPtrA = hostMatA.getMatValPtr();
    HIGH_PRECISION_TYPE* hostMatPtrB = hostMatB.getMatValPtr();
    HIGH_PRECISION_TYPE* hostMatPtrC = hostMatC.getMatValPtr();
    HIGH_PRECISION_TYPE* hostMatPtrC_copy = hostMatC_copy.getMatValPtr();
    HOST::generateArrayRandom1D(hostMatPtrA, rowNumA * colNumA);
    HOST::generateArrayRandom1D(hostMatPtrB, colNumA * colNumB);

    CPU_TIMER_FUNC()
    CPU_TIMER_BEGIN()
    hostMatA.MatMatMul(hostMatB, hostMatC);
    CPU_TIMER_END()
    std::cout << " --- omp original mat-mat mul executes: " << CPU_EXEC_TIME() << " ms." << std::endl;

    CPU_TIMER_BEGIN()
    // 设置线程数为系统最大可用线程
    mkl_set_num_threads(40);
    // 使用最优参数的cblas_dgemm调用
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rowNumA, colNumB, colNumA, 1.0, hostMatPtrA,
                rowNumA, hostMatPtrB, colNumA, 0.0, hostMatPtrC_copy, rowNumA);
    CPU_TIMER_END()
    std::cout << " --- mkl mat-mat mul executes: " << CPU_EXEC_TIME() << " ms." << std::endl;

    /* check */
    HOST::checkAnswer(hostMatPtrC_copy, hostMatPtrC, rowNumA * colNumB, "check gemm");
}

/** @brief 计算转置的稠密矩阵向量乘法
 * @attention 这个不能用并行，会导致性能下降，不如用串行 */
void test_tgemv() {
    /* 对于一般的稠密矩阵向量乘法，提升程度不如串行的CPU代码 */
    INT32 rowNum = 100000, colNum = 180;
    HOST::DenseMatrix<HIGH_PRECISION_TYPE> hostMat(HOST::DenseMatColumnFirst, rowNum, colNum, memoryAligned);
    HOST::DenseVector<HIGH_PRECISION_TYPE> hostVec(rowNum);
    HOST::DenseVector<HIGH_PRECISION_TYPE> hostRes(colNum);
    HOST::DenseVector<HIGH_PRECISION_TYPE> hostResCopy(colNum);

    HIGH_PRECISION_TYPE* hostVecPtr = hostVec.getRawValPtr();
    HIGH_PRECISION_TYPE* hostMatPtr = hostMat.getMatValPtr();
    HIGH_PRECISION_TYPE* hostResPtr = hostRes.getRawValPtr();
    HIGH_PRECISION_TYPE* hostResPtr_copy = hostResCopy.getRawValPtr();
    HOST::generateArrayRandom1D(hostVecPtr, rowNum);
    HOST::generateArrayRandom1D(hostMatPtr, rowNum * colNum);

    CPU_TIMER_FUNC()
    CPU_TIMER_BEGIN()
    hostRes.fillVector(0, colNum, 0);
    hostMat.transposeVec(hostVec, hostRes);
    CPU_TIMER_END()
    std::cout << " --- sequential transMatPVec executes: " << CPU_EXEC_TIME() << " ms." << std::endl;

    CPU_TIMER_BEGIN()
    mkl_set_num_threads(8);
    cblas_dgemv(CblasColMajor, CblasTrans, rowNum, colNum, 1.0, hostMatPtr, rowNum, hostVecPtr, 1, 0.0,
                hostResPtr_copy, 1);
    CPU_TIMER_END()
    std::cout << " --- mkl transMatPVec executes: " << CPU_EXEC_TIME() << " ms." << std::endl;

    /* check */
    HOST::checkAnswer(hostResPtr_copy, hostResPtr, colNum, "check trans gemv");
}

/** @brief 计算稠密矩阵向量乘法，列主序 */
void test_gemv_col_major() {
    INT32 rowNum = 100000, colNum = 180;
    HOST::DenseMatrix<HIGH_PRECISION_TYPE> hostMat(HOST::DenseMatColumnFirst, rowNum, colNum, memoryAligned);
    HOST::AlignedVector<HIGH_PRECISION_TYPE> hostVec(colNum);
    HOST::AlignedVector<HIGH_PRECISION_TYPE> hostRes(rowNum);
    HOST::AlignedVector<HIGH_PRECISION_TYPE> hostResCopy(rowNum);

    HIGH_PRECISION_TYPE* hostVecPtr = hostVec.getRawValPtr();
    HIGH_PRECISION_TYPE* hostMatPtr = hostMat.getMatValPtr();
    HIGH_PRECISION_TYPE* hostResPtr = hostRes.getRawValPtr();
    HIGH_PRECISION_TYPE* hostResPtr_copy = hostResCopy.getRawValPtr();

    HOST::generateArrayRandom1D(hostVecPtr, colNum);
    HOST::generateArrayRandom1D(hostMatPtr, rowNum * colNum);

    // CPU_TIMER_FUNC()
    // CPU_TIMER_BEGIN()
    FLOAT64 startTime = omp_get_wtime();
    hostRes.fillVector(0, rowNum, 0);
    hostMat.MatVec(hostVec, hostRes);
    FLOAT64 endTime = omp_get_wtime();
    FLOAT64 executeTime = (endTime - startTime) * 1000;
    // CPU_TIMER_END()
    std::cout << " --- sequential MatPVec executes: " << executeTime << " ms." << std::endl;
    hostRes.printVector("hostRes");

    // CPU_TIMER_BEGIN()
    startTime = omp_get_wtime();
    mkl_set_num_threads(24);
    cblas_dgemv(CblasColMajor, CblasNoTrans, rowNum, colNum, 1.0, hostMatPtr, rowNum, hostVecPtr, 1, 0.0,
                hostResPtr_copy, 1);
    endTime = omp_get_wtime();
    executeTime = (endTime - startTime) * 1000;
    // CPU_TIMER_END()
    std::cout << " --- mkl MatPVec executes: " << executeTime << " ms." << std::endl;

    /* check */
    HOST::checkAnswer(hostResPtr_copy, hostResPtr, rowNum, "check gemv");
}

/** @brief 计算稠密矩阵向量乘法，列主序 */
void test_gemv_row_major() {
    std::cout << "[INFO] test row major dense matPvec:" << std::endl;
    INT32 rowNum = 100000, colNum = 180;
    HOST::DenseMatrix<HIGH_PRECISION_TYPE> hostMat(HOST::DenseMatRowFirst, rowNum, colNum, memoryAligned);
    HOST::AlignedVector<HIGH_PRECISION_TYPE> hostVec(colNum);
    HOST::AlignedVector<HIGH_PRECISION_TYPE> hostRes(rowNum);
    HOST::AlignedVector<HIGH_PRECISION_TYPE> hostResCopy(rowNum);

    HIGH_PRECISION_TYPE* hostVecPtr = hostVec.getRawValPtr();
    HIGH_PRECISION_TYPE* hostMatPtr = hostMat.getMatValPtr();
    HIGH_PRECISION_TYPE* hostResPtr = hostRes.getRawValPtr();
    HIGH_PRECISION_TYPE* hostResPtr_copy = hostResCopy.getRawValPtr();

    HOST::generateArrayRandom1D(hostVecPtr, colNum);
    HOST::generateArrayRandom1D(hostMatPtr, rowNum * colNum);

    // CPU_TIMER_FUNC()
    // CPU_TIMER_BEGIN()
    FLOAT64 startTime = omp_get_wtime();
    // hostRes.fillVector(0, rowNum, 0);
    hostMat.MatVec(hostVec, hostRes);
    FLOAT64 endTime = omp_get_wtime();
    FLOAT64 executeTime = (endTime - startTime) * 1000;
    // CPU_TIMER_END()
    std::cout << " --- sequential MatPVec executes: " << executeTime << " ms." << std::endl;
    hostRes.printVector("hostRes");

    // CPU_TIMER_BEGIN()
    startTime = omp_get_wtime();
    mkl_set_num_threads(24);
    // mkl_enable_instructions(MKL_ENABLE_AVX2);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, rowNum, colNum, 1.0, hostMatPtr, colNum, hostVecPtr, 1, 0.0,
                hostResPtr_copy, 1);
    endTime = omp_get_wtime();
    executeTime = (endTime - startTime) * 1000;
    // CPU_TIMER_END()
    std::cout << " --- mkl MatPVec executes: " << executeTime << " ms." << std::endl;

    /* check */
    HOST::checkAnswer(hostResPtr_copy, hostResPtr, rowNum, "check gemv");
}

void test_spmv() {
    std::cout << "[INFO] test spmv:" << std::endl;
    HOST::CSRMatrix<HIGH_PRECISION_TYPE> testCSR;
    INT32 gridSize = 64;
    HOST::generateLaplacianCSR(testCSR, gridSize, gridSize, gridSize, 0.0, 0.0, 0.0, 0.0);
    testCSR.printMatrix("testCSR");

    HOST::AlignedVector<HIGH_PRECISION_TYPE> testVec(testCSR.getColNum());
    HOST::AlignedVector<HIGH_PRECISION_TYPE> testRes(testCSR.getColNum());
    HOST::AlignedVector<HIGH_PRECISION_TYPE> testResCopy(testCSR.getColNum());

    UINT32* matRowOffsetPtr = testCSR.getRowOffsetPtr(0);
    UINT32* matColIndicesPtr = testCSR.getColIndicesPtr(0);
    HIGH_PRECISION_TYPE* matCSRValuesPtr = testCSR.getCSRValuesPtr(0);
    HIGH_PRECISION_TYPE* testVecPtr = testVec.getRawValPtr();
    HIGH_PRECISION_TYPE* testResPtr = testRes.getRawValPtr();
    HIGH_PRECISION_TYPE* testResPtr_copy = testResCopy.getRawValPtr();

    HOST::generateArrayRandom1D(testVecPtr, testCSR.getColNum());

    testVec.printVector("testVec");

    CPU_TIMER_FUNC()
    CPU_TIMER_BEGIN()
    testCSR.MatPVec(testVec, testRes);
    CPU_TIMER_END()
    std::cout << " --- omp CSR MatPVec executes: " << CPU_EXEC_TIME() << " ms." << std::endl;


    // Create sparse_matrix_t object
    mkl_set_num_threads(24);
    mkl_set_interface_layer(MKL_INTERFACE_GNU);
    mkl_set_threading_layer(MKL_THREADING_GNU);
    INT32 n = testCSR.getRowNum();
    sparse_matrix_t A;
    sparse_status_t status;

    // Create CSR format matrix
    status = mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, n, n, (INT32*)matRowOffsetPtr,
                                     (INT32*)matRowOffsetPtr + 1, (INT32*)matColIndicesPtr, matCSRValuesPtr);
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Error creating sparse matrix: %d\n", status);
        exit(-1);
    }

    // Perform matrix-vector multiplication
    double alpha = 1.0, beta = 0.0;

    // Create matrix descriptor and explicitly assign values
    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL; // Matrix type is general (non-symmetric)

    // Perform matrix-vector multiplication
    CPU_TIMER_BEGIN()
    status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, descr, testVecPtr, beta, testResPtr_copy);
    CPU_TIMER_END()
    std::cout << " --- mkl CSR MatPVec executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Error in matrix-vector multiplication: %d\n", status);
        exit(-1);
    }


    /* check */
    testRes.printVector("testRes");
    testResCopy.printVector("testResCopy");
    HOST::checkAnswer(testResPtr_copy, testResPtr, n, "check spmv");
}

int main() {
    test_spmv();
    return 0;
}
