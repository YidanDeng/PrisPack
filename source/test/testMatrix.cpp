/*
 * @author  邓轶丹
 * @date    2024/5/8
 * @details 测试矩阵
 */
#include <random>
#include "../../include/MatrixClass/CSRMatrix.h"
#include "../../include/MatrixClass/DenseMatrix.h"
#include "../../include/MatrixClass/ModelProblem.h"
#include "../../include/utils/TestTools/checkTools.hpp"
#include "../../include/utils/TestTools/generateTools.hpp"
#include "../../include/utils/TimerTools/CPUtimer.hpp"


#ifdef CUDA_ENABLED

#include "../../include/MatrixClass/DeviceCSRMatrix.cuh"

#endif //CUDA_ENABLED


#define DATA_TYPE double


void testModelProblem() {
    /* 测试一下泊松方程 */
    HOST::CSRMatrix<DATA_TYPE> test(memoryAligned);
    HOST::generatePoissonCSR(test, 2); // 生成一个具有3x3子块的方程组系数矩阵
    test.printMatrix("test mat"); // 打印CSR格式的矩阵

    /* 测试一下拉普拉斯 */
    HOST::generateLaplacianCSR(test, 2, 2, 1, 0.0, 0.0, 0.0, 0.0);
    std::cout << "[INFO] test mat row: " << test.getRowNum() << ", test mat col: " << test.getColNum() << std::endl;
    test.printMatrix("test mat"); // 打印CSR格式的矩阵
}

void testSubMat() {
    UINT32 dim = 4;
    HOST::CSRMatrix<DATA_TYPE> test(memoryAligned);
    HOST::CSRMatrix<DATA_TYPE> subTest(memoryAligned);
    HOST::generatePoissonCSR(test, 2); // 生成一个具有2x2子块(矩阵维数为4)的方程组系数矩阵
    test.printMatrix("test mat"); // 打印CSR格式的矩阵
    /* 一个直接提取子矩阵的用例 */
    test.getSubMatrix(0, 1, 2, 3, subTest);
    subTest.printMatrix("sub mat(range: [0:1, 2:3])"); // 即取原矩阵中右上角的维数为2的子块

    /* 一个全重排用例 */
    HOST::DenseVector<UINT32> rowPerm(dim);
    HOST::DenseVector<UINT32> colPerm(dim);
    rowPerm[0] = 3, rowPerm[1] = 1, rowPerm[2] = 2, rowPerm[3] = 0;
    colPerm[0] = 2, colPerm[1] = 1, colPerm[2] = 0, colPerm[3] = 3;
    test.getSubMatrix(rowPerm, colPerm, subTest);
    subTest.formatStructure();
    subTest.printMatrix("test mat (full perm)");
    /* 部分重排 */
    HOST::DenseVector<UINT32> rowPerm_sub(2);
    HOST::DenseVector<UINT32> colPerm_sub(2);
    rowPerm_sub[0] = 3, rowPerm_sub[1] = 1;
    colPerm_sub[0] = 2, colPerm_sub[1] = 1;
    test.getSubMatrix(rowPerm_sub, colPerm_sub, subTest);
    subTest.formatStructure();
    subTest.printMatrix("test mat (partial perm)");

    /* 测试按行提取子块的同时再将子块划分成更小的子块 */
    HOST::DenseVector<UINT32> colOffset(3); // 子块的列偏移
    // 针对列数为4的矩阵，如把子块按列划分为2个分区，可有[0,2,4]，列数位于[0,1]属于第一个子分区，列数[2,3]属于第二个子分区
    colOffset[1] = 2;
    colOffset[2] = 4;
    // 取原矩阵前两行进行分块
    test.getSubMatrix(0, 1, colOffset, subTest);
    subTest.printMatrix("multi-separate sub-matrix");
}

/** @brief 用于测试管理移动CSR内部私有成员向量移动（rowOffset/colIndices/values）的类*/
void testAutoMoveVector() {
    HOST::CSRMatrix<DATA_TYPE> test(memoryAligned);
    HOST::generatePoissonCSR(test, 3);
    test.printMatrix("test mat");
    // 声明针对CSR矩阵中的rowOffset向量的移动管理工具，第2、第3参数分别对应CSR中写好的移出/移入函数指针
    HOST::MovePrivateVector<CSR_MATRIX(DATA_TYPE), UINT32> testMove(test, &CSR_MATRIX(DATA_TYPE)::moveRowOffsetTo,
                                                                    &CSR_MATRIX(DATA_TYPE)::moveRowOffsetFrom);
    std::cout << "[INFO] Move row-offset to testMove:" << std::endl;
    // 操作移动出的rowoffset对象
    testMove->printVector("row-offset vector");
    // 打印移动rowOffset之后的CSR矩阵
    test.printMatrix("test mat");
    // 将移动出的rowOffset向量归还至原CSR对象
    testMove.moveBack();
    std::cout << "[INFO] Move row-offset back to test matrix:" << std::endl;
    // 打印归还后的CSR矩阵
    test.printMatrix("test mat");
}


#ifdef CUDA_ENABLED

void testDeviceCSR() {
    // 生成CPU上的测试数据
    HOST::CSRMatrix<DATA_TYPE> hostCSR(memoryPageLocked);
    UINT32 blockSize = 2;
    HOST::generatePoissonCSR(hostCSR, blockSize);
    hostCSR.printMatrix("test mat");
    // 创建GPU上的矩阵对象
    DEVICE::DeviceCSRMatrix<DATA_TYPE> deviceCsrMatrix(hostCSR.getRowNum(), hostCSR.getColNum(),
                                                       hostCSR.getNNZnum(0, hostCSR.getRowNum() - 1), DEFAULT_GPU);
    // 拷贝CPU上的值到GPU
    deviceCsrMatrix.copyMatFromHost(hostCSR);
    // 打印GPU上的矩阵，查看值是否都已正确拷贝
    deviceCsrMatrix.printMatrix("dev CSR");
}

#endif //CUDA_ENABLED

void testCSR() {
    /* 测试一下泊松方程 */
    HOST::CSRMatrix<DATA_TYPE> test(memoryAligned);
    HOST::generatePoissonCSR(test, 3); // 生成一个具有3x3子块的方程组系数矩阵
    test.printMatrix("test mat"); // 打印CSR格式的矩阵
    /* 放缩矩阵（保留原值）*/
    test.resize(4, 4, test.getNNZnum(0, 3), RESERVE_DATA);
    // 若resize操作保留原值，参数“非零元个数”只是一个预估值，实际存储的有效元素小于或等于该值。实际存储有效非零元个数以rowOffset为准
    test.printMatrix("resized mat");
    // 因此，针对这个测试用例，若colIndices和values多打了一些无效值是正常现象
}

void testCSR_setValues() {
    HOST::CSRMatrix<DATA_TYPE> test(3, 3, 6, memoryBase);
    UINT32 colIdxPtr[2] = {1, 2};
    DATA_TYPE values[2] = {3.0, 4.0};
    for (UINT32 i = 0; i < 3; ++i) {
        test.setColsValsByRow(i, colIdxPtr, values, 2);
    }
    test.printMatrix("test mat");
}


void testBlockCSR() {
    HOST::CSRMatrix<DATA_TYPE> test(memoryAligned);
    HOST::generatePoissonCSR(test, 2); // 生成一个具有3x3子块的方程组系数矩阵
    test.printMatrix("test mat"); // 打印CSR格式的矩阵
    HOST::AutoAllocateVector<UINT32> colOffset(3, memoryBase);
    // 划分两块，当前colOffset = [0, 4, 9]，即列号位于[0,3]的非零元在第一个子块，列号在[4,8]的非零元位于第二个子块
    colOffset[0] = 2, colOffset[1] = 3, colOffset[2] = 4;
    // 用来承接分好的块CSR
    UniquePtr1D<HOST::CSRMatrix<DATA_TYPE>> blockCSRPtr;
    // 该方法允许切全零块，假设左闭右开区间[colOffset[a], colOffset[b])中无任何非零元，对应的子块为空CSR对象
    // 现提取原CSR矩阵的前6行数据，然后将其根据colOffset划分成子矩阵
    test.getSubMatrix(0, 3, *colOffset, blockCSRPtr, test.getMemoryType());
    for (INT32 i = 0; i < blockCSRPtr.getDim(); ++i) {
        std::string temp = "block " + std::to_string(i);
        blockCSRPtr[i]->printMatrix(temp.c_str());
    }
}


void testDenseMat() {
    UINT32 row = 4, col = 4;
    HOST::DenseMatrix<DATA_TYPE> testDense(HOST::DenseMatColumnFirst, row, col, memoryBase);
    HOST::AutoAllocateVector<DATA_TYPE> testCol(row, memoryBase);
    for (UINT32 i = 0; i < col; ++i) {
        testCol->fillVector(0, row, i + 1);
        testDense.setValsByCol(i, *testCol);
    }
    testDense.printMatrix("test insert val by col");

    testDense.transpose();
    testDense.printMatrix("test transpose");
}


#pragma GCC push_options
#pragma GCC optimize ("O0")

void CSRMatPVecNoOpt(const HOST::CSRMatrix<DATA_TYPE>& mat, const HOST::AutoAllocateVector<DATA_TYPE>& in,
                     HOST::AutoAllocateVector<DATA_TYPE>& out) {
    UINT32 k1, k2, col;
    const UINT32* rowOffsetPtr = mat.getRowOffsetPtr(0);
    const UINT32* colIndicesPtr = mat.getColIndicesPtr(0);
    const DATA_TYPE* valuesPtr = mat.getCSRValuesPtr(0);
    DATA_TYPE val;
    for (UINT32 i = 0; i < mat.getRowNum(); ++i) {
        k1 = rowOffsetPtr[i];
        k2 = rowOffsetPtr[i + 1];
        out[i] = 0;
        for (UINT32 j = k1; j < k2; ++j) {
            col = colIndicesPtr[j];
            val = valuesPtr[j];
            out[i] += in[col] * val;
        }
    }
}

#pragma GCC pop_options


void CSRMatPVec(const HOST::CSRMatrix<DATA_TYPE>& mat, const HOST::AutoAllocateVector<DATA_TYPE>& in,
                HOST::AutoAllocateVector<DATA_TYPE>& out) {
    UINT32 k1, k2, col;
    const UINT32* rowOffsetPtr = mat.getRowOffsetPtr(0);
    const UINT32* colIndicesPtr = mat.getColIndicesPtr(0);
    const DATA_TYPE* valuesPtr = mat.getCSRValuesPtr(0);
    DATA_TYPE val;
    for (UINT32 i = 0; i < mat.getRowNum(); ++i) {
        k1 = rowOffsetPtr[i];
        k2 = rowOffsetPtr[i + 1];
        out[i] = 0;
        for (UINT32 j = k1; j < k2; ++j) {
            col = colIndicesPtr[j];
            val = valuesPtr[j];
            out[i] += in[col] * val;
        }
    }
}

void testEfficiency() {
    HOST::CSRMatrix<DATA_TYPE> testMat;
    UINT32 diagDim = 1000;
    HOST::generatePoissonCSR(testMat, diagDim);
    testMat.printMatrix("test mat");
    HOST::AutoAllocateVector<DATA_TYPE> in(testMat.getRowNum(), memoryBase);
    HOST::AutoAllocateVector<DATA_TYPE> out(testMat.getRowNum(), memoryBase);

    CPU_TIMER_FUNC()
    // 预加载
    for (int i = 0; i < testMat.getRowNum(); ++i) {
        in[i] = 1;
    }
    CPU_TIMER_BEGIN()
    CSRMatPVecNoOpt(testMat, in, out);
    CPU_TIMER_END()
    std::cout << " --- no opt: " << CPU_EXEC_TIME() << " ms." << std::endl;
    CPU_TIMER_BEGIN()
    CSRMatPVec(testMat, in, out);
    CPU_TIMER_END()
    std::cout << " --- auto opt: " << CPU_EXEC_TIME() << " ms." << std::endl;
    CPU_TIMER_BEGIN()
    testMat.MatPVec(*in, *out);
    CPU_TIMER_END()
    std::cout << " --- mul thread: " << CPU_EXEC_TIME() << " ms." << std::endl;
}


void testPivot() {
    HOST::CSRMatrix<DATA_TYPE> oriMat;
    UINT32 diagDim = 2048;
    HOST::generatePoissonCSR(oriMat, diagDim);
    oriMat.printMatrix("test mat");
    HOST::AutoAllocateVector<UINT32> pivotArr;
    HOST::AutoAllocateVector<UINT32> colPerm(oriMat.getColNum(), memoryBase);
    HOST::AutoAllocateVector<UINT32> rowPerm(oriMat.getColNum(), memoryBase);
    for (UINT32 i = 0; i < oriMat.getColNum(); ++i) {
        rowPerm[i] = i;
        colPerm[i] = i;
    }
    std::shuffle(&colPerm[0], &colPerm[0] + oriMat.getColNum(), std::mt19937(std::random_device()()));
    colPerm.printVector("col perm");
    oriMat.printMatrix("origin mat");

    HOST::CSRMatrix<DATA_TYPE> testMat;
    oriMat.getSubMatrix(*rowPerm, *colPerm, testMat);
    testMat.printMatrix("shuffled mat");
    testMat.pivotReorderByRow(*pivotArr);
    testMat.printMatrix("reorder pivot mat");
    pivotArr.printVector("pivot arr");
    HOST::checkAnswer(&pivotArr[0], &colPerm[0], oriMat.getColNum(), "Check pivot reorder");
}

void testMatrixNorm() {
    HOST::CSRMatrix<DATA_TYPE> oriMat;
    HOST::generateLaplacianCSR(oriMat, 64, 64, 64, 0.1, 0.1, 0.1, 0.05);
    HOST::AutoAllocateVector<DATA_TYPE> v(oriMat.getRowNum(), memoryAligned);
    HOST::AutoAllocateVector<DATA_TYPE> spmvResV(oriMat.getRowNum(), memoryAligned);
    HOST::generateArrayRandom1D(v.getRawValPtr(), oriMat.getRowNum());
    oriMat.MatPVec(*v, *spmvResV);
    FLOAT64 res = sqrt(v->innerProduct(*spmvResV));
    FLOAT64 resNoExtraMem = oriMat.getMatrixNorm(*v, 0);
    FLOAT64 residual = fabs(res - resNoExtraMem);
    std::cout << "residual: " << residual << std::endl;
}

int main() {
    testModelProblem();

    return 0;
}
