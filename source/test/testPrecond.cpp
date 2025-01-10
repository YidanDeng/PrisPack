/*
 * @author  邓轶丹、袁心磊
 * @date    2024/6/10
 * @details 测试各种预条件
 */
#include "../../include/Preconditioner/IncompleteCholesky.h"
#include "../../include/Preconditioner/IncompleteLU.h"
#include "../../include/Preconditioner/IncompleteLDLT.h"
#include "../../include/Preconditioner/AMSEDPrecondition.h"
#include "../../include/utils/TestTools/generateTools.hpp"
#include "../../include/utils/ExternalTools/MatrixReorderTools.h"
#include "../../include/MatrixClass/ModelProblem.h"
#include "../../include/utils/TestTools/WriteMtxTools.h"

#define HIGH_PRECISION_TYPE double
#define LOW_PRECISION_TYPE float

void testIC() {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> testMat; // 这里只是声明了一个空智能指针，需要实例化
    testMat.construct(); // 指针实例化，调用对应类的构造函数初始化对象，这里主要调的是无参构造
    INT32 dim = 64;
    const HIGH_PRECISION_TYPE h = 1.0 / dim; // 网格间距
    const HIGH_PRECISION_TYPE c = 0.0; // c的值
    HIGH_PRECISION_TYPE s = h * h * c; // 计算移位参数s
    HOST::generateLaplacianCSR(*testMat, dim, dim, dim, 0.0, 0.0, 0.0, s);
    testMat->printMatrix("original mat A");

    /* 构建预条件 */
    HOST::IncompleteCholesky<HIGH_PRECISION_TYPE, HIGH_PRECISION_TYPE> testIC(testMat.get(), 1e-3);
    testIC.setup(); // 声明了预条件子后必须调用这个进行预条件的计算，构造函数只负责开内存空间
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> transL = testIC.getTransL();
    transL->printMatrix("trans L");

    /* 求解y = M^{-1} x */
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testB(testMat->getRowNum(), memoryBase);
    HIGH_PRECISION_TYPE vecFillVal = 1;
    testB->fillVector(0, testB.getLength(), vecFillVal);
    testB->printVector("original B");
    // 求稀疏下三角方程组
    testIC.MSolveLower(*testB);
    testB->printVector("solve lower spsv");
    // 求稀疏上三角方程组（这个结果相当于y = M^{-1} x）
    testIC.MSolveUpper(*testB);
    testB->printVector("solve upper spsv(final MInvResult)");
    // 验证答案
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> tempB(testMat->getRowNum(), memoryBase);
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> residual(testMat->getRowNum(), memoryBase);
    residual->fillVector(0, residual.getLength(), vecFillVal);
    // 直接用原矩阵乘结果向量，观察残差
    testMat->MatPVec(*testB, *tempB);
    residual.add(-1, *tempB);
    FLOAT64 res = sqrt(residual->sumKahan(0, residual.getLength(), [](const FLOAT64 x) { return x * x; }));
    residual->printVector("residual vector(norm 1)");
    std::cout << "final residual: " << res << std::endl;
}

void testIC_withPerm(char* argv) {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> testMat, amdPermMat, rcmPermMat; // 这里只是声明了一个空智能指针，需要实例化
    testMat.construct(); // 指针实例化，调用对应类的构造函数初始化对象，这里主要调的是无参构造
    amdPermMat.construct();
    rcmPermMat.construct();
    INT32 diagDim = 8;
    HOST::generatePoissonCSR(*testMat, diagDim);
    testMat->printMatrix("original mat A");

    HOST::AutoAllocateVector<UINT32> perm(testMat->getRowNum(), memoryBase);

    /* 初始化写入工具 */
    HOST::WriteMtxTools<FLOAT64> oriMtx(argv, "../source/test/testResults/originMat.mtx", 1, MTX_STORAGE_GENERAL);
    HOST::WriteMtxTools<FLOAT64> rcmOrderMtx(argv, "../source/test/testResults/rcmReorder.mtx", 1, MTX_STORAGE_GENERAL);
    HOST::WriteMtxTools<FLOAT64> amdOrderMtx(argv, "../source/test/testResults/amdReorder.mtx", 1, MTX_STORAGE_GENERAL);

    HOST::WriteMtxTools<LOW_PRECISION_TYPE> oriMtx_IC(argv, "../source/test/testResults/originMat_IC.mtx", 1,
                                                      MTX_STORAGE_SYMMETRIC);
    HOST::WriteMtxTools<LOW_PRECISION_TYPE> rcmOrderMtx_IC(argv, "../source/test/testResults/rcmReorder_IC.mtx", 1,
                                                           MTX_STORAGE_SYMMETRIC);
    HOST::WriteMtxTools<LOW_PRECISION_TYPE> amdOrderMtx_IC(argv, "../source/test/testResults/amdReorder_IC.mtx", 1,
                                                           MTX_STORAGE_SYMMETRIC);

    /* 写入原始矩阵（用于对比）*/
    oriMtx.writeMatrix(*testMat);

    /* AMD 排序 */
    HOST::amdReorderCSR(*testMat, *perm);
    testMat->getSubMatrix(*perm, *perm, *amdPermMat);
    amdOrderMtx.writeMatrix(*amdPermMat);

    /* RCM 排序 */
    HOST::rcmReorderCSR(*testMat, *perm);
    testMat->getSubMatrix(*perm, *perm, *rcmPermMat);
    rcmOrderMtx.writeMatrix(*rcmPermMat);


    /* 构建预条件 */
    HOST::IncompleteCholesky<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE> testOriginalIC(testMat.get(), 1e-2);
    HOST::IncompleteCholesky<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE> testAmdIC(amdPermMat.get(), 1e-2);
    HOST::IncompleteCholesky<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE> testRcmIC(rcmPermMat.get(), 1e-2);

    testOriginalIC.setup(); // 声明了预条件子后必须调用这个进行预条件的计算，构造函数只负责开内存空间
    testAmdIC.setup();
    testRcmIC.setup();
    SharedObject<HOST::CSRMatrix<LOW_PRECISION_TYPE>> transOriginL = testOriginalIC.getTransL();
    SharedObject<HOST::CSRMatrix<LOW_PRECISION_TYPE>> transAmdL = testAmdIC.getTransL();
    SharedObject<HOST::CSRMatrix<LOW_PRECISION_TYPE>> transRcmL = testRcmIC.getTransL();

    oriMtx_IC.writeMatrix(*transOriginL);
    amdOrderMtx_IC.writeMatrix(*transAmdL);
    rcmOrderMtx_IC.writeMatrix(*transRcmL);
}


void testILUT() {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> testMat; // 这里只是声明了一个空智能指针，需要实例化
    testMat.construct(); // 指针实例化，调用对应类的构造函数初始化对象，这里主要调的是无参构造
    INT32 diagDim = 1024;
    HOST::generatePoissonCSR(*testMat, diagDim);
    testMat->printMatrix("original mat A");

    /* 构建预条件 */
    HOST::IncompleteLU<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE> testILU(testMat.get(), 20, 1e-2);
    testILU.setup(); // 声明了预条件子后必须调用这个进行预条件的计算，构造函数只负责开内存空间
    SharedObject<HOST::CSRMatrix<LOW_PRECISION_TYPE>> L = testILU.getL();
    L->printMatrix("L");
    SharedObject<HOST::AutoAllocateVector<LOW_PRECISION_TYPE>> diag = testILU.getDiagVals();
    diag->printVector("diag");
    SharedObject<HOST::CSRMatrix<LOW_PRECISION_TYPE>> U = testILU.getU();
    U->printMatrix("U");

    /* 求解 */
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testB(testMat->getRowNum(), memoryBase);
    HIGH_PRECISION_TYPE vecFillVal = 1;
    testB->fillVector(0, testB.getLength(), vecFillVal);
    testB->printVector("original B");
    // 求稀疏下三角方程组
    testILU.MSolveLower(*testB);
    testB->printVector("solve lower spsv");
    // 求稀疏上三角方程组（这个结果相当于y = M^{-1} x）
    testILU.MSolveUpper(*testB);
    testB->printVector("solve upper spsv(final MInvResult)");
    // 验证答案
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> tempB(testMat->getRowNum(), memoryBase);
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> residual(testMat->getRowNum(), memoryBase);
    residual->fillVector(0, residual.getLength(), vecFillVal);
    // 直接用原矩阵乘结果向量，观察残差
    testMat->MatPVec(*testB, *tempB);
    residual.add(-1, *tempB);
    FLOAT64 res = residual->norm_2();
    residual->printVector("residual vector");
    std::cout << "final residual: " << res << std::endl;
}

void testILDLT() {
    /* 生成测试矩阵A */
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> testMat; // 这里只是声明了一个空智能指针，需要实例化
    testMat.construct(); // 指针实例化，调用对应类的构造函数初始化对象，这里主要调的是无参构造
    INT32 diagDim = 1024;
    HOST::generatePoissonCSR(*testMat, diagDim);
    testMat->printMatrix("original mat A");

    /* 生成随机标准答案x，并构造B = Ax */
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX(testMat->getRowNum(), memoryBase),
                                                  testB(testMat->getRowNum(), memoryBase);
    HOST::generateArrayRandom1D(testX->getRawValPtr(), testX.getLength());
    testX.printVector("original X");
    testMat->MatPVec(*testX, *testB);
    testB->printVector("original B");

    /* 初始化ILDL分解 */
    HOST::IncompleteLDLT<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE> preILDLT(testMat.get(), 1e-3);
    preILDLT.setup();

    /* 求解 x = M^{-1} B */
    preILDLT.MInvSolve(*testB);
    testB->printVector("x = M^{-1} B");

    /* 检查答案精度 */
    testX.add(-1, *testB);
    HIGH_PRECISION_TYPE residual = sqrt(testX->sumKahanOMP(0, testX.getLength(), [](HIGH_PRECISION_TYPE x) {
        return x * x;
    }));
    // HIGH_PRECISION_TYPE residual = testX->norm_2();
    std::cout << "residual: " << residual << std::endl;
}

void testAMSED() {
    /* 生成测试矩阵A */
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> testMat; // 这里只是声明了一个空智能指针，需要实例化
    testMat.construct(); // 指针实例化，调用对应类的构造函数初始化对象，这里主要调的是无参构造
    INT32 dim = 64;
    const HIGH_PRECISION_TYPE h = 1.0 / dim; // 网格间距
    const HIGH_PRECISION_TYPE c = 0.4; // c的值
    HIGH_PRECISION_TYPE s = h * h * c; // 计算移位参数s
    HOST::generateLaplacianCSR(*testMat, dim, dim, dim, 0.0, 0.0, 0.0, 0.0);
    // HOST::generatePoissonCSR(*testMat, 90);
    testMat->printMatrix("original mat A");

    /* 生成随机标准答案x，并构造B = Ax */
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX(testMat->getRowNum(), memoryBase),
                                                  testB(testMat->getRowNum(), memoryBase), testB_copy(
                                                      testMat->getRowNum(), memoryBase);
    HOST::generateArrayRandom1D(testX->getRawValPtr(), testX.getLength());
    testX.printVector("original X");
    testMat->MatPVec(*testX, *testB);
    testB->printVector("original B");
    testB_copy.copy(*testB);
    /* 初始化计时器 */
    CPU_TIMER_FUNC()

    /* 初始化AMSED预条件 */
    HOST::AMSEDPrecondition<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE> preAMSED(
        testMat.get(), 4, MatrixReorderRCM, AMSEDEigenvalueDeflation);
    CPU_TIMER_BEGIN()
    preAMSED.setup();
    CPU_TIMER_END()
    std::cout << "AMSED setup(reorder and build MSLR structure) executes: " << CPU_EXEC_TIME() << " ms." << std::endl;

    /* 求解 x = M^{-1} B */
    CPU_TIMER_BEGIN()
    preAMSED.MInvSolve(*testB);
    CPU_TIMER_END()
    std::cout << "AMSED solve x = M^{-1} B executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    testB->printVector("x = M^{-1} B");

    /* 检查答案精度 */
    HOST::check_correctness(static_cast<INT32>(testMat->getRowNum()), (int*)testMat->getRowOffsetPtr(0),
                            (int*)testMat->getColIndicesPtr(0),
                            testMat->getCSRValuesPtr(0), testB->getRawValPtr(), testB_copy->getRawValPtr());

    testX.add(-1, *testB);
    HIGH_PRECISION_TYPE residual = sqrt(testX->sumKahanOMP(0, testX.getLength(), [](HIGH_PRECISION_TYPE x) {
        return x * x;
    }));
    std::cout << "residual: " << residual << std::endl;
}

void testMSLR() {
    /* 生成测试矩阵A */
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> testMat; // 这里只是声明了一个空智能指针，需要实例化
    testMat.construct(); // 指针实例化，调用对应类的构造函数初始化对象，这里主要调的是无参构造
    INT32 dim = 64;
    const HIGH_PRECISION_TYPE h = 1.0 / dim; // 网格间距
    const HIGH_PRECISION_TYPE c = 0.4; // c的值
    HIGH_PRECISION_TYPE s = h * h * c; // 计算移位参数s
    HOST::generateLaplacianCSR(*testMat, dim, dim, dim, 0.0, 0.0, 0.0, 0.0);
    // HOST::generatePoissonCSR(*testMat, 90);
    testMat->printMatrix("original mat A");

    /* 生成随机标准答案x，并构造B = Ax */
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX(testMat->getRowNum(), memoryBase),
                                                  testB(testMat->getRowNum(), memoryBase), testB_copy(
                                                      testMat->getRowNum(), memoryBase);
    HOST::generateArrayRandom1D(testX->getRawValPtr(), testX.getLength());
    testX.printVector("original X");
    testMat->MatPVec(*testX, *testB);
    testB->printVector("original B");
    testB_copy.copy(*testB);
    /* 初始化计时器 */
    CPU_TIMER_FUNC()

    /* 初始化AMSED预条件 */
    HOST::AMSEDPrecondition<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE> preAMSED(
        testMat.get(), 4, MatrixReorderRCM, AMSEDBasic);
    CPU_TIMER_BEGIN()
    preAMSED.setup();
    CPU_TIMER_END()
    std::cout << "MSLR setup(reorder and build MSLR structure) executes: " << CPU_EXEC_TIME() << " ms." << std::endl;

    /* 求解 x = M^{-1} B */
    CPU_TIMER_BEGIN()
    preAMSED.MInvSolve(*testB);
    CPU_TIMER_END()
    std::cout << "MSLR solve x = M^{-1} B executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    testB->printVector("x = M^{-1} B");

    /* 检查答案精度 */
    HOST::check_correctness(static_cast<INT32>(testMat->getRowNum()), (int*)testMat->getRowOffsetPtr(0),
                            (int*)testMat->getColIndicesPtr(0),
                            testMat->getCSRValuesPtr(0), testB->getRawValPtr(), testB_copy->getRawValPtr());

    testX.add(-1, *testB);
    HIGH_PRECISION_TYPE residual = sqrt(testX->sumKahanOMP(0, testX.getLength(), [](HIGH_PRECISION_TYPE x) {
        return x * x;
    }));
    std::cout << "residual: " << residual << std::endl;
}


int main(int argc, char** argv) {
    /* 测试各种预条件求 x = M^{-1} y */
    // testIC();
    testAMSED();
    return 0;
}
