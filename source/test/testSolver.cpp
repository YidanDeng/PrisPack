/*
 * @author  袁心磊、邓轶丹
 * @date    2024/6/15
 * @details 测试solver
 */

#include "../../include/Solver/GMRES.h"
#include "../../include/Solver/CG.h"
#include "../../include/Preconditioner/IncompleteLU.h"
#include "../../include/Preconditioner/IncompleteCholesky.h"
#include "../../include/Preconditioner/AMSEDPrecondition.h"
#include "../../include/MatrixClass/ModelProblem.h"
#include "../../include/utils/TestTools/ReadMtxTools.h"
#include "../../include/utils/TestTools/checkTools.hpp"
#include "../../include/utils/TimerTools/CPUtimer.hpp"
#include "../../include/utils/ExternalTools/MetisTools.h"
#include <filesystem>
#include <fstream>

#define HIGH_PRECISION_TYPE double
#define LOW_PRECISION_TYPE float

void testGmres() {
    SharedObject<HOST::CSRMatrix<FLOAT64> > testMat; // 这里只是声明了一个空智能指针，需要实例化
    testMat.construct(); // 指针实例化，调用对应类的构造函数初始化对象，这里主要调的是无参构造
    HOST::generateLaplacianCSR(*testMat, 50, 50, 50, 0.1, 0.1, 0.1, 0.0);
    UINT32 dim = testMat->getRowNum();
    HOST::AutoAllocateVector<FLOAT64> right_hand(dim, memoryBase);
    HOST::AutoAllocateVector<FLOAT64> temp(dim, memoryBase);
    temp->fillVector(0, dim, 1);
    testMat->MatPVec(*temp, *right_hand);

    HOST::AutoAllocateVector<FLOAT64> res(dim, memoryBase);
    double pre_start = omp_get_wtime();
    HOST::GMRES<FLOAT64> gmres(dim);
    gmres.setRestart(true, 20);
    gmres.solve(*testMat, *right_hand, *res);
    double pre_end = omp_get_wtime();
    double time = pre_end - pre_start;
    std::cout << "Converged : " << gmres.getConvergence() << std::endl;
    std::cout << "Number of iterations: " << gmres.getNumIter() << std::endl;
    std::cout << "Error2 : " << gmres.getError() << std::endl;
    std::cout << "time: " << time << std::endl;
    res.printVector("res");
}

void testPrecondGMRES() {
    SharedObject<HOST::CSRMatrix<FLOAT64> > testMat, pre_testMat; // 这里只是声明了一个空智能指针，需要实例化
    testMat.construct();
    pre_testMat.construct();
    //    INT32 diagDim = 1000;
    //    HOST::generatePoissonCSR(*testMat, diagDim);
    HOST::generateLaplacianCSR(*testMat, 128, 128, 128, 0.0, 0.0, 0.0, 0.02);
    testMat->printMatrix("test mat");
    UINT32 dim = testMat->getRowNum();
    HOST::AutoAllocateVector<UINT32> perm(dim, memoryBase);
    HOST::rcmReorderCSR(*testMat.get(), *perm);
    testMat->getSubMatrix(*perm, *perm, *pre_testMat.get());
    testMat = pre_testMat;
//    SharedObject<HOST::IncompleteLU<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE> > testILU(testMat.get(), 40, 1e-2);
    SharedObject<HOST::IncompleteLDLT<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE> > testILU(testMat.get(), 1e-2);

    SharedObject<HOST::GMRES<FLOAT64> > testGMRES(dim, testILU.get(), 1e-6, HOST::FlexibleGMRES);

    HOST::AutoAllocateVector<FLOAT64> right_hand(dim, memoryBase);
    HOST::AutoAllocateVector<FLOAT64> temp(dim, memoryBase);
    /* 生成随机标准答案x，并构造B = Ax */
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX(testMat->getRowNum(), memoryBase),
            testB(testMat->getRowNum(), memoryBase);
    HOST::generateArrayRandom1D(testX->getRawValPtr(), testX.getLength());
    testX.printVector("original X");
    testMat->MatPVec(*testX, *right_hand);
    right_hand->printVector("original B");

    HOST::AutoAllocateVector<FLOAT64> res(dim, memoryBase);
    testGMRES->setRestart(true, 100);
    testGMRES->solve(*testMat, *right_hand, *res);
    std::cout << "fill : " << testILU->getPreconditionFillinRatio() << std::endl;
    std::cout << "Converged : " << testGMRES->getConvergence() << std::endl;
    std::cout << "Number of iterations: " << testGMRES->getNumIter() << std::endl;
    std::cout << "Error2 : " << testGMRES->getError() << std::endl;
    HOST::check_correctness(dim, (int *) testMat->getRowOffsetPtr(0), (int *) testMat->getColIndicesPtr(0),
                            testMat->getCSRValuesPtr(0), res->getRawValPtr(), right_hand->getRawValPtr());
    res.printVector("res");

}

void testPrecondCG() {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE> > testMat; // 这里只是声明了一个空智能指针，需要实例化
    testMat.construct();    // 对象实例化，调用对应的无参构造函数
    HOST::generateLaplacianCSR(*testMat, 32, 32, 32, 0.0, 0.0, 0.0, 0.0);
    testMat->printMatrix("test mat");
    SharedObject<HOST::IncompleteCholesky<HIGH_PRECISION_TYPE, HIGH_PRECISION_TYPE> > testIC(testMat.get(), 1e-3);
    UINT32 dim = testMat->getRowNum();
    SharedObject<HOST::CG<HIGH_PRECISION_TYPE> > testCG(dim, testIC.get(), HOST::FlexibleCG, 1e-6, MAX_ITER_NUM_SOLVER);
    HOST::AutoAllocateVector<FLOAT64> right_hand(dim, memoryBase);
    HOST::AutoAllocateVector<FLOAT64> temp(dim, memoryBase);
    temp->fillVector(0, dim, 1);
    testMat->MatPVec(*temp, *right_hand);
    HOST::AutoAllocateVector<FLOAT64> res(dim, memoryBase);
    CPU_TIMER_FUNC()
    CPU_TIMER_BEGIN()
    testCG->solve(*testMat, *right_hand, *res);
    CPU_TIMER_END()
    std::cout << " --- Solver totally executes: " << CPU_EXEC_TIME() << std::endl;
    std::cout << "Converged : " << testCG->getConvergence() << std::endl;
    std::cout << "Number of iterations: " << testCG->getNumIter() << std::endl;
    std::cout << "Error2 : " << testCG->getError() << std::endl;
    res.printVector("res");
}

void testSolverNoRightHand(char **argv) {
    CPU_TIMER_FUNC()
    CPU_TIMER_BEGIN()
    HOST::ReadMtxTools<FLOAT64> readMtxTools(argv[0], "../datasets/paper/symmetric/cfd1.mtx", 1);
    SharedObject<HOST::CSRMatrix<FLOAT64> > testCSR(memoryAligned);
    readMtxTools.loadMatrix(*testCSR);
    testCSR->printMatrix("testMat");
    CPU_TIMER_END()
    std::cout << " --- Read mtx and rhs executes: " << CPU_EXEC_TIME() / 1000 << " s." << std::endl;

    /* 求解 */
    UINT32 dim = testCSR->getRowNum();
    HOST::AutoAllocateVector<FLOAT64> rightHand(dim, memoryBase);
    HOST::AutoAllocateVector<FLOAT64> temp(dim, memoryBase);
    /* 生成随机标准答案x，并构造B = Ax */
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX(testCSR->getRowNum(), memoryBase),
            testB(testCSR->getRowNum(), memoryBase);
    HOST::generateArrayRandom1D(testX->getRawValPtr(), testX.getLength());
    testX.printVector("original X");
    testCSR->MatPVec(*testX, *rightHand);
    rightHand->printVector("original B");
    SharedObject<HOST::IncompleteLDLT<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE> > testILU(testCSR.get(), 1e-2);
    SharedObject<HOST::GMRES<FLOAT64> > testGMRES(dim, testILU.get(), 1e-6, HOST::FlexibleGMRES);

    HOST::AutoAllocateVector<FLOAT64> res(dim, memoryBase);
    testGMRES->setRestart(true, 100);
    CPU_TIMER_BEGIN()
    testGMRES->solve(*testCSR, *rightHand, *res);
    CPU_TIMER_END()
    std::cout << "fill : " << testILU->getPreconditionFillinRatio() << std::endl;
    std::cout << " --- Solver totally executes: " << CPU_EXEC_TIME() << std::endl;
    std::cout << "Converged : " << testGMRES->getConvergence() << std::endl;
    std::cout << "Number of iterations: " << testGMRES->getNumIter() << std::endl;
    std::cout << "Error2 : " << testGMRES->getError() << std::endl;
    HOST::check_correctness(dim, (int *) testCSR->getRowOffsetPtr(0), (int *) testCSR->getColIndicesPtr(0),
                            testCSR->getCSRValuesPtr(0), res->getRawValPtr(), rightHand->getRawValPtr());
    res.printVector("res");
    testCSR->MatPVec(*res, *rightHand);
    rightHand.printVector("print right hand");
}

void testAMSEDPrecondGMRES() {
    /* 生成测试矩阵A */
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE> > testMat; // 这里只是声明了一个空智能指针，需要实例化
    testMat.construct(); // 指针实例化，调用对应类的构造函数初始化对象，这里主要调的是无参构造
    INT32 dim = 64;
    HOST::generateLaplacianCSR(*testMat, dim, dim, dim, 0.0, 0.0, 0.0, 0.01);
    testMat->printMatrix("original mat A");

    /* 生成随机标准答案x，并构造B = Ax */
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX(testMat->getRowNum(), memoryBase),
            testB(testMat->getRowNum(), memoryBase);
    HOST::generateArrayRandom1D(testX->getRawValPtr(), testX.getLength());
    testX.printVector("original X");
    testMat->MatPVec(*testX, *testB);
    testB->printVector("original B");
    /* 初始化计时器 */
    CPU_TIMER_FUNC()

    /* 初始化AMSED预条件 */
    SharedObject<HOST::AMSEDPrecondition<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE>> preAMSED(
            testMat.get(), 4, MatrixReorderRCM, AMSEDEigenvalueDeflation);
    CPU_TIMER_BEGIN()
    preAMSED->setup();
    CPU_TIMER_END()
    std::cout << " --- AMSED setup(reorder and build MSLR structure) executes: " << CPU_EXEC_TIME() << " ms." <<
            std::endl;
    std::cout << " --- AMSED fill: " << preAMSED->getPreconditionFillinRatio() << std::endl;
    /* 初始化求解器 */
    SharedObject<HOST::GMRES<FLOAT64> > testGMRES(testMat->getRowNum(), preAMSED.get(), 1e-6, HOST::FlexibleGMRES);
    HOST::AutoAllocateVector<FLOAT64> res(testMat->getRowNum(), memoryBase);
    testGMRES->setRestart(true, 100);
    CPU_TIMER_BEGIN()
    testGMRES->solve(*testMat, *testB, *res);
    CPU_TIMER_END()
    std::cout << " --- AMSED fill: " << preAMSED->getPreconditionFillinRatio() << std::endl;
    std::cout << " --- Solver totally executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    std::cout << "Converged : " << testGMRES->getConvergence() << std::endl;
    std::cout << "Number of iterations: " << testGMRES->getNumIter() << std::endl;
    std::cout << "Error2 : " << testGMRES->getError() << std::endl;
    HOST::check_correctness(dim, (int *) testMat->getRowOffsetPtr(0), (int *) testMat->getColIndicesPtr(0),
                            testMat->getCSRValuesPtr(0), res->getRawValPtr(), testB->getRawValPtr());
    res.printVector("res");
}

void testILDLTPrecondGMRES() {
    /* 生成测试矩阵A */
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE> > testMat; // 这里只是声明了一个空智能指针，需要实例化
    testMat.construct(); // 指针实例化，调用对应类的构造函数初始化对象，这里主要调的是无参构造
    INT32 dim = 64;
    HOST::generateLaplacianCSR(*testMat, dim, dim, dim, 0.0, 0.0, 0.0, 0.05);
    testMat->printMatrix("original mat A");

    /* 生成随机标准答案x，并构造B = Ax */
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX(testMat->getRowNum(), memoryBase),
            testB(testMat->getRowNum(), memoryBase);
    HOST::generateArrayRandom1D(testX->getRawValPtr(), testX.getLength());
    testX.printVector("original X");
    testMat->MatPVec(*testX, *testB);
    testB->printVector("original B");
    /* 初始化计时器 */
    CPU_TIMER_FUNC()
    /* 初始化ILDLT预条件 */
    SharedObject<HOST::IncompleteLDLT<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE> > preILDLT(testMat.get(), 1e-2);
    CPU_TIMER_BEGIN()
    preILDLT->setup();
    CPU_TIMER_END()
    std::cout << " --- ILDLT setup executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    std::cout << " --- ILDLT fill: " << preILDLT->getPreconditionFillinRatio() << std::endl;
    /* 初始化求解器 */
    SharedObject<HOST::GMRES<HIGH_PRECISION_TYPE> >
            testGMRES(testMat->getRowNum(), preILDLT.get(), 1e-6, HOST::FlexibleGMRES);
    HOST::AutoAllocateVector<FLOAT64> res(testMat->getRowNum(), memoryBase);
    testGMRES->setRestart(true, 100);
    CPU_TIMER_BEGIN()
    testGMRES->solve(*testMat, *testB, *res);
    CPU_TIMER_END()
    std::cout << " --- Solver totally executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    std::cout << "Converged : " << testGMRES->getConvergence() << std::endl;
    std::cout << "Number of iterations: " << testGMRES->getNumIter() << std::endl;
    std::cout << "Error2 : " << testGMRES->getError() << std::endl;
    HOST::check_correctness(dim, (int *) testMat->getRowOffsetPtr(0), (int *) testMat->getColIndicesPtr(0),
                            testMat->getCSRValuesPtr(0), res->getRawValPtr(), testB->getRawValPtr());
    res.printVector("res");

}

int main(int argc, char **argv) {
    testPrecondCG();

    return 0;
}
