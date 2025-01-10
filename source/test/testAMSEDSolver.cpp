/*
 * @author  邓轶丹
 * @date    2024/11/30
 * @details 测试AMSED
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

#define HIGH_PRECISION_TYPE double
#define LOW_PRECISION_TYPE double

#define LAPLACIAN_X_SHIFT 0.0
#define LAPLACIAN_Y_SHIFT 0.0
#define LAPLACIAN_Z_SHIFT 0.0

#define GMRES_RESTART_STEPS 60
#define GMRES_CONVERGENCE_TOLERANCE (1e-6)

#define CG_CONVERGENCE_TOLERANCE (1e-6)


void generateRandomTestData(INT32 problemType, INT32 gridDim, HIGH_PRECISION_TYPE shift,
                            HOST::CSRMatrix<HIGH_PRECISION_TYPE>& csrMat, HostVector<HIGH_PRECISION_TYPE>& resultX,
                            HostVector<HIGH_PRECISION_TYPE>& rightHand) {
    /* 生成测试矩阵A */
    if (problemType == 2) {
        HOST::generateLaplacianCSR(csrMat, gridDim, gridDim, 1, LAPLACIAN_X_SHIFT, LAPLACIAN_Y_SHIFT, LAPLACIAN_Z_SHIFT,
                                   shift);
    } else if (problemType == 3) {
        HOST::generateLaplacianCSR(csrMat, gridDim, gridDim, gridDim, LAPLACIAN_X_SHIFT, LAPLACIAN_Y_SHIFT,
                                   LAPLACIAN_Z_SHIFT, shift);
    }
#ifndef NDEBUG
    else {
        THROW_LOGIC_ERROR("Problem type is incorrect!");
    }
#endif

    csrMat.printMatrix("original mat A");

    /* 生成随机标准答案x，并构造B = Ax */
    resultX.resize(csrMat.getRowNum(), RESERVE_NO_DATA);
    rightHand.resize(csrMat.getRowNum(), RESERVE_NO_DATA);
    HOST::generateArrayRandom1D(resultX.getRawValPtr(), resultX.getLength());
    resultX.printVector("original X");
    csrMat.MatPVec(resultX, rightHand);
    rightHand.printVector("original B");
}

void generateSteadyTestData(INT32 problemType, INT32 gridDim, HIGH_PRECISION_TYPE shift,
                            HOST::CSRMatrix<HIGH_PRECISION_TYPE>& csrMat, HostVector<HIGH_PRECISION_TYPE>& rightHand) {
    /* 生成测试矩阵A */
    if (problemType == 2) {
        HOST::generateLaplacianCSR(csrMat, gridDim, gridDim, 1, LAPLACIAN_X_SHIFT, LAPLACIAN_Y_SHIFT, LAPLACIAN_Z_SHIFT,
                                   shift);
    } else if (problemType == 3) {
        HOST::generateLaplacianCSR(csrMat, gridDim, gridDim, gridDim, LAPLACIAN_X_SHIFT, LAPLACIAN_Y_SHIFT,
                                   LAPLACIAN_Z_SHIFT, shift);
    }
#ifndef NDEBUG
    else {
        THROW_LOGIC_ERROR("Problem type is incorrect!");
    }
#endif
    csrMat.printMatrix("original mat A");

    /* 生成全为1的标准答案x，并构造B = Ax */
    HOST::DenseVector<HIGH_PRECISION_TYPE> resultX;
    resultX.resize(csrMat.getRowNum(), RESERVE_NO_DATA);
    rightHand.resize(csrMat.getRowNum(), RESERVE_NO_DATA);
    resultX.fillVector(0, csrMat.getRowNum(), 1);
    resultX.printVector("original X");
    csrMat.MatPVec(resultX, rightHand);
    rightHand.printVector("original B");
}

void writeData2Files() {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> csrMat;
    csrMat.construct();
    HOST::generateLaplacianCSR(*csrMat, 20, 20, 20, LAPLACIAN_X_SHIFT, LAPLACIAN_Y_SHIFT, LAPLACIAN_Z_SHIFT,
                               0.0);
    /* 初始化AMSED预条件 */
    SharedObject<HOST::AMSEDPrecondition<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE>> preAMSED(
        csrMat.get(), 4, 100, 16, 0.8, MatrixReorderRCM, AMSEDEigenvalueDeflation, UseClassicLanczos);
    preAMSED->setup(); // 写入文件的过程在setup中
}

void testAMSEDPrecondCG(INT32 level, INT32 lanczosSteps, INT32 lowRankSize, MatrixReorderOption_t localReorderType,
                        LanczosType_t lanczosType, HIGH_PRECISION_TYPE eigCorrectBound,
                        const std::shared_ptr<HOST::CSRMatrix<HIGH_PRECISION_TYPE>>& testMat,
                        HostVector<HIGH_PRECISION_TYPE>& rightHand) {
    UINT32 dim = testMat->getRowNum();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testB(dim, memoryAligned), res(dim, memoryAligned);
    testB.copy(rightHand);
    res->fillVector(0, dim, 0);
    /* 初始化计时器 */
    CPU_TIMER_FUNC()
    /* 初始化AMSED预条件 */
    SharedObject<HOST::AMSEDPrecondition<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE>> preAMSED(
        testMat, level, lanczosSteps, lowRankSize, eigCorrectBound, localReorderType,
        AMSEDEigenvalueDeflation, lanczosType);
    CPU_TIMER_BEGIN()
    preAMSED->setup();
    CPU_TIMER_END()
    std::cout << " --- AMSED setup(reorder and build MSLR structure) executes: " << CPU_EXEC_TIME() << " ms." <<
        std::endl;
    std::cout << " --- AMSED total fill: " << preAMSED->getPreconditionFillinRatio() << std::endl;
    std::cout << " --- AMSED IC fill: " << preAMSED->getIncompleteFactorRatio() << std::endl;
    std::cout << " --- AMSED LRC fill: " << preAMSED->getLowRankCorrectionRatio() << std::endl;
    std::cout << " --- AMSED implicit Lanczos executes: " << preAMSED->getCompLanczosTime() << " ms." <<
        std::endl;
    std::cout << " --- AMSED block decomposition executes: " << preAMSED->getBlockDecompositionTime() << " ms." <<
        std::endl;

    /* 初始化求解器 */
    SharedObject<HOST::CG<HIGH_PRECISION_TYPE>> testCG(dim, preAMSED.get(), HOST::FlexibleCG, CG_CONVERGENCE_TOLERANCE,
                                                       MAX_ITER_NUM_SOLVER);
    CPU_TIMER_BEGIN()
    testCG->solve(*testMat, *testB, *res);
    CPU_TIMER_END()
    std::cout << " --- Solver totally executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    std::cout << " --- CG converged : " << testCG->getConvergence() << std::endl;
    std::cout << " --- Number of iterations: " << testCG->getNumIter() << std::endl;
    std::cout << " --- Error2 : " << testCG->getError() << std::endl;
    HOST::check_correctness(static_cast<INT32>(dim), (int*)testMat->getRowOffsetPtr(0),
                            (int*)testMat->getColIndicesPtr(0),
                            testMat->getCSRValuesPtr(0), res->getRawValPtr(), testB->getRawValPtr());
    res.printVector("res");
}

void testMSLRPrecondCG(INT32 level, INT32 lanczosSteps, INT32 lowRankSize, MatrixReorderOption_t localReorderType,
                       LanczosType_t lanczosType, const std::shared_ptr<HOST::CSRMatrix<HIGH_PRECISION_TYPE>>& testMat,
                       HostVector<HIGH_PRECISION_TYPE>& rightHand) {
    UINT32 dim = testMat->getRowNum();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testB(dim, memoryBase), res(dim, memoryBase);
    testB.copy(rightHand);
    /* 初始化计时器 */
    CPU_TIMER_FUNC()
    /* 初始化MSLR预条件 */
    SharedObject<HOST::AMSEDPrecondition<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE>> preMSLR(
        testMat, level, lanczosSteps, lowRankSize, localReorderType, AMSEDBasic, lanczosType);
    CPU_TIMER_BEGIN()
    preMSLR->setup();
    CPU_TIMER_END()
    std::cout << " --- MSLR setup(reorder and build MSLR structure) executes: " << CPU_EXEC_TIME() << " ms." <<
        std::endl;
    std::cout << " --- MSLR total fill: " << preMSLR->getPreconditionFillinRatio() << std::endl;
    std::cout << " --- MSLR IC fill: " << preMSLR->getIncompleteFactorRatio() << std::endl;
    std::cout << " --- MSLR LRC fill: " << preMSLR->getLowRankCorrectionRatio() << std::endl;
    std::cout << " --- MSLR implicit Lanczos executes: " << preMSLR->getCompLanczosTime() << " ms." << std::endl;
    std::cout << " --- MSLR block decomposition executes: " << preMSLR->getBlockDecompositionTime() << " ms." <<
        std::endl;

    /* 初始化求解器 */
    SharedObject<HOST::CG<HIGH_PRECISION_TYPE>> testCG(dim, preMSLR.get(), HOST::FlexibleCG, CG_CONVERGENCE_TOLERANCE,
                                                       MAX_ITER_NUM_SOLVER);
    CPU_TIMER_BEGIN()
    testCG->solve(*testMat, *testB, *res);
    CPU_TIMER_END()
    std::cout << " --- Parallel compute B executes: " << preMSLR->getParallelBInvSolveTime() << " ms." << std::endl;
    std::cout << " --- Sequential compute low-rank correction executes: " << preMSLR->getCompLowRankCorrectTime() <<
        " ms." << std::endl;
    std::cout << " --- Solver totally executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    std::cout << " --- CG converged : " << testCG->getConvergence() << std::endl;
    std::cout << " --- Number of iterations: " << testCG->getNumIter() << std::endl;
    std::cout << " --- Error2 : " << testCG->getError() << std::endl;
    HOST::check_correctness(static_cast<INT32>(dim), (int*)testMat->getRowOffsetPtr(0),
                            (int*)testMat->getColIndicesPtr(0),
                            testMat->getCSRValuesPtr(0), res->getRawValPtr(), testB->getRawValPtr());
    res.printVector("res");
}

void testILDLTPrecondCG(MatrixReorderOption_t reorderType,
                        const std::shared_ptr<HOST::CSRMatrix<HIGH_PRECISION_TYPE>>& testMat,
                        HostVector<HIGH_PRECISION_TYPE>& rightHand) {
    UINT32 dim = testMat->getRowNum();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testB(dim, memoryBase), res(dim, memoryBase);
    testB.copy(rightHand);
    HOST::AutoAllocateVector<UINT32> perm(dim, memoryBase);
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> reorderMat;
    reorderMat.construct();
    if (reorderType == MatrixReorderNo) reorderMat.get() = testMat;
    else if (reorderType == MatrixReorderAMD) {
        HOST::amdReorderCSR(*testMat, *perm);
    } else if (reorderType == MatrixReorderRCM) {
        HOST::rcmReorderCSR(*testMat, *perm);
    }
    if (reorderType != MatrixReorderNo) {
        testMat->getSubMatrix(*perm, *perm, *reorderMat);
        for (UINT32 i = 0; i < dim; i++) {
            testB[i] = rightHand[perm[i]];
        }
    }
    /* 初始化计时器 */
    CPU_TIMER_FUNC()
    /* 初始化AMSED预条件 */
    SharedObject<HOST::IncompleteLDLT<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE>> preILDLT(reorderMat.get(), 1e-3);
    CPU_TIMER_BEGIN()
    preILDLT->setup();
    CPU_TIMER_END()
    std::cout << " --- ILDLT setup executes: " << CPU_EXEC_TIME() << " ms." <<
        std::endl;
    std::cout << " --- ILDLT fill: " << preILDLT->getPreconditionFillinRatio() << std::endl;

    /* 初始化求解器 */
    SharedObject<HOST::CG<HIGH_PRECISION_TYPE>> testCG(dim, preILDLT.get(), HOST::FlexibleCG, 1e-6,
                                                       MAX_ITER_NUM_SOLVER);
    CPU_TIMER_BEGIN()
    testCG->solve(*reorderMat, *testB, *res);
    CPU_TIMER_END()
    std::cout << " --- Solver totally executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    std::cout << " --- CG converged : " << testCG->getConvergence() << std::endl;
    std::cout << " --- Number of iterations: " << testCG->getNumIter() << std::endl;
    std::cout << " --- Error2 : " << testCG->getError() << std::endl;
    HOST::check_correctness(static_cast<INT32>(dim), (int*)reorderMat->getRowOffsetPtr(0),
                            (int*)reorderMat->getColIndicesPtr(0),
                            reorderMat->getCSRValuesPtr(0), res->getRawValPtr(), testB->getRawValPtr());
    res.printVector("res");
}


void testICTPrecondCG(MatrixReorderOption_t reorderType, HIGH_PRECISION_TYPE dropTolerace,
                      const std::shared_ptr<HOST::CSRMatrix<HIGH_PRECISION_TYPE>>& testMat,
                      HostVector<HIGH_PRECISION_TYPE>& rightHand) {
    UINT32 dim = testMat->getRowNum();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testB(dim, memoryBase), res(dim, memoryBase);
    testB.copy(rightHand);
    HOST::AutoAllocateVector<UINT32> perm(dim, memoryBase);
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> reorderMat;
    reorderMat.construct();
    if (reorderType == MatrixReorderNo) reorderMat.get() = testMat;
    else if (reorderType == MatrixReorderAMD) {
        HOST::amdReorderCSR(*testMat, *perm);
        testMat->getSubMatrix(*perm, *perm, *reorderMat);
    } else if (reorderType == MatrixReorderRCM) {
        HOST::rcmReorderCSR(*testMat, *perm);
        testMat->getSubMatrix(*perm, *perm, *reorderMat);
    }
    if (reorderType != MatrixReorderNo) {
        for (UINT32 i = 0; i < dim; i++) {
            testB[i] = rightHand[perm[i]];
        }
    }
    /* 初始化计时器 */
    CPU_TIMER_FUNC()
    /* 初始化ICT预条件 */
    SharedObject<HOST::IncompleteCholesky<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE>> preICT(
        reorderMat.get(), dropTolerace);
    CPU_TIMER_BEGIN()
    preICT->setup();
    CPU_TIMER_END()
    std::cout << " --- ICT setup executes: " << CPU_EXEC_TIME() << " ms." <<
        std::endl;
    std::cout << " --- ICT fill: " << preICT->getPreconditionFillinRatio() << std::endl;

    /* 初始化求解器 */
    SharedObject<HOST::CG<HIGH_PRECISION_TYPE>> testCG(dim, preICT.get(), HOST::FlexibleCG, CG_CONVERGENCE_TOLERANCE,
                                                       MAX_ITER_NUM_SOLVER);
    CPU_TIMER_BEGIN()
    testCG->solve(*reorderMat, *testB, *res);
    CPU_TIMER_END()
    std::cout << " --- Solver totally executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    std::cout << " --- CG converged : " << testCG->getConvergence() << std::endl;
    std::cout << " --- Number of iterations: " << testCG->getNumIter() << std::endl;
    std::cout << " --- Error2 : " << testCG->getError() << std::endl;
    HOST::check_correctness(static_cast<INT32>(dim), (int*)reorderMat->getRowOffsetPtr(0),
                            (int*)reorderMat->getColIndicesPtr(0),
                            reorderMat->getCSRValuesPtr(0), res->getRawValPtr(), testB->getRawValPtr());
    res.printVector("res");
}


void testAMSEDPrecondGMRES(INT32 level, INT32 lanczosSteps, INT32 lowRankSize, MatrixReorderOption_t localReorderType,
                           LanczosType_t lanczosType, HIGH_PRECISION_TYPE eigCorrectBound,
                           const std::shared_ptr<HOST::CSRMatrix<HIGH_PRECISION_TYPE>>& testMat,
                           HostVector<HIGH_PRECISION_TYPE>& rightHand) {
    UINT32 dim = testMat->getRowNum();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testB(dim, memoryBase), res(dim, memoryBase);
    testB.copy(rightHand);
    /* 初始化计时器 */
    CPU_TIMER_FUNC()
    /* 初始化AMSED预条件 */
    SharedObject<HOST::AMSEDPrecondition<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE>> preAMSED(
        testMat, level, lanczosSteps, lowRankSize, eigCorrectBound, localReorderType,
        AMSEDEigenvalueDeflation, lanczosType);
    // CPU_TIMER_BEGIN()
    preAMSED->setup();
    // CPU_TIMER_END()
    // std::cout << " --- AMSED setup(reorder and build MSLR structure) executes: " << CPU_EXEC_TIME() << " ms." <<
    //         std::endl;
    std::cout << " --- AMSED fill: " << preAMSED->getPreconditionFillinRatio() << std::endl;
    /* 初始化求解器 */
    SharedObject<HOST::GMRES<HIGH_PRECISION_TYPE>>
        testGMRES(testMat->getRowNum(), preAMSED.get(), GMRES_CONVERGENCE_TOLERANCE, HOST::FlexibleGMRES);
    testGMRES->setRestart(true, GMRES_RESTART_STEPS);
    CPU_TIMER_BEGIN()
    testGMRES->solve(*testMat, *testB, *res);
    CPU_TIMER_END()
    std::cout << " --- Solver totally executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    std::cout << " --- GMRES converged : " << testGMRES->getConvergence() << std::endl;
    std::cout << " --- Number of iterations: " << testGMRES->getNumIter() << std::endl;
    std::cout << " --- Error2 : " << testGMRES->getError() << std::endl;
    HOST::check_correctness(static_cast<INT32>(dim), (int*)testMat->getRowOffsetPtr(0),
                            (int*)testMat->getColIndicesPtr(0),
                            testMat->getCSRValuesPtr(0), res->getRawValPtr(), testB->getRawValPtr());
    res.printVector("res");
}

void testMSLRPrecondGMRES(INT32 level, INT32 lanczosSteps, INT32 lowRankSize, MatrixReorderOption_t localReorderType,
                          LanczosType_t lanczosType,
                          const std::shared_ptr<HOST::CSRMatrix<HIGH_PRECISION_TYPE>>& testMat,
                          HostVector<HIGH_PRECISION_TYPE>& rightHand) {
    UINT32 dim = testMat->getRowNum();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testB(dim, memoryBase), res(dim, memoryBase);
    testB.copy(rightHand);
    /* 初始化计时器 */
    CPU_TIMER_FUNC()
    /* 初始化MSLR预条件 */
    SharedObject<HOST::AMSEDPrecondition<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE>> preMSLR(
        testMat, level, lanczosSteps, lowRankSize, localReorderType, AMSEDBasic, lanczosType);
    preMSLR->setup();
    std::cout << " --- MSLR fill: " << preMSLR->getPreconditionFillinRatio() << std::endl;
    /* 初始化求解器 */
    SharedObject<HOST::GMRES<HIGH_PRECISION_TYPE>>
        testGMRES(testMat->getRowNum(), preMSLR.get(), GMRES_CONVERGENCE_TOLERANCE, HOST::FlexibleGMRES);
    testGMRES->setRestart(true, GMRES_RESTART_STEPS);
    CPU_TIMER_BEGIN()
    testGMRES->solve(*testMat, *testB, *res);
    CPU_TIMER_END()
    std::cout << " --- Solver totally executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    std::cout << " --- GMRES converged : " << testGMRES->getConvergence() << std::endl;
    std::cout << " --- Number of iterations: " << testGMRES->getNumIter() << std::endl;
    std::cout << " --- Error2 : " << testGMRES->getError() << std::endl;
    HOST::check_correctness(static_cast<INT32>(dim), (int*)testMat->getRowOffsetPtr(0),
                            (int*)testMat->getColIndicesPtr(0),
                            testMat->getCSRValuesPtr(0), res->getRawValPtr(), testB->getRawValPtr());
    res.printVector("res");
}

void testILDLTPrecondGMRES(MatrixReorderOption_t reorderType,
                           const std::shared_ptr<HOST::CSRMatrix<HIGH_PRECISION_TYPE>>& testMat,
                           HostVector<HIGH_PRECISION_TYPE>& rightHand, HIGH_PRECISION_TYPE dropToleraceICT) {
    UINT32 dim = testMat->getRowNum();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testB(dim, memoryBase), res(dim, memoryBase);
    testB.copy(rightHand);
    HOST::AutoAllocateVector<UINT32> perm(dim, memoryBase);
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> reorderMat;
    reorderMat.construct();
    if (reorderType == MatrixReorderNo) reorderMat.get() = testMat;
    else if (reorderType == MatrixReorderAMD) {
        HOST::amdReorderCSR(*testMat, *perm);
        testMat->getSubMatrix(*perm, *perm, *reorderMat);
    } else if (reorderType == MatrixReorderRCM) {
        HOST::rcmReorderCSR(*testMat, *perm);
        testMat->getSubMatrix(*perm, *perm, *reorderMat);
    }
    if (reorderType != MatrixReorderNo) {
        for (UINT32 i = 0; i < dim; i++) {
            testB[i] = rightHand[perm[i]];
        }
    }
    /* 初始化计时器 */
    CPU_TIMER_FUNC()
    /* 初始化ILDLT预条件 */
    SharedObject<HOST::IncompleteLDLT<LOW_PRECISION_TYPE, HIGH_PRECISION_TYPE>> preILDLT(
        reorderMat.get(), dropToleraceICT);
    CPU_TIMER_BEGIN()
    preILDLT->setup();
    CPU_TIMER_END()
    std::cout << " --- ILDLT setup executes: " << CPU_EXEC_TIME() << " ms." <<
        std::endl;
    std::cout << " --- ILDLT fill: " << preILDLT->getPreconditionFillinRatio() << std::endl;

    /* 初始化求解器 */
    SharedObject<HOST::GMRES<HIGH_PRECISION_TYPE>>
        testGMRES(reorderMat->getRowNum(), preILDLT.get(), GMRES_CONVERGENCE_TOLERANCE, HOST::FlexibleGMRES);
    testGMRES->setRestart(true, GMRES_RESTART_STEPS);
    CPU_TIMER_BEGIN()
    testGMRES->solve(*reorderMat, *testB, *res);
    CPU_TIMER_END()
    std::cout << " --- Solver totally executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    std::cout << " --- GMRES converged : " << testGMRES->getConvergence() << std::endl;
    std::cout << " --- Number of iterations: " << testGMRES->getNumIter() << std::endl;
    std::cout << " --- Error2 : " << testGMRES->getError() << std::endl;
    HOST::check_correctness(static_cast<INT32>(dim), (int*)reorderMat->getRowOffsetPtr(0),
                            (int*)reorderMat->getColIndicesPtr(0),
                            reorderMat->getCSRValuesPtr(0), res->getRawValPtr(), testB->getRawValPtr());
    res.printVector("res");
}


void testModelProblemCG(INT32 modelType, INT32 gridDim, HIGH_PRECISION_TYPE s, INT32 level, INT32 lanczosSteps,
                        MatrixReorderOption_t reorderOption, LanczosType_t lanczosType,
                        INT32 rk, HIGH_PRECISION_TYPE eigCorrectBound, HIGH_PRECISION_TYPE dropToleraceICT) {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> testMat;
    testMat.construct();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX, testB;

    std::cout << " ================= Testing Model problem CG Begin =================" << std::endl;
    /* 生成标准答案x，并构造B = Ax */
    // generateSteadyTestData(modelType, gridDim, s, *testMat, *testB);
    generateRandomTestData(modelType, gridDim, s, *testMat, *testX, *testB);
    /* 测试CG方法 */
    std::cout << " Test ICT-CG: " << std::endl;
    testICTPrecondCG(reorderOption, dropToleraceICT, testMat.get(), *testB);
    std::cout << std::endl;
    std::cout << " Test MSLR-CG: " << std::endl;
    std::cout << " --- Lanczos steps: " << lanczosSteps << ", Low-rank size: " << rk << std::endl;
    testMSLRPrecondCG(level, lanczosSteps, rk, reorderOption, lanczosType, testMat.get(), *testB);
    std::cout << std::endl;
    std::cout << " Test AMSED-CG: " << std::endl;
    std::cout << " --- Lanczos steps: " << lanczosSteps << ", Low-rank size: " << rk <<
        ", eigen-value correction bound: " << eigCorrectBound << std::endl;
    testAMSEDPrecondCG(level, lanczosSteps, rk, reorderOption, lanczosType, eigCorrectBound, testMat.get(), *testB);
    std::cout << " ================= Testing Model problem CG End =================" << std::endl;
    std::cout << std::endl;
}

void testModelProblemGMRES(INT32 modelType, INT32 gridDim, HIGH_PRECISION_TYPE s, INT32 level, INT32 lanczosSteps,
                           MatrixReorderOption_t reorderOption, LanczosType_t lanczosType,
                           INT32 rk, HIGH_PRECISION_TYPE eigCorrectBound, HIGH_PRECISION_TYPE dropToleraceICT) {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> testMat;
    testMat.construct();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX, testB;

    std::cout << " ================= Testing GMRES Begin =================" << std::endl;
    /* 生成标准答案x，并构造B = Ax */
    // generateSteadyTestData(modelType, gridDim, s, *testMat, *testB);
    generateRandomTestData(modelType, gridDim, s, *testMat, *testX, *testB);
    /* 测试GMRES方法 */
    // std::cout << " Test ILDLT-GMRES: " << std::endl;
    // testILDLTPrecondGMRES(reorderOption, testMat.get(), *testB, dropToleraceICT);
    // std::cout << std::endl;
    std::cout << " Test MSLR-GMRES: " << std::endl;
    std::cout << " --- Lanczos steps: " << lanczosSteps << ", Low-rank size: " << rk << std::endl;
    testMSLRPrecondGMRES(level, lanczosSteps, rk, reorderOption, lanczosType, testMat.get(), *testB);
    std::cout << std::endl;
    std::cout << " Test AMSED-GMRES: " << std::endl;
    std::cout << " --- Lanczos steps: " << lanczosSteps << ", Low-rank size: " << rk <<
        ", eigen-value correction bound: " << eigCorrectBound << std::endl;
    testAMSEDPrecondGMRES(level, lanczosSteps, rk, reorderOption, lanczosType, eigCorrectBound, testMat.get(), *testB);
    std::cout << " ================= Testing GMRES End =================" << std::endl;
    std::cout << std::endl;
}


void testEffectOfMatrixOrderType(INT32 modelType, INT32 gridDim, HIGH_PRECISION_TYPE s, INT32 level, INT32 lanczosSteps,
                                 INT32 rkMax, HIGH_PRECISION_TYPE eigCorrectBound) {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> testMat;
    testMat.construct();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX, testB;

    std::cout << " ================= Testing Effect of Levels by AMSED-CG Begin =================" << std::endl;
    /* 生成随机标准答案x，并构造B = Ax */
    generateRandomTestData(modelType, gridDim, s, *testMat, *testX, *testB);

    std::cout << " Test AMSED-CG, Current level size: " << 4 << ", Lanczos steps: " << lanczosSteps <<
        ", Low-rank size: " << rkMax << ", eigen-value correction bound: " << eigCorrectBound <<
        ", reorder type: MatrixReorderNo" << std::endl;
    testAMSEDPrecondCG(level, lanczosSteps, rkMax, MatrixReorderNo, UseRestartLanczos, eigCorrectBound, testMat.get(),
                       *testB);
    std::cout << std::endl;


    std::cout << " Test AMSED-CG, Current level size: " << 4 << ", Lanczos steps: " << lanczosSteps <<
        ", Low-rank size: " << rkMax << ", eigen-value correction bound: " << eigCorrectBound <<
        ", reorder type: MatrixReorderAMD" << std::endl;
    testAMSEDPrecondCG(level, lanczosSteps, rkMax, MatrixReorderAMD, UseRestartLanczos, eigCorrectBound, testMat.get(),
                       *testB);
    std::cout << std::endl;

    std::cout << " Test AMSED-CG, Current level size: " << 4 << ", Lanczos steps: " << lanczosSteps <<
        ", Low-rank size: " << rkMax << ", eigen-value correction bound: " << eigCorrectBound <<
        ", reorder type: MatrixReorderRCM" << std::endl;
    testAMSEDPrecondCG(level, lanczosSteps, rkMax, MatrixReorderRCM, UseRestartLanczos, eigCorrectBound, testMat.get(),
                       *testB);
    std::cout << std::endl;

    std::cout << " ================= Testing Effect of Levels by AMSED-CG End =================" << std::endl;
    std::cout << std::endl;
}

void testEffectOfLevelsAMSED(INT32 modelType, INT32 gridDim, HIGH_PRECISION_TYPE s, INT32 level, INT32 lanczosSteps,
                             INT32 rkMax, HIGH_PRECISION_TYPE eigCorrectBound) {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> testMat;
    testMat.construct();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX, testB;

    std::cout << " ================= Testing Effect of Levels by AMSED-CG Begin =================" << std::endl;
    /* 生成随机标准答案x，并构造B = Ax */
    generateRandomTestData(modelType, gridDim, s, *testMat, *testX, *testB);

    /* 测试CG方法 */
    for (UINT32 i = level; i < level + 8; ++i) {
        std::cout << " Test AMSED-CG, Current level size: " << i << ", Lanczos steps: " << lanczosSteps <<
            ", Low-rank size: " << rkMax << ", eigen-value correction bound: " << eigCorrectBound << std::endl;
        testAMSEDPrecondCG(i, lanczosSteps, rkMax, MatrixReorderRCM, UseRestartLanczos, eigCorrectBound, testMat.get(),
                           *testB);
        std::cout << std::endl;
    }

    std::cout << " ================= Testing Effect of Levels by AMSED-CG End =================" << std::endl;
    std::cout << std::endl;
}


void testEffectOfEigenBoundAMSED(INT32 modelType, INT32 gridDim, HIGH_PRECISION_TYPE s, INT32 level, INT32 lanczosSteps,
                                 INT32 rkMax, HIGH_PRECISION_TYPE eigCorrectStartBound) {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> testMat;
    testMat.construct();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX, testB;

    std::cout << " ================= Testing Effect of Eigen-value Bound by AMSED-CG Begin =================" <<
        std::endl;
    /* 生成随机标准答案x，并构造B = Ax */
    generateRandomTestData(modelType, gridDim, s, *testMat, *testX, *testB);

    /* 测试CG方法 */
    for (HIGH_PRECISION_TYPE i = eigCorrectStartBound; i < 1.0; i = i + 0.2) {
        std::cout << " Test AMSED-CG, Current level size: " << level << ", Lanczos steps: " << lanczosSteps <<
            ", Low-rank size: " << rkMax << ", eigen-value correction bound: " << i << std::endl;
        testAMSEDPrecondCG(level, lanczosSteps, rkMax, MatrixReorderRCM, UseRestartLanczos, i, testMat.get(), *testB);
        std::cout << std::endl;
    }

    std::cout << " ================= Testing Effect of Eigen-value Bound by AMSED-CG End =================" <<
        std::endl;
    std::cout << std::endl;
}

void testEffectOfLowRankBoundAMSEDPrecondCG(INT32 modelType, INT32 gridDim, HIGH_PRECISION_TYPE s, INT32 level,
                                            INT32 rk, HIGH_PRECISION_TYPE eigCorrectStartBound) {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> testMat;
    testMat.construct();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX, testB;

    std::cout << " ================= Testing Effect of Eigen-value Bound by AMSED-CG Begin =================" <<
        std::endl;
    /* 生成随机标准答案x，并构造B = Ax */
    generateRandomTestData(modelType, gridDim, s, *testMat, *testX, *testB);

    /* 测试CG方法 */
    for (UINT32 i = rk; i < 100; i = i + 10) {
        std::cout << " Test AMSED-CG, Current level size: " << level << ", target rank: " << rk <<
            ", Low-rank bound: " << i << ", eigen-value correction bound: " << eigCorrectStartBound << std::endl;
        testAMSEDPrecondCG(level, i, rk, MatrixReorderNo, UseRestartLanczos, eigCorrectStartBound,
                           testMat.get(), *testB);
        std::cout << std::endl;
    }

    std::cout << " ================= Testing Effect of Eigen-value Bound by AMSED-CG End =================" <<
        std::endl;
    std::cout << std::endl;
}


void testEffectOfLowRankBoundAMSEDPrecondGMRES(INT32 modelType, INT32 gridDim, HIGH_PRECISION_TYPE s, INT32 level,
                                               INT32 lanczosSteps,
                                               INT32 rkMax, HIGH_PRECISION_TYPE eigCorrectStartBound) {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> testMat;
    testMat.construct();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> testX, testB;

    std::cout << " ================= Testing Effect of Eigen-value Bound by AMSED-GMRES Begin =================" <<
        std::endl;
    /* 生成随机标准答案x，并构造B = Ax */
    generateRandomTestData(modelType, gridDim, s, *testMat, *testX, *testB);

    /* 测试CG方法 */
    for (UINT32 i = rkMax; i < 70; i = i + 10) {
        std::cout << " Test AMSED-GMRES, Current level size: " << level << ", Lanczos steps: " << lanczosSteps <<
            ", Low-rank size: " << i << ", eigen-value correction bound: " << eigCorrectStartBound << std::endl;
        testAMSEDPrecondGMRES(level, lanczosSteps, i, MatrixReorderNo, UseClassicLanczos, eigCorrectStartBound,
                              testMat.get(), *testB);
        std::cout << std::endl;
    }

    std::cout << " ================= Testing Effect of Eigen-value Bound by AMSED-GMRES End =================" <<
        std::endl;
    std::cout << std::endl;
}


void testMatrixMarketGMRES(char* argv, const char* dataPathA, const char* dataPathB, INT32 level, INT32 lanczosSteps,
                           MatrixReorderOption_t reorderOption, LanczosType_t lanczosType, INT32 rk,
                           HIGH_PRECISION_TYPE eigCorrectBound, HIGH_PRECISION_TYPE dropToleraceICT) {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> originCSR; ///< 从文件中读取的CSR矩阵
    originCSR.construct();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> rhs; ///< 对应的右端项（如果有dataPathB，就读入，否则用随机向量生成右端项）
    if (dataPathB != nullptr && strcmp(dataPathB, "") != 0) {
        HOST::ReadMtxTools<HIGH_PRECISION_TYPE> readMtxTools(argv, dataPathA, dataPathB, 1);
        readMtxTools.loadMatrix(*originCSR);
        readMtxTools.loadRightHand(*rhs);
    } else {
        HOST::ReadMtxTools<HIGH_PRECISION_TYPE> readMtxTools(argv, dataPathA, 1);
        HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> res;
        readMtxTools.loadMatrix(*originCSR);
        readMtxTools.loadRightHand(*res, *rhs, 1);
    }

    originCSR->printMatrix("origin matrix");
    rhs->printVector("origin right hand");
    std::cout << std::endl;

    std::cout << " ================= Testing GMRES Begin =================" << std::endl;
    /* 测试GMRES方法 */
    std::cout << " Test ILDLT-GMRES: " << std::endl;
    testILDLTPrecondGMRES(reorderOption, originCSR.get(), *rhs, dropToleraceICT);
    std::cout << std::endl;
    std::cout << " Test MSLR-GMRES: " << std::endl;
    std::cout << " --- Lanczos steps: " << lanczosSteps << ", Low-rank size: " << rk << std::endl;
    testMSLRPrecondGMRES(level, lanczosSteps, rk, reorderOption, lanczosType, originCSR.get(), *rhs);
    std::cout << std::endl;
    std::cout << " Test AMSED-GMRES: " << std::endl;
    std::cout << " --- Lanczos steps: " << lanczosSteps << ", Low-rank size: " << rk <<
        ", eigen-value correction bound: " << eigCorrectBound << std::endl;
    testAMSEDPrecondGMRES(level, lanczosSteps, rk, reorderOption, lanczosType, eigCorrectBound, originCSR.get(), *rhs);
    std::cout << " ================= Testing GMRES End =================" << std::endl;
    std::cout << std::endl;
}


void testMatrixMarketCG(char* argv, const char* dataPathA, const char* dataPathB, INT32 level, INT32 lanczosSteps,
                        MatrixReorderOption_t reorderOption, LanczosType_t lanczosType, INT32 rk,
                        HIGH_PRECISION_TYPE eigCorrectBound, HIGH_PRECISION_TYPE dropToleraceICT) {
    SharedObject<HOST::CSRMatrix<HIGH_PRECISION_TYPE>> originCSR; ///< 从文件中读取的CSR矩阵
    originCSR.construct();
    HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> rhs; ///< 对应的右端项（如果有dataPathB，就读入，否则用随机向量生成右端项）
    if (dataPathB != nullptr && strcmp(dataPathB, "") != 0) {
        HOST::ReadMtxTools<HIGH_PRECISION_TYPE> readMtxTools(argv, dataPathA, dataPathB, 1);
        readMtxTools.loadMatrix(*originCSR);
        readMtxTools.loadRightHand(*rhs);
    } else {
        HOST::ReadMtxTools<HIGH_PRECISION_TYPE> readMtxTools(argv, dataPathA, 1);
        HOST::AutoAllocateVector<HIGH_PRECISION_TYPE> res;
        readMtxTools.loadMatrix(*originCSR);
        readMtxTools.loadRightHand(*res, *rhs, 1);
    }

    originCSR->printMatrix("origin matrix");
    rhs->printVector("origin right hand");
    std::cout << std::endl;

    std::cout << " ================= Testing CG Begin =================" << std::endl;
    /* 测试CG方法 */
    std::cout << " Test ICT-CG: " << std::endl;
    testICTPrecondCG(reorderOption, dropToleraceICT, originCSR.get(), *rhs);
    std::cout << std::endl;
    std::cout << " Test MSLR-CG: " << std::endl;
    std::cout << " --- Lanczos steps: " << lanczosSteps << ", Low-rank size: " << rk << std::endl;
    testMSLRPrecondCG(level, lanczosSteps, rk, reorderOption, lanczosType, originCSR.get(), *rhs);
    std::cout << std::endl;
    std::cout << " Test AMSED-CG: " << std::endl;
    std::cout << " --- Lanczos steps: " << lanczosSteps << ", Low-rank size: " << rk <<
        ", eigen-value correction bound: " << eigCorrectBound << std::endl;
    testAMSEDPrecondCG(level, lanczosSteps, rk, reorderOption, lanczosType, eigCorrectBound, originCSR.get(), *rhs);
    std::cout << " ================= Testing CG End =================" << std::endl;
    std::cout << std::endl;
}


int main(int argc, char* argv[]) {
    /* ======================================== 测试CG方法 ======================================== */
    /* 测试模型问题 */
    /* 二维正定问题 */
    testModelProblemCG(2, 256, 0.0, 3, 20, MatrixReorderRCM, UseRestartLanczos, 20, 0.8, 1e-3);

    /* 三维正定问题 */
    testModelProblemCG(3, 20, 0.0, 4, 100, MatrixReorderRCM, UseRestartLanczos, 16, 0.8, 1.115e-3);

    /* ======================================== 测试GMRES方法 ======================================== */
    /* 测试模型问题 */
    /* 二维不定问题 */
    testModelProblemGMRES(2, 256, 0.01, 3, 64, MatrixReorderRCM, UseRestartLanczos, 64, 0.9, 1e-3);

    /* 三维不定问题 */
    testModelProblemGMRES(3, 32, 0.04, 4, 20, MatrixReorderRCM, UseRestartLanczos, 20, 0.8, 7e-4);

    return 0;
}
