/**
 * @author  邓轶丹
 * @date    2024/12/24
 * @details 测试Cublas工具
 */
#include "../../../include/utils/TimerTools/CPUtimer.hpp"
// #include "../../../include/CUDA/BLAS/MultiCUBLAStools.cuh"
#include "../../../include/CUDA/BLAS/CUSPARSEtools.cuh"
#include "../../../include/CUDA/BLAS/CUBLAStools.cuh"
#include "../../../include/MatrixClass/CSRMatrix.h"
// #include "../../../include/CUDA/BLAS/MultiCUSPARSEtools.cuh"
#include "../../../include/MatrixClass/ModelProblem.h"
#include "../../../include/utils/TestTools/checkTools.hpp"
#include "../../../include/utils/TestTools/generateTools.hpp"
#include "../../../include/MatrixClass/DenseMatrix.h"

#define HIGH_PRECISION_TYPE double
#define LOW_PRECISION_TYPE double

void testDenseMatrixVectorMul() {
    /* 当矩阵行数较多，列数也较多时，有较大提升，但随着列数增加，会有精度损失 */
    UINT32 rowNum = 100000, colNum = 180;
    HOST::DenseMatrix<HIGH_PRECISION_TYPE> hostMat(HOST::DenseMatColumnFirst, rowNum, colNum, memoryPageLocked);
    HOST::PageLockedVector<HIGH_PRECISION_TYPE> hostVec(colNum);
    HOST::PageLockedVector<HIGH_PRECISION_TYPE> hostRes(rowNum);
    HOST::PageLockedVector<HIGH_PRECISION_TYPE> devResHostCopy(rowNum);

    HIGH_PRECISION_TYPE* hostVecPtr = hostVec.getRawValPtr();
    HIGH_PRECISION_TYPE* hostMatPtr = hostMat.getMatValPtr();

    HOST::generateArrayRandom1D(hostVecPtr, colNum);
    HOST::generateArrayRandom1D(hostMatPtr, rowNum * colNum);
    // hostMat.printMatrix("host matrix");
    hostRes.fillVector(0, rowNum, 0);
    CPU_TIMER_FUNC()
    CPU_TIMER_BEGIN()
    hostMat.MatVec(hostVec, hostRes);
    CPU_TIMER_END()
    std::cout << " --- host matPvec executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    hostRes.printVector("host result");

    SharedObject<DEVICE::StreamController> streamController(DEFAULT_GPU);

    DEVICE::SyncDeviceVector<HIGH_PRECISION_TYPE> deviceVec(colNum);
    DEVICE::SyncDeviceVector<HIGH_PRECISION_TYPE> deviceRes(rowNum);
    DEVICE::SyncDeviceVector<HIGH_PRECISION_TYPE> deviceMat(rowNum * colNum);
    DEVICE::CUBLAStools<HIGH_PRECISION_TYPE> cublasTools(streamController.get());

    CPU_TIMER_BEGIN()
    deviceVec.asyncCopyFromHost(hostVec, 0, 0, colNum, *streamController);
    deviceMat.asyncCopyFromHost(hostMatPtr, 0, rowNum * colNum, *streamController);
    // streamController->synchronize();
    // deviceVec.printVector("device vec");
    // deviceMat.printVector("device mat");

    cublasTools.cublasMatVecMul(1.0, CUBLAS_OP_N, deviceMat, deviceVec, 0.0, deviceRes, rowNum, colNum);
    streamController->synchronize();
    CPU_TIMER_END()
    std::cout << " --- dev matPvec executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    deviceRes.printVector("device result");
    deviceRes.asyncCopyToHost(devResHostCopy, 0, 0, rowNum, *streamController);
    streamController->synchronize();
    devResHostCopy.printVector("device result host copy");

    /* check */
    HOST::checkAnswer(devResHostCopy.getRawValPtr(), hostRes.getRawValPtr(), rowNum, "check gemv");
}

void testTransDenseMatrixVectorMul() {
    /* 对于一般的稠密矩阵向量乘法，提升程度不如串行的CPU代码 */
    UINT32 rowNum = 100000, colNum = 180;
    HOST::DenseMatrix<HIGH_PRECISION_TYPE> hostMat(HOST::DenseMatColumnFirst, rowNum, colNum, memoryPageLocked);
    HOST::PageLockedVector<HIGH_PRECISION_TYPE> hostVec(rowNum);
    HOST::PageLockedVector<HIGH_PRECISION_TYPE> hostRes(colNum);
    HOST::PageLockedVector<HIGH_PRECISION_TYPE> devResHostCopy(colNum);

    HIGH_PRECISION_TYPE* hostVecPtr = hostVec.getRawValPtr();
    HIGH_PRECISION_TYPE* hostMatPtr = hostMat.getMatValPtr();

    HOST::generateArrayRandom1D(hostVecPtr, rowNum);
    HOST::generateArrayRandom1D(hostMatPtr, rowNum * colNum);
    // hostMat.printMatrix("host matrix");
    CPU_TIMER_FUNC()
    CPU_TIMER_BEGIN()
    hostRes.fillVector(0, colNum, 0);
    hostMat.transposeVec(hostVec, hostRes);
    CPU_TIMER_END()
    std::cout << " --- host transMatPVec executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    hostRes.printVector("host result");

    SharedObject<DEVICE::StreamController> streamController(DEFAULT_GPU);

    DEVICE::SyncDeviceVector<HIGH_PRECISION_TYPE> deviceVec(rowNum);
    DEVICE::SyncDeviceVector<HIGH_PRECISION_TYPE> deviceRes(colNum);
    DEVICE::SyncDeviceVector<HIGH_PRECISION_TYPE> deviceMat(rowNum * colNum);
    DEVICE::CUBLAStools<HIGH_PRECISION_TYPE> cublasTools(streamController.get());
    CPU_TIMER_BEGIN()
    deviceVec.asyncCopyFromHost(hostVec, 0, 0, rowNum, *streamController);
    deviceMat.asyncCopyFromHost(hostMatPtr, 0, rowNum * colNum, *streamController);
    // streamController->synchronize();
    // deviceVec.printVector("device vec");
    // deviceMat.printVector("device mat");

    cublasTools.cublasMatVecMul(1.0, CUBLAS_OP_T, deviceMat, deviceVec, 0.0, deviceRes, rowNum, colNum);
    streamController->synchronize();
    CPU_TIMER_END()
    std::cout << " --- device transMatPVec executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    deviceRes.printVector("device result");
    deviceRes.asyncCopyToHost(devResHostCopy, 0, 0, colNum, *streamController);
    streamController->synchronize();
    devResHostCopy.printVector("device result host copy");

    /* check */
    HOST::checkAnswer(devResHostCopy.getRawValPtr(), hostRes.getRawValPtr(), colNum, "check trans gemv");
}

void testDenseMatrixMatMul() {
    /* 对于大型的稠密矩阵，提升较大 */
    UINT32 rowNumA = 80000, colNumA = 32, colNumB = 160;
    HOST::DenseMatrix<HIGH_PRECISION_TYPE> hostMatA(HOST::DenseMatColumnFirst, rowNumA, colNumA, memoryPageLocked);
    HOST::DenseMatrix<HIGH_PRECISION_TYPE> hostMatB(HOST::DenseMatColumnFirst, colNumA, colNumB, memoryPageLocked);
    HOST::DenseMatrix<HIGH_PRECISION_TYPE> hostMatC(HOST::DenseMatColumnFirst, rowNumA, colNumB, memoryBase);

    HOST::PageLockedVector<HIGH_PRECISION_TYPE> devMatHostCopy(rowNumA * colNumB);

    HIGH_PRECISION_TYPE* hostMatPtrA = hostMatA.getMatValPtr();
    HIGH_PRECISION_TYPE* hostMatPtrB = hostMatB.getMatValPtr();
    HIGH_PRECISION_TYPE* hostMatPtrC = hostMatC.getMatValPtr();

    HOST::generateArrayRandom1D(hostMatPtrA, rowNumA * colNumA);
    HOST::generateArrayRandom1D(hostMatPtrB, colNumA * colNumB);

    // hostMatA.printMatrix("host matrix A");
    // hostMatB.printMatrix("host matrix B");

    CPU_TIMER_FUNC()
    CPU_TIMER_BEGIN()
    hostMatA.MatMatMul(hostMatB, hostMatC);
    CPU_TIMER_END()
    std::cout << " --- host mat-mat mul executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    // hostMatC.printMatrix("host matrix C");


    SharedObject<DEVICE::StreamController> streamController(DEFAULT_GPU);

    DEVICE::SyncDeviceVector<HIGH_PRECISION_TYPE> deviceMatA(rowNumA * colNumA);
    DEVICE::SyncDeviceVector<HIGH_PRECISION_TYPE> deviceMatB(colNumA * colNumB);
    DEVICE::SyncDeviceVector<HIGH_PRECISION_TYPE> deviceMatC(rowNumA * colNumB);
    DEVICE::CUBLAStools<HIGH_PRECISION_TYPE> cublasTools(streamController.get());
    CPU_TIMER_BEGIN()
    deviceMatA.asyncCopyFromHost(hostMatPtrA, 0, rowNumA * colNumA, *streamController);
    deviceMatB.asyncCopyFromHost(hostMatPtrB, 0, colNumA * colNumB, *streamController);
    // streamController->synchronize();
    // deviceMatA.printVector("device mat A");
    // deviceMatB.printVector("device mat B");

    cublasTools.cublasMatMatMul(1.0, CUBLAS_OP_N, deviceMatA, CUBLAS_OP_N, deviceMatB, 0.0, deviceMatC, rowNumA,
                                colNumA, colNumB);
    // streamController->synchronize();
    // deviceMatC.printVector("device result");
    deviceMatC.asyncCopyToHost(devMatHostCopy, 0, 0, rowNumA * colNumB, *streamController);
    streamController->synchronize();
    CPU_TIMER_END()
    std::cout << " --- device mat-mat mul executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    devMatHostCopy.printVector("device result host copy");

    /* check */
    HOST::checkAnswer(devMatHostCopy.getRawValPtr(), hostMatPtrC, rowNumA * colNumB, "check gemm");
}


int main() {
    // testDenseMatrixVectorMul();
    // testTransDenseMatrixVectorMul();
    testDenseMatrixMatMul();
    return 0;
}
