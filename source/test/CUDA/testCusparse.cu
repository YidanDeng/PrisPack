//
// Created by Hp on 24-11-3.
//

#include "../../../include/VectorClass/SyncDeviceVector.cuh"
#include "../../../include/VectorClass/ASyncDeviceVector.cuh"
#include "../../../include/CUDA/BLAS/CUBLAStools.cuh"
#include "../../../include/CUDA/BLAS/CUSPARSEtools.cuh"
#include "../../../include/MatrixClass/CSRMatrix.h"
#include "../../../include/utils/MemoryTools/SharedPtrTools.h"
#include "../../../include/MatrixClass/MatrixTools.h"
#include "../../../include/utils/TestTools/generateTools.hpp"
#include "../../../include/MatrixClass/ModelProblem.h"
#include "../../../include/Preconditioner/IncompleteCholesky.h"
#include "../../../include/Preconditioner/IncompleteCholesky.cuh"
#include "../../../include/utils/TimerTools/CPUtimer.hpp"


#define DATA_TYPE double

void testCuSparseSPMV() {
    // 生成CPU上的测试数据
    HOST::CSRMatrix<DATA_TYPE> hostCSR(memoryPageLocked);
    UINT32 blockSize = 2;
    HOST::generatePoissonCSR(hostCSR, blockSize);
    // hostCSR.printMatrix("test mat");
    UINT32 vecDim = hostCSR.getRowNum();
    HOST::PageLockedVector<DATA_TYPE> hostVec1(vecDim), hostVec2(vecDim), hostRes(vecDim);
    HOST::generateArrayRandom1D(hostVec1.getRawValPtr(), vecDim);
    HOST::generateArrayRandom1D(hostVec2.getRawValPtr(), vecDim);

    // 创建GPU上的矩阵对象
    SharedObject<DEVICE::DeviceCSRMatrix<DATA_TYPE> > deviceCsr(hostCSR.getRowNum(), hostCSR.getColNum(),
                                                                hostCSR.getNNZnum(0, hostCSR.getRowNum() - 1),
                                                                DEFAULT_GPU);
    // 拷贝CPU上的值到GPU
    deviceCsr->copyMatFromHost(hostCSR);
    // 打印GPU上的矩阵，查看值是否都已正确拷贝
    deviceCsr->printMatrix("dev CSR");
    SharedObject<DEVICE::SyncDeviceVector<DATA_TYPE> > devVec1(vecDim, DEFAULT_GPU),
            devVec2(vecDim, DEFAULT_GPU), devRes(vecDim, DEFAULT_GPU);
    devVec1->copyFromHost(hostVec1);
    devVec1->printVector("dev vec 1");
    devVec2->copyFromHost(hostVec2);
    devVec2->printVector("dev vec 2");

    // 创建CSR矩阵的Descriptor
    SharedObject<DEVICE::CusparseCSRDescriptor<DATA_TYPE> > devMatDesc(deviceCsr.get());

    // 创建device vector的Descriptor
    SharedObject<DEVICE::CusparseDnVectorDescriptor<DATA_TYPE> > devVecDesc1(devVec1.get()),
            devVecDesc2(devVec2.get()), devResDesc(devRes.get());

    // 创建SpMV
    SharedObject<DEVICE::CusparseHandler> handler(DEFAULT_GPU);
    DEVICE::CusparseCsrSpMV<DATA_TYPE> devSpMV(devMatDesc.get(), handler.get());
    // 计算第一个矩阵向量乘法
    devSpMV.csrMultiplyVec(devVecDesc1.get(), devResDesc.get());
    devRes->printVector("dev result 1");
    hostCSR.MatPVec(hostVec1, hostRes);
    hostRes.printVector("host result 1");
    devSpMV.csrMultiplyVec(devVecDesc2.get(), devResDesc.get());
    hostCSR.MatPVec(hostVec2, hostRes);
    hostRes.printVector("host result 2");
    devRes->printVector("dev result 2");
}

void testCusparseSpSV() {
    INT32 diagDim = 512;
    SharedObject<HOST::CSRMatrix<DATA_TYPE> > hostCSR(memoryPageLocked);
    HOST::generatePoissonCSR(*hostCSR, diagDim); // 生成diagDim^2 x diagDim^2矩阵
    hostCSR->printMatrix("test mat HOST");
    HOST::AutoAllocateVector<DATA_TYPE> hostVec(hostCSR->getRowNum(), memoryPageLocked);
    hostVec->fillVector(0, hostCSR->getRowNum(), 1);
    /* 计算预条件 */
    SharedObject<HOST::IncompleteCholesky<DATA_TYPE> > testIC(hostCSR.get(), 1e-4);
    testIC->setup();
    SharedObject<HOST::CSRMatrix<DATA_TYPE> > icTransL = testIC->getTransL();

    /* 在GPU上创建向量和矩阵，并从CPU拷贝值到GPU */
    SharedObject<DEVICE::DeviceCSRMatrix<DATA_TYPE> > devMat(icTransL->getRowNum(), icTransL->getColNum(),
                                                             icTransL->getNNZnum(0, icTransL->getRowNum() - 1),
                                                             DEFAULT_GPU);
    SharedObject<DEVICE::SyncDeviceVector<DATA_TYPE> > devVecIN(icTransL->getRowNum(), DEFAULT_GPU);
    SharedObject<DEVICE::SyncDeviceVector<DATA_TYPE> > devVecOUT(icTransL->getRowNum(), DEFAULT_GPU);
    SharedObject<DEVICE::StreamController> stream(DEFAULT_GPU);
    devMat->asyncCopyFromHost(*icTransL, *stream);
    stream->synchronize();
    devMat->printMatrix("dev mat");

    SharedObject<DEVICE::CusparseHandler> cuHandler(stream.get());
    SharedObject<DEVICE::CusparseCSRDescriptor<DATA_TYPE> > matDesc(devMat.get());
    SharedObject<DEVICE::CusparseDnVectorDescriptor<DATA_TYPE> > vecDescIN(devVecIN.get());
    SharedObject<DEVICE::CusparseDnVectorDescriptor<DATA_TYPE> > vecDescOUT(devVecOUT.get());
    DEVICE::CusparseCSRTriSolve<DATA_TYPE> spsvLower(matDesc.get(),
                                                     CUSPARSE_FILL_MODE_UPPER,
                                                     CUSPARSE_DIAG_TYPE_NON_UNIT,
                                                     cuHandler.get());
    DEVICE::CusparseCSRTriSolve<DATA_TYPE> spsvUpper(matDesc.get(),
                                                     CUSPARSE_FILL_MODE_UPPER,
                                                     CUSPARSE_DIAG_TYPE_NON_UNIT,
                                                     cuHandler.get());
    /* warmup */
    spsvLower.csrTriSolve(CUSPARSE_OPERATION_TRANSPOSE, vecDescIN.get(), vecDescOUT.get());
    spsvUpper.csrTriSolve(CUSPARSE_OPERATION_NON_TRANSPOSE, vecDescOUT.get(), vecDescIN.get());
    devVecIN->asyncCopyFromHost(*hostVec, 0, 0, icTransL->getRowNum(), *stream);
    CPU_TIMER_FUNC()
    CPU_TIMER_BEGIN()
    // for (INT32 i = 0; i < 100; ++i) {
    spsvLower.csrTriSolve(CUSPARSE_OPERATION_TRANSPOSE, vecDescIN.get(), vecDescIN.get());
    spsvUpper.csrTriSolve(CUSPARSE_OPERATION_NON_TRANSPOSE, vecDescIN.get(), vecDescIN.get());
    // }
    stream->synchronize();
    CPU_TIMER_END()
    devVecIN->printVector("MInvSolve(GPU)");
    std::cout << " --- MInvSolve(GPU) executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    CPU_TIMER_BEGIN()
    // for (INT32 i = 0; i < 100; ++i) {
    hostVec->fillVector(0, hostVec.getLength(), 1);
    testIC->MInvSolve(*hostVec);
    // }
    CPU_TIMER_END()
    hostVec.printVector("MInvSolve(CPU)");
    std::cout << " --- MInvSolve(CPU) executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    HOST::AutoAllocateVector<DATA_TYPE> hostDevRes(hostCSR->getRowNum(), memoryPageLocked);
    devVecIN->copyToHost(*hostDevRes, 0, 0, hostCSR->getRowNum());
    hostVec.add(-1, *hostDevRes);
    std::cout << "--- result residual between CPU and GPU: " << hostVec->norm_2() << std::endl;
}

void testCusparseSpSV_2() {
    /* 在CPU端生成数据 */
    INT32 diagDim = 1024;
    SharedObject<HOST::CSRMatrix<DATA_TYPE> > hostCSR(memoryPageLocked);
    HOST::generatePoissonCSR(*hostCSR, diagDim); // 生成diagDim^2 x diagDim^2矩阵
    hostCSR->printMatrix("test mat HOST");
    HOST::AutoAllocateVector<DATA_TYPE> hostVec(hostCSR->getRowNum(), memoryPageLocked);
    hostVec->fillVector(0, hostCSR->getRowNum(), 1);

    /* 初始化流控制器和Cusparse Handler*/
    SharedObject<DEVICE::StreamController> cuStream(DEFAULT_GPU);
    SharedObject<DEVICE::CusparseHandler> cuHandler(cuStream.get());

    /* 初始化GPU上的存储空间 */
    SharedObject<DEVICE::SyncDeviceVector<DATA_TYPE> > devVec(hostVec.getLength(), DEFAULT_GPU);

    /* 初始化CusparseDnVecDescriptor*/
    DEVICE::CusparseDnVectorDescriptor<DATA_TYPE> devDesc(devVec.get());

    /* 初始化GPU上向量的初始值 */
    devVec->asyncCopyFromHost(*hostVec, 0, 0, hostVec.getLength(), *cuStream);

    /* 初始化GPU上的IC预条件，并warmup */
    DEVICE::IncompleteCholesky<DATA_TYPE> devICT(hostCSR.get(), 1e-4, SPSV_COMPUTE_ON_DEVICE, cuHandler.get());
    devICT.setup();
    devICT.warmup(devDesc);

    /* 计算 GPU上的 y = M^{-1} x */
    CPU_TIMER_FUNC()
    CPU_TIMER_BEGIN()
    devICT.MInvSolve(devDesc);
    cuStream->synchronize();
    CPU_TIMER_END()
    std::cout << " --- MInvSolve(Device) executes: " << CPU_EXEC_TIME() << " ms." << std::endl;

    /* 打印GPU上的答案 */
    devVec->printVector("MInvSolve(Device)");

    /* 在CPU上再算一遍答案 */
    devICT.setComputeType(SPSV_COMPUTE_ON_HOST);
    CPU_TIMER_BEGIN()
    devICT.MInvSolve(*hostVec);
    CPU_TIMER_END()
    std::cout << " --- MInvSolve(Host) executes: " << CPU_EXEC_TIME() << " ms." << std::endl;
    hostVec.printVector("MInvSolve(Host)");

    /* 验证答案 */
    HOST::AutoAllocateVector<DATA_TYPE> hostDevRes(hostCSR->getRowNum(), memoryPageLocked);
    devVec->copyToHost(*hostDevRes, 0, 0, hostCSR->getRowNum());
    hostVec.add(-1, *hostDevRes);
    std::cout << "--- result residual between CPU and GPU: " << hostVec->norm_2() << std::endl;
}


int main() {
    // testCuSparseSPMV();
    // testCusparseSpSV();
    testCusparseSpSV_2();
    return 0;
}
