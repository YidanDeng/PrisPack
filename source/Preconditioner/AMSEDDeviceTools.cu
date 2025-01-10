/*
 * @author  邓轶丹
 * @date    2024/12/27
 */
#include "../../include/Preconditioner/AMSEDDeviceTools.cuh"

/* 为了满足CPP和CUDA混合编译只能这样分离出来 */
template <typename HighPrecisionType>
__global__ void transposedGivensApplied2ColumnsKernel(HighPrecisionType c, HighPrecisionType s,
                                                      HighPrecisionType* VColPtr1, HighPrecisionType* VColPtr2,
                                                      HighPrecisionType* auxVColPtr1, HighPrecisionType* auxVColPtr2,
                                                      UINT32 vectorLanczosLength) {
    // 使用线程索引计算行索引
    UINT32 rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // 确保不超出数组范围
    if (rowIdx < vectorLanczosLength) {
        // 更新a_i
        auxVColPtr1[rowIdx] = VColPtr1[rowIdx] * c + VColPtr2[rowIdx] * s;
        // 更新b_i
        auxVColPtr2[rowIdx] = -1.0 * VColPtr1[rowIdx] * s + VColPtr2[rowIdx] * c;
    }
}

template <typename HighPrecisionType>
void transposedGivensApplied2ColumnsDEVICE(
    HighPrecisionType c, HighPrecisionType s, UINT32 colIdx1, UINT32 colIdx2, DeviceVector<HighPrecisionType>& devV,
    DeviceVector<HighPrecisionType>& devAuxV, DeviceVector<HighPrecisionType>& devZ,
    DeviceVector<HighPrecisionType>& devAuxZ, INT32 vectorLanczosLength,
    const DEVICE::StreamController& cudaStream) {
    // 将Givens变换（转置）右乘到Lanczos向量组的列上，相当于colIdx1和colIdx2提取出来的两列[a_i, b_i] => [a_i * c + b_i * s, -a_i * s + b_i * c]
    // 设置CUDA执行配置
    int threadsPerBlock = 1024; // 根据GPU性能调整
    int blocksPerGrid = (vectorLanczosLength + threadsPerBlock - 1) / threadsPerBlock;

    UINT32 actualStartPos1 = colIdx1 * vectorLanczosLength, actualStartPos2 = colIdx2 * vectorLanczosLength;
    HighPrecisionType* devVPtr1 = devV.getRawValPtr() + actualStartPos1;
    HighPrecisionType* devVPtr2 = devV.getRawValPtr() + actualStartPos2;
    HighPrecisionType* devAuxVPtr1 = devAuxV.getRawValPtr() + actualStartPos1;
    HighPrecisionType* devAuxVPtr2 = devAuxV.getRawValPtr() + actualStartPos2;
    CHECK_CUDA(cudaSetDevice(cudaStream.getDeviceID()))
    transposedGivensApplied2ColumnsKernel<<<blocksPerGrid, threadsPerBlock, 0, *cudaStream>>>(
        c, s, devVPtr1, devVPtr2, devAuxVPtr1, devAuxVPtr2, vectorLanczosLength);

    HighPrecisionType* devZPtr1 = devZ.getRawValPtr() + actualStartPos1;
    HighPrecisionType* devZPtr2 = devZ.getRawValPtr() + actualStartPos2;
    HighPrecisionType* devAuxZPtr1 = devAuxZ.getRawValPtr() + actualStartPos1;
    HighPrecisionType* devAuxZPtr2 = devAuxZ.getRawValPtr() + actualStartPos2;
    transposedGivensApplied2ColumnsKernel<<<blocksPerGrid, threadsPerBlock, 0, *cudaStream>>>(
        c, s, devZPtr1, devZPtr2, devAuxZPtr1, devAuxZPtr2, vectorLanczosLength);
}


/* ======================================= 显式实例化 ======================================= */
template void transposedGivensApplied2ColumnsDEVICE<FLOAT32>(FLOAT32, FLOAT32, UINT32, UINT32,
                                                             DeviceVector<FLOAT32>&, DeviceVector<FLOAT32>&,
                                                             DeviceVector<FLOAT32>&, DeviceVector<FLOAT32>&, INT32,
                                                             const DEVICE::StreamController&);

template void transposedGivensApplied2ColumnsDEVICE<FLOAT64>(FLOAT64, FLOAT64, UINT32, UINT32,
                                                             DeviceVector<FLOAT64>&, DeviceVector<FLOAT64>&,
                                                             DeviceVector<FLOAT64>&, DeviceVector<FLOAT64>&,
                                                             INT32, const DEVICE::StreamController&);
