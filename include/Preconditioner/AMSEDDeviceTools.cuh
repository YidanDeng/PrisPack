/*
 * @author  邓轶丹
 * @date    2024/12/27
 * @details GPU上的辅助函数，用于在AMSED预条件中将Givens变换应用到矩阵的列上。
 *          论文标题：基于特征值放缩的代数多水平Schur补预条件子
 *          Title：  An Algebraic Multilevel Schur complement preconditioner based on Eigenvalue Deflation
 */

#ifndef AMSEDDEVICETOOLS_CUH
#define AMSEDDEVICETOOLS_CUH

#include "../VectorClass/DeviceVector.cuh"
#include "../CUDA/StreamController.cuh"

/** @brief 用于求V_k * Y * Q^T */
template <typename HighPrecisionType>
void transposedGivensApplied2ColumnsDEVICE(HighPrecisionType c, HighPrecisionType s,
                                           UINT32 colIdx1, UINT32 colIdx2,
                                           DeviceVector<HighPrecisionType>& devV,
                                           DeviceVector<HighPrecisionType>& devAuxV,
                                           DeviceVector<HighPrecisionType>& devZ,
                                           DeviceVector<HighPrecisionType>& devAuxZ,
                                           INT32 vectorLanczosLength,
                                           const DEVICE::StreamController& cudaStream);
#endif //AMSEDDEVICETOOLS_CUH
