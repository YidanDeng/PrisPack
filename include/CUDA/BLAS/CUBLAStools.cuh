/**
 * @author  邓轶丹
 * @date    2024/4/3
 * @details 用于控制单GPU上使用多个流异步求解cublas相关的计算，所有控制变量自动释放，待计算的值通过函数参数进行传递。 */

#ifndef PMSLS_DEV_ASYCCUBLASTOOLS_H
#define PMSLS_DEV_ASYCCUBLASTOOLS_H

#include "../../../include/VectorClass/PageLockedVector.cuh"
#include "../../../include/CUDA/cuThrust.cuh"
#include "../../../include/VectorClass/SyncDeviceVector.cuh"


namespace DEVICE {
    /** @brief cublas针对稠密向量的计算工具
     * @attention: 如果将其绑定到特定的流上，就是异步计算，否则是同步计算 */
    template <typename ValType>
    class CUBLAStools {
    private:
        INT32 m_deviceID{0};
        cublasHandle_t m_handler{nullptr};
        std::shared_ptr<StreamController> m_stream; ///< 转存变量，不在内部初始化
    public:
        CUBLAStools();

        ~CUBLAStools();

        /** @brief 自定义构造函数，在当前GPU上使用若干个非默认cuda流，但是cuda流必须在外部申请 */
        explicit CUBLAStools(const std::shared_ptr<StreamController>& stream);

        inline INT32 getDeviceID() const {
            return m_deviceID;
        }

        inline void resetStream(const std::shared_ptr<StreamController>& newStream) {
            // 如果handler和newStream不在一个GPU上，先删掉旧的
            if (m_deviceID != newStream->getDeviceID() && m_handler != nullptr) {
                CHECK_CUDA(cudaSetDevice(m_deviceID))
                CHECK_CUBLAS(cublasDestroy(m_handler))
                m_handler = nullptr;
            }
            // 切换到newStream所在的GPU
            CHECK_CUDA(cudaSetDevice(newStream->getDeviceID()))
            // 如果此时的handler是空的，在当前GPU上创建，并更新GPU编号
            if (m_handler == nullptr) {
                m_deviceID = newStream->getDeviceID();
                CHECK_CUBLAS(cublasCreate(&m_handler))
            }
            // 更新流
            m_stream = newStream;
            // 将新的流和handler绑定
            CHECK_CUBLAS(cublasSetStream(m_handler, **newStream))
        }

        /* 异步(同步)操作函数 */
        void cublasVecAdd(const ValType& alpha, const DeviceVector<ValType>& dev_x, DeviceVector<ValType>& dev_y);

        ValType cublasInnerProduct(const DeviceVector<ValType>& dev_x, const DeviceVector<ValType>& dev_y);

        ValType cublasVecNorm2(const DeviceVector<ValType>& dev_vec);

        // 新增函数：稠密矩阵与向量乘法
        void cublasMatVecMul(ValType alpha, cublasOperation_t matOption, const DeviceVector<ValType>& dev_matrix,
                             const DeviceVector<ValType>& dev_vector,
                             ValType beta, DeviceVector<ValType>& dev_result, INT32 rows, INT32 cols);

        // 新增函数：稠密矩阵与矩阵乘法
        void cublasMatMatMul(ValType alpha, cublasOperation_t transa, const DeviceVector<ValType>& dev_matrixA,
                             cublasOperation_t transb, const DeviceVector<ValType>& dev_matrixB,
                             ValType beta, DeviceVector<ValType>& dev_result, INT32 rowsA, INT32 colsA, INT32 colsB);
    };


    template class CUBLAStools<FLOAT32>;
    template class CUBLAStools<FLOAT64>;
}


#endif //PMSLS_DEV_ASYCCUBLASTOOLS_H
