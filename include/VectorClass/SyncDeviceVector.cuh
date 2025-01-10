/*
 * @author  邓轶丹
 * @date    2024/5/3
 * @details GPU上的向量类，显存以同步的方式分配
 */

#ifndef PMSLS_NEW_SYNCDEVICEVECTOR_CUH
#define PMSLS_NEW_SYNCDEVICEVECTOR_CUH

#include "DeviceVector.cuh"
#include "../utils/MemoryTools/DeviceMemoryController.cuh"
#include "../utils/ErrorHandler.h"
#include "PageLockedVector.cuh"
#include "../CUDA/cuThrust.cuh"

namespace DEVICE {
    template<typename ValType>
    class SyncDeviceVector final : public DeviceVector<ValType> {
    public:
        /* 构造函数与析构函数 */
        SyncDeviceVector() {
            this->m_location = DEFAULT_GPU;
            this->m_memoryType = memoryDeviceSync;
        }

        /** @brief 在默认GPU上根据指定长度初始化向量 */
        explicit SyncDeviceVector(UINT32 length);

        SyncDeviceVector(UINT32 length, INT32 deviceID);

        SyncDeviceVector(const SyncDeviceVector<ValType> &pre_vec) = delete;

        SyncDeviceVector(SyncDeviceVector<ValType> &pre_vec) = delete;

        SyncDeviceVector(SyncDeviceVector<ValType> &&pre_vec) noexcept;

        SyncDeviceVector<ValType> &operator=(const SyncDeviceVector<ValType> &pre_vec) = delete;

        SyncDeviceVector<ValType> &operator=(SyncDeviceVector<ValType> &pre_vec) = delete;

        SyncDeviceVector<ValType> &operator=(SyncDeviceVector<ValType> &&pre_vec) noexcept;

        ~SyncDeviceVector() override;

        /** @brief GPU上同步申请内存的resize函数
         * @attention CPU会等待GPU内存处理函数执行结束
         * @param  newLen [in]: 新内存的长度
         * @param [in]
         */
        void resize(const UINT32 &newLen, UINT8 needReserve) override;


        /* 其他操作函数 */
        inline cudaError_t initSpace(UINT32 data_length) {
#ifndef NDEBUG
            if (BaseVector<ValType>::m_valuesPtr != nullptr) {
                SHOW_ERROR("The device memory has already been acquired!")
                return cudaErrorAlreadyAcquired;
            }
#endif
            if (data_length == 0) return cudaSuccess;
            CHECK_CUDA(cudaSetDevice(BaseVector<ValType>::m_location))
            size_t byte_size = data_length * sizeof(ValType);
            this->m_length = data_length;
            this->m_byteSize = byte_size;
            return allocMemoryDevice(BaseVector<ValType>::m_valuesPtr, byte_size);
        }

        inline cudaError_t initSpace(UINT32 data_length, INT32 deviceID) {
#ifndef NDEBUG
            if (BaseVector<ValType>::m_valuesPtr != nullptr) {
                SHOW_ERROR("The device memory has already been acquired!")
                return cudaErrorAlreadyAcquired;
            }
#endif
            BaseVector<ValType>::m_location = deviceID;
            if (data_length == 0) return cudaSuccess;
            CHECK_CUDA(cudaSetDevice(deviceID))
            size_t byte_size = data_length * sizeof(ValType);
            this->m_length = data_length;
            this->m_byteSize = byte_size;
            return allocMemoryDevice(BaseVector<ValType>::m_valuesPtr, byte_size);
        }

        inline void clear() override {
            CHECK_CUDA(cudaSetDevice(BaseVector<ValType>::m_location))
            TRY_CATCH(freeAndResetDevicePointer(this->m_valuesPtr))
            this->m_byteSize = 0;
            this->m_length = 0;
        }
    };

    template class SyncDeviceVector<INT32>;
    template class SyncDeviceVector<UINT32>;
    template class SyncDeviceVector<FLOAT32>;
    template class SyncDeviceVector<FLOAT64>;
} // DEVICE

#endif //PMSLS_NEW_SYNCDEVICEVECTOR_CUH
