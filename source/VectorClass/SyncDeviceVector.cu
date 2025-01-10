/*
 * @author  邓轶丹
 * @date    2024/5/3
 */

#include "../../include/VectorClass/SyncDeviceVector.cuh"

namespace DEVICE {
    template<typename ValType>
    SyncDeviceVector<ValType>::~SyncDeviceVector() {
        CHECK_CUDA(cudaSetDevice(this->m_location))
        CHECK_CUDA(freeAndResetDevicePointer(this->m_valuesPtr))
#ifndef NINFO
        SHOW_INFO("Auto-free Device vector.")
#endif
    }

    template<typename ValType>
    SyncDeviceVector<ValType>::SyncDeviceVector(UINT32 length) {
#ifndef NINFO
        SHOW_INFO("Allocation for Sync Device Vector Begin!")
#endif
        this->m_length = length;
        this->m_memoryType = memoryDeviceSync;
        this->m_location = DEFAULT_GPU;
        if (this->m_length > 0) {
            this->m_byteSize = length * sizeof(ValType);
            CHECK_CUDA(cudaSetDevice(DEFAULT_GPU))
            INT32 status = allocMemoryDevice(this->m_valuesPtr, this->m_byteSize);
#ifndef NDEBUG
            if (status != memoryOptionSuccess) {
                SHOW_INFO("Allocation for Device vector failed!")
                exit(memoryAllocationFailed);
            }
#endif
        }

#ifndef NINFO
        SHOW_INFO("Allocation for Device Vector finished!")
#endif
    }


    template<typename ValType>
    SyncDeviceVector<ValType>::SyncDeviceVector(UINT32 length, INT32 deviceID) {
#ifndef NINFO
        SHOW_INFO("Allocation for Device Vector Begin!")
#endif
        this->m_location = deviceID;
        this->m_memoryType = memoryDeviceSync;
        this->m_length = length;
        if (length > 0) {
            CHECK_CUDA(cudaSetDevice(deviceID))
            this->m_byteSize = length * sizeof(ValType);
            INT32 status = allocMemoryDevice(this->m_valuesPtr, this->m_byteSize);
#ifndef NDEBUG
            if (status != memoryOptionSuccess) {
                SHOW_INFO("Allocation for Device vector failed!")
                exit(memoryAllocationFailed);
            }
#endif
        }

#ifndef NINFO
        SHOW_INFO("Allocation for Device Vector finished!")
#endif
    }


    template<typename ValType>
    SyncDeviceVector<ValType>::SyncDeviceVector(SyncDeviceVector<ValType> &&pre_vec) noexcept {
        CHECK_CUDA(cudaSetDevice(pre_vec.m_location))
        this->m_memoryType = memoryDeviceSync;
        this->m_location = pre_vec.m_location;
        this->m_byteSize = pre_vec.m_byteSize;
        this->m_length = pre_vec.m_length;
        this->m_valuesPtr = pre_vec.m_valuesPtr;
        pre_vec.m_byteSize = 0;
        pre_vec.m_length = 0;
        pre_vec.m_valuesPtr = nullptr;
    }


    template<typename ValType>
    SyncDeviceVector<ValType> &SyncDeviceVector<ValType>::operator=(SyncDeviceVector<ValType> &&pre_vec) noexcept {
        if (this == &pre_vec)
            return *this;
        if (this->m_valuesPtr) clear();
        this->m_valuesPtr = pre_vec.m_valuesPtr;
        this->m_byteSize = pre_vec.m_byteSize;
        this->m_length = pre_vec.m_length;
        this->m_location = pre_vec.m_location;
        pre_vec.m_byteSize = 0;
        pre_vec.m_length = 0;
        pre_vec.m_valuesPtr = nullptr;
        return *this;
    }

    template<typename ValType>
    void SyncDeviceVector<ValType>::resize(const UINT32 &newLen, UINT8 needReserve) {
        if (newLen != this->m_length) {
            if (needReserve == RESERVE_NO_DATA) {
                clear();
                initSpace(newLen);
            } else if (needReserve == RESERVE_DATA) {
                CHECK_CUDA(cudaSetDevice(this->m_location))
                ValType *dev_temp_ptr;
                copyFromDeviceToDevice(this->m_valuesPtr, dev_temp_ptr, this->m_byteSize, AllocDevice);
                CHECK_CUDA(freeAndResetDevicePointer(this->m_valuesPtr));
                this->m_valuesPtr = dev_temp_ptr;
            }
#ifndef NDEBUG
            else {
                THROW_INVALID_ARGUMENT("Wrong param for \"needReserve\"");
            }
#endif
        }
    }
} // DEVICE
