/*
 * @author  邓轶丹
 * @date    2024/8/12
 */
#include "../../include/VectorClass/DeviceVector.cuh"

template<class ValType>
void DeviceVector<ValType>::printVector(const char *message) const {
    std::cout << L_GREEN << "[Device " << BaseVector<ValType>::m_location << "]" << L_PURPLE << "(" << message
            << ")" << COLOR_NONE;
    CHECK_CUDA(cudaSetDevice(BaseVector<ValType>::m_location))
    thrust::device_ptr<ValType> valPtr = PACK_RAW_PTR(BaseVector<ValType>::m_valuesPtr);
    UINT32 actualLength = this->m_length <= 30 ? this->m_length : 30;
    for (UINT32 i = 0; i < actualLength; ++i) {
        std::cout << "  " << valPtr[i];
    }
    if (this->m_length > actualLength) std::cout << L_CYAN << " ... (the rest values were folded)" << COLOR_NONE;
    std::cout << std::endl;
}

template<class ValType>
void DeviceVector<ValType>::copyFromHost(const ValType *host_ptr, UINT32 dstStartIdx, UINT32 data_length) {
    if (data_length == 0) return;
    size_t copyBytes = data_length * sizeof(ValType);
#ifndef NDEBUG
    if (!BaseVector<ValType>::m_valuesPtr || this->m_length == 0 || this->m_byteSize == 0) {
        SHOW_ERROR("The vector memory space was not allocated!")
        exit(memoryAllocationFailed);
    }
    if (this->m_length < data_length || this->m_byteSize < copyBytes) {
        SHOW_ERROR("The vector memory space was not enough!")
        exit(EXIT_FAILURE);
    }
#endif
    ValType *dstPtr = this->m_valuesPtr + dstStartIdx;
    CHECK_CUDA(cudaSetDevice(this->m_location))
    DEVICE::copyFromHostToDevice(host_ptr, dstPtr, copyBytes, DoNotAlloc);
}

template<class ValType>
void DeviceVector<ValType>::copyToHost(ValType *host_ptr, UINT32 srcStartIdx, UINT32 data_length) const {
    if (data_length == 0) return;
    size_t copyBytes = data_length * sizeof(ValType);
#ifndef NDEBUG
    if (!BaseVector<ValType>::m_valuesPtr || this->m_length == 0 || this->m_byteSize == 0) {
        SHOW_ERROR("The vector memory space was not allocated!")
        exit(memoryAllocationFailed);
    }
    if (this->m_length < data_length || this->m_byteSize < copyBytes) {
        SHOW_ERROR("The vector memory space was not enough!")
        exit(EXIT_FAILURE);
    }
#endif
    CHECK_CUDA(cudaSetDevice(this->m_location))
    const ValType *srcPtr = this->m_valuesPtr + srcStartIdx;
    DEVICE::copyFromDeviceToHost(srcPtr, host_ptr, copyBytes, DoNotAlloc);
}

template<class ValType>
void DeviceVector<ValType>::copyFromPeer(const ValType *peer_ptr, INT32 peerID, UINT32 dstStartIdx,
                                         UINT32 data_length) {
    if (data_length == 0) return;
    size_t copyBytes = data_length * sizeof(ValType);
#ifndef NDEBUG
    if (!BaseVector<ValType>::m_valuesPtr || this->m_length == 0 || this->m_byteSize == 0) {
        SHOW_ERROR("The vector memory space was not allocated!")
        exit(memoryAllocationFailed);
    }
    if (this->m_length < data_length || this->m_byteSize < copyBytes) {
        SHOW_ERROR("The vector memory space was not enough!")
        exit(EXIT_FAILURE);
    }
#endif
    CHECK_CUDA(cudaSetDevice(this->m_location))
    ValType *dstPtr = this->m_valuesPtr + dstStartIdx;
    CHECK_CUDA(cudaMemcpyPeer(dstPtr, this->m_location, peer_ptr, peerID, copyBytes))
}

template<class ValType>
void DeviceVector<ValType>::asyncCopyFromCurrentDevice(const ValType *dev_ptr, UINT32 srcStartIdx, UINT32 dstStartIdx,
                                                       UINT32 dataLength, const DEVICE::StreamController &stream) {
    const ValType *src_dev_ptr = dev_ptr + srcStartIdx;
    ValType *dst_dev_ptr = this->m_valuesPtr + dstStartIdx;
    UINT32 byte_size = dataLength * sizeof(ValType);
    CHECK_CUDA(cudaSetDevice(this->m_location))
    CHECK_CUDA(cudaMemcpyAsync(dst_dev_ptr, src_dev_ptr, byte_size, cudaMemcpyDeviceToDevice, *stream))
}

template<class ValType>
void DeviceVector<ValType>::asyncCopyFromHost(const ValType *host_ptr, UINT32 deviceStartIdx,
                                              UINT32 dataLength, const DEVICE::StreamController &stream) {
    ValType *dst_dev_ptr = BaseVector<ValType>::m_valuesPtr + deviceStartIdx;
    UINT32 byte_size = dataLength * sizeof(ValType);
    CHECK_CUDA(cudaSetDevice(this->m_location))
    CHECK_CUDA(cudaMemcpyAsync(dst_dev_ptr, host_ptr, byte_size, cudaMemcpyHostToDevice, *stream))
}

template<class ValType>
void DeviceVector<ValType>::asyncCopyToHost(ValType *host_ptr,  UINT32 deviceStartIdx,
                                            UINT32 dataLength, const DEVICE::StreamController &stream) const {
    ValType *src_dev_ptr = BaseVector<ValType>::m_valuesPtr + deviceStartIdx;
    UINT32 byte_size = dataLength * sizeof(ValType);
    CHECK_CUDA(cudaSetDevice(this->m_location))
    CHECK_CUDA(cudaMemcpyAsync(host_ptr, src_dev_ptr, byte_size, cudaMemcpyDeviceToHost, *stream))
}

template<class ValType>
void DeviceVector<ValType>::asyncCopyFromPeer(const ValType *devPtr, INT32 srcDevID,
                                              const DEVICE::EventController &srcEvent,
                                              UINT32 dstStartIdx, UINT32 dataLength,
                                              const DEVICE::StreamController &dstStream) {
    ValType *dst_ptr = BaseVector<ValType>::m_valuesPtr + dstStartIdx;
    UINT32 byte_size = dataLength * sizeof(ValType);
    CHECK_CUDA(cudaSetDevice(this->m_location))
    // peer access 启动需要在外部调用，用MultiDeviceStream类的对应函数，这里只是检测能否使用peer access
    INT32 canAccessPeer;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccessPeer, srcDevID,this->m_location));
    // 等待前置任务完成（srcEvent记录前置任务），否则async拷贝可能拷贝到不正确的值（当前GPU上的流和源GPU上的流是异步执行的）
    CHECK_CUDA(cudaStreamWaitEvent(*dstStream, *srcEvent, cudaEventWaitDefault))
    if (canAccessPeer) {
        CHECK_CUDA(cudaMemcpyPeerAsync(dst_ptr, this->m_location, devPtr,srcDevID, byte_size,
            *dstStream))
    }
#ifndef NWARN
    else {
        // 如果不支持peer access，就必须通过CPU进行中转，即：发送端GPU -> CPU -> 接收端GPU
        SHOW_WARN("Peer access failed, try to transform data via Host...")
        HOST::PageLockedVector<ValType> temp(dataLength);
        CHECK_CUDA(cudaMemcpyAsync(&temp[0], devPtr, byte_size, cudaMemcpyDeviceToHost, *dstStream))
        CHECK_CUDA(cudaMemcpyAsync(dst_ptr, &temp[0], byte_size, cudaMemcpyHostToDevice, *dstStream))
    }
#endif
}
