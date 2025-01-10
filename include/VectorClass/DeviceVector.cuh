/*
 * @author  邓轶丹
 * @date    2024/8/12
 * @details GPU上的向量基类
 */

#ifndef DEVICEVECTOR_CUH
#define DEVICEVECTOR_CUH

#include "../../include/CUDA/StreamController.cuh"
#include "../../include/CUDA/EventController.cuh"
#include "BaseVector.h"
#include "PageLockedVector.cuh"
#include "../utils/MemoryTools/DeviceMemoryController.cuh"

/** @brief  继承自BaseVector的基类，将作为新的基类给GPU上向量，提供一些GPU上的共有方法 */
template<class ValType>
class DeviceVector : public BaseVector<ValType> {
public:
    DeviceVector() = default;

    void printVector(const char *message) const override;

    void copy(const BaseVector<ValType> &vec) override {
        copy(vec, 0, 0, vec.getLength());
    }

    void copy(const BaseVector<ValType> &vec, UINT32 src_start, UINT32 dst_start, UINT32 length) override {
#ifndef NDEBUG
        if (src_start >= vec.getLength() || src_start + length > vec.getLength() ||
            dst_start >= this->m_length || dst_start + length > this->m_length) {
            SHOW_ERROR("The start position or copied length is out-of-range!")
            std::cerr << " --- dst_start: " << dst_start << ", dst_length: " << this->m_length << ", src_start: "
                    << src_start << ", src_length: " << vec.getLength() << ", copied length: " << length
                    << std::endl;
            exit(EXIT_FAILURE);
        }
#endif
        const ValType *srcPtr = vec.getRawValPtr() + src_start;
        if (vec.getLocation() < 0) {
            // 向量在CPU上
#ifndef NDEBUG
            THROW_EXCEPTION(vec.getMemoryType() != memoryPageLocked,
                            THROW_INVALID_ARGUMENT("The copied host vector should be page-locked type!"))
#endif
            copyFromHost(srcPtr, dst_start, vec.getLength());
        } else if (vec.getLocation() == this->m_location) {
            // 向量在当前GPU上
            copyFromCurrentDevice(srcPtr, dst_start, vec.getLength());
        } else {
            // 向量在其他GPU上
            copyFromPeer(srcPtr, vec.getLocation(), dst_start, vec.getLength());
        }
    }

    ~DeviceVector() override = default;


    inline DEVICE_PTR(ValType) getThrustPtr() const {
        return PACK_RAW_PTR(BaseVector<ValType>::m_valuesPtr);
    }

    /* ============================= 同步拷贝 =============================
     * 以下操作不涉及流，各拷贝操作相对于CPU来说是同步的 */
    /** @brief 在当前GPU上拷贝向量 */
    void copyFromCurrentDevice(const DeviceVector<ValType> &dev_vec) {
        copyFromCurrentDevice(dev_vec, 0, 0, dev_vec.getLength());
    }

    void copyFromCurrentDevice(const DeviceVector<ValType> &dev_vec, UINT32 srcStartIdx, UINT32 dstStartIdx,
                               UINT32 data_length) {
        const ValType *srcPtr = dev_vec.getRawValPtr() + srcStartIdx;
        copyFromCurrentDevice(srcPtr, dstStartIdx, data_length);
    }

    void copyFromCurrentDevice(const ValType *dev_ptr, UINT32 dstStartIdx, UINT32 data_length) {
        if (data_length == 0) return;
#ifndef NDEBUG
        THROW_EXCEPTION(dstStartIdx + data_length > this->m_length,
                        THROW_INVALID_ARGUMENT("The copied vector was longer than current vec! Copy operation denied!"))
#endif
        ValType *dstPtr = this->m_valuesPtr + dstStartIdx;
        CHECK_CUDA(cudaSetDevice(this->m_location))
        DEVICE::copyFromDeviceToDevice(dev_ptr, dstPtr, data_length * sizeof(ValType), DoNotAlloc);
    }

    /** @brief 同步操作，从CPU拷贝现成的值到GPU（Host -> Device） */
    void copyFromHost(const HostVector<ValType> &host_vec) {
        copyFromHost(host_vec, 0, 0, host_vec.getLength());
    }

    void copyFromHost(const HostVector<ValType> &host_vec, UINT32 srcStartIdx, UINT32 dstStartIdx, UINT32 data_length) {
        const ValType *srcPtr = host_vec.getRawValPtr() + srcStartIdx;
        copyFromHost(srcPtr, dstStartIdx, data_length);
    }

    void copyFromHost(const ValType *host_ptr, UINT32 dstStartIdx, UINT32 data_length);

    /** @brief 同步操作，从GPU拷贝现成的值到CPU（Device -> Host） */
    void copyToHost(HostVector<ValType> &host_vec) const {
        copyToHost(host_vec, 0, 0, this->m_length);
    }

    void copyToHost(HostVector<ValType> &host_vec, UINT32 srcStartIdx, UINT32 dstStartIdx, UINT32 data_length) const {
        ValType *dstPtr = host_vec.getRawValPtr() + dstStartIdx;
        copyToHost(dstPtr, srcStartIdx, data_length);
    }

    void copyToHost(ValType *host_ptr, UINT32 srcStartIdx, UINT32 data_length) const;

    /** @brief 从其他GPU上拷贝向量到当前GPU */
    void copyFromPeer(const DeviceVector<ValType> &peer_vec) {
        copyFromPeer(peer_vec, 0, 0, peer_vec.getLength());
    }

    void copyFromPeer(const DeviceVector<ValType> &peer_vec, UINT32 srcStartIdx, UINT32 dstStartIdx,
                      UINT32 data_length) {
        const ValType *srcPtr = peer_vec.getRawValPtr() + srcStartIdx;
        copyFromPeer(srcPtr, peer_vec.getLocation(), dstStartIdx, data_length);
    }

    void copyFromPeer(const ValType *peer_ptr, INT32 peerID, UINT32 dstStartIdx, UINT32 data_length);


    /* ============================= 异步拷贝 =============================
        *各拷贝操作相对于CPU来说是异步的，每个操作由对应的流控制，不同GPU之间的操作也可以是异步的 */

    /** @brief 异步操作，从GPU上的另一个向量拷贝到本向量（Device -> Device）
     * @attention 如果拷贝向量与被拷贝向量不被同一个流控制，则当前拷贝操作不检查同步操作
     * @param [in] deviceVector:    另一个Device向量（必须和当前向量在同一个GPU上）；
     * @param [in] srcStartIdx:     待拷贝的向量在源向量中的起始下标；
     * @param [in] dstStartIdx:     待拷贝的向量在目标向量中的起始下标；
     * @param [in] dataLength:      待拷贝的向量长度；
     * @param [in] stream:          控制异步操作的流（必须和当前向量在同一个GPU上）*/
    void asyncCopyFromCurrentDevice(const DeviceVector<ValType> &deviceVector, UINT32 srcStartIdx,
                                    UINT32 dstStartIdx, UINT32 dataLength, const DEVICE::StreamController &stream) {
#ifndef NDEBUG
        ERROR_CHECK(*stream == nullptr, DEBUG_MESSEGE_OPTION,
                    "You are trying to use stream-copy, but the stream was not initialized!");
        ERROR_CHECK(deviceVector.getByteSize() == 0 || deviceVector.getRawValPtr() == nullptr, DEBUG_MESSEGE_OPTION,
                    "The src dev vector was not initialized!");
        ERROR_CHECK(this->m_byteSize == 0 || BaseVector<ValType>::m_valuesPtr == nullptr, DEBUG_MESSEGE_OPTION,
                    "The src dev vector was not initialized!");
        ERROR_CHECK(srcStartIdx + dataLength > deviceVector.getLength() || dstStartIdx + dataLength > this->m_length,
                    DEBUG_MESSEGE_OPTION, "The vector index is out-of-range!");
        THROW_EXCEPTION(this->m_location != deviceVector.m_location || this->m_location != stream.getDeviceID(),
                        THROW_LOGIC_ERROR("The src device vector was not on the current GPU or "
                            "stream was not on current GPU! Copy rejected!"))
#endif
        asyncCopyFromCurrentDevice(deviceVector.getRawValPtr(), srcStartIdx, dstStartIdx, dataLength, stream);
    }

    /** @brief 异步操作，从GPU上的另一个向量拷贝到本向量（Device -> Device）
     * @attention 裸指针指向的内存区域必须保证和当前向量在同一个GPU上 */
    void asyncCopyFromCurrentDevice(const ValType *dev_ptr, UINT32 srcStartIdx,
                                    UINT32 dstStartIdx, UINT32 dataLength, const DEVICE::StreamController &stream);

    /** @brief 从CPU拷贝向量到当前GPU
     * @attention  CPU上的内存必须以锁页方式存储 */
    void asyncCopyFromHost(const HostVector<ValType> &host_memory, UINT32 hostStartIdx, UINT32 deviceStartIdx,
                           UINT32 dataLength, const DEVICE::StreamController &stream) {
#ifndef NDEBUG
        ERROR_CHECK(*stream == nullptr, DEBUG_MESSEGE_OPTION,
                    "You are trying to use stream-copy, but the stream was not initialized!");
        ERROR_CHECK(host_memory.getByteSize() == 0, DEBUG_MESSEGE_OPTION,
                    "The host vector was not initialized!");
        ERROR_CHECK(hostStartIdx + dataLength > host_memory.getLength(), DEBUG_MESSEGE_OPTION,
                    "The vector index is out-of-range!");
        ERROR_CHECK(deviceStartIdx + dataLength > this->m_length, DEBUG_MESSEGE_OPTION,
                    "The host vector length is not compatible to device vector!");
        ERROR_CHECK(host_memory.getMemoryType() != memoryPageLocked, DEBUG_MESSEGE_OPTION,
                    "You are trying to use async-stream, but the vector was not the page-locked type!");
        THROW_EXCEPTION(this->m_location != stream.getDeviceID(),
                        THROW_LOGIC_ERROR("Stream was not on current GPU! Copy rejected!"))
#endif
        const ValType *srcPtr = host_memory.getRawValPtr() + hostStartIdx;
        asyncCopyFromHost(srcPtr, deviceStartIdx, dataLength, stream);
    }

    /** @brief 从CPU拷贝向量到当前GPU
     * @attention  裸指针指向的CPU上的内存必须以锁页方式存储，这里不做检查 */
    void asyncCopyFromHost(const ValType *host_ptr, UINT32 deviceStartIdx, UINT32 dataLength,
                           const DEVICE::StreamController &stream);

    void asyncCopyToHost(HostVector<ValType> &host_memory, UINT32 hostStartIdx, UINT32 deviceStartIdx,
                         UINT32 dataLength, const DEVICE::StreamController &stream) const {
#ifndef NDEBUG
        ERROR_CHECK(*stream == nullptr, DEBUG_MESSEGE_OPTION,
                    "You are trying to use stream-copy, but the stream was not initialized!");
        ERROR_CHECK(host_memory.getByteSize() == 0, DEBUG_MESSEGE_OPTION,
                    "The host vector was not initialized!");
        ERROR_CHECK(hostStartIdx + dataLength > host_memory.getLength(), DEBUG_MESSEGE_OPTION,
                    "The vector index is out-of-range!");
        ERROR_CHECK(deviceStartIdx + dataLength > this->m_length, DEBUG_MESSEGE_OPTION,
                    "The host vector length is not compatible to device vector!");
        ERROR_CHECK(host_memory.getMemoryType() != memoryPageLocked, DEBUG_MESSEGE_OPTION,
                    "You are trying to use async-stream, but the vector was not the page-locked type!");
        THROW_EXCEPTION(this->m_location != stream.getDeviceID(),
                        THROW_LOGIC_ERROR("Stream was not on current GPU! Copy rejected!"))
#endif
        ValType *dstPtr = host_memory.getRawValPtr() + hostStartIdx;
        asyncCopyToHost(dstPtr, deviceStartIdx, dataLength, stream);
    }


    void asyncCopyToHost(ValType *host_ptr, UINT32 deviceStartIdx, UINT32 dataLength,
                         const DEVICE::StreamController &stream) const;


    /** @brief 从其他GPU上拷贝向量到当前GPU
     * @param [in] deviceVector:        被拷贝的向量（位于另一块GPU上）
     * @param [in] srcEvent:            用于记录被拷贝向量所在的GPU上与之相关的前置事件（即拷贝操作发生前必须完成的一些异步事件）
     * @param [in] srcStartIdx:         被拷贝向量中需要拷贝的连续元素起始下标
     * @param [in] dstStartIdx:         最终拷贝到当前向量中的起始下标
     * @param [in] dataLength:          被拷贝元素总长度
     * @param [in] dstStream:           控制拷贝操作的流，必须位于当前GPU上
     * */
    void asyncCopyFromPeer(const DeviceVector<ValType> &deviceVector, const DEVICE::EventController &srcEvent,
                           UINT32 srcStartIdx, UINT32 dstStartIdx, UINT32 dataLength,
                           const DEVICE::StreamController &dstStream) {
#ifndef NDEBUG
        ERROR_CHECK(*dstStream == nullptr, DEBUG_MESSEGE_OPTION,
                    "You are trying to use dstStream-copy, but the dstStream was not initialized!");
        ERROR_CHECK(deviceVector.getByteSize() == 0, DEBUG_MESSEGE_OPTION,
                    "The src device vector was not initialized!");
        ERROR_CHECK(srcStartIdx + dataLength > deviceVector.getLength(), DEBUG_MESSEGE_OPTION,
                    "The vector index is out-of-range!");
        ERROR_CHECK(dstStartIdx + dataLength > this->m_length, DEBUG_MESSEGE_OPTION,
                    "The host vector length is not compatible to device vector!");
        THROW_EXCEPTION(this->m_location != dstStream.getDeviceID(),
                        THROW_LOGIC_ERROR("Stream was not on current GPU! Copy rejected!"))
        THROW_EXCEPTION(*srcEvent == nullptr, THROW_LOGIC_ERROR("The event on src device was not initialized!"))
        THROW_EXCEPTION(srcEvent.getDeviceID() != deviceVector.getLocation(),
                        THROW_LOGIC_ERROR("The src event was not on src device!"))
#endif

#ifndef NWARN
        if (this->m_location == deviceVector.m_location) {
            SHOW_WARN("The current device ID is equal to src device ID.")
            this->asyncCopyFromCurrentDevice(deviceVector, srcStartIdx, dstStartIdx, dataLength, dstStream);
            return;
        }
#endif
        const ValType *srcPtr = deviceVector.getRawValPtr() + srcStartIdx;
        asyncCopyFromPeer(srcPtr, deviceVector.getLocation(), srcEvent, dstStartIdx, dataLength, dstStream);
    }

    void asyncCopyFromPeer(const ValType *devPtr, INT32 srcDevID, const DEVICE::EventController &srcEvent,
                           UINT32 dstStartIdx, UINT32 dataLength,
                           const DEVICE::StreamController &dstStream);
};

template class DeviceVector<FLOAT32>;
template class DeviceVector<FLOAT64>;
template class DeviceVector<INT32>;
template class DeviceVector<UINT32>;


#endif //DEVICEVECTOR_CUH
