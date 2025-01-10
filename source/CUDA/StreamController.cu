/**
* @author  邓轶丹
 * @date    2024/3/25
 * @details GPU上的Stream控制器
 */

#include "../../include/CUDA/StreamController.cuh"

namespace DEVICE {
    StreamController::StreamController() {
        switch2Device(DEFAULT_GPU);
        CHECK_CUDA(cudaStreamCreate(&m_stream))
    }

    StreamController::StreamController(const INT32 &deviceID) {
        m_deviceID = deviceID;
        switch2Device(deviceID);
        CHECK_CUDA(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking))
    }

    StreamController::StreamController(const INT32 &deviceID, const UINT32 &streamType) {
        m_deviceID = deviceID;
        switch2Device(deviceID);
        m_streamType = streamType;
        CHECK_CUDA(cudaStreamCreateWithFlags(&m_stream, streamType))
    }

    StreamController::~StreamController() {
        switch2Device(m_deviceID);
        CHECK_CUDA(cudaStreamDestroy(m_stream))
    }


}