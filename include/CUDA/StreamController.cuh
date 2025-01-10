/**
 * @author  邓轶丹
 * @date    2024/3/25
 * @details GPU上的Stream控制器
 */
#ifndef PMSLS_DEV_CUTOOLS_CUH
#define PMSLS_DEV_CUTOOLS_CUH


#include "../../config/headers.h"
#include "../../config/config.h"
#include "../../config/debug.h"
#include "../../config/CUDAheaders.cuh"
#include "../../include/utils/MemoryTools/UniquePtrTools.h"
#include "../../include/utils/MemoryTools/SharedPtrTools.h"


namespace DEVICE {
    class StreamController {
    private:
        INT32 m_deviceID{DEFAULT_GPU};
        cudaStream_t m_stream{nullptr};
        UINT32 m_streamType{cudaStreamDefault};

    public:
        StreamController();

        explicit StreamController(const INT32 &deviceID);

        StreamController(const INT32 &deviceID, const UINT32 &streamType);

        ~StreamController();

        StreamController(const StreamController &pre) = delete;

        StreamController(StreamController &pre) = delete;

        StreamController(StreamController &&pre) = delete;

        StreamController &operator=(const StreamController &pre) = delete;

        StreamController &operator=(StreamController &pre) = delete;

        StreamController &operator=(StreamController &&pre) = delete;

        inline cudaStream_t &operator*() {
            return m_stream;
        }

        inline const cudaStream_t &operator*() const {
            return m_stream;
        }

        inline INT32 getDeviceID() const {
            return m_deviceID;
        }

        inline UINT32 getStreamType() const {
            return m_streamType;
        }

        static inline void switch2Device(INT32 deviceID) {
            CHECK_CUDA(cudaSetDevice(deviceID))
        }

        inline void synchronize() const {
            switch2Device(m_deviceID);
            CHECK_CUDA(cudaStreamSynchronize(m_stream))
        }

        /** @brief 使当前流与指定事件同步 */
        inline void waitEvent(const cudaEvent_t &cudaEvent) {
            CHECK_CUDA(cudaSetDevice(m_deviceID))
            CHECK_CUDA(cudaStreamWaitEvent(m_stream, cudaEvent, cudaEventWaitDefault))
        }

    };



}

#endif //PMSLS_DEV_CUTOOLS_CUH
