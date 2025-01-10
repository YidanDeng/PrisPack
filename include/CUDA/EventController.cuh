/**
 * @author  邓轶丹
 * @date    2024/5/27
 * @details GPU上的Event控制器
 */

#ifndef PMSLS_NEW_EVENTCONTROLLER_H
#define PMSLS_NEW_EVENTCONTROLLER_H

#include "../../config/headers.h"
#include "../../config/config.h"
#include "../../config/debug.h"
#include "../../config/CUDAheaders.cuh"
#include "../../include/utils/ErrorHandler.h"

namespace DEVICE {
    // 只能记录当前设备上的事件
    class EventController {
    private:
        INT32 m_deviceID{DEFAULT_GPU};
        cudaEvent_t m_event{nullptr};

    public:
        EventController() {
            CHECK_CUDA(cudaSetDevice(DEFAULT_GPU))
            CHECK_CUDA(cudaEventCreate(&m_event))
        }

        explicit EventController(const INT32& deviceID) {
            CHECK_CUDA(cudaSetDevice(deviceID))
            m_deviceID = deviceID;
            CHECK_CUDA(cudaEventCreate(&m_event))
        }

        /* CudaEven一经创建不能随意更改，一旦修改可能导致很严重的错误，所以几种复制构造和赋值操作都被禁掉了 */
        EventController(const EventController& pre) = delete;

        EventController(EventController& pre) = delete;

        EventController(EventController&& pre) = delete;

        EventController& operator=(const EventController& pre) = delete;

        EventController& operator=(EventController& pre) = delete;

        EventController& operator=(EventController&& pre) = delete;


        /**@brief 记录流的状态到本事件
         * @attention 流和事件必须在一个设备上 */
        inline void record(const cudaStream_t& stream) {
#ifndef NDEBUG
            THROW_EXCEPTION(checkEvent() != cudaSuccess, THROW_LOGIC_ERROR("Captured work was incomplete! "
                                                                           "New event can not been recorded!"))
#endif
            CHECK_CUDA(cudaSetDevice(m_deviceID))
            CHECK_CUDA(cudaEventRecord(m_event, stream))
        }


        inline cudaError_t checkEvent() {
            cudaError_t status = cudaEventQuery(m_event);
            return status;
        }

        inline INT32 getDeviceID() const {
            return m_deviceID;
        }

        ~EventController() {
            CHECK_CUDA(cudaSetDevice(m_deviceID))
            CHECK_CUDA(cudaEventDestroy(m_event))
        }


        cudaEvent_t& operator*() {
            return m_event;
        }

        const cudaEvent_t& operator*() const {
            return m_event;
        }
    };
}


#endif //PMSLS_NEW_EVENTCONTROLLER_H
