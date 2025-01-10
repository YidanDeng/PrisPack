/*
 * @author  邓轶丹
 * @date    2023/12/28
 * @details GPU上的计时工具
 */

#ifndef CUDATOOLS_GPUTIMER_CUH
#define CUDATOOLS_GPUTIMER_CUH

#include "../../../config/CUDAheaders.cuh"

#define GPU_TIMER_FUNC(timerName)          DEVICE::GPUtimer time_##timerName(__func__);
#define GPU_TIMER_BEGIN(timerName)         time_##timerName.gpuTimerStart();
#define GPU_TIMER_END(timerName)           time_##timerName.gpuTimerStop(); cudaDeviceSynchronize();
#define GPU_EXEC_TIME(timerName)           time_##timerName.computeGPUtime()

namespace DEVICE {


    class GPUtimer {
    private:
        std::string m_message;
        cudaEvent_t m_init{nullptr}, m_start{nullptr}, m_stop{nullptr};
        float m_execute_time{0};

    public:
        GPUtimer() {
            cudaEventCreate(&m_init);
            cudaEventCreate(&m_start);
            cudaEventCreate(&m_stop);
            cudaEventRecord(m_init, nullptr);  // 记录开始时间
        }

        explicit GPUtimer(std::string msg) : m_message(std::move(msg)) {
            cudaEventCreate(&m_init);
            cudaEventCreate(&m_start);
            cudaEventCreate(&m_stop);
            cudaEventRecord(m_init, nullptr);  // 记录开始时间
        }


        inline void gpuTimerStart() {
            cudaEventRecord(m_start, nullptr);  // 记录开始时间
            cudaEventSynchronize(m_start);
        }

        inline void gpuTimerStop() {
            cudaEventRecord(m_stop, nullptr);  // 记录结束时间
            cudaEventSynchronize(m_stop);
        }

        inline float computeGPUtime() {
            cudaEventElapsedTime(&m_execute_time, m_start, m_stop);  // 计算时间差
            return m_execute_time;
        }


        ~GPUtimer() {
            cudaEventRecord(m_stop, nullptr);  // 记录结束时间
            cudaEventSynchronize(m_stop);
            cudaEventElapsedTime(&m_execute_time, m_init, m_stop);  // 计算时间差
            cudaEventDestroy(m_start);
            cudaEventDestroy(m_stop);
            cudaEventDestroy(m_init);
            if (!this->m_message.empty())
                std::cout << L_BLUE << "[INFO] " << m_message << " func totally executes: " << COLOR_NONE
                          << m_execute_time << " ms." << std::endl;
        }
    };

}

#endif //CUDATOOLS_GPUTIMER_CUH
