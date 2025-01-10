/*
 * @author  邓轶丹
 * @date    2023/12/4
 * @details CPU上的计时工具
 */

/**
 * @author: Deng Yidan
 * @date: 2023/12/4
 * @details: 实现针对CPU的计时器
 */

#ifndef TESTCUDAMEMORY_CPUTIMER_H
#define TESTCUDAMEMORY_CPUTIMER_H

#include "../../../config/headers.h"
#include "../../../config/debug.h"

#define CPU_TIMER_FUNC(timerName)      HOST::CPUtimer timer_##timerName(__func__);
#define CPU_TIMER_BEGIN(timerName)     timer_##timerName.cpuTimerStart();
#define CPU_TIMER_END(timerName)       timer_##timerName.cpuTimerEnd();
#define CPU_EXEC_TIME(timerName)       timer_##timerName.computeCPUtime()


namespace HOST {
    class CPUtimer {
    private:
        std::string m_Tag;                  ///< 记录输出信息
        std::chrono::high_resolution_clock::time_point m_initTime, m_start, m_end;      ///< 一些时间戳

    public:
        CPUtimer() : m_initTime(std::chrono::high_resolution_clock::now()) {};

        explicit CPUtimer(std::string tag) : m_Tag(std::move(tag)),
                                             m_initTime(std::chrono::high_resolution_clock::now()) {}

        inline void cpuTimerStart() {
            this->m_start = std::chrono::high_resolution_clock::now();
        }

        inline void cpuTimerEnd() {
            this->m_end = std::chrono::high_resolution_clock::now();
        }

        inline double computeCPUtime() {
            return 1.0 * (std::chrono::duration_cast<std::chrono::microseconds>(this->m_end - this->m_start).count()) /
                   1000;
        }

        ~CPUtimer() {
            this->m_end = std::chrono::high_resolution_clock::now();
            auto exectime = std::chrono::duration_cast<std::chrono::microseconds>(
                    this->m_end - this->m_initTime).count();
            if (!this->m_Tag.empty())
                std::cout << L_BLUE << "[INFO] " << m_Tag << " func totally executes: " << COLOR_NONE
                          << 1.0 * exectime / 1000 << " ms." << std::endl;
        }
    };
}


#endif //TESTCUDAMEMORY_CPUTIMER_H