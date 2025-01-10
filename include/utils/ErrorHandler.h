/*
 * @author  邓轶丹
 * @date    2024/5/25
 * @details 错误处理工具
 */

#ifndef PMSLS_NEW_ERRORHANDLER_H
#define PMSLS_NEW_ERRORHANDLER_H

#include "../../config/config.h"
#include "../../config/debug.h"
#include "../../config/headers.h"

// 表示访问容器时超出范围。例如，访问一个数组中不存在的元素。
#define THROW_OUT_OF_RANGE(message) throw std::out_of_range(message)

// 表示试图创建超出允许最大长度的对象。例如，创建一个过大的数组
#define THROW_LENGTH_ERROR(message) throw std::length_error(message)

// 表示逻辑错误异常，它通常用于提示程序内部的逻辑错误。
#define THROW_LOGIC_ERROR(message) throw std::logic_error(message)

// 表示传递给函数的参数无效。例如，传递一个负数给需要正数的函数。
#define THROW_INVALID_ARGUMENT(message) throw std::invalid_argument(message)

// 表示参数超出了函数定义的有效域。例如，对负数使用平方根函数。
#define THROW_DOMAIN_ERROR(message) throw std::domain_error(message)

// 表示产生了值域错误。通常在数值计算中使用。
#define THROW_RANGE_ERROR(message) throw std::range_error(message)

// 表示算术运算产生溢出。例如，整数加法溢出。
#define THROW_OVERFLOW_ERROR(message) throw std::overflow_error(message)

// 表示算术运算产生下溢。例如，浮点数减法下溢。
#define THROW_UNDERFLOW_ERROR(message) throw std::underflow_error(message)

// 表示内存分配失败。例如，new 操作符无法分配足够的内存。
#define THROW_BAD_ALLOC(message) throw std::bad_alloc(); std::cerr << message << std::endl;


/** @brief 一个结构体，用来记录函数信息 */
struct FunctionEntry {
    FunctionEntry(std::string fn, std::string file, int line)
            : funcName(std::move(fn)), fileName(std::move(file)), lineNumber(line) {}

    std::string funcName;
    std::string fileName;
    int lineNumber;
};


class ErrorHandler {
private:
    static thread_local std::vector<FunctionEntry> functionStack;
    static thread_local bool errorLogged;

public:
    static std::mutex mutexErrorMessage;        ///< 线程互斥锁，用来处理多线程互斥操作

    /** @brief 用来将当前函数相关信息入栈 */
    inline static void pushFunction(const std::string &funcName, const std::string &fileName, int lineNumber) {
        functionStack.emplace_back(funcName, fileName, lineNumber);
    }

    /** @brief 用来将当前函数相关信息出栈 */
    inline static void popFunction() {
        if (!functionStack.empty()) {
            functionStack.pop_back();
        }
    }

    /** @brief 用来修改标志变量，控制消息的输出次数 */
    inline static void clearErrorLogged() {
        errorLogged = false;
    }

    /** @brief 用来打印错误日志 */
    static void logError(const std::string &errorMessage, const std::string &fileName, int lineNumber);
};



// 这是针对串行程序的宏
#define TRY         ErrorHandler::clearErrorLogged(); \
                    try { \
                        ErrorHandler::pushFunction(__FUNCTION__, __FILE__, __LINE__);

#define  CATCH                  } catch (std::exception &e) { \
                                    ErrorHandler::logError(e.what(), __FILE__, __LINE__); \
                                    ErrorHandler::popFunction(); \
                                    throw; \
                                }


#define TRY_CATCH(errorFunc)    ErrorHandler::clearErrorLogged(); \
                                try { \
                                    ErrorHandler::pushFunction(__FUNCTION__, __FILE__, __LINE__); \
                                    errorFunc;                    \
                                } catch (std::exception &e) { \
                                    ErrorHandler::logError(e.what(), __FILE__, __LINE__); \
                                    ErrorHandler::popFunction(); \
                                    throw; \
                                }

// 这是针对多线程环境的宏
#define TRY_CATCH_THREAD(errorFunc, lockName)   {\
                                                std::lock_guard<std::mutex> lock##lockName(ErrorHandler::mutexErrorMessage);\
                                                ErrorHandler::clearErrorLogged(); \
                                                try { \
                                                    ErrorHandler::pushFunction(__FUNCTION__, __FILE__, __LINE__); \
                                                    errorFunc;                    \
                                                } catch (std::exception &e) { \
                                                    ErrorHandler::logError(e.what(), __FILE__, __LINE__); \
                                                    ErrorHandler::popFunction(); \
                                                    throw; \
                                                }\
                                                }
#define THROW_EXCEPTION(errorCondition, exception)   if (errorCondition){ \
                                                                            TRY_CATCH(exception) \
                                                                        }

// 以上宏的用法见测试文件

#endif //PMSLS_NEW_ERRORHANDLER_H
