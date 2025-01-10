/*
 * @author  邓轶丹
 * @date    2024/5/26
 * @details 错误处理工具
 */
#include "../../include/utils/ErrorHandler.h"

thread_local std::vector<FunctionEntry> ErrorHandler::functionStack;
thread_local bool ErrorHandler::errorLogged = false;
std::mutex ErrorHandler::mutexErrorMessage;


void ErrorHandler::logError(const std::string &errorMessage, const std::string &fileName, int lineNumber) {
    if (!errorLogged) { // 当前错误日志还未打印过
        std::ostringstream oss;
        oss << L_RED << "[ERROR] " << errorMessage << COLOR_NONE << "\n";
        oss << L_PURPLE << "Occurred at: " << COLOR_NONE << fileName << ":" << lineNumber << "\n";
        oss << L_PURPLE << "Function stack:\n" << COLOR_NONE;
        for (const auto &entry: functionStack) {
            oss << "--- at " << L_CYAN << entry.funcName << COLOR_NONE << " (" << entry.fileName << ":"
                << entry.lineNumber << ")\n";
        }
        std::cerr << oss.str();
        errorLogged = true;     // 修改标记变量
    }
}