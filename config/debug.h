#ifndef PMSLS_DEV_DEBUG_H
#define PMSLS_DEV_DEBUG_H

extern "C++" {
#include "headers.h"
/* debug功能宏定义开关 */
#ifdef OPENMP_FOUND
#define _openmp_vector  //开启向量内的并行（后面逐渐替换为USE_OMP_VECTOR_FUNC）
#define USE_OMP_VECTOR_FUNC
#define _openmp        //开启矩阵向量乘/矩阵转置向量乘的并行（后面逐渐替换为USE_OMP_MATRIX_FUNC）
#define USE_OMP_MATRIX_FUNC
#define _openmp_ddcsg   // 这个建议统一为USE_OMP_PRECOND_FUNC
#define USE_OMP_PRECOND_FUNC
#endif


#ifdef NDEBUG // Release版会自动定义这个宏，Debug模式则不会定义
#define NINFO         // 取消注释则关闭所有INFO提示
#define NWARN       // 取消注释则关闭所有warn提示
#endif

#ifndef NDEBUG // Release版会自动定义这个宏，Debug模式则不会定义
#define NINFO         // 取消注释则关闭所有INFO提示
#define NWARN       // 取消注释则关闭所有warn提示
// #define NTIME       //取消注释就关闭计时操作
#endif

/* test功能宏定义开关 */
#define TEST           // 开启测试，用来控制所有test文件
#define TEST_CSR        // 开启测试，用来控制CSR矩阵的test文件
#define TEST_COO        // 开启测试，用来控制COO矩阵的test文件
#define TEST_GEN_VEC    // 开启测试，用来控制GeneralVector的test文件
#define TEST_IC         //开启测试，用来测试IC分解是否正确
#define TEST_MSLR       //开启测试，用来测试MSLR预处理的构造
#define TEST_CG         //开启测试，用来测试CG加速器的求解
#define TEST_ICCG       //开启测试，用来测试ICCG求解器
#define TEST_gmres      //开启测试，用来测试Gmres加速器
#define TEST_ILU        //开启测试，用来测试ILUT分解
/* 控制台输出文字颜色宏定义 */
// 定义宏以控制控制台输出文字的颜色 参考：https://blog.csdn.net/sexyluna/article/details/119119218
#define COLOR_NONE          "\e[0m"             //清除颜色，即之后的打印为正常输出，之前的不受影响
#define BLACK               "\e[0;30m"          //深黑
#define L_BLACK             "\e[1;30m"          //亮黑，偏灰褐
#define RED                 "\e[0;31m"          //深红，暗红
#define L_RED               "\e[1;31m"          //鲜红
#define GREEN               "\e[0;32m"          //深绿，暗绿
#define L_GREEN             "\e[1;32m"          //鲜绿
#define BROWN               "\e[0;33m"          //深黄，暗黄
#define YELLOW              "\e[1;33m"          //鲜黄
#define BLUE                "\e[0;34m"          //深蓝，暗蓝
#define L_BLUE              "\e[1;34m"          //亮蓝，偏白灰
#define PURPLE              "\e[0;35m"          //深粉，暗粉，偏暗紫
#define L_PURPLE            "\e[1;35m"          //亮粉，偏白灰
#define CYAN                "\e[0;36m"          //暗青色
#define L_CYAN              "\e[1;36m"          //鲜亮青色
#define GRAY                "\e[0;37m"          //灰色
#define WHITE               "\e[1;37m"          //白色，字体粗一点，比正常大，比bold小
#define BOLD                "\e[1m"             //白色，粗体
#define UNDERLINE           "\e[4m"             //下划线，白色，正常大小
#define BLINK               "\e[5m"             //闪烁，白色，正常大小
#define REVERSE             "\e[7m"             //反转，即字体背景为白色，字体为黑色
#define HIDE                "\e[8m"             //隐藏
#define CLEAR               "\e[2J"             //清除
#define CLRLINE             "\r\e[K"            //清除行

/* debug功能函数 */
#define DEBUG_MESSEGE_OPTION __FILE__, __func__, __LINE__


#define SHOW_INFO(message) \
        std::cout << YELLOW << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << L_CYAN << " [INFO] " << message\
        << COLOR_NONE << std::endl;


#define SHOW_WARN(message) \
    std::cout << YELLOW << "File: " << __FILE__ << ", Function: " << __func__ \
    << ", Line " << __LINE__ << ": " << L_PURPLE << "[WARN] "<< message << COLOR_NONE << std::endl;


#define SHOW_ERROR(message) \
    std::cerr << YELLOW << "File: " << __FILE__ << ", Function: " << __func__ \
    << ", Line " << __LINE__ << ": " << L_RED << "[ERROR] "<< message << COLOR_NONE << std::endl;


inline void
BAD_ALLOC_CHECK(void *pointer, const char *fileName, const char *funcName, const int errorLine, const char *message) {
    if (!pointer) {
#ifndef NDEBUG
        std::cerr << YELLOW << "File: " << fileName << ", Function: " << funcName
                  << ", Line " << errorLine << ": " << L_RED << message << COLOR_NONE << std::endl;
#endif
        exit(-1);
    }
}

inline void ERROR_CHECK(const bool errorCondition, const char *fileName, const char *funcName, const int errorLine,
                        const char *message) {
    if (errorCondition) {
#ifndef NDEBUG
        std::cerr << YELLOW << "File: " << fileName << ", Function: " << funcName
                  << ", Line " << errorLine << ": " << L_RED << "[ERROR] " << message << COLOR_NONE << std::endl;
#endif
        exit(-1);
    }
}


#define CHECK_CUBLAS(func)   { const cublasStatus_t status = (func);         \
                                if(status != CUBLAS_STATUS_SUCCESS){        \
                                    printf("CUBLAS function failed! Error occurred in file: %s, ", __FILE__);    \
                                    printf("at function: %s -> %s, line: %d\n", __func__, #func, __LINE__);       \
                                    exit(EXIT_FAILURE);                     \
                                }                                           \
                            }


#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(EXIT_FAILURE);                                                   \
    }                                                                          \
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(EXIT_FAILURE);                                                   \
    }                                                                          \
}


}


#endif //PMSLS_DEV_DEBUG_H
