/*
 * @author  邓轶丹
 * @date    2023/12/25
 * @details 用来检查计算答案是否满足精度要求的一些函数
 */

#ifndef CUDATOOLS_CHECK_TOOLS_H
#define CUDATOOLS_CHECK_TOOLS_H

#include "../../../config/headers.h"
#include "../../../config/config.h"
#include "../../../config/debug.h"

namespace HOST {
    template<typename ValType>
    inline void checkAnswer(ValType *values, ValType *res, UINT32 size, const char *message) {  // 这里无需返回值，第一遍写成bool了，导致程序卡死
        volatile bool isCorrect = true;     // 使用volatile关键字强制要求编译器保留该值，避免开启最高级别优化时被移除，导致cout陷入死循环
        FLOAT64 epsilon = std::is_same<ValType, FLOAT64>::value ? 1e-10 : 1e-6;  // float最高精度到1e-6，1e-7或更高精度将判断错误
        FLOAT64 total_err = 0.0, error;
        for (INT32 i = 0; i < size; ++i) {
            error = values[i] - res[i];
            total_err += fabs(error);
            if (fabs(error) > epsilon) {
                isCorrect = false;
                std::cout << L_RED << "[ERROR]" << i << " -- wrong: " << std::setprecision(12) << values[i]
                          << ", correct: " << res[i]
                          << COLOR_NONE << std::endl;
            }
            if (total_err > epsilon) {
                isCorrect = false;
                std::cout << L_RED << "[ERROR]" << i
                          << " -- The total error cannot meet the accuracy requirement! error: "
                          << total_err << COLOR_NONE << std::endl;
                break;
            }
        }
        // 清除所有与域相关，与基数相关，与浮点相关的设置
        std::cout.unsetf(std::ios::adjustfield | std::ios::basefield | std::ios::floatfield);
        std::cout << L_PURPLE << "[INFO] " << message << (isCorrect ? " -- pass." : " -- failed!") << COLOR_NONE
                  << std::endl;
    }

    template<typename ValType>
    bool checkAnswerWithReturnValue(ValType *values, ValType *res, UINT32 size, const char *message) {
        volatile bool isCorrect = true;     // 使用volatile关键字强制要求编译器保留该值，避免开启最高级别优化时被移除，导致cout陷入死循环
        FLOAT64 epsilon = std::is_same<ValType, FLOAT64>::value ? 1e-10 : 1e-6;  // float最高精度到1e-6，1e-7或更高精度将判断错误
        FLOAT64 total_err = 0.0, error;
        for (INT32 i = 0; i < size; ++i) {
            error = values[i] - res[i];
            total_err += fabs(error);
            if (fabs(error) > epsilon) {
                isCorrect = false;
                std::cout << L_RED << "[ERROR]" << i << " -- wrong: " << std::setprecision(12) << values[i]
                          << ", correct: " << res[i]
                          << COLOR_NONE << std::endl;
            }
            if (total_err > epsilon) {
                isCorrect = false;
                std::cout << L_RED << "[ERROR]" << i
                          << " -- The total error cannot meet the accuracy requirement! error: "
                          << total_err << COLOR_NONE << std::endl;
                break;
            }
        }
        // 清除所有与域相关，与基数相关，与浮点相关的设置
        std::cout.unsetf(std::ios::adjustfield | std::ios::basefield | std::ios::floatfield);
        std::cout << L_PURPLE << "[INFO] " << message << (isCorrect ? " -- pass." : " -- failed!") << COLOR_NONE
                  << std::endl;
        return isCorrect;
    }

    template<typename ValType>
    inline INT32 checkAnswerPrecision(ValType result, ValType actual_result) {
        FLOAT64 threshold = 1.0;
        FLOAT64 error = actual_result - result;
        while (fabs(error) < threshold && threshold > 1e-10) threshold /= 10;
        return (INT32) log10(threshold) + 1;
    }

    // Calculate the 2-norm of a vector ,sum use kekan sum
    inline double vec2norm(double *x, int n) {
        double sum = 0.0;
        double c = 0.0;
        for (int i = 0; i < n; i++) {
            double num = x[i] * x[i];
            double z = num - c;
            double t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }

        return sqrt(sum);
    }

    // Calculate the 2-norm of a vector ,sum use kekan sum
    inline long double vec2norm_ld(long double *x, int n) {
        long double sum = 0.0;
        long double c = 0.0;
        for (int i = 0; i < n; i++) {
            long double num = x[i] * x[i];
            long double z = num - c;
            long double t = sum + z;
            c = (t - sum) - z;
            sum = t;
        }

        return sqrtl(sum);
    }

    inline double max_check(double *x, int n) {
        double max = DBL_MIN;
        for (int i = 0; i < n; i++) {
            double x_fabs = fabs(x[i]);
            max = max > x_fabs ? max : x_fabs;
        }
        return max;
    }

    inline long double max_check_ld(long double *x, int n) {
        long double max = DBL_MIN;
        for (int i = 0; i < n; i++) {
            long double x_fabs = fabsl(x[i]);
            max = max > x_fabs ? max : x_fabs;
        }
        return max;
    }

    // Multiply a csr matrix with a vector x, and get the resulting vector y ,sum use kekan sum
    inline void spmv(int n, int *row_ptr, int *col_idx, double *val, double *x, double *y) {
        for (int i = 0; i < n; i++) {
            y[i] = 0.0;
            double c = 0.0;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                double num = val[j] * x[col_idx[j]];
                double z = num - c;
                double t = y[i] + z;
                c = (t - y[i]) - z;
                y[i] = t;
            }
        }
    }

    // Multiply a csr matrix with a vector x, and get the resulting vector y ,sum use kekan
    // sum
    inline void spmv_ld(int n, int *row_ptr, int *col_idx, long double *val, long double *x, long double *y) {
        for (int i = 0; i < n; i++) {
            y[i] = 0.0;
            long double c = 0.0;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                long double num = val[j] * x[col_idx[j]];
                long double z = num - c;
                long double t = y[i] + z;
                c = (t - y[i]) - z;
                y[i] = t;
            }
        }
    }


    inline void check_correctness(int n, int *row_ptr, int *col_idx, double *val, double *x, double *b) {
        double *b_new = (double *) malloc(sizeof(double) * n);
        double *check_b = (double *) malloc(sizeof(double) * n);
        // double* r_b     = (double*)malloc(sizeof(double) * n);

        spmv(n, row_ptr, col_idx, val, x, b_new);
        for (int i = 0; i < n; i++) {
            check_b[i] = b_new[i] - b[i];
            // r_b[i]     = fabs(check_b[i]) / MAX((b[i]), 1e-20);
        }

        double answer1 = vec2norm(check_b, n);
        double answer2 = max_check(check_b, n);
        double answer3 = answer1 / vec2norm(b, n);
        // double answer4 = max_check(r_b, n);
        fprintf(stdout, "Check || b - Ax || 2             =  %12.6e\n", answer1);
        fprintf(stdout, "Check || b - Ax || MAX           =  %12.6e\n", answer2);
        fprintf(stdout, "Check || b - Ax || 2 / || b || 2 =  %12.6e\n", answer3);
        // fprintf(stdout, "Check MAX { |b - Ax|_i / |b_i| } =  %12.6e\n", answer4);

        free(b_new);
        free(check_b);
        // free(r_b);
    }


    inline void check_correctness_ld_d2ld(int n, int *row_ptr, int *col_idx, double *val, double *x, double *b) {
        //! Step 1: data type transformation: double -> long double
        int nnz = row_ptr[n] - row_ptr[0];
        // printf("nnz = %d\n", nnz);
        long double *val_ld = (long double *) malloc(sizeof(long double) * nnz);
        long double *x_ld = (long double *) malloc(sizeof(long double) * n);
        long double *b_ld = (long double *) malloc(sizeof(long double) * n);

        for (int i = 0; i < nnz; i++) val_ld[i] = (long double) val[i];

        for (int i = 0; i < n; i++) {
            x_ld[i] = (long double) x[i];
            b_ld[i] = (long double) b[i];
        }

        //! Step 2: Check
        long double *b_new = (long double *) malloc(sizeof(long double) * n);
        long double *check_b = (long double *) malloc(sizeof(long double) * n);
        // long double* r_b     = (long double*)malloc(sizeof(long double) * n);
        spmv_ld(n, row_ptr, col_idx, val_ld, x_ld, b_new);
        for (int i = 0; i < n; i++) {
            check_b[i] = b_new[i] - b_ld[i];
            // r_b[i]     = fabsl(check_b[i]) / MAX(fabsl(b_ld[i]), 1e-20);
        }

        long double answer1 = vec2norm_ld(check_b, n);
        long double answer2 = max_check_ld(check_b, n);
        long double answer3 = answer1 / vec2norm_ld(b_ld, n);
        // long double answer4 = max_check_ld(r_b, n);

        fprintf(stdout, "LD-Check || b - Ax || 2             =  %12.6Le\n", answer1);
        fprintf(stdout, "LD-Check || b - Ax || MAX           =  %12.6Le\n", answer2);
        fprintf(stdout, "LD-Check || b - Ax || 2 / || b || 2 =  %12.6Le\n", answer3);
        // fprintf(stdout, "LD-Check MAX { |b - Ax|_i / |b_i| } =  %12.6Le\n", answer4);

        //! Step 3: free memory
        free(val_ld);
        free(x_ld);
        free(b_ld);

        free(b_new);
        free(check_b);
        // free(r_b);
    }

    template<typename ValType>
    void checkNotANumberError(ValType * vecPtr, UINT32 vecLength) {
        for (UINT32 i = 0; i < vecLength; i++) {
            if (std::isnan(vecPtr[i])) {
                std::cerr << " --- check NaN detected at index: " << i << std::endl;
                break;
            }
        }
    }


}


#endif //CUDATOOLS_CHECK_TOOLS_H
