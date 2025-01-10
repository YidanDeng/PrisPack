/*
 * @author  邓轶丹
 * @date    2023/12/21
 * @details 生成数据的函数
 */

#ifndef CUDATOOLS_GENERATE_TOOLS_H
#define CUDATOOLS_GENERATE_TOOLS_H

#include "../../../config/headers.h"
#include "../../../config/config.h"
#include "../../../config/debug.h"

namespace HOST {
    template<typename ValType, typename CoefType>
    void normalScaling(CoefType alpha, ValType *values, INT32 size) {
        for (INT32 i = 0; i < size; ++i) {
            values[i] *= alpha;
        }
    }

    template<typename ValType, typename CoefType>
    void normalScaling_2(CoefType alpha, ValType *values, ValType *res, INT32 size) {
        for (INT32 i = 0; i < size; ++i) {
            res[i] = values[i] * alpha;
        }
    }

    template<typename ValType>
    ValType normalSumReduction(ValType *values, INT32 size) {
        ValType res = 0;
        for (INT32 i = 0; i < size; ++i) {
            res += values[i];
        }
        return res;
    }

    /** @brief 基于标量的Kahan求和 */
    template<typename T>
    T normalKahanSum(T *values, INT32 size) {
        FLOAT64 local_sum{0.0};
        FLOAT64 local_correction{0.0};
        FLOAT64 corrected_next_term{0.0}, new_sum{0.0};
        for (INT32 i = 0; i < size; ++i) {
            corrected_next_term = values[i] + local_correction;
            new_sum = local_sum + local_correction;
            // 更新局部校正值，以进行下一次迭代
            local_correction = corrected_next_term - (new_sum - local_sum);
            local_sum = new_sum;
        }
        return local_sum + local_correction;
    }

    template<typename ValType>
    void normalAdd(ValType *values1, ValType *values2, ValType *res, INT32 size) {
        for (INT32 i = 0; i < size; ++i) {
            res[i] = values1[i] + values2[i];
        }
    }

    template<typename ValType, typename CoefType1, typename CoefType2>
    void normalAdd_2(CoefType1 alpha, ValType *values1, CoefType2 beta, ValType *values2, ValType *res, INT32 size) {
        for (INT32 i = 0; i < size; ++i) {
            res[i] = alpha * values1[i] + beta * values2[i];
        }
    }

    template<typename T>
    T normalInnerProduct(T *values1, T *values2, INT32 size) {
        T res = 0;
        for (INT32 i = 0; i < size; ++i) {
            res += values1[i] * values2[i];
        }
        return res;
    }


/** @brief 用来生成一维随机数组成的数组*/
    template<typename T>
    void generateArrayRandom1D(T *values, UINT32 size) {
        std::random_device rd;
        std::default_random_engine generator(rd());
        if (std::is_same<T, FLOAT64>::value || std::is_same<T, float>::value) {
            std::uniform_real_distribution<FLOAT32> distribution(-1.0, 1.0);
            for (INT32 i = 0; i < size; i++) {
                values[i] = distribution(generator);
            }
        } else if (std::is_same<T, INT32>::value) {
            std::uniform_int_distribution<INT32> distribution(-10, 10);
            for (INT32 i = 0; i < size; i++) {
                values[i] = distribution(generator);
            }
        } else {
#ifndef NWARN
            SHOW_WARN("The generation for test array should be applied by FLOAT64, float or INT32 type.")
#endif
            std::uniform_int_distribution<INT32> distribution(-10, 10);
            for (INT32 i = 0; i < size; i++) {
                values[i] = distribution(generator);
            }
        }
    }

    /**@brief 用来生成前半段为1，后半段为一个极小数字的向量 */
    template<typename T>
    void generateArraySteady1D(T *values, UINT32 size) {
        for (INT32 i = 0; i < size; i++) {
            if (i < size / 2) values[i] = 1;
            else values[i] = 0.000001;
        }
    }
}

#endif //CUDATOOLS_GENERATE_TOOLS_H