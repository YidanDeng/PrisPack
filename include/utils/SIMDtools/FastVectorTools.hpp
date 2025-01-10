/*
* @author  邓轶丹
 * @date    2023/11/10
 * @details SIMD指令操作，支持128、256、512 bit向量化操作
 *          实测编码过程有些复杂，如果是一般的向量计算，直接通过GNU编译器的向量化选项完成
 */

#ifndef CUDATOOLS_FASTVECTORTOOLS_API_H
#define CUDATOOLS_FASTVECTORTOOLS_API_H

#include "../../../config/headers.h"
#include "../../../config/config.h"
#include "../../../config/debug.h"
#include "../TimerTools/CPUtimer.hpp"

/**
 * @author Deng Yidan
 * @attention 所有向量化操作必须使用对齐的内存（linux和windows的内存申请方式不同，注意区分），否则计算值可能有误！
 **/

#if MAX_VECTOR_SIZE == 512
typedef vcl::Vec8d VCL_DOUBLE64;
typedef vcl::Vec16f VCL_FLOAT32;
typedef vcl::Vec16i VCL_INT32;
#elif  MAX_VECTOR_SIZE == 256
typedef vcl::Vec4d VCL_DOUBLE64;
typedef vcl::Vec8f VCL_FLOAT32;
typedef vcl::Vec8i VCL_INT32;
#elif  MAX_VECTOR_SIZE == 128
typedef vcl::Vec2d VCL_DOUBLE64;
typedef vcl::Vec4f VCL_FLOAT32;
typedef vcl::Vec4i VCL_INT32;
#endif

#define ALIGNED_BYTE_128 16         // 对齐字节数
#define ALIGNED_BYTE_256 32
#define ALIGNED_BYTE_512 64


namespace SIMD {
    /* ==========================[SSE: 128 bit]========================== */
    /** @brief 基于SSE指令集的向量数乘操作： values *= m_alpha */
    template<typename ValType, typename CoefType>
    void vecScalingSSE(CoefType alpha, ValType *values, int size) {
//        CPUtimer time("vecScalingSSE[value *= m_alpha]");
        // SSE指令以128字节为单位并行处理所有元素（按16字节对齐）
        int parallel_width = ALIGNED_BYTE_128 / sizeof(ValType);
        int loop = size / parallel_width;        // 开启向量并行的部分实际需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        if (std::is_same<ValType, double>::value) {
            __m128d coef = _mm_set_pd(alpha, alpha);
            __m128d *simd_vec = (__m128d *) values;
            for (int i = 0; i < loop; ++i, ++simd_vec) {
                *simd_vec = _mm_mul_pd(*simd_vec, coef);
            }
        } else if (std::is_same<ValType, float>::value) {
            __m128 coef = _mm_set_ps(alpha, alpha, alpha, alpha);
            __m128 *simd_vec = (__m128 *) values;
            for (int i = 0; i < loop; ++i, ++simd_vec) {
                *simd_vec = _mm_mul_ps(*simd_vec, coef);
            }
        } else if (std::is_same<ValType, int>::value) {
            __m128i coef = _mm_set_epi32(alpha, alpha, alpha, alpha);
            __m128i *simd_vec = (__m128i *) values;
            for (int i = 0; i < loop; ++i, ++simd_vec) {
                *simd_vec = _mm_mullo_epi32(*simd_vec, coef);
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) values[i] *= alpha;
        }
        // 处理剩下的元素
        for (int i = parallel_zone; i < size; ++i) values[i] *= alpha;
    }

    /** @brief 基于SSE指令集的向量数乘操作：values2 = values1 * m_alpha  */
    template<typename ValType, typename CoefType>
    void vecScalingSSE(CoefType alpha, ValType *values1, ValType *values2, int size) {
//        CPUtimer time("vecScalingSSE[value2 = m_alpha * value1");
        // SSE指令以128字节为单位并行处理所有元素（按16字节对齐）
        int parallel_width = ALIGNED_BYTE_128 / sizeof(ValType);
        int loop = size / parallel_width;        // 开启向量并行的部分实际需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        if (std::is_same<ValType, double>::value) {
            __m128d coef = _mm_set_pd(alpha, alpha);
            __m128d *simd_vec1 = (__m128d *) values1;
            __m128d *simd_vec2 = (__m128d *) values2;
            for (int i = 0; i < loop; ++i, ++simd_vec1, ++simd_vec2) {
                *simd_vec2 = _mm_mul_pd(*simd_vec1, coef);
            }
        } else if (std::is_same<ValType, float>::value) {
            __m128 coef = _mm_set_ps(alpha, alpha, alpha, alpha);
            __m128 *simd_vec1 = (__m128 *) values1;
            __m128 *simd_vec2 = (__m128 *) values2;
            for (int i = 0; i < loop; ++i, ++simd_vec1, ++simd_vec2) {
                *simd_vec2 = _mm_mul_ps(*simd_vec1, coef);
            }
        } else if (std::is_same<ValType, int>::value) {
            __m128i coef = _mm_set_epi32(alpha, alpha, alpha, alpha);
            __m128i *simd_vec1 = (__m128i *) values1;
            __m128i *simd_vec2 = (__m128i *) values2;
            for (int i = 0; i < loop; ++i, ++simd_vec1, ++simd_vec2) {
                *simd_vec2 = _mm_mullo_epi32(*simd_vec1, coef);
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) values2[i] = values1[i] * alpha;
        }
        // 处理剩下的元素
        for (int i = parallel_zone; i < size; ++i) values2[i] = values1[i] * alpha;
    }

    /** @brief 基于SSE指令集的向量相加操作：res += values */
    template<typename ValType>
    void vecAddSSE(ValType *values, ValType *res, int size) {
//        CPUtimer time("vecAddSSE[res += values]");     // 计时
        // SSE指令以128字节为单位并行处理所有元素（按16字节对齐）
        int parallel_width = ALIGNED_BYTE_128 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        if (std::is_same<ValType, double>::value) {
            __m128d *simd_vec = (__m128d *) values;
            __m128d *simd_res = (__m128d *) res;
            for (int i = 0; i < loop; ++i) {
                *simd_res = _mm_add_pd(*simd_vec, *simd_res);
                simd_vec++;
                simd_res++;
            }
        } else if (std::is_same<ValType, float>::value) {
            __m128 *simd_vec = (__m128 *) values;
            __m128 *simd_res = (__m128 *) res;
            for (int i = 0; i < loop; ++i) {
                *simd_res = _mm_add_ps(*simd_vec, *simd_res);
                simd_vec++;
                simd_res++;
            }
        } else if (std::is_same<ValType, int>::value) {
            __m128i *simd_vec = (__m128i *) values;
            __m128i *simd_res = (__m128i *) res;
            for (int i = 0; i < loop; ++i) {
                *simd_res = _mm_add_epi32(*simd_vec, *simd_res);
                simd_vec++;
                simd_res++;
            }
        } else {
            std::cout
                    << "[WARNING] if you want to use vector-parallelism, the values type must be double, float or int."
                    << std::endl;
            for (int i = 0; i < parallel_zone; ++i) res[i] += values[i];
        }
        for (int i = parallel_zone; i < size; ++i) res[i] += values[i];
    }


    /** @brief 基于SSE指令集的向量相加操作：res = values1 + values2 */
    template<typename ValType>
    void vecAddSSE(ValType *value1, ValType *value2, ValType *res, int size) {
//        CPUtimer time("vecAddSSE[res = value1 + value2]");
        // SSE指令以128字节为单位并行处理所有元素（按16字节对齐）
        int parallel_width = ALIGNED_BYTE_128 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;

        if (std::is_same<ValType, double>::value) {
            __m128d *simd_vec1 = (__m128d *) value1;
            __m128d *simd_vec2 = (__m128d *) value2;
            __m128d *simd_res = (__m128d *) res;
            for (int i = 0; i < loop; ++i) {
                *simd_res = _mm_add_pd(*simd_vec1, *simd_vec2);
                simd_vec1++;
                simd_vec2++;
                simd_res++;
            }
        } else if (std::is_same<ValType, float>::value) {
            __m128 *simd_vec1 = (__m128 *) value1;
            __m128 *simd_vec2 = (__m128 *) value2;
            __m128 *simd_res = (__m128 *) res;
            for (int i = 0; i < loop; ++i) {
                *simd_res = _mm_add_ps(*simd_vec1, *simd_vec2);
                simd_vec1++;
                simd_vec2++;
                simd_res++;
            }
        } else if (std::is_same<ValType, int>::value) {
            __m128i *simd_vec1 = (__m128i *) value1;
            __m128i *simd_vec2 = (__m128i *) value2;
            __m128i *simd_res = (__m128i *) res;
            for (int i = 0; i < loop; ++i) {
                *simd_res = _mm_add_epi32(*simd_vec1, *simd_vec2);
                simd_vec1++;
                simd_vec2++;
                simd_res++;
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) res[i] = value1[i] + value2[i];
        }
        for (int i = parallel_zone; i < size; ++i) res[i] = value1[i] + value2[i];
    }

    /** @brief 基于SSE指令集的向量相加操作：res = m_alpha * values1 + beta * values2 */
    template<typename ValType, typename CoefType1, typename CoefType2>
    void vecAddSSE(CoefType1 alpha, ValType *value1, CoefType2 beta, ValType *value2, ValType *res, int size) {
//        CPUtimer time("vecAddSSE[res = m_alpha * value1 + beta * value2]");
        // SSE指令以128字节为单位并行处理所有元素（按16字节对齐）
        int parallel_width = ALIGNED_BYTE_128 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;

        if (std::is_same<ValType, double>::value) {
            __m128d *simd_vec1 = (__m128d *) value1;
            __m128d *simd_vec2 = (__m128d *) value2;
            __m128d *simd_res = (__m128d *) res;
            __m128d coef1 = _mm_set_pd(alpha, alpha), coef2 = _mm_set_pd(beta, beta);
            __m128d temp_mul1, temp_mul2;
            for (int i = 0; i < loop; ++i) {
                temp_mul1 = _mm_mul_pd(coef1, *simd_vec1);
                temp_mul2 = _mm_mul_pd(coef2, *simd_vec2);
                *simd_res = _mm_add_pd(temp_mul1, temp_mul2);
                simd_vec1++;
                simd_vec2++;
                simd_res++;
            }
        } else if (std::is_same<ValType, float>::value) {
            __m128 *simd_vec1 = (__m128 *) value1;
            __m128 *simd_vec2 = (__m128 *) value2;
            __m128 *simd_res = (__m128 *) res;
            __m128 coef1 = _mm_set_ps(alpha, alpha, alpha, alpha), coef2 = _mm_set_ps(beta, beta, beta, beta);
            __m128 temp_mul1, temp_mul2;
            for (int i = 0; i < loop; ++i) {
                temp_mul1 = _mm_mul_ps(coef1, *simd_vec1);
                temp_mul2 = _mm_mul_ps(coef2, *simd_vec2);
                *simd_res = _mm_add_ps(temp_mul1, temp_mul2);
                simd_vec1++;
                simd_vec2++;
                simd_res++;
            }
        } else if (std::is_same<ValType, int>::value) {
            __m128i *simd_vec1 = (__m128i *) value1;
            __m128i *simd_vec2 = (__m128i *) value2;
            __m128i *simd_res = (__m128i *) res;
            __m128i coef1 = _mm_set_epi32(alpha, alpha, alpha, alpha), coef2 = _mm_set_epi32(beta, beta, beta, beta);
            __m128i temp_mul1, temp_mul2;
            for (int i = 0; i < loop; ++i) {
                temp_mul1 = _mm_mullo_epi32(coef1, *simd_vec1);
                temp_mul2 = _mm_mullo_epi32(coef2, *simd_vec2);
                *simd_res = _mm_add_epi32(temp_mul1, temp_mul2);
                simd_vec1++;
                simd_vec2++;
                simd_res++;
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) res[i] = alpha * value1[i] + beta * value2[i];
        }
        for (int i = parallel_zone; i < size; ++i) res[i] = alpha * value1[i] + beta * value2[i];
    }

    /** @brief 基于SSE指令集的向量约简求和操作: sum_{i=1}^{size} values_i */
    template<typename ValType>
    ValType sumReductionSSE(ValType *values, int size) {
//        CPUtimer time("sumReductionSSE[res = sum(values[:])]");
        ValType total_sum = 0;
        // SSE指令以128字节为单位并行处理所有元素（按16字节对齐）
        int parallel_width = ALIGNED_BYTE_128 / sizeof(ValType);
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;

        if (std::is_same<ValType, double>::value) {
            __m128d simd_sum = _mm_setzero_pd();// 累加和
            __m128d *p = (__m128d *) values;
            for (int i = 0; i < loop; i++) {
                simd_sum = _mm_add_pd(simd_sum, *p);
                p++;
            }
            // 合并simd向量上的答案
            for (int i = 0; i < parallel_width; ++i) total_sum += ((double *) &simd_sum)[i];
        } else if (std::is_same<ValType, float>::value) {
            __m128 simd_sum = _mm_setzero_ps();// 累加和
            __m128 *p = (__m128 *) values;
            for (int i = 0; i < loop; i++) {
                simd_sum = _mm_add_ps(simd_sum, *p);
                p++;
            }
            for (int i = 0; i < parallel_width; ++i) total_sum += ((float *) &simd_sum)[i];

        } else if (std::is_same<ValType, int>::value) {
            __m128i *p = (__m128i *) values;
            __m128i simd_sum = _mm_setzero_si128();// 累加和

            for (size_t i = 0; i < loop; i++) {
                simd_sum = _mm_add_epi32(simd_sum, *p);
                p++;
            }
            for (int i = 0; i < parallel_width; ++i) total_sum += ((int *) &simd_sum)[i];
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) total_sum += values[i];
        }
        for (int i = parallel_zone; i < size; ++i) total_sum += values[i];

        return total_sum;
    }

    /** @brief 基于SSE2指令集的向量内积操作 */
    template<typename ValType>
    ValType innerProductSSE(ValType *value1, ValType *value2, int size) {
//        CPUtimer time("innerProductSSE[values1^T values2]");
        // SSE指令以128字节为单位并行处理所有元素（按16字节对齐）
        int parallel_width = ALIGNED_BYTE_128 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;

        ValType res = 0;
        if (std::is_same<ValType, double>::value) {
            __m128d simd_inner_product = _mm_setzero_pd();
            __m128d *a = (__m128d *) value1;
            __m128d *b = (__m128d *) value2;
            for (int i = 0; i < loop; ++i, ++a, ++b) {
                __m128d temp_inner_product = _mm_mul_pd(*a, *b);
                simd_inner_product = _mm_add_pd(simd_inner_product, temp_inner_product);
            }
            // 合并simd向量上的值
            for (int i = 0; i < parallel_width; ++i) res += ((double *) &simd_inner_product)[i];
        } else if (std::is_same<ValType, float>::value) {
            __m128 simd_inner_product = _mm_setzero_ps();
            __m128 *a = (__m128 *) value1;
            __m128 *b = (__m128 *) value2;
            for (int i = 0; i < loop; ++i, ++a, ++b) {
                __m128 temp_inner_product = _mm_mul_ps(*a, *b);
                simd_inner_product = _mm_add_ps(simd_inner_product, temp_inner_product);
            }
            // 合并simd向量上的值
            for (int i = 0; i < parallel_width; ++i) res += ((float *) &simd_inner_product)[i];
        } else if (std::is_same<ValType, int>::value) {
            __m128i simd_inner_product = _mm_setzero_si128();
            __m128i *a = (__m128i *) value1;
            __m128i *b = (__m128i *) value2;
            for (int i = 0; i < loop; ++i, ++a, ++b) {
                __m128i temp_inner_product = _mm_mullo_epi32(*a, *b);
                simd_inner_product = _mm_add_epi32(simd_inner_product, temp_inner_product);
            }
            // 合并simd向量上的值
            for (int i = 0; i < parallel_width; ++i) res += ((int *) &simd_inner_product)[i];
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) res += (value1[i] * value2[i]);
        }

        for (int i = parallel_zone; i < size; ++i) res += (value1[i] * value2[i]);
        return res;
    }


/* ==========================[AVX: 256 bit]========================== */
    /** @brief 基于AVX指令集的向量数乘操作： values *= m_alpha */
    template<typename ValType, typename CoefType>
    void vecScalingAVX(CoefType alpha, ValType *values, int size) {
//        CPUtimer time("vecScalingAVX[value *= m_alpha]");
        // AVX指令以256字节为单位并行处理所有元素（按32字节对齐）
        int parallel_width = ALIGNED_BYTE_256 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        if (std::is_same<ValType, double>::value) {
            __m256d coef = _mm256_set_pd(alpha, alpha, alpha, alpha);
            __m256d *simd_vec = (__m256d *) values;
            for (int i = 0; i < loop; ++i, ++simd_vec) {
                *simd_vec = _mm256_mul_pd(*simd_vec, coef);
            }
        } else if (std::is_same<ValType, float>::value) {
            __m256 coef = _mm256_set_ps(alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha);
            __m256 *simd_vec = (__m256 *) values;
            for (int i = 0; i < loop; ++i) {
                *simd_vec = _mm256_mul_ps(*simd_vec, coef);
                simd_vec++;
            }
        } else if (std::is_same<ValType, int>::value) {
            __m256i coef = _mm256_set_epi32(alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha);
            __m256i *simd_vec = (__m256i *) values;
            for (int i = 0; i < loop; ++i) {
                *simd_vec = _mm256_mullo_epi32(*simd_vec, coef);
                simd_vec++;
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) values[i] *= alpha;
        }
        for (int i = parallel_zone; i < size; ++i) values[i] *= alpha;
    }

    /** @brief 基于AVX指令集的向量数乘操作：values2 = values1 * m_alpha  */
    template<typename ValType, typename CoefType>
    void vecScalingAVX(CoefType alpha, ValType *values1, ValType *values2, int size) {
//        CPUtimer time("vecScalingAVX[value2 = m_alpha * value1]");
        // AVX指令以256字节为单位并行处理所有元素（按32字节对齐）
        int parallel_width = ALIGNED_BYTE_256 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        if (std::is_same<ValType, double>::value) {
            __m256d coef = _mm256_set_pd(alpha, alpha, alpha, alpha);
            __m256d *simd_vec1 = (__m256d *) values1;
            __m256d *simd_vec2 = (__m256d *) values2;
            for (int i = 0; i < loop; ++i) {
                *simd_vec2 = _mm256_mul_pd(*simd_vec1, coef);
                simd_vec1++;
                simd_vec2++;
            }
        } else if (std::is_same<ValType, float>::value) {
            __m256 coef = _mm256_set_ps(alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha);
            __m256 *simd_vec1 = (__m256 *) values1;
            __m256 *simd_vec2 = (__m256 *) values2;
            for (int i = 0; i < loop; ++i) {
                *simd_vec2 = _mm256_mul_ps(*simd_vec1, coef);
                simd_vec1++;
                simd_vec2++;
            }
        } else if (std::is_same<ValType, int>::value) {
            __m256i coef = _mm256_set_epi32(alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha);
            __m256i *simd_vec1 = (__m256i *) values1;
            __m256i *simd_vec2 = (__m256i *) values2;
            for (int i = 0; i < loop; ++i) {
                *simd_vec2 = _mm256_mullo_epi32(*simd_vec1, coef);
                simd_vec1++;
                simd_vec2++;
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) values2[i] = values1[i] * alpha;
        }
        for (int i = parallel_zone; i < size; ++i) values2[i] = values1[i] * alpha;
    }

    /** @brief 基于AVX指令集的向量相加操作：res += values */
    template<typename ValType>
    void vecAddAVX(ValType *values, ValType *res, int size) {
//        CPUtimer time("vecAddAVX[res += value]");
        // AVX指令以256字节为单位并行处理所有元素（按32字节对齐）
        int parallel_width = ALIGNED_BYTE_256 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        if (std::is_same<ValType, double>::value) {
            __m256d *simd_vec = (__m256d *) values;
            __m256d *simd_res = (__m256d *) res;
            for (int i = 0; i < loop; ++i, ++simd_vec, ++simd_res)
                *simd_res = _mm256_add_pd(*simd_vec, *simd_res);
        } else if (std::is_same<ValType, float>::value) {
            __m256 *simd_vec = (__m256 *) values;
            __m256 *simd_res = (__m256 *) res;
            for (int i = 0; i < loop; ++i, ++simd_vec, ++simd_res)
                *simd_res = _mm256_add_ps(*simd_vec, *simd_res);
        } else if (std::is_same<ValType, int>::value) {
            __m256i *simd_vec = (__m256i *) values;
            __m256i *simd_res = (__m256i *) res;
            for (int i = 0; i < loop; ++i, ++simd_vec, ++simd_res)
                *simd_res = _mm256_add_epi32(*simd_vec, *simd_res);
        } else {
            std::cout
                    << "[WARNING] if you want to use vector-parallelism, the values type must be double, float or int."
                    << std::endl;
            for (int i = 0; i < parallel_zone; ++i) res[i] += values[i];
        }
        for (int i = parallel_zone; i < size; ++i) res[i] += values[i];
    }

    /** @brief 基于AVX指令集的向量相加操作：res = values1 + values2 */
    template<typename ValType>
    void vecAddAVX(ValType *value1, ValType *value2, ValType *res, int size) {
//        CPUtimer time("vecAddAVX[res = value1 + value2]");
        // AVX指令以256字节为单位并行处理所有元素（按32字节对齐）
        int parallel_width = ALIGNED_BYTE_256 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        if (std::is_same<ValType, double>::value) {
            __m256d *simd_vec1 = (__m256d *) value1;
            __m256d *simd_vec2 = (__m256d *) value2;
            __m256d *simd_res = (__m256d *) res;
            for (int i = 0; i < loop; ++i, ++simd_vec1, ++simd_vec2, ++simd_res)
                *simd_res = _mm256_add_pd(*simd_vec1, *simd_vec2);
        } else if (std::is_same<ValType, float>::value) {
            __m256 *simd_vec1 = (__m256 *) value1;
            __m256 *simd_vec2 = (__m256 *) value2;
            __m256 *simd_res = (__m256 *) res;
            for (int i = 0; i < loop; ++i, ++simd_vec1, ++simd_vec2, ++simd_res)
                *simd_res = _mm256_add_ps(*simd_vec1, *simd_vec2);
        } else if (std::is_same<ValType, int>::value) {
            __m256i *simd_vec1 = (__m256i *) value1;
            __m256i *simd_vec2 = (__m256i *) value2;
            __m256i *simd_res = (__m256i *) res;
            for (int i = 0; i < loop; ++i, ++simd_vec1, ++simd_vec2, ++simd_res)
                *simd_res = _mm256_add_epi32(*simd_vec1, *simd_vec2);
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) res[i] = value1[i] + value2[i];
        }
        for (int i = parallel_zone; i < size; ++i) res[i] = value1[i] + value2[i];
    }

    /** @brief 基于AVX指令集的向量相加操作：res = m_alpha * values1 + beta * values2 */
    template<typename ValType, typename CoefType1, typename CoefType2>
    void vecAddAVX(CoefType1 alpha, ValType *value1, CoefType2 beta, ValType *value2, ValType *res, int size) {
//        CPUtimer time("vecAddAVX[res = m_alpha * value1 + beta * value2]");
        // AVX指令以256字节为单位并行处理所有元素（按32字节对齐）
        int parallel_width = ALIGNED_BYTE_256 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        if (std::is_same<ValType, double>::value) {
            __m256d coef1 = _mm256_set_pd(alpha, alpha, alpha, alpha), coef2 = _mm256_set_pd(beta, beta, beta, beta);
            __m256d *simd_vec1 = (__m256d *) value1;
            __m256d *simd_vec2 = (__m256d *) value2;
            __m256d *simd_res = (__m256d *) res;
            __m256d temp_mul1, temp_mul2;
            for (int i = 0; i < loop; ++i, ++simd_vec1, ++simd_vec2, ++simd_res) {
                temp_mul1 = _mm256_mul_pd(coef1, *simd_vec1);
                temp_mul2 = _mm256_mul_pd(coef2, *simd_vec2);
                *simd_res = _mm256_add_pd(temp_mul1, temp_mul2);
            }
        } else if (std::is_same<ValType, float>::value) {
            __m256 coef1 = _mm256_set_ps(alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha),
                    coef2 = _mm256_set_ps(beta, beta, beta, beta, beta, beta, beta, beta);
            __m256 *simd_vec1 = (__m256 *) value1;
            __m256 *simd_vec2 = (__m256 *) value2;
            __m256 *simd_res = (__m256 *) res;
            __m256 temp_mul1, temp_mul2;
            for (int i = 0; i < loop; ++i, ++simd_vec1, ++simd_vec2, ++simd_res) {
                temp_mul1 = _mm256_mul_ps(coef1, *simd_vec1);
                temp_mul2 = _mm256_mul_ps(coef2, *simd_vec2);
                *simd_res = _mm256_add_ps(temp_mul1, temp_mul2);
            }
        } else if (std::is_same<ValType, int>::value) {
            __m256i coef1 = _mm256_set_epi32(alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha),
                    coef2 = _mm256_set_epi32(beta, beta, beta, beta, beta, beta, beta, beta);
            __m256i *simd_vec1 = (__m256i *) value1;
            __m256i *simd_vec2 = (__m256i *) value2;
            __m256i *simd_res = (__m256i *) res;
            __m256i temp_mul1, temp_mul2;
            for (int i = 0; i < loop; ++i, ++simd_vec1, ++simd_vec2, ++simd_res) {
                temp_mul1 = _mm256_mullo_epi32(coef1, *simd_vec1);
                temp_mul2 = _mm256_mullo_epi32(coef2, *simd_vec2);
                *simd_res = _mm256_add_epi32(temp_mul1, temp_mul2);
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) res[i] = alpha * value1[i] + beta * value2[i];
        }
        for (int i = parallel_zone; i < size; ++i) res[i] = alpha * value1[i] + beta * value2[i];
    }

    template<typename ValType>
    ValType sumReductionAVX(ValType *values, int size) {
//        CPUtimer time("sumReductionAVX[res = sum(values[:])]");
        // AVX指令以256字节为单位并行处理所有元素（按32字节对齐）
        int parallel_width = ALIGNED_BYTE_256 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        ValType total_sum = 0;
        if (std::is_same<ValType, double>::value) {
            __m256d simd_sum = _mm256_setzero_pd();
            __m256d *simd_vec = (__m256d *) values;
            for (size_t i = 0; i < loop; i++) {
                simd_sum = _mm256_add_pd(simd_sum, *simd_vec);
                simd_vec++;
            }
            for (int i = 0; i < parallel_width; ++i) total_sum += ((double *) &simd_sum)[i];
        } else if (std::is_same<ValType, float>::value) {
            __m256 simd_sum = _mm256_setzero_ps();
            __m256 *simd_vec = (__m256 *) values;
            for (size_t i = 0; i < loop; i++) {
                simd_sum = _mm256_add_ps(simd_sum, *simd_vec);
                simd_vec++;
            }
            for (int i = 0; i < parallel_width; ++i) total_sum += ((float *) &simd_sum)[i];
        } else if (std::is_same<ValType, int>::value) {
            __m256i simd_sum = _mm256_setzero_si256();
            __m256i *simd_vec = (__m256i *) values;
            for (size_t i = 0; i < loop; i++) {
                simd_sum = _mm256_add_epi32(simd_sum, *simd_vec);
                simd_vec++;
            }
            for (int i = 0; i < parallel_width; ++i) total_sum += ((int *) &simd_sum)[i];
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) total_sum += values[i];
        }
        for (int i = parallel_zone; i < size; ++i) total_sum += values[i];
        return total_sum;
    }

    template<typename ValType>
    ValType innerProductAVX(ValType *values1, ValType *values2, int size) {
//        CPUtimer time("innerProductAVX[values1^T values2]");
        // AVX指令以256字节为单位并行处理所有元素（按32字节对齐）
        int parallel_width = ALIGNED_BYTE_256 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        ValType total_sum = 0;
        if (std::is_same<ValType, double>::value) {
            __m256d simd_sum = _mm256_setzero_pd();
            __m256d *simd_vec1 = (__m256d *) values1;
            __m256d *simd_vec2 = (__m256d *) values2;
            __m256d temp_mul_res;
            for (int i = 0; i < loop; i++) {
                temp_mul_res = _mm256_mul_pd(*simd_vec1, *simd_vec2);
                simd_sum = _mm256_add_pd(simd_sum, temp_mul_res);
                simd_vec1++;
                simd_vec2++;
            }
            for (int i = 0; i < parallel_width; ++i) total_sum += ((double *) &simd_sum)[i];
        } else if (std::is_same<ValType, float>::value) {
            __m256 simd_sum = _mm256_setzero_ps();
            __m256 *simd_vec1 = (__m256 *) values1;
            __m256 *simd_vec2 = (__m256 *) values2;
            __m256 temp_mul_res;
            for (int i = 0; i < loop; i++) {
                temp_mul_res = _mm256_mul_ps(*simd_vec1, *simd_vec2);
                simd_sum = _mm256_add_ps(simd_sum, temp_mul_res);
                simd_vec1++;
                simd_vec2++;
            }
            for (int i = 0; i < parallel_width; ++i) total_sum += ((float *) &simd_sum)[i];
        } else if (std::is_same<ValType, int>::value) {
            __m256i simd_sum = _mm256_setzero_si256();
            __m256i *simd_vec1 = (__m256i *) values1;
            __m256i *simd_vec2 = (__m256i *) values2;
            __m256i temp_mul_res;
            for (int i = 0; i < loop; i++) {
                temp_mul_res = _mm256_mullo_epi32(*simd_vec1, *simd_vec2);
                simd_sum = _mm256_add_epi32(simd_sum, temp_mul_res);
                simd_vec1++;
                simd_vec2++;
            }
            for (int i = 0; i < parallel_width; ++i) total_sum += ((int *) &simd_sum)[i];
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) total_sum += (values1[i] * values2[i]);
        }
        for (int i = parallel_zone; i < size; ++i) total_sum += (values1[i] * values2[i]);
        return total_sum;
    }

/* ==========================[FOG: 第三方向量库]========================== */
    template<typename ValType, typename CoefType>
    void vecScalingFOG(CoefType alpha, ValType *values, int size) {
//        CPUtimer time("vecScalingFOG[value *= m_alpha]");
        // FOG向量库考虑到了不同框架之间的兼容性，所以可以直接使用512 bit的做法，就算不适配512长度的向量，也能自动转换成256 bit的向量或128 bit向量
        const int parallel_width = ALIGNED_BYTE_512 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        if (std::is_same<ValType, double>::value) {
            VCL_DOUBLE64 coef(alpha);
            VCL_DOUBLE64 simd_vec;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec.load((const double *) (values + i));
                simd_vec *= coef;
                simd_vec.store((double *) (values + i));
            }
        } else if (std::is_same<ValType, float>::value) {
            VCL_FLOAT32 coef(alpha);
            VCL_FLOAT32 simd_vec;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec.load((const float *) (values + i));
                simd_vec *= coef;
                simd_vec.store((float *) (values + i));
            }
        } else if (std::is_same<ValType, int>::value) {
            VCL_INT32 coef(alpha);
            VCL_INT32 simd_vec;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec.load((const int *) (values + i));
                simd_vec *= coef;
                simd_vec.store((int *) (values + i));
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) values[i] *= alpha;
        }
        for (int i = parallel_zone; i < size; ++i) values[i] *= alpha;
    }

    template<typename ValType, typename CoefType>
    void vecScalingFOG(CoefType alpha, ValType *values1, ValType *values2, int size) {
//        CPUtimer time("vecScalingFOG[value2 = m_alpha * value1]");
        // FOG向量库考虑到了不同框架之间的兼容性，所以可以直接使用512 bit的做法，就算不适配512长度的向量，也能自动转换成256 bit的向量或128 bit向量
        const int parallel_width = ALIGNED_BYTE_512 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        if (std::is_same<ValType, double>::value) {
            VCL_DOUBLE64 coef(alpha);
            VCL_DOUBLE64 simd_vec;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec.load((const double *) (values1 + i));
                simd_vec *= coef;
                simd_vec.store((double *) (values2 + i));
            }
        } else if (std::is_same<ValType, float>::value) {
            VCL_FLOAT32 coef(alpha);
            VCL_FLOAT32 simd_vec;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec.load((const float *) (values1 + i));
                simd_vec *= coef;
                simd_vec.store((float *) (values2 + i));
            }
        } else if (std::is_same<ValType, int>::value) {
            VCL_INT32 coef(alpha);
            VCL_INT32 simd_vec;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec.load((const int *) (values1 + i));
                simd_vec *= coef;
                simd_vec.store((int *) (values2 + i));
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) values2[i] = values1[i] * alpha;
        }
        for (int i = parallel_zone; i < size; ++i) values2[i] = values1[i] * alpha;
    }

    template<typename ValType>
    void vecAddFOG(ValType *values, ValType *res, int size) {
//        CPUtimer time("vecAddFOG[res += value]");
        // FOG向量库考虑到了不同框架之间的兼容性，所以可以直接使用512 bit的做法，就算不适配512长度的向量，也能自动转换成256 bit的向量或128 bit向量
        const int parallel_width = ALIGNED_BYTE_512 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        if (std::is_same<ValType, double>::value) {
            VCL_DOUBLE64 simd_vec, simd_res;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec.load((const double *) (values + i));
                simd_res.load((const double *) (res + i));
                simd_res += simd_vec;
                simd_res.store((double *) (res + i));
            }
        } else if (std::is_same<ValType, float>::value) {
            VCL_FLOAT32 simd_vec, simd_res;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec.load((const float *) (values + i));
                simd_res.load((const float *) (res + i));
                simd_res += simd_vec;
                simd_res.store((float *) (res + i));
            }
        } else if (std::is_same<ValType, int>::value) {
            VCL_INT32 simd_vec, simd_res;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec.load((const int *) (values + i));
                simd_res.load((const int *) (res + i));
                simd_res += simd_vec;
                simd_res.store((int *) (res + i));
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) res[i] += values[i];
        }
        for (int i = parallel_zone; i < size; ++i) res[i] += values[i];
    }

    template<typename ValType>
    void vecAddFOG(ValType *values1, ValType *values2, ValType *res, int size) {
//        CPUtimer time("vecAddFOG");
        // FOG向量库考虑到了不同框架之间的兼容性，所以可以直接使用512 bit的做法，就算不适配512长度的向量，也能自动转换成256 bit的向量或128 bit向量
        const int parallel_width = ALIGNED_BYTE_512 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        if (std::is_same<ValType, double>::value) {
            VCL_DOUBLE64 simd_vec1, simd_vec2, simd_res;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec1.load((const double *) (values1 + i));
                simd_vec2.load((const double *) (values2 + i));
                simd_res = simd_vec1 + simd_vec2;
                simd_res.store((double *) (res + i));
            }
        } else if (std::is_same<ValType, float>::value) {
            VCL_FLOAT32 simd_vec1, simd_vec2, simd_res;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec1.load((const float *) (values1 + i));
                simd_vec2.load((const float *) (values2 + i));
                simd_res = simd_vec1 + simd_vec2;
                simd_res.store((float *) (res + i));
            }
        } else if (std::is_same<ValType, int>::value) {
            VCL_INT32 simd_vec1, simd_vec2, simd_res;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec1.load((const int *) (values1 + i));
                simd_vec2.load((const int *) (values2 + i));
                simd_res = simd_vec1 + simd_vec2;
                simd_res.store((int *) (res + i));
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) res[i] = values1[i] + values2[i];
        }
        for (int i = parallel_zone; i < size; ++i) res[i] = values1[i] + values2[i];
    }


    template<typename ValType, typename CoefType1, typename CoefType2>
    void vecAddFOG(CoefType1 alpha, ValType *values1, CoefType2 beta, ValType *values2, ValType *res, int size) {
//        CPUtimer time("vecAddFOG-2");
        // FOG向量库考虑到了不同框架之间的兼容性，所以可以直接使用512 bit的做法，就算不适配512长度的向量，也能自动转换成256 bit的向量或128 bit向量
        const int parallel_width = ALIGNED_BYTE_512 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        if (std::is_same<ValType, double>::value) {
            VCL_DOUBLE64 simd_vec1, simd_vec2, simd_res, simd_coef1(alpha), simd_coef2(beta);
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec1.load((const double *) (values1 + i));
                simd_vec2.load((const double *) (values2 + i));
                simd_res = simd_coef1 * simd_vec1 + simd_coef2 * simd_vec2;
                simd_res.store((double *) (res + i));
            }
        } else if (std::is_same<ValType, float>::value) {
            VCL_FLOAT32 simd_vec1, simd_vec2, simd_res, simd_coef1(alpha), simd_coef2(beta);
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec1.load((const float *) (values1 + i));
                simd_vec2.load((const float *) (values2 + i));
                simd_res = simd_coef1 * simd_vec1 + simd_coef2 * simd_vec2;
                simd_res.store((float *) (res + i));
            }
        } else if (std::is_same<ValType, int>::value) {
            VCL_INT32 simd_vec1, simd_vec2, simd_res, simd_coef1(alpha), simd_coef2(beta);
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec1.load((const int *) (values1 + i));
                simd_vec2.load((const int *) (values2 + i));
                simd_res = simd_coef1 * simd_vec1 + simd_coef2 * simd_vec2;
                simd_res.store((int *) (res + i));
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) res[i] = alpha * values1[i] + beta * values2[i];
        }
        for (int i = parallel_zone; i < size; ++i) res[i] = alpha * values1[i] + beta * values2[i];
    }

    template<typename ValType>
    ValType sumReductionFOG(ValType *values, int size) {
//        CPUtimer time("sumReductionFOG[res = sum(values[:])]");
        // FOG向量库考虑到了不同框架之间的兼容性，所以可以直接使用512 bit的做法，就算不适配512长度的向量，也能自动转换成256 bit的向量或128 bit向量
        const int parallel_width = ALIGNED_BYTE_512 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        ValType total_sum = 0;
        if (std::is_same<ValType, double>::value) {
            VCL_DOUBLE64 simd_vec;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec.load((const double *) (values + i));
                total_sum += horizontal_add(simd_vec);
            }
        } else if (std::is_same<ValType, float>::value) {
            VCL_FLOAT32 simd_vec;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec.load((const float *) (values + i));
                total_sum += horizontal_add(simd_vec);
            }
        } else if (std::is_same<ValType, int>::value) {
            VCL_INT32 simd_vec;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec.load((const int *) (values + i));
                total_sum += horizontal_add(simd_vec);
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) total_sum += values[i];
        }
        for (int i = parallel_zone; i < size; ++i) total_sum += values[i];
        return total_sum;
    }

    template<typename ValType>
    ValType innerProductFOG(ValType *values1, ValType *values2, int size) {
        // FOG向量库考虑到了不同框架之间的兼容性，所以可以直接使用512 bit的做法，就算不适配512长度的向量，也能自动转换成256 bit的向量或128 bit向量
        const int parallel_width = ALIGNED_BYTE_512 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;        // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        ValType total_sum = 0;
        if (std::is_same<ValType, double>::value) {
            VCL_DOUBLE64 simd_vec1, simd_vec2, temp_product;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec1.load((const double *) (values1 + i));
                simd_vec2.load((const double *) (values2 + i));
                temp_product = simd_vec1 * simd_vec2;
                total_sum += horizontal_add(temp_product);
            }
        } else if (std::is_same<ValType, float>::value) {
            VCL_FLOAT32 simd_vec1, simd_vec2, temp_product;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec1.load((const float *) (values1 + i));
                simd_vec2.load((const float *) (values2 + i));
                temp_product = simd_vec1 * simd_vec2;
                total_sum += horizontal_add(temp_product);
            }
        } else if (std::is_same<ValType, int>::value) {
            VCL_INT32 simd_vec1, simd_vec2, temp_product;
            for (int i = 0; i < parallel_zone; i += parallel_width) {
                simd_vec1.load((const int *) (values1 + i));
                simd_vec2.load((const int *) (values2 + i));
                temp_product = simd_vec1 * simd_vec2;
                total_sum += horizontal_add(temp_product);
            }
        } else {
            std::cout << "[WARNING] if you want to use vector-parallelism, the value type must be double, float or int."
                      << std::endl;
            for (int i = 0; i < parallel_zone; ++i) total_sum += values1[i] * values2[i];
        }
        for (int i = parallel_zone; i < size; ++i) total_sum += values1[i] * values2[i];
        return total_sum;
    }


    /** @brief 超高精度求和，适用于上亿个浮点数相加，对精度较低的数据类型很有效，但整体效率很低，除非必要，慎用
     * @details 使用了Kahan求和算法 */
    template<typename ValType>
    ValType sumReductionKahan(ValType *values, int size) {
        // FOG向量库考虑到了不同框架之间的兼容性，所以可以直接使用512 bit的做法，就算不适配512长度的向量，也能自动转换成256 bit的向量或128 bit向量
        const int parallel_width = ALIGNED_BYTE_512 / sizeof(ValType);     // 并行带宽
        int loop = size / parallel_width;                   // 一共需要执行多少次循环
        int parallel_zone = loop * parallel_width;
        ValType total_sum = 0;
        int remainder = size - parallel_zone;
        if (std::is_same<ValType, double>::value) {
            VCL_DOUBLE64 sum(0.0);
            // 用0填充带宽为8的双精度向量变量
            VCL_DOUBLE64 local_sum(0.0);
            VCL_DOUBLE64 local_correction(0.0);
            VCL_DOUBLE64 var_v, new_sum, corrected_next_term;
            for (long i = 0; i < parallel_zone; i += parallel_width) {
                var_v.load((const double *) (values + i));   // 将var+i以packed形式加载到带宽为4的双精度向量变量中
                corrected_next_term = var_v + local_correction;  // 以packed形式做带宽为8的双精度向量变量加法
                new_sum = local_sum + local_correction;                       // 原理同上
                local_correction = corrected_next_term - (new_sum - local_sum);     // 原理同上
                local_sum = new_sum;
            }
            if (remainder > 0) {
                var_v.load_partial(remainder,
                                   (const double *) (values + parallel_zone));               // 加载剩余元素，未填满的地方补0
                corrected_next_term = var_v + local_correction;       // 以packed形式做带宽为8的双精度向量变量加法
                new_sum = local_sum + local_correction;
                local_correction = corrected_next_term - (new_sum - local_sum);
                local_sum = new_sum;
            }
            sum = local_correction;
            sum += local_sum;   // 以packed形式存储带宽为parallel_width的双精度向量，sum_v中的结果保存至sum

            double simd_sum = 0.0, simd_correction = 0.0, corrected_next_term_s, new_sum_s;
            // 对来自parallel_width个通道的结果进行最终的求和计算
            for (int i = 0; i < parallel_width; ++i) {
                corrected_next_term_s = sum[i] + simd_correction;
                new_sum_s = simd_sum + simd_correction;
                simd_correction = corrected_next_term_s - (new_sum_s - simd_sum);
                simd_sum = new_sum_s;
            }
            total_sum = simd_sum + simd_correction;
        } else if (std::is_same<ValType, float>::value) {
            VCL_FLOAT32 sum(0.0);
            // 用0填充带宽为8的双精度向量变量
            VCL_FLOAT32 local_sum(0.0);
            VCL_FLOAT32 local_correction(0.0);
            VCL_FLOAT32 var_v, corrected_next_term, new_sum;
            for (long i = 0; i < parallel_zone; i += parallel_width) {
                var_v.load((const float *) (values + i));   // 将var+i以packed形式加载到带宽为4的双精度向量变量中
                corrected_next_term = var_v + local_correction;  // 以packed形式做带宽为4的双精度向量变量加法
                new_sum = local_sum + local_correction;                       // 原理同上
                local_correction = corrected_next_term - (new_sum - local_sum);     // 原理同上
                local_sum = new_sum;
            }
            if (remainder > 0) {
                var_v.load_partial(remainder, (const float *) (values + parallel_zone)); // 加载剩余元素，未填满的地方补0
                corrected_next_term = var_v + local_correction;       // 以packed形式做带宽为4的双精度向量变量加法
                new_sum = local_sum + local_correction;
                local_correction = corrected_next_term - (new_sum - local_sum);
                local_sum = new_sum;
            }

            sum = local_correction;
            sum += local_sum; // 以packed形式存储带宽为parallel_width的双精度向量，sum_v中的结果保存至sum

            float simd_sum = 0.0, simd_correction = 0.0;
            // 对来自parallel_width个通道的结果进行最终的求和计算
            for (int i = 0; i < parallel_width; ++i) {
                float corrected_next_term_s = sum[i] + simd_correction;
                float new_sum_s = simd_sum + simd_correction;
                simd_correction = corrected_next_term_s - (new_sum_s - simd_sum);
                simd_sum = new_sum_s;
            }
            total_sum = simd_sum + simd_correction;
        } else {
            VCL_INT32 sum(0.0);
            // 用0填充带宽为8的双精度向量变量
            VCL_INT32 local_sum(0.0);
            VCL_INT32 local_correction(0.0);
            VCL_INT32 var_v, corrected_next_term, new_sum;
            for (long i = 0; i < parallel_zone; i += parallel_width) {
                var_v.load((const int *) (values + i));   // 将var+i以packed形式加载到带宽为4的双精度向量变量中
                corrected_next_term = var_v + local_correction;  // 以packed形式做带宽为4的双精度向量变量加法
                new_sum = local_sum + local_correction;                       // 原理同上
                local_correction = corrected_next_term - (new_sum - local_sum);     // 原理同上
                local_sum = new_sum;
            }
            if (remainder > 0) {
                var_v.load_partial(remainder, (const int *) (values + parallel_zone)); // 加载剩余元素，未填满的地方补0
                corrected_next_term = var_v + local_correction;       // 以packed形式做带宽为4的双精度向量变量加法
                new_sum = local_sum + local_correction;
                local_correction = corrected_next_term - (new_sum - local_sum);
                local_sum = new_sum;
            }

            sum = local_correction;
            sum += local_sum;   // 以packed形式存储带宽为parallel_width的双精度向量，sum_v中的结果保存至sum

            int simd_sum = 0.0, simd_correction = 0.0;
            // 对来自parallel_width个通道的结果进行最终的求和计算
            for (int i = 0; i < parallel_width; ++i) {
                int corrected_next_term_s = sum[i] + simd_correction;
                int new_sum_s = simd_sum + simd_correction;
                simd_correction = corrected_next_term_s - (new_sum_s - simd_sum);
                simd_sum = new_sum_s;
            }
            total_sum = simd_sum + simd_correction;
        }
        return total_sum;
    }


}


#endif //CUDATOOLS_FASTVECTORTOOLS_API_H
