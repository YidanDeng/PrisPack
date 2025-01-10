/**
 * @author  邓轶丹
 * @date    2024/4/16
 * @details 通过Thrust库在GPU上计算，性能一般，仅用作初期实验部分
 */

#ifndef PMSLS_DEV_CUTHRUST_CUH
#define PMSLS_DEV_CUTHRUST_CUH

#include "../../config/headers.h"
#include "../../config/config.h"
#include "../../config/debug.h"
#include "../../config/CUDAheaders.cuh"

namespace DEVICE {
    // square<T> computes the square of a number f(x) -> x*x
    template<typename ValType, typename CoefType>
    struct Square {
        const CoefType a{1};

        explicit Square(CoefType _a) : a(_a) {}

        __host__ __device__
        ValType operator()(const ValType &x) const {
            return a * x * x;
        }
    };

    template<typename ValType, typename CoefType1, typename CoefType2>
    struct VecAdd_func {
        const CoefType1 a{1};
        const CoefType2 b{1};

        explicit VecAdd_func(CoefType1 _a, CoefType2 _b) : a(_a), b(_b) {}

        __host__ __device__ ValType operator()(ValType &x, ValType &y) {
            return a * x + b * y;
        }
    };

    template<typename ValType>
    void printVector(ValType *dev_val, UINT32 size, const std::string &prefix) {
        std::cout << prefix << ":";
        thrust::device_ptr<ValType> devicePtr = PACK_RAW_PTR(dev_val);
        UINT32 upper_bound = (size <= 40 ? size : 40);
        for (int i = 0; i < upper_bound; ++i) {
            std::cout << "  " << devicePtr[i];
        }
        std::cout << std::endl;
    }

    template<typename ValType, typename CoefType>
    inline ValType vecNorm2(DEVICE_PTR(ValType) dev_vec, CoefType alpha, int size) {
        Square<ValType, CoefType> func(alpha);
        thrust::plus<ValType> binary_op;
        ValType init = 0;
        return std::sqrt(thrust::transform_reduce(thrust::device, dev_vec, dev_vec + size, func, init, binary_op));
    }

    template<typename ValType>
    inline ValType vecReduce(DEVICE_PTR(ValType) dev_ptr, UINT32 data_size) {
        return thrust::reduce(dev_ptr, dev_ptr + data_size);
    }

    template<typename ValType, typename BinFunc, typename InitType>
    inline ValType vecReduce(DEVICE_PTR(ValType) dev_ptr, BinFunc func, InitType init, UINT32 data_size) {
        return thrust::reduce(dev_ptr, dev_ptr + data_size, init, func);
    }

    template<typename ValType>
    inline void vecCopy(DEVICE_PTR(ValType) dev_in, UINT32 num, DEVICE_PTR(ValType) dev_out) {
        thrust::copy_n(thrust::device, dev_in, num, dev_out);
    }


    template<typename ValType, typename CoefType1, typename CoefType2>
    inline void
    vecAdd(CoefType1 alpha, DEVICE_PTR(ValType) dev_x, CoefType2 beta, DEVICE_PTR(ValType) dev_y,
           DEVICE_PTR(ValType) dev_out,
           UINT32 size) {
        VecAdd_func<ValType, CoefType1, CoefType2> func(alpha, beta);
        thrust::transform(thrust::device, dev_x, dev_x + size, dev_y, dev_out, func);
    }


    template<typename ValType, typename CoefType1, typename CoefType2>
    inline ValType vecInnerProduct(CoefType1 alpha, DEVICE_PTR(ValType) d_in1, CoefType2 beta,
                                   DEVICE_PTR(ValType) d_in2, int size) {
        ValType init{0};
        return alpha * beta * thrust::inner_product(thrust::device, d_in1, d_in1 + size, d_in2, init);
    }


    template<typename ValType, typename MapType>
    inline void
    vecGather(DEVICE_PTR(ValType) d_src, DEVICE_PTR(MapType) d_map, DEVICE_PTR(ValType) d_out, int map_size) {
//        thrust::copy(thrust::device,
//                     thrust::make_permutation_iterator(d_src.begin(), d_map.begin()),
//                     thrust::make_permutation_iterator(d_src.begin(), d_map.end()),
//                     d_out.begin());
        thrust::gather(thrust::device, d_map, d_map + map_size, d_src, d_out);
    }

    template<typename ValType, typename MapType>
    inline void
    vecScatter(DEVICE_PTR(ValType) d_in, DEVICE_PTR(MapType) d_map, DEVICE_PTR(ValType) d_out, UINT32 map_size) {
        thrust::scatter(thrust::device, d_in, d_in + map_size, d_map, d_out);
    }

    template<typename ValType>
    inline void vecSwap(DEVICE_PTR(ValType) d_x, DEVICE_PTR(ValType) d_y, int size) {
        thrust::swap_ranges(thrust::device, d_x, d_x + size, d_y);
    }

    /** @brief 针对向量的一元变换 */
    template<typename ValType, typename TransFunc>
    inline void vecTransform(DEVICE_PTR(ValType) d_in,      ///< 输入向量：Trust的Device指针类型
                             DEVICE_PTR(ValType) d_out,     ///< 输出结果向量：Trust的Device指针类型
                             TransFunc &func,               ///< 传入对应的变换操作：自定义结构体（主要重载了圆括号）
                             UINT32 size                    ///< 输入、输出向量的长度：长度必须一致，本函数不对长度做检查，使用时需注意
    ) {
        thrust::transform(thrust::device, d_in, d_in + size, d_out, func);
    }

    /** @brief 针对向量的二元变换 */
    template<typename ValType, typename TransFunc>
    inline void
    vecTransform(DEVICE_PTR(ValType) d_in1,     ///< 输入向量1：Trust的Device指针类型
                 DEVICE_PTR(ValType) d_in2,     ///< 输入向量2：Trust的Device指针类型
                 DEVICE_PTR(ValType) d_out,     ///< 输出结果向量：Trust的Device指针类型
                 TransFunc &func,               ///< 传入对应的变换操作：自定义结构体对象（主要重载了圆括号）
                 UINT32 size                    ///< 输入、输出向量的长度：长度必须一致，本函数不对长度做检查，使用时需注意
    ) {
        thrust::transform(thrust::device, d_in1, d_in1 + size, d_in2, d_out, func);
    }


    template<typename ValType>
    inline void vecSequence(DEVICE_PTR(ValType) d_vec, const INT32 &devID, UINT32 size) {
        CHECK_CUDA(cudaSetDevice(devID))
        thrust::sequence(thrust::device, d_vec, d_vec + size);
    }

    template<typename ValType, typename FillType>
    inline void vecFill(const DEVICE_PTR(ValType) &d_vec, const INT32 &devID, FillType val, UINT32 size) {
        CHECK_CUDA(cudaSetDevice(devID))
        thrust::fill(thrust::device, d_vec, d_vec + size, (ValType) val);
    }

}

#endif //PMSLS_DEV_CUTHRUST_CUH
