#ifndef PMSLS_DEV_CUDAHEADERS_CUH
#define PMSLS_DEV_CUDAHEADERS_CUH

#include "headers.h"
#include "config.h"
#include "debug.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/adjacent_difference.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <cublas_v2.h>
#define DISABLE_CUSPARSE_DEPRECATED         // 用来取消编译器关于cuSPARSE中过时API的警告
#include <cusparse.h>

#endif //PMSLS_DEV_CUDAHEADERS_CUH
