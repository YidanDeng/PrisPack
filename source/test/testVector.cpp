/*
 * @author  邓轶丹
 * @date    2024/5/9
 * @details 测试vector相关的工具函数
 */
#include "../../include/utils/TestTools/generateTools.hpp"
#include "../../include/VectorClass/DenseVector.h"
#include "../../include/VectorClass/AlignedVector.h"
#include "../../include/VectorClass/AutoAllocateVector.h"
#include "../../include/utils/TimerTools/CPUtimer.hpp"

#ifdef CUDA_ENABLED

#include "../../include/VectorClass/SyncDeviceVector.cuh"
#include "../../include/VectorClass/PageLockedVector.cuh"
#include "../../include/CUDA/MultiDeviceStream.cuh"

#endif

#define DATA_TYPE double

void testDenseVector() {
    HOST::DenseVector<DATA_TYPE> testVec;
    testVec.resize(10, RESERVE_DATA);
    testVec.fillVector(0, testVec.getLength(), 2);
    testVec[0] = 5;
    testVec.printVector("test vec");
}

void testAutoVector() {
    UINT32 vecLength = 10;
    /* 方式一：直接使用智能指针 */
    std::unique_ptr<HostVector<DATA_TYPE> > vecPtr = std::make_unique<HOST::DenseVector<DATA_TYPE> >(vecLength);
    vecPtr->resize(vecLength + 2, RESERVE_NO_DATA);
    vecPtr->fillVector(0, vecPtr->getLength(), 3);
    (*vecPtr)[0] = 4;
    vecPtr->printVector("test vec");

    /* 方式二：使用封装好的类，根据指定类型动态分配对应空间 */
    HOST::AutoAllocateVector<DATA_TYPE> testAutoAlloc(vecLength, memoryAligned);
    (*testAutoAlloc).fillVector(0, vecLength, 1);
    (*testAutoAlloc).printVector("test auto alloc vec(fill with \"1\")");
    testAutoAlloc[0] = 0;
    (*testAutoAlloc).printVector("test auto alloc vec(change first value to \"0\")");
    testAutoAlloc->resize(vecLength + 3, RESERVE_NO_DATA);
    testAutoAlloc->fillVector(0, testAutoAlloc->getLength(), 4);
    testAutoAlloc->printVector("test auto alloc vec(resize and fill with \"4\")");
}


/* ============================= 一些GPU上的代码 ============================= */
#ifdef CUDA_ENABLED

void testDeviceVector() {
    UINT32 dim = 20;
    // 创建锁页内存，用于异步拷贝
    HOST::PageLockedVector<DATA_TYPE> testVec(20);
    testVec.fillVector(0, dim, 2);
    testVec.printVector("host vec");
    // 先测试同步拷贝
    DEVICE::SyncDeviceVector<DATA_TYPE> devVec(dim);
    devVec.copyFromHost(testVec);
    devVec.printVector("dev vec");
    // 测试异步拷贝，先创建GPU上的流，流的数量由用户指定；若有多个GPU，则在每个GPU上都分配指定数量的流
    UINT32 streamNum = 2;
    DEVICE::MultiDeviceStream deviceStream(streamNum);
    testVec.fillVector(0, dim, 3);
    testVec.printVector("host vec");

    //    devVec.asyncCopyFromHost(testVec, 0, dim, deviceStream[DEFAULT_GPU], streamNum);
    deviceStream.syncAllStreams(DEFAULT_GPU);
    devVec.printVector("async vec");
}

#endif //CUDA_ENABLED

#pragma GCC push_options
#pragma GCC optimize ("O0")

void unoptimizedFunction(const HOST::AutoAllocateVector<DATA_TYPE> &a, HOST::AutoAllocateVector<DATA_TYPE> &b) {
    // 需要关闭优化的代码段
    for (UINT32 i = 0; i < a.getLength(); ++i) {
        b[i] += a[i];
    }
}

void unoptimizedFunction(HOST::AutoAllocateVector<DATA_TYPE> &a) {
    // 需要关闭优化的代码段
    for (UINT32 i = 0; i < a.getLength(); ++i) {
        a[i] *= 2;
    }
}

void innerPro(HOST::AutoAllocateVector<DATA_TYPE> &a, HOST::AutoAllocateVector<DATA_TYPE> &b) {
    // 需要关闭优化的代码段
    DATA_TYPE res = 0;
    for (UINT32 i = 0; i < a.getLength(); ++i) {
        res += a[i] * b[i];
    }
}

void sumNoOpt(HOST::AutoAllocateVector<DATA_TYPE> &a) {
    // 需要关闭优化的代码段
    DATA_TYPE res = 0;
    for (UINT32 i = 0; i < a.getLength(); ++i) {
        res += a[i];
    }
}

#pragma GCC pop_options

/* 向量加法 */
void testEfficiency1() {
    UINT32 dim = 1000000;
    HOST::AutoAllocateVector<DATA_TYPE> a(dim, memoryBase), b(dim, memoryBase);
    a->fillVector(0, dim, 1);
    CPU_TIMER_FUNC()
    // 预加载到缓存
    for (UINT32 i = 0; i < dim; ++i) {
        b[i] += a[i];
    }
    CPU_TIMER_BEGIN()
    unoptimizedFunction(a, b);
    CPU_TIMER_END()
    std::cout << " --- no opt: " << CPU_EXEC_TIME() << " ms." << std::endl;
    CPU_TIMER_BEGIN()

    for (UINT32 i = 0; i < dim; ++i) {
        b[i] += a[i];
    }
    CPU_TIMER_END()
    std::cout << " --- auto opt: " << CPU_EXEC_TIME() << " ms." << std::endl;
    /* 多线程 b += a */
    CPU_TIMER_BEGIN()
    b.add(1, *a);
    CPU_TIMER_END()
    std::cout << " --- mul thread: " << CPU_EXEC_TIME() << " ms." << std::endl;
}

/* 向量数乘 */
void testEfficiency2() {
    UINT32 dim = 1000000;
    HOST::AutoAllocateVector<DATA_TYPE> a(dim, memoryBase);
    a->fillVector(0, dim, 1);
    CPU_TIMER_FUNC()
    // 预加载到缓存
    for (UINT32 i = 0; i < dim; ++i) {
        a[i] *= 2;
    }
    CPU_TIMER_BEGIN()
    /* 无优化 */
    unoptimizedFunction(a);
    CPU_TIMER_END()
    std::cout << " --- no opt: " << CPU_EXEC_TIME() << " ms." << std::endl;
    CPU_TIMER_BEGIN()
    /* 向量化 */
    for (UINT32 i = 0; i < dim; ++i) {
        a[i] *= 2;
    }
    CPU_TIMER_END()
    std::cout << " --- auto opt: " << CPU_EXEC_TIME() << " ms." << std::endl;
    /* 多线程 */
    CPU_TIMER_BEGIN()
    a->scale(2);
    CPU_TIMER_END()
    std::cout << " --- mul thread: " << CPU_EXEC_TIME() << " ms." << std::endl;
}

/* 向量内积*/
void testEfficiency3() {
    UINT32 dim = 1000000;
    HOST::AutoAllocateVector<DATA_TYPE> a(dim, memoryBase), b(dim, memoryBase);

    CPU_TIMER_FUNC()
    // 预加载到缓存
    DATA_TYPE innerProRes;
    for (UINT32 i = 0; i < dim; ++i) {
        a[i] = i;
        b[i] = i % 10;
    }
    CPU_TIMER_BEGIN()
    /* 无优化 */
    innerPro(a, b);
    CPU_TIMER_END()
    std::cout << " --- no opt: " << CPU_EXEC_TIME() << " ms." << std::endl;
    CPU_TIMER_BEGIN()
    /* 向量化 */
    innerProRes = 0;
    for (UINT32 i = 0; i < dim; ++i) {
        innerProRes += a[i] * b[i];
    }
    CPU_TIMER_END()
    std::cout << innerProRes << std::endl;
    std::cout << " --- auto opt: " << CPU_EXEC_TIME() << " ms." << std::endl;
    /* 多线程 */
    CPU_TIMER_BEGIN()
    a->innerProduct(*b);
    CPU_TIMER_END()
    std::cout << " --- mul thread: " << CPU_EXEC_TIME() << " ms." << std::endl;
}

/* 向量求和 */
void testEfficiency4() {
    UINT32 dim = 1000000;
    HOST::AutoAllocateVector<DATA_TYPE> a(dim, memoryBase);
    CPU_TIMER_FUNC()
    // 预加载到缓存
    for (UINT32 i = 0; i < dim; ++i) {
        a[i] = i;
    }
    CPU_TIMER_BEGIN()
    /* 无优化 */
    sumNoOpt(a);
    CPU_TIMER_END()
    std::cout << " --- no opt: " << CPU_EXEC_TIME() << " ms." << std::endl;
    CPU_TIMER_BEGIN()
    /* 向量化 */
    DATA_TYPE sum = 0;
    for (UINT32 i = 0; i < dim; ++i) {
        sum += a[i];
    }
    CPU_TIMER_END()
    std::cout << sum << std::endl;
    std::cout << " --- auto opt: " << CPU_EXEC_TIME() << " ms." << std::endl;
    /* 多线程 */
    CPU_TIMER_BEGIN()
    sum = a->sum();
    CPU_TIMER_END()
    std::cout << sum << std::endl;
    std::cout << " --- mul thread: " << CPU_EXEC_TIME() << " ms." << std::endl;
}

void testKahanSum() {
    UINT32 dim = 10000000;
    HOST::AutoAllocateVector<DATA_TYPE> testVec(dim, memoryBase);
    HOST::generateArraySteady1D(testVec->getRawValPtr(), dim);
    CPU_TIMER_FUNC()
    CPU_TIMER_BEGIN()
    DATA_TYPE sum1 = testVec->sumKahan(0, dim);
    CPU_TIMER_END()
    std::cout << " --- no omp Kahan: " << CPU_EXEC_TIME() << " ms." << std::endl;
    FLOAT64 globalSum{0}, globalCorrection{0};
    CPU_TIMER_BEGIN()
#pragma omp parallel default(none) num_threads(THREAD_NUM) proc_bind(master) shared(testVec, dim, globalSum, globalCorrection)
    {
        testVec->sumKahanOuterOMP(0, dim, globalSum, globalCorrection, [](DATA_TYPE x)-> DATA_TYPE { return x; });
    }
    CPU_TIMER_END()
    std::cout << " --- omp Kahan(version 2): " << CPU_EXEC_TIME() << " ms." << std::endl;

    CPU_TIMER_BEGIN()
    DATA_TYPE sum2 = testVec->sumKahanOMP(0, dim, [](DATA_TYPE x)-> DATA_TYPE { return x; });
    CPU_TIMER_END()
    std::cout << " --- omp Kahan(version 1): " << CPU_EXEC_TIME() << " ms." << std::endl;

    std::cout << "res1: " << std::setprecision(14) << sum1 << std::endl;
    std::cout << "res2: " << sum2 << std::endl;
    std::cout << "res3: " << globalSum << std::endl;
    DATA_TYPE result = fabs(sum1 - sum2);
    std::cout << "residual result(version 1): " << result << std::endl;
    result = fabs(sum1 - globalSum);
    std::cout << "residual result(version 2): " << result << std::endl;
}


int main() {
    testKahanSum();

    return 0;
}
