/*
 * @author  邓轶丹
 * @date    2024/5/25
 * @details 测试各种工具函数
 */

#include "../../include/utils/MemoryTools/UniquePtrTools.h"
#include "../../include/utils/MemoryTools/SharedPtrTools.h"
#include "../../include/utils/TestTools/ReadMtxTools.h"
#include "../../include/utils/TestTools/WriteMtxTools.h"
#include "../../include/utils/TestTools/checkTools.hpp"
#include "../../include/utils/ErrorHandler.h"
#include <thread>

#include "../../include/MatrixClass/ModelProblem.h"


class Test {
private:
    INT32 m_id{0};
    INT32 m_id2{0};

public:
    explicit Test(INT32 id) {
        m_id = id;
        std::cout << "construction! id:" << id << std::endl;
    }

    explicit Test(INT32 id1, INT32 id2) {
        m_id = id1;
        m_id2 = id2;
        std::cout << "construction! id1:" << id1 << ", id2:" << id2 << std::endl;
    }

    void hello() const {
        std::cout << "hello! id: " << m_id << ", id2:" << m_id2 << std::endl;
    }

    ~Test() {
        std::cout << "destruction! id:" << m_id << ", id2:" << m_id2 << std::endl;
    }
};


void testUniquePtr() {
    /* 用智能指针管理对象，无需手动释放新创建的对象，在指针生存期结束后，所指向的对象自动析构，只需确保该对象的析构函数无异常 */
    INT32 rowDim = 2, colDim = 3;
    /* 测试一维智能指针数组（每个指针指向一个类对象） */
    UniquePtr1D<Test> test(rowDim);
    for (INT32 i = 0; i < rowDim; ++i) {
        // 当前智能指针指向一个新构造的对象
        test[i] = std::make_unique<Test>(i);
    }
    // 测试调整一维智能指针数组的大小
    test.realloc(rowDim + 2);

    /* 测试二维智能指针数组 */
    UniquePtr2D<Test> test2D(rowDim, colDim);
    for (INT32 i = 0; i < rowDim; ++i) {
        for (INT32 j = 0; j < colDim; ++j) {
            // 当前智能指针指向一个新构造的对象
            //            test2D[i][j] = std::make_unique<Test>(i, j);
            test2D[i].construct(j, i, j);
            // 调取该对象对应的操作函数
            test2D(i, j).hello();
        }
    }
    // 测试调整二维智能指针数组的大小
    test2D.realloc(rowDim + 2);
    // 给新生成的指针分配一个新的对象
    test2D[2] = UniquePtr1D<Test>(colDim);
    std::cout << std::endl;
}

void testSharedPtr() {
    int dim1 = 3;
    SharedPtr1D<Test> test(dim1);
    for (INT32 i = 0; i < dim1; ++i) {
        // 当前智能指针指向一个新构造的对象
        test[i] = std::make_shared<Test>(i);
    }

    test.realloc(dim1 + 2);

    SharedPtr1D<Test> test2;
    test2 = test;
    test2[0]->hello();

    int dim2 = 3;
    SharedPtr2D<Test> test2D(dim1, dim2);
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            //            test2D[i][j] = std::make_shared<Test>(i, j);
            test2D[i].construct(j, i, j);
            test2D(i, j).hello();
        }
    }
}


/* =================== 测试错误异常处理函数 ===================
 * 概述：假设函数A调用了函数B，函数B里抛了一个异常，在接收到异常后终止程序，并打印出错代码所在文件、代码行号等信息，记录函数调用顺序 */
void functionB() {
    // 假设函数B出现了数组越界异常
    int test = 0;
#ifndef NDEBUG
    // errorCondition, exception
    THROW_EXCEPTION(test == 0, THROW_OUT_OF_RANGE("Trow exception!"))
#endif
}

void functionA() {
    // 函数A接收B传来的异常
    TRY_CATCH(functionB())
}

void testError() {
    TRY_CATCH(functionA());
}

void testThreadError() {
    TRY_CATCH_THREAD(functionA(),)
}

void testMultiThread() {
    std::thread t1(testThreadError);
    std::thread t2(testThreadError);
    std::cout << "The current thread ID is: " << std::this_thread::get_id() << std::endl;
    // 后面就会抛一个异常，这里只能输出第一个接收到异常的线程ID
    t1.join();
    t2.join();
}

void testMtxTools(char *argv) {
    HOST::ReadMtxTools<FLOAT64> readMtxTools(argv, "../datasets/atmosmodd.mtx", "../datasets/atmosmodd_b.mtx", 1);
    HOST::DenseMatrix<FLOAT64> rhs;
    HOST::CSRMatrix<FLOAT64> outCSR;
    readMtxTools.loadMatrix(outCSR);
    readMtxTools.loadRightHand(rhs);
}

void testMtxTools_complex(char *argv) {
    HOST::ReadMtxTools<FLOAT64> readMtxTools(argv, "../datasets/complexMat/output1_S.mtx",
                                             "../datasets/complexMat/output1_b.mtx", 1);
    HOST::DenseVector<FLOAT64> rhsReal, rhsImag;
    HOST::CSRMatrix<FLOAT64> outCSRreal, outCSRimag;
    readMtxTools.loadMatrix(outCSRreal, outCSRimag);
    readMtxTools.loadRightHand(rhsReal, rhsImag);
}

void testMtxTools_withoutRhsFile(char *argv) {
    HOST::ReadMtxTools<FLOAT64> readMtxTools(argv, "../datasets/cfd1.mtx", 1);
    HOST::AutoAllocateVector<FLOAT64> res, rhs, spmvRes;
    HOST::CSRMatrix<FLOAT64> outCSR;
    readMtxTools.loadMatrix(outCSR);
    readMtxTools.loadRightHand(*res, *rhs, 1);
    /* 验证矩阵是否读入正确、右端项是否正确生成 */
    spmvRes.resize(res.getLength(), RESERVE_NO_DATA);
    outCSR.MatPVec(*res, *spmvRes);
    HOST::checkAnswer(rhs->getRawValPtr(), spmvRes->getRawValPtr(), rhs.getLength(), "check Ax == b");
    HOST::check_correctness(outCSR.getRowNum(), (int *) outCSR.getRowOffsetPtr(0), (int *) outCSR.getColIndicesPtr(0),
                            outCSR.getCSRValuesPtr(0), res->getRawValPtr(), spmvRes->getRawValPtr());
}

void testWriteMtxTools(char *argv) {
    HOST::WriteMtxTools<FLOAT64> writeMtxTools(argv, "../datasets/testWriteMat.mtx", 1, MTX_STORAGE_GENERAL);
    HOST::CSRMatrix<FLOAT64> testMat;

    UINT32 diagBlockSize = 4;
    HOST::generatePoissonCSR(testMat, diagBlockSize);

    writeMtxTools.writeMatrix(testMat);
}


int main(int argc, char **argv) {
    /* 测试mtx工具 */
    testMtxTools(argv[0]);

    return 0;
}
