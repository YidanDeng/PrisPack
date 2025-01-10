/*
 * @author  邓轶丹
 * @date    2024/11/16
 * @details 测试矩阵重排序
 */
#include <iostream>
#include <vector>
#include "rcm.hpp"
#include "../../include/utils/ExternalTools/MatrixReorderTools.h"
#include "../../include/utils/TestTools/WriteMtxTools.h"
#include "../../include/MatrixClass/CSRMatrix.h"
#include "../../include/MatrixClass/MatrixTools.h"
#include "../../include/MatrixClass/ModelProblem.h"

void testPermSymmetric() {
    // 定义一个稀疏对称矩阵的结构
    int node_num = 5; // 矩阵的行/列数量
    int adj_num = 11; // 非零元素数量

    // 稀疏对称矩阵的CSR格式的row_ptr和col_index
    // 这个和目前用的csr格式略有不同，起始下标默认从1开始（c++风格默认0开始）
    // 这里传入完整的一个对称矩阵，借助rcm算法导出重排标识数组（这个重排是同时行重排+列重排）
    int adj_row[] = {1, 4, 6, 8, 10, 12}; // row_ptr数组，大小为node_num + 1
    int adj[] = {1, 2, 4, 1, 2, 3, 5, 1, 4, 3, 5}; // col_index数组，大小为adj_num

    // 创建一个数组用于存储RCM排序的结果
    std::vector<int> perm(node_num);
    // 调用RCM排序函数
    genrcm(node_num, adj_num, adj_row, adj, perm.data());
    // 输出排序结果
    // 最后基于perm矩阵对结构对称的矩阵同时行+列重排（行重排和列重排用的同一个perm）
    std::cout << "RCM ordered vertices: ";
    for (int i = 0; i < node_num; ++i) {
        std::cout << perm[i] << " ";
    }
    std::cout << std::endl;
}

void testPermSymmetric2() {
    int node_num = 9; // 矩阵的行/列数量
    HOST::COOMatrix<double> testCOO;
    testCOO.resize(node_num, node_num, node_num * node_num, RESERVE_NO_DATA);
    // 第1行
    testCOO.pushBack(0, 0, 1);
    // 第2行
    testCOO.pushBack(1, 1, 9);
    // 第3行
    testCOO.pushBack(2, 2, 1);
    // 第4行
    testCOO.pushBack(3, 0, 3);
    testCOO.pushBack(0, 3, 3);
    testCOO.pushBack(3, 1, 6);
    testCOO.pushBack(1, 3, 6);
    testCOO.pushBack(3, 3, 9);
    // 第5行
    testCOO.pushBack(4, 2, 2);
    testCOO.pushBack(2, 4, 2);
    testCOO.pushBack(4, 4, 5);
    // 第6行
    testCOO.pushBack(5, 4, 2);
    testCOO.pushBack(4, 5, 2);
    testCOO.pushBack(5, 5, 40);
    // 第7行
    testCOO.pushBack(6, 0, 2);
    testCOO.pushBack(0, 6, 2);
    testCOO.pushBack(6, 3, 22);
    testCOO.pushBack(3, 6, 22);
    testCOO.pushBack(6, 6, 24);
    // 第8行
    testCOO.pushBack(7, 0, 1);
    testCOO.pushBack(0, 7, 1);
    testCOO.pushBack(7, 1, 3);
    testCOO.pushBack(1, 7, 3);
    testCOO.pushBack(7, 3, 13);
    testCOO.pushBack(3, 7, 13);
    testCOO.pushBack(7, 5, 6);
    testCOO.pushBack(5, 7, 6);
    testCOO.pushBack(7, 6, 10);
    testCOO.pushBack(6, 7, 10);
    testCOO.pushBack(7, 7, 16);
    // 第9行
    testCOO.pushBack(8, 0, 4);
    testCOO.pushBack(0, 8, 4);
    testCOO.pushBack(8, 2, 2);
    testCOO.pushBack(2, 8, 2);
    testCOO.pushBack(8, 3, 12);
    testCOO.pushBack(3, 8, 12);
    testCOO.pushBack(8, 4, 4);
    testCOO.pushBack(4, 8, 4);
    testCOO.pushBack(8, 6, 10);
    testCOO.pushBack(6, 8, 10);
    testCOO.pushBack(8, 7, 16);
    testCOO.pushBack(7, 8, 16);
    testCOO.pushBack(8, 8, 41);
    // 将coo转换为csr
    HOST::CSRMatrix<double> testCSR, permMat;
    HOST::transCOO2CSR(testCOO, testCSR);
    testCSR.printMatrix("testCSR");
    /* RCM排序 */
    HOST::AutoAllocateVector<UINT32> perm(node_num, memoryBase);
    HOST::rcmReorderCSR(testCSR, *perm);
    perm.printVector("perm(RCM)");
    testCSR.getSubMatrix(*perm, *perm, permMat);
    permMat.printMatrix("permMat(RCM)");
}

void testPermASymmetric() {
    // 定义一个稀疏对称矩阵的结构
    int node_num = 5; // 矩阵的行/列数量
    int adj_num = 11; // 非零元素数量（仅存储上三角或下三角部分）

    // 稀疏对称矩阵的CSR格式的row_ptr和col_index
    // 这个和目前用的csr格式略有不同，起始下标默认从1开始（c++风格默认0开始）
    // 这里传入完整的一个对称矩阵，借助rcm算法导出重排标识数组（这个重排是同时行重排+列重排）
    int adj_row[] = {1, 4, 6, 8, 10, 12}; // row_ptr数组，大小为node_num + 1
    int adj[] = {1, 2, 3, 1, 2, 4, 5, 1, 3, 4, 5}; // col_index数组，大小为adj_num

    // 创建一个数组用于存储RCM排序的结果
    std::vector<int> perm(node_num);
    // 调用RCM排序函数
    genrcm(node_num, adj_num, adj_row, adj, perm.data());
    // 输出排序结果
    // 最后基于perm矩阵对结构对称的矩阵同时行+列重排（行重排和列重排用的同一个perm）
    std::cout << "RCM ordered vertices: ";
    for (int i = 0; i < node_num; ++i) {
        std::cout << perm[i] << " ";
    }
    std::cout << std::endl;
}

void testLargeMat_structureSymmetric() {
    HOST::CSRMatrix<FLOAT64> testMat, permMat;
    HOST::generateLaplacianCSR(testMat, 2, 2, 2, 0.0, 0.0, 0.0, 0.05);
    testMat.printMatrix("testMat");
    HOST::AutoAllocateVector<UINT32> perm(testMat.getRowNum(), memoryBase);

    /* RCM排序 */
    HOST::rcmReorderCSR(testMat, *perm);
    perm.printVector("perm(RCM)");
    testMat.getSubMatrix(*perm, *perm, permMat);
    permMat.printMatrix("permMat(RCM)");

    /* AMD排序 */
    HOST::amdReorderCSR(testMat, *perm);
    perm.printVector("perm(AMD)");
    testMat.getSubMatrix(*perm, *perm, permMat);
    permMat.printMatrix("permMat(AMD)");
}


/** @brief 测试将重排后数据写入mtx文件 */
void testLargeMat_structureASymmetric_write2file(char* argv) {
    HOST::CSRMatrix<FLOAT64> testMat, permMat;
    HOST::generateLaplacianCSR(testMat, 20, 20, 20, 0.0, 0.0, 0.0, 0.05);
    testMat.printMatrix("testMat");
    HOST::AutoAllocateVector<UINT32> perm(testMat.getRowNum(), memoryBase);

    HOST::WriteMtxTools<FLOAT64> oriMtx(argv, "../source/test/testResults/originMat.mtx", 1, MTX_STORAGE_GENERAL);

    oriMtx.writeMatrix(testMat);

    /* RCM排序 */
    HOST::WriteMtxTools<FLOAT64> wMtx1(argv, "../source/test/testResults/rcmReorder.mtx", 1, MTX_STORAGE_GENERAL);
    HOST::rcmReorderCSR(testMat, *perm);
    perm.printVector("perm(RCM)");
    testMat.getSubMatrix(*perm, *perm, permMat);
    permMat.printMatrix("permMat(RCM)");
    wMtx1.writeMatrix(permMat);

    /* AMD排序 */
    HOST::WriteMtxTools<FLOAT64> wMtx2(argv, "../source/test/testResults/amdReorder.mtx", 1, MTX_STORAGE_GENERAL);
    HOST::amdReorderCSR(testMat, *perm);
    perm.printVector("perm(AMD)");
    testMat.getSubMatrix(*perm, *perm, permMat);
    permMat.printMatrix("permMat(AMD)");
    wMtx2.writeMatrix(permMat);
}

int main(int argc, char** argv) {
    testPermSymmetric();

    return 0;
}
