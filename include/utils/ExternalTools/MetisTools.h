/**
 * @author  袁心磊、刘玉琴、邓轶丹
 * @date    2024/6/29
 * @details 封装高级矩阵重排序方法
 */

#ifndef SOLVERCHALLENGE_METISTOOLS_H
#define SOLVERCHALLENGE_METISTOOLS_H

#include <metis.h>
#include "../../VectorClass/BaseVector.h"
#include "../../MatrixClass/CSRMatrix.h"
#include "../MemoryTools/SharedPtrTools.h"

namespace HOST {
    //metis图划分
    /**
    * @brief 图划分
    * @param A 对称矩阵
    * @param perm 排序向量，现在编号---->原始编号
    * @param dom_ptr 各个分区节点个数，偏移量【0，x,x+y,.....】
    * @return 返回-1，则输入有错，返回0，运行成功，返回1，则有子域没有内点
 */
    template<typename ValType>
    void
    CsrMatrixMetisKway(std::shared_ptr<CSRMatrix<ValType> > &A, HostVector<UINT32> &perm, HostVector<UINT32> &dom_ptr,
                       const UINT32 &block_num) {
        UINT32 size = A->getRowNum(), nnz = A->getNNZnum(0, size - 1), jj = 0;
        idx_t lnrows = (idx_t) size, edgecut = 0, ncon = 1, num_dom = block_num;
        std::vector<idx_t> xadj(size + 1, memoryBase), adjncy(nnz - size, memoryBase), vwgt(size, memoryBase);
        std::vector<idx_t> adjwgt(nnz - size, memoryBase), lmap(size, memoryBase);
        dom_ptr.resize(num_dom * 2 + 1, RESERVE_NO_DATA);
        const UINT32 *rowIdxPtr = A->getRowOffsetPtr(0);
        const UINT32 *colIdxPtr = A->getColIndicesPtr(0);
        for (UINT32 i = 0; i < size; ++i) {
            for (UINT32 j = rowIdxPtr[i]; j < rowIdxPtr[i + 1]; ++j) {
                if (colIdxPtr[j] != i) {
                    adjncy[jj] = (idx_t) colIdxPtr[j];
                    adjwgt[jj] = 1;
                    jj++;
                } else {
                    vwgt[i] = 6;
                }
                if (vwgt[i] == 0) vwgt[i] = 6;
                xadj[i + 1] = (idx_t) jj;
            }
        }
        // 图划分，得到节点与所属子域的映射关系（数组下标：原来节点的编号，数组中的值：所属子域的编号）
        if (num_dom < 8) {
            METIS_PartGraphRecursive(&lnrows, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr,
                                     nullptr, &num_dom, nullptr, nullptr, nullptr, &edgecut, lmap.data());
        } else {
            METIS_PartGraphKway(&lnrows, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr, nullptr,
                                &num_dom, nullptr, nullptr, nullptr, &edgecut, lmap.data());
        }
        jj = 0;
        for (UINT32 i = 0; i < num_dom; ++i) {
            for (UINT32 j = 0; j < size; ++j) {
                if (lmap[j] == i) {
                    perm[jj] = j;
                    jj++;
                }
            }
            dom_ptr[i + 1] = jj;
        }

        for (UINT32 i = 0; i < num_dom; ++i) {
            if ((dom_ptr[i + 1] - dom_ptr[i]) == 0)
                exit(-1);
        }
        std::cout << "edgecut: " << edgecut << std::endl;
        for (UINT32 i = 0; i < num_dom; ++i) {
            for (UINT32 j = dom_ptr[i]; j < dom_ptr[i + 1]; ++j) {
                for (UINT32 k = xadj[perm[j]]; k < xadj[perm[j] + 1]; ++k) {
                    if (lmap[adjncy[k]] != i) {
                        // 如果是外点，就将其放在C块（右下角子块）
                        for (UINT32 l = 0; l < num_dom; ++l) {
                            if (lmap[adjncy[k]] == l)
                                lmap[adjncy[k]] = num_dom + l;
                        }
                    }
                }
            }
        }
        jj = 0;
        for (UINT32 i = 0; i < num_dom * 2; ++i) {
            for (UINT32 j = 0; j < size; ++j) {
                if (lmap[j] == i) {
                    perm[jj] = j;
                    jj++;
                }
            }
            dom_ptr[i + 1] = jj;
        }
//        perm.printVector("prem: ");
//        A->getSubMatrix(perm, perm, Ai);
//        *A = Ai;
        //        lmap.clear();
        //        xadj.clear();
        //        adjncy.clear();
        //        vwgt.clear();
        //        adjwgt.clear();
    }

    template<typename ValType>
    INT32 csrMatrixMetisKwayHID(CSRMatrix<ValType> &A, UINT32 &num_dom, HostVector<UINT32> &map,
                                  HostVector<UINT32> &seq,
                                  UINT32 &edgecut, HostVector<UINT32> &perm, HostVector<UINT32> &dom_ptr) {
        UINT32 nrows, ncols, nnz, j, i, jj, i1, i2, col, p, err = 0;
        nrows = A.getRowNum();
        ncols = A.getColNum();
        if (nrows != ncols) {
            SHOW_WARN("METIS partition only works for square matrix.")
            return -1;
        }
        if (num_dom == 1) {
            SHOW_WARN("The total number of domain is 1.")
            return -1;
        }
        nnz = A.getNNZnum(0, nrows - 1);
        //准备metis需要的数据结构:
        std::vector<idx_t> xadj(nrows + 1); //长度为节点数+1
        std::vector<idx_t> adjncy(nnz); //长度为2*边数
        std::vector<idx_t> vwgt(nrows, 0); //存储点的权重,size为n*ncon
        // vwgt.fillVector(0);
        std::vector<idx_t> adjwgt(nnz); //存储边的权重,size为2*边数
        std::vector<idx_t> lmap(nrows); //长度为节点数，存储各个节点被分到哪个子域

        map.resize(nrows, RESERVE_NO_DATA);
        perm.resize(nrows, RESERVE_NO_DATA);
        dom_ptr.resize(num_dom + 1, RESERVE_NO_DATA);
        if (dom_ptr.getMemoryType() != memoryBase) dom_ptr.fillVector(0, dom_ptr.getLength(), 0);

        xadj[0] = 0;
        jj = 0;
        UINT32 *A_i, *A_j;
        A_i = A.getRowOffsetPtr(0);
        A_j = A.getColIndicesPtr(0);

        for (i = 0; i < nrows; i++) {
            i1 = A_i[i];
            i2 = A_i[i + 1];
            for (j = i1; j < i2; j++) {
                col = A_j[j];
                if (col != i) //非对角线元素
                {
                    adjncy[jj] = col;
                    adjwgt[jj] = 1;
                    jj++;
                } else {
                    vwgt[i] = 6;
                }
            }
            if (vwgt[i] == 0) {
                //此时说明该行（该顶点）对角线元素为0
                vwgt[i] = 6;
            }
            xadj[i + 1] = jj;
        }

        //参数设置
        idx_t lnrows = (idx_t) nrows;
        idx_t lnum_dom = (idx_t) num_dom;
        idx_t ledgecut;
        idx_t ncon = 1;

        //调用metis接口
        if (lnum_dom > lnrows) {
            //若区域数大于顶点数
            lnum_dom = lnrows;
        }

        //官方建议划分子图个数超过8，建议使用METIS_PartGraphKway;小规模划分使用METIS_PartGraphRecursive，这样可以得到更高质量的解
        if (lnum_dom >= 8) {
            METIS_PartGraphKway(&lnrows, &ncon, xadj.data(), adjncy.data(), vwgt.data(), NULL, adjwgt.data(),
                                &lnum_dom, NULL, NULL, NULL,
                                &ledgecut, lmap.data());
        } else {
            METIS_PartGraphRecursive(&lnrows, &ncon, xadj.data(), adjncy.data(), vwgt.data(), NULL, adjwgt.data(),
                                     &lnum_dom, NULL, NULL, NULL, &ledgecut, lmap.data());
        }

        //将idx_t类型转换为UINT32
        num_dom = (UINT32) lnum_dom;
        for (i = 0; i < nrows; i++) {
            map[i] = (UINT32) lmap[i];
        }
        //确定每一个子域（分区）的节点个数  dom_ptr = [0,x,y]，第一个子域有x个节点，第二个有y个节点
        for (i = 0; i < nrows; i++) {
            dom_ptr[map[i] + 1]++;
        }
        //累计节点个数 dom_ptr = [0,x,x+y]
        UINT32 num_dom2 = 0;
        for (i = 0; i < num_dom; i++) {
            if (dom_ptr[i + 1] > 0) {
                num_dom2++; //累计非空子域的个数
            }
            dom_ptr[i + 1] += dom_ptr[i];
        }

        //若每个子域都有节点，则应该num_dom==num_dom2；若num_dom2<num_dom，则说明有空的子域，删掉这些空的子域
        if (num_dom2 < num_dom) {
            AutoAllocateVector<UINT32> dom_ptr2(dom_ptr.getLength(), memoryBase); //临时变量，存放之前的dom_ptr
            dom_ptr2.copy(dom_ptr);
            AutoAllocateVector<INT32> map2(num_dom, memoryBase);
            map2->fillVector(0, num_dom, -1);
            dom_ptr.resize(num_dom2 + 1, RESERVE_NO_DATA);
            num_dom2 = 0;
            dom_ptr[0] = 0;
            for (i = 0; i < num_dom; i++) {
                if (dom_ptr2[i + 1] > dom_ptr2[i]) {
                    map2[i] = num_dom2;
                    dom_ptr[++num_dom2] = dom_ptr2[i + 1];
                }
            }
            for (i = 0; i < nrows; i++) {
                map[i] = (UINT32) map2[map[i]];
            }
            // dom_ptr2.clear();
            // map2.clear();
            num_dom = num_dom2;
        }

        //check dom_ptr[num_dom]==nrows
        THROW_EXCEPTION(dom_ptr[num_dom] != nrows, THROW_LOGIC_ERROR("dom_ptr fill error!"))

        //perm向量记录划分区域后现编号（perm下标索引）对应的原来的编号（perm的值）
        for (i = 0; i < nrows; i++) {
            p = map[i]; //顶点i对应的分区/子域
            perm[dom_ptr[p]++] = i; //此时对dom_ptr做了修改，由【0,x,x+y,...】变成了【x,x+y,...】
        }
        //dom_ptr还原
        for (i = num_dom; i > 0; i--) {
            dom_ptr[i] = dom_ptr[i - 1];
        }
        dom_ptr[0] = 0;
        //edge seperator
        seq.resize(nrows, RESERVE_NO_DATA);
        if (seq.getMemoryType() != memoryBase) seq.fillVector(0, seq.getLength(), 0);
        edgecut = 0;
        AutoAllocateVector<UINT32> sep_size(num_dom, memoryBase); // 使用base自动给内存置为0
        // sep_size.fillVector(0);
        for (i = 0; i < nrows; i++) {
            i1 = A_i[i];
            i2 = A_i[i + 1];
            p = map[i];
            for (j = i1; j < i2; j++) {
                col = A_j[j];
                if (p != map[col]) {
                    seq[i] = 1;
                    sep_size[p]++;
                    edgecut++;
                    break;
                }
            }
        }
        //检查是否有子域没有内点
        for (i = 0; i < num_dom; i++) {
            if (sep_size[i] == dom_ptr[i + 1] - dom_ptr[i]) {
                SHOW_WARN("The idx: " << i << ", this sub-domain has no inner-point.")
                // std::cout << "第" << i << "个子域没有内点" << std::endl;
                err = 1;
                break;
            }
        }

        return err;
    }

    template<typename ValType>
    void
    CsrMatrixMetisKway(std::shared_ptr<CSRMatrix<ValType>> &A, HostVector<UINT32> &perm, HostVector<UINT32> &dom_ptr,
                       const UINT32 &block_num, std::shared_ptr<CSRMatrix<ValType>> &A_out) {
        UINT32 size = A->getRowNum(), nnz = A->getNNZnum(0, size - 1), jj = 0;
        idx_t lnrows = (idx_t) size, edgecut = 0, ncon = 1, num_dom = block_num;
        std::vector<idx_t> xadj(size + 1, memoryBase), adjncy(nnz - size, memoryBase), vwgt(size, memoryBase);
        std::vector<idx_t> adjwgt(nnz - size, memoryBase), lmap(size, memoryBase);
        CSRMatrix<ValType> Ai;
        dom_ptr.resize(num_dom * 2 + 1, RESERVE_NO_DATA);
        const UINT32 *rowIdxPtr = A->getRowOffsetPtr(0);
        const UINT32 *colIdxPtr = A->getColIndicesPtr(0);
        for (UINT32 i = 0; i < size; ++i) {
            for (UINT32 j = rowIdxPtr[i]; j < rowIdxPtr[i + 1]; ++j) {
                if (colIdxPtr[j] != i) {
                    adjncy[jj] = (idx_t) colIdxPtr[j];
                    adjwgt[jj] = 1;
                    jj++;
                } else {
                    vwgt[i] = 6;
                }
                if (vwgt[i] == 0) vwgt[i] = 6;
                xadj[i + 1] = (idx_t) jj;
            }
        }
//    图划分
        if (num_dom < 8) {
            METIS_PartGraphRecursive(&lnrows, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr,
                                     nullptr, &num_dom, nullptr, nullptr, nullptr, &edgecut, lmap.data());
        } else {
            METIS_PartGraphKway(&lnrows, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr, nullptr,
                                &num_dom, nullptr, nullptr, nullptr, &edgecut, lmap.data());
        }
        jj = 0;
        for (UINT32 i = 0; i < num_dom; ++i) {
            for (UINT32 j = 0; j < size; ++j) {
                if (lmap[j] == i) {
                    perm[jj] = j;
                    jj++;
                }
            }
            dom_ptr[i + 1] = jj;
        }

        for (UINT32 i = 0; i < num_dom; ++i) {
            if ((dom_ptr[i + 1] - dom_ptr[i]) == 0)
                exit(-1);
        }
        std::cout << "edgecut: " << edgecut << std::endl;
        for (UINT32 i = 0; i < num_dom; ++i) {
            for (UINT32 j = dom_ptr[i]; j < dom_ptr[i + 1]; ++j) {
                for (UINT32 k = xadj[perm[j]]; k < xadj[perm[j] + 1]; ++k) {
                    if (lmap[adjncy[k]] != i) {
                        for (UINT32 l = 0; l < num_dom; ++l) {
                            if (lmap[adjncy[k]] == l)
                                lmap[adjncy[k]] = num_dom + l;
                        }
                    }
                }
            }
        }
        jj = 0;
        for (UINT32 i = 0; i < num_dom * 2; ++i) {
            for (UINT32 j = 0; j < size; ++j) {
                if (lmap[j] == i) {
                    perm[jj] = j;
                    jj++;
                }
            }
            dom_ptr[i + 1] = jj;
        }
        perm.printVector("prem: ");
        A->getSubMatrix(perm, perm, Ai);
        *A_out = Ai;
    }
}
#endif //SOLVERCHALLENGE_METISTOOLS_H
