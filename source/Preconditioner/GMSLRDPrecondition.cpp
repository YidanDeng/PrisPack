/*
 * @author  袁心磊、邓轶丹
 * @date    2024/11/16
 */

#include "../../include/Preconditioner/GMSLRPrecondition.h"

#include "../../include/utils/ExternalTools/MetisTools.h"

namespace HOST {
    template<typename LowPrecisionType, typename HighPrecisionType>
    GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::GMSLRPrecondition(
            const std::shared_ptr<CSRMatrix<HighPrecisionType> > &matA, INT32 levelNum,
            MatrixReorderOption_t localReorderType) {
        this->m_precondType = PreconditionGMSLR;
        m_matA = matA;
        this->m_ArowNum = matA->getRowNum();
        this->m_AcolNum = matA->getColNum();
        this->m_Annz = matA->getNNZnum(0, this->m_ArowNum - 1);
        m_nlev_setup = levelNum;
        m_local_ordering_setup = localReorderType;
        m_pperm.resize(this->m_ArowNum, RESERVE_NO_DATA);
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::GMSLRPrecondition(
            const std::shared_ptr<CSRMatrix<HighPrecisionType> > &matA, INT32 levelNum, INT32 lowRankSize,
            MatrixReorderOption_t localReorderType) {
        this->m_precondType = PreconditionGMSLR;
        m_matA = matA;
        this->m_ArowNum = matA->getRowNum();
        this->m_AcolNum = matA->getColNum();
        this->m_Annz = matA->getNNZnum(0, this->m_ArowNum - 1);
        m_nlev_setup = levelNum;
        m_lowRankSize = lowRankSize;
        m_local_ordering_setup = localReorderType;
        m_pperm.resize(this->m_ArowNum, RESERVE_NO_DATA);
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::GMSLRPrecondition(
            const std::shared_ptr<CSRMatrix<HighPrecisionType> > &matA, INT32 levelNum, INT32 lanczosSteps,
            INT32 lowRankSize, MatrixReorderOption_t localReorderType) {
        this->m_precondType = PreconditionGMSLR;
        m_matA = matA;
        this->m_ArowNum = matA->getRowNum();
        this->m_AcolNum = matA->getColNum();
        this->m_Annz = matA->getNNZnum(0, this->m_ArowNum - 1);
        m_nlev_setup = levelNum;
        m_lowRankSize = lowRankSize;
        m_lanczosSteps = lanczosSteps;
        m_local_ordering_setup = localReorderType;
        m_pperm.resize(this->m_ArowNum, RESERVE_NO_DATA);
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::GMSLRPrecondition(
            const std::shared_ptr<CSRMatrix<HighPrecisionType> > &matA, INT32 levelNum, INT32 lanczosSteps,
            INT32 lowRankSize, HighPrecisionType eigCorrectBound, MatrixReorderOption_t localReorderType) {
        this->m_precondType = PreconditionGMSLR;
        m_matA = matA;
        this->m_ArowNum = matA->getRowNum();
        this->m_AcolNum = matA->getColNum();
        this->m_Annz = matA->getNNZnum(0, this->m_ArowNum - 1);
        m_nlev_setup = levelNum;
        m_lowRankSize = lowRankSize;
        m_lanczosSteps = lanczosSteps;
        m_local_ordering_setup = localReorderType;
        m_pperm.resize(this->m_ArowNum, RESERVE_NO_DATA);
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::setupPermutationND(HostVector<INT32> &map_v,
                                                                                    HostVector<INT32> &mapptr_v) {
        UINT32 tlvl, clvl, nrows, i, j, k, size1, size2;
        INT32 domi;
        SharedPtr2D<AutoAllocateVector<UINT32> > level_str;
        nrows = m_matA->getRowNum();
        tlvl = m_nlev_setup;
        clvl = 0;
        //递归划分，level_str传入时为空，传出时给每个子指针分配对应的对象
        setupPermutationNDRecursive(*m_matA, clvl, tlvl, level_str);
        //将level_str填入map_v和mapptr_v
        map_v.resize(nrows, RESERVE_NO_DATA); //每个节点所在区域
        mapptr_v.resize(tlvl + 1, RESERVE_NO_DATA);
        //每个level包含几个区域【0，8，12，14，15】level 1有8个区域mapptr_v[1]-mapptr_v[0]
        m_nlev_max = tlvl;
        m_nlev_used = tlvl;
        domi = 0;
        mapptr_v[0] = 0;
        for (i = 0; i < m_nlev_used; ++i) {
            size1 = level_str[i].getDim();
            for (j = 0; j < size1; ++j) {
                size2 = level_str[i][j]->getLength();
                for (k = 0; k < size2; ++k) {
                    map_v[level_str(i, j)[k]] = domi;
                }
                domi++;
            }
            mapptr_v[i + 1] = domi;
        }
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::buildLevelStructure(HostVector<INT32> &map_v,
                                                                                     HostVector<INT32> &mapptr_v) {
        INT32 n, i, j, nlev_used, nlev_max, ndom, level, ncomp, maps, mape;
        UINT32 ni, n_start, n_end, n_local, n_remain;
        AutoAllocateVector<UINT32> temp_perm, local_perm;
        CSRMatrix<HighPrecisionType> &A = *m_matA;
        CSRMatrix<HighPrecisionType> Apq;

        n = A.getRowNum();
        nlev_used = m_nlev_used;
        nlev_max = m_nlev_max;

        ndom = mapptr_v[nlev_max]; //一共ndom个区域
        AutoAllocateVector<INT32> domptr_v(ndom + 1, memoryBase); //
        AutoAllocateVector<INT32> domsize_v(ndom, memoryBase); //每个区域的size（有几个节点）
        for (i = 0; i < n; ++i) {
            domsize_v[map_v[i]]++;
        }
        domptr_v[0] = 0;
        domptr_v[1] = 0;
        for (i = 2; i < ndom + 1; ++i) {
            domptr_v[i] = domptr_v[i - 1] + domsize_v[i - 2];
        }
        for (i = 0; i < n; ++i) {
            m_pperm[domptr_v[map_v[i] + 1]++] = i; //pperm为重排序向量，下标为现在的节点编号，值为对应的节点原始的编号
        }

        m_levs_all.realloc(nlev_used);
        m_lev_ptr_v.resize(nlev_used + 1, RESERVE_NO_DATA); //记录每个level开始的节点
        if (m_lev_ptr_v->getMemoryType() != memoryBase) m_lev_ptr_v->fillVector(0, m_lev_ptr_v.getLength(), 0);
        m_dom_ptr_v2.realloc(nlev_used); //SharedPtr1D<AutoAllocateVector>结构，记录每个level每个区域开始的节点
        m_lev_ptr_v[nlev_used] = n;

        for (level = 0; level < nlev_used; ++level) {
            m_levs_all.construct(level); // 创建空对象，其中的成员变量全部使用默认构造函数
            m_dom_ptr_v2.construct(level); // 构造空的vector对象，内存模式使用base（即calloc分配的内存）
            GMSLRLevelStructure<HighPrecisionType> &level_str = *m_levs_all[level];
            // 如果是最后一个level
            if (level == nlev_used - 1) {
                mape = mapptr_v[nlev_max];
                maps = mapptr_v[level]; // 最后一个level是从第几个区域开始的
                ncomp = mape - maps; // 最后一个level的区域个数
                level_str.ncomps = ncomp;
                m_lev_ptr_v[level] = domptr_v[maps]; // domptr_v是每个区域开始的节点编号，这里求出最后一个level开始的节点编号
                m_dom_ptr_v2[level]->resize(ncomp + 1, RESERVE_NO_DATA);
                for (i = 0, j = maps; i <= ncomp; ++i, ++j) {
                    (*m_dom_ptr_v2[level])[i] = domptr_v[j];
                }
            } else {
                // 其他level
                mape = mapptr_v[level + 1];
                maps = mapptr_v[level];
                ncomp = mape - maps; // 这个level的区域个数
                level_str.ncomps = ncomp;
                m_lev_ptr_v[level] = domptr_v[maps];
                m_dom_ptr_v2[level]->resize(ncomp + 1, RESERVE_NO_DATA);
                for (i = 0, j = maps; i <= ncomp; ++i, ++j) {
                    (*m_dom_ptr_v2[level])[i] = domptr_v[j];
                }
            }
            // 添加局部排序使得分块子矩阵带宽变小，例如RCM排序或AMD排序（目前只支持这两种）
            for (i = 0; i < ncomp; ++i) {
                ni = (*m_dom_ptr_v2[level])[i + 1] - (*m_dom_ptr_v2[level])[i]; //该区域的节点编号
                reorderSubMatrixCSR(m_local_ordering_setup, A, *m_pperm, (*m_dom_ptr_v2[level])[i], ni);
            }
        }

        // 根据重排向量，重新排序整个A矩阵
        A.getSubMatrix(*m_pperm, *m_pperm, Apq);
#ifndef NDEBUG
        INT32 err = checkPermutation(Apq);
        THROW_EXCEPTION(err != GMSLR_SUCCESS, THROW_LOGIC_ERROR("The permutation of HID failed!"))
        writePermMat(Apq);
#endif
        //提取子矩阵
        UINT32 subMatStartRowNo, subMatEndRowNo, subMatStartColNo, subMatEndColNo;
        UINT32 maxSubdomainSize = 0;
        for (level = 0; level < nlev_used; ++level) {
            GMSLRLevelStructure<HighPrecisionType> &level_str = *m_levs_all[level];
            if (level == nlev_used - 1) {
                //最后一个level
                mape = mapptr_v[nlev_max];
                maps = mapptr_v[level];
                ncomp = mape - maps;
                level_str.ncomps = ncomp;

                n_start = domptr_v[maps]; //这一level开始的节点
                n_end = domptr_v[mape]; //这一level结束的节点
                n_local = n_end - n_start;
                maxSubdomainSize = std::max(maxSubdomainSize, n_local);
                // n_remain = n - n_end;
                std::cout << " --- build level structure, level: " << level << ", comps: " << ncomp << std::endl;

                level_str.B_mat.realloc(ncomp);
                for (i = 0; i < ncomp; ++i) {
                    level_str.B_mat.construct(i);
                    subMatStartRowNo = (*m_dom_ptr_v2[level])[i];
                    subMatEndRowNo = (*m_dom_ptr_v2[level])[i + 1] - 1;
                    subMatStartColNo = subMatStartRowNo;
                    subMatEndColNo = subMatEndRowNo;
                    Apq.getSubMatrix(subMatStartRowNo, subMatEndRowNo, subMatStartColNo, subMatEndColNo,
                                     *level_str.B_mat[i]);
                }
            } else {
                //其他level
                mape = mapptr_v[level + 1];
                maps = mapptr_v[level];
                ncomp = mape - maps;
                level_str.ncomps = ncomp;
                std::cout << " --- build level structure, level: " << level << ", comps: " << ncomp << std::endl;

                n_start = domptr_v[maps]; //这一level开始的节点
                n_end = domptr_v[mape]; //这一level结束的节点
                n_local = n_end - n_start;
                n_remain = n - n_end;
                maxSubdomainSize = std::max(maxSubdomainSize, n_local);

                level_str.B_mat.realloc(ncomp);
                for (i = 0; i < ncomp; ++i) {
                    level_str.B_mat.construct(i);
                    subMatStartRowNo = (*m_dom_ptr_v2[level])[i];
                    subMatEndRowNo = (*m_dom_ptr_v2[level])[i + 1] - 1;
                    subMatStartColNo = subMatStartRowNo;
                    subMatEndColNo = subMatEndRowNo;
                    Apq.getSubMatrix(subMatStartRowNo, subMatEndRowNo, subMatStartColNo, subMatEndColNo,
                                     *level_str.B_mat[i]);
                }
                //提取E矩阵
                //E的size是行数：n_end-n_start(n_local)列数：n-n_end(n_remain)
                subMatStartRowNo = n_start;
                subMatEndRowNo = n_start + n_local - 1;
                subMatStartColNo = n_end;
                subMatEndColNo = n_end + n_remain - 1;
                Apq.getSubMatrix(subMatStartRowNo, subMatEndRowNo, subMatStartColNo, subMatEndColNo, level_str.E_mat);
                Apq.getSubMatrix(subMatStartColNo, subMatEndColNo, subMatStartRowNo, subMatEndRowNo, level_str.F_mat);
                //提取C矩阵
                subMatStartRowNo = n_end;
                subMatEndRowNo = n_end + n_remain - 1;
                subMatStartColNo = n_end;
                subMatEndColNo = n_end + n_remain - 1;
                Apq.getSubMatrix(subMatStartRowNo, subMatEndRowNo, subMatStartColNo, subMatEndColNo, level_str.C_mat);
            }
        }
        // 这里分配辅助变量空间，为后续并行求解B块做准备，分配原则是在不同块之间留够两个缓存行的间隙
        // 这个大小取最大B块行数加两倍最大并行域个缓存行大小，这个大小足以涵盖后续所有有关B逆求解的部分，因此后续无需再重新调整空间大小
        m_auxParallelSolveB.reset(
                maxSubdomainSize + m_levs_all[0]->ncomps * 2 * ALIGNED_BYTES / sizeof(HighPrecisionType),
                memoryAligned);
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::setupPermutationNDRecursive(
            CSRMatrix<HighPrecisionType> &A, UINT32 clvl, UINT32 &tlvl,
            SharedPtr2D<AutoAllocateVector<UINT32> > &level_str) {
        UINT32 i, j, ncomps, nS, ndom, edgecut, idx, size, k, k2, k1;
        DenseVector<UINT32> map, sep, perm, dom_ptr, row_perm, col_perm;
        SharedPtr1D<AutoAllocateVector<UINT32> > comp_indices;
        CSRMatrix<HighPrecisionType> B, C;
        /* 若只有一个水平 */
        if (tlvl < 2) {
            tlvl = 1;
            level_str.realloc(1);
            level_str[0].realloc(1);
            level_str[0][0]->resize(A.getRowNum(), RESERVE_NO_DATA);
            // 使用 std::iota 填充序列，该函数会将范围 [first, last) 中的元素填充为从 value 开始的递增序列。
            std::iota(level_str(0, 0).getRawValPtr(), level_str(0, 0).getRawValPtr() + m_matA->getRowNum(), 0);
            // level_str[0][0].UnitPerm();
            return;
        }

        if (clvl >= tlvl - 1) {
            //找到原始矩阵所有的连通分支
            ncomps = 0;
            getConnectedComponents(A, comp_indices, ncomps);
            tlvl = 1;
            level_str.realloc(1);
            level_str[0].realloc(ncomps);
            for (i = 0; i < ncomps; ++i) {
                level_str[0][i] = comp_indices[i];
            }
            return;
        }
        UINT32 levelStrDim = tlvl - clvl;
        level_str.realloc(levelStrDim);
        AutoAllocateVector<UINT32> currLevelStrCount(levelStrDim, memoryBase);
        //第一步，找到原始矩阵所有连通分支
        ncomps = 0;
        getConnectedComponents(A, comp_indices, ncomps); // 得到最终ncomps
        //第二步，对每个连通分支进行图划分
        AutoAllocateVector<UINT32> tlvls(ncomps, memoryBase);
        // 给指针开辟足够的空间
        for (i = 0; i < levelStrDim; ++i) {
            level_str[i].realloc(ncomps);
        }
        for (i = 0; i < ncomps; ++i) {
            //检查当前连通分支的size，若很小，则stop
            nS = comp_indices[i]->getLength();
            if (nS <= m_minsep || nS <= 2) {
                tlvls[i] = 1;
                if (currLevelStrCount[0] >= level_str[0].getDim())
                    level_str[0].realloc(std::max(ncomps, currLevelStrCount[0]) * 2);
                level_str[0][currLevelStrCount[0]++] = comp_indices[i];
                // level_str[0].push_back(std::move(comp_indices[i]));
                // continue;
            } else {
                A.getSubMatrix(**comp_indices[i], **comp_indices[i], C);
                ndom = 2;
                if (csrMatrixMetisKwayHID(C, ndom, map, sep, edgecut, perm, dom_ptr) == 1) {
                    //此时至少有一个子域没有内点，则应该停止
                    edgecut = nS;
                }
                //没有两个子域或者至少有一个子域没有内点，stop
                if (ndom < 2 || edgecut == nS) {
                    tlvls[i] = 1;
                    if (currLevelStrCount[0] >= level_str[0].getDim())
                        level_str[0].realloc(std::max(ncomps, currLevelStrCount[0]) * 2);
                    level_str[0][currLevelStrCount[0]++] = comp_indices[i];
                    // level_str[0].push_back(std::move(comp_indices[i]));
                    continue;
                }
                // 根据标记向量（里面只有0和1）来提取对应的行和列
                getSubMatrixNoPerm(C, sep, sep, row_perm, col_perm, true, B);
                // C.getSubMatrixNoPerm(sep, sep, row_perm, col_perm, true, B);
                //把seperator移除得到矩阵B,true为提取sep为零元素位置，即不在分割符内的
                //下一水平
                tlvls[i] = tlvl;
                // std::vector<std::vector<vec::GeneralVector<UINT32> > > sub_level_str;
                SharedPtr2D<AutoAllocateVector<UINT32> > sub_level_str;
                setupPermutationNDRecursive(B, clvl + 1, tlvls[i], sub_level_str);

                //放入内点
                for (j = 0; j < tlvls[i]; j++) {
                    idx = j;
                    size = sub_level_str[idx].getDim();
                    for (k = 0; k < size; k++) {
                        if (currLevelStrCount[idx] >= level_str[idx].getDim())
                            level_str[idx].realloc(std::max(ncomps, currLevelStrCount[idx]) * 2);
                        level_str[idx][currLevelStrCount[idx]++] = sub_level_str[idx][k];
                        // level_str[idx].push_back(std::move(sub_level_str[idx][k]));
                        // 这里直接使用shared指针，所以老代码的back就是sub_level_str[idx][k]指向的对象
                        AutoAllocateVector<UINT32> &nodes = *sub_level_str[idx][k];
                        // vec::GeneralVector<UINT32> &nodes = level_str[idx].back();
                        k2 = nodes.getLength();
                        for (k1 = 0; k1 < k2; k1++) {
                            nodes[k1] = (*comp_indices[i])[row_perm[nodes[k1]]];
                        }
                    }
                }
                //放入边分隔符
                idx = tlvls[i];
                // level_str[idx].push_back(vec::GeneralVector<UINT32>());
                if (currLevelStrCount[idx] >= level_str[idx].getDim())
                    level_str[idx].realloc(std::max(ncomps, currLevelStrCount[idx]) * 2);
                level_str[idx].construct(currLevelStrCount[idx], 0, memoryBase);
                // vec::GeneralVector<UINT32> &nodes = level_str[idx].back();
                AutoAllocateVector<UINT32> &nodes = level_str(idx, currLevelStrCount[idx]);
                currLevelStrCount[idx]++;
                nodes.resize(edgecut, RESERVE_NO_DATA);
                UINT32 currIdx = 0;
                for (j = 0; j < nS; j++) {
                    if (sep[j] != 0) {
                        if (currIdx >= nodes.getLength())
                            nodes.resize(std::max(currIdx, (UINT32) 1024) * 2,
                                         RESERVE_DATA);
                        nodes[currIdx++] = (*comp_indices[i])[j];
                        // nodes.push_back(comp_indices[i][j], 1);
                    }
                }
                // 把nodes的大小还原为实际大小
                nodes.resize(currIdx, RESERVE_DATA);
                tlvls[i] += 1;
            }
        }
        // tlvl = tlvls.max();
        // 调整level_str为实际大小
        for (i = 0; i < levelStrDim; ++i) {
            level_str[i].realloc(currLevelStrCount[i]);
        }
        tlvl = *std::max_element(tlvls.getRawValPtr(), tlvls.getRawValPtr() + tlvls.getLength());
    }


    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::getConnectedComponents(CSRMatrix<HighPrecisionType> &A,
                                                                                        SharedPtr1D<AutoAllocateVector<UINT32> > &comp_indices,
                                                                                        UINT32 &ncomps) {
        UINT32 n, i, j, current_comps, qs, qe, size, idx, idx2, accumcomp, index1, index2;
        n = A.getRowNum();
        AutoAllocateVector<INT32> marker(n, memoryBase); // 记录节点是否被访问
        marker->fillVector(0, n, -1); // 初始化marker向量为-1
        AutoAllocateVector<UINT32> queue(n, memoryBase); // 记录遍历顺序（BFS寻找连通分支）
        AutoAllocateVector<UINT32> comps_size(10, memoryBase); // 记录每个连通分支的size

        // 当前连通分支数，从0开始
        current_comps = 0; // 统计实际写入comps的元素个数
        // UINT32 comps_size_count = 0;
        for (i = 0; i < n; ++i) {
            if (marker[i] < 0) {
                queue[0] = i; // 连通分支的第一个节点为i，qe代表此时放入队列的点个数（已访问的节点个数）
                qs = 0;
                qe = 1;
                size = 0;
                marker[i] = current_comps;
                size++;
                while (qe > qs) {
                    idx = queue[qs];
                    // 寻找idx相邻接的节点，若还没有被访问，则放入queue中
                    index1 = *A.getRowOffsetPtr(idx);
                    index2 = *A.getRowOffsetPtr(idx + 1);
                    // index1 = this->m_rowOffset[idx];
                    // index2 = this->m_rowOffset[idx + 1];
                    UINT32 *colIdxPtr = A.getColIndicesPtr(0);
                    for (j = index1; j < index2; ++j) {
                        idx2 = colIdxPtr[j];
                        if (marker[idx2] < 0) {
                            queue[qe++] = idx2;
                            marker[idx2] = current_comps;
                            size++;
                        }
                    }
                    // 寻找下一个节点及其相邻接的点
                    qs++;
                }

                // 此时该连通分支已经全部遍历（访问）
                // comps_size.push_back(size, 1);
                if (current_comps >= comps_size.getLength())
                    comps_size.resize(
                            std::max(current_comps, static_cast<UINT32>(1024)) * 2, RESERVE_DATA);
                comps_size[current_comps] = size;
                current_comps++;
            }
        }
        // 若找到的连通分支总数比需要的连通分支多，则合并连通分支
        AutoAllocateVector<UINT32> currIdxCount; // 用来统计comp_indices[i]实际写入元素个数
        if (ncomps > 1 && ncomps < current_comps) {
            accumcomp = 0;
            j = 0;
            AutoAllocateVector<UINT32> comps_size_adj(ncomps, memoryBase); // 合并后的分支有几个元素（记录合并后的每个连通分支的size）
            AutoAllocateVector<UINT32> comps_map(current_comps, memoryBase);
            // 记录映射：第marker[i]个连通分支--->现在是第几（comps_map[marker[i]]）个连通分支
            // 使用 std::iota 填充序列，该函数会将范围 [first, last) 中的元素填充为从 value 开始的递增序列。
            AutoAllocateVector<UINT32> order(current_comps, memoryBase); // current_comps即为comps_size中实际存储的元素个数
            AutoAllocateVector<UINT32> orderTmp1(current_comps, memoryBase);
            AutoAllocateVector<UINT32> orderTmp2(current_comps, memoryBase);
            std::iota(order->getRawValPtr(), order->getRawValPtr() + current_comps, 0);
            // 将连通分支的size升序排序，order记录原下标，两个数组一起排序，以便选择合并策略
            mergeSortVectorPair(&comps_size[0], &order[0], &orderTmp1[0], &orderTmp2[0], current_comps);
            // comps_size.Sort(order, true,false);
            // 用完临时空间就清空
            orderTmp1.clear();
            orderTmp2.clear();
            for (i = 0; i < current_comps; ++i) {
                accumcomp += comps_size[order[i]]; // 记录节点个数
                comps_map[order[i]] = j;

                if (accumcomp > (n / ncomps + 1)) {
                    comps_size_adj[j] = accumcomp;
                    accumcomp = 0;
                    j++; // j代表合并后的第几个连通分支
                }
                // ncomps-j为还要生成几个连通分支，current_comps-i为现在还未合并的连通分支数,若两者相等，则停止合并
                if ((current_comps - i) == (ncomps - j)) {
                    comps_size_adj[j] = accumcomp;
                    i++;
                    j++;
                    for (; i < current_comps; ++i) {
                        comps_size_adj[j] = comps_size[order[i]];
                        comps_map[order[i]] = j;
                        j++;
                    }
                }
            }
            // 将结果填入comp_indices中
            comp_indices.realloc(ncomps); // 先重新调整comp_indices集合的大小
            currIdxCount.resize(ncomps, RESERVE_NO_DATA); // 用来统计comp_indices[i]实际写入元素个数
            for (i = 0; i < ncomps; ++i) {
                /* 这里的写法和老代码不一样，这里先预分配存储空间大小 */
                comp_indices.construct(i, comps_size_adj[i], memoryBase);
                // comp_indices[i].resize(0,comps_size_adj[i], 0, 1);
            }
            for (i = 0; i < n; ++i) {
                idx = comps_map[marker[i]];
                if (currIdxCount[idx] >= comp_indices[idx]->getLength())
                    comp_indices[idx]->resize(
                            std::max(currIdxCount[idx], static_cast<UINT32>(1024)) * 2, RESERVE_DATA);
                (*comp_indices[idx])[currIdxCount[idx]++] = i;
                // comp_indices[idx].push_back(i, 0);
            }
            for (i = 0; i < ncomps; ++i) {
                // comp_indices[i].Sort(true);
                std::sort(comp_indices[i]->getRawValPtr(), comp_indices[i]->getRawValPtr() + currIdxCount[i],
                          std::less<>());
            }
        } else {
            comp_indices.realloc(current_comps);
            currIdxCount.resize(current_comps, RESERVE_NO_DATA); // 用来统计comp_indices[i]实际写入元素个数
            for (i = 0; i < current_comps; ++i) {
                comp_indices.construct(i, comps_size[i], memoryBase);
                // comp_indices[i].resize(0,comps_size[i], 0, 1);
            }
            for (i = 0; i < n; ++i) {
                idx = marker[i];
                if (currIdxCount[idx] >= comp_indices[idx]->getLength())
                    comp_indices[idx]->resize(
                            std::max(currIdxCount[idx], static_cast<UINT32>(1024)) * 2, RESERVE_DATA);
                (*comp_indices[idx])[currIdxCount[idx]++] = i;
                // comp_indices[idx].push_back(i, 0);
            }
            for (i = 0; i < current_comps; ++i) {
                std::sort(comp_indices[i]->getRawValPtr(), comp_indices[i]->getRawValPtr() + currIdxCount[i],
                          std::less<>());
                // std::sort(&(*comp_indices[i])[0], &(*comp_indices[i])[currIdxCount[i]], std::less<>());
                // comp_indices[i].Sort(true);
            }
            ncomps = current_comps;
        }
        // 将每个向量还原为实际存储大小
        UINT32 comp_indices_dim = comp_indices.getDim();
        for (UINT32 i = 0; i < comp_indices_dim; ++i) {
            comp_indices[i]->resize(currIdxCount[i], RESERVE_DATA);
        }
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::getSubMatrixNoPerm(CSRMatrix<HighPrecisionType> &A,
                                                                                    HostVector<UINT32> &rows,
                                                                                    HostVector<UINT32> &cols,
                                                                                    HostVector<UINT32> &row_perm,
                                                                                    HostVector<UINT32> &col_perm,
                                                                                    bool complement,
                                                                                    CSRMatrix<HighPrecisionType> &csr_outmat) {
        UINT32 i, nr, nc, rowNum, colNum;
        rowNum = A.getRowNum();
        colNum = A.getColNum();
        row_perm.resize(rowNum, RESERVE_NO_DATA);
        col_perm.resize(colNum, RESERVE_NO_DATA);
        if (row_perm.getMemoryType() != memoryBase) row_perm.fillVector(0, rowNum, 0);
        if (col_perm.getMemoryType() != memoryBase) col_perm.fillVector(0, colNum, 0);
        nr = 0;
        nc = 0;
        if (complement) {
            for (i = 0; i < rowNum; ++i) {
                if (rows[i] == 0) {
                    row_perm[nr] = i;
                    nr++;
                }
            }
            for (i = 0; i < colNum; ++i) {
                if (cols[i] == 0) {
                    col_perm[nc] = i;
                    nc++;
                }
            }
        } else {
            for (i = 0; i < rowNum; ++i) {
                if (rows[i] != 0) {
                    row_perm[nr] = i;
                    nr++;
                }
            }
            for (i = 0; i < colNum; ++i) {
                if (cols[i] != 0) {
                    col_perm[nc] = i;
                    nc++;
                }
            }
        }
        row_perm.resize(nr, RESERVE_DATA);
        col_perm.resize(nc, RESERVE_DATA);
        A.getSubMatrix(row_perm, col_perm, csr_outmat);
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    INT32 GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::checkPermutation(
            const CSRMatrix<HighPrecisionType> &permA) {
        if (m_pperm.getLength() != m_matA->getRowNum() && m_pperm.getLength() != m_matA->getColNum()) {
            SHOW_ERROR("The permutation array length was incorrect!")
            return GMSLR_FAILED;
        }
        AutoAllocateVector<UINT32> checkPerm(m_pperm.getLength(), memoryBase);
        for (UINT32 i = 0; i < m_pperm.getLength(); ++i) {
            if (checkPerm[m_pperm[i]] != 0) {
                SHOW_ERROR("The same row and column of original matrix A was used more than once!")
                return GMSLR_FAILED;
            }
            checkPerm[m_pperm[i]]++;
        }
        AutoAllocateVector<UINT32> pperm_reverse(this->m_ArowNum, memoryBase);
        // 生成重排向量的逆过程
        for (UINT32 i = 0; i < this->m_ArowNum; ++i) {
            pperm_reverse[m_pperm[i]] = i;
        }
        CSRMatrix<HighPrecisionType> recoverMat;
        permA.getSubMatrix(*pperm_reverse, *pperm_reverse, recoverMat);

        UINT32 rowNum = m_matA->getRowNum();
        UINT32 colNum = m_matA->getColNum();
        UINT32 nnzNum = m_matA->getNNZnum(0, rowNum - 1);
        UINT32 *A_rowOffsetPtr = m_matA->getRowOffsetPtr(0);
        UINT32 *A_colIdxPtr = m_matA->getColIndicesPtr(0);
        HighPrecisionType *A_valuesPtr = m_matA->getCSRValuesPtr(0);
        UINT32 *recoverA_rowOffsetPtr = recoverMat.getRowOffsetPtr(0);
        UINT32 *recoverA_colIdxPtr = recoverMat.getColIndicesPtr(0);
        HighPrecisionType *recoverA_valuesPtr = recoverMat.getCSRValuesPtr(0);
        if (recoverMat.getRowNum() != rowNum || recoverMat.getColNum() != colNum) {
            SHOW_ERROR("The dim of permuted matrix is not equal to original matrix!")
            return GMSLR_FAILED;
        }

        bool isPass = checkAnswerWithReturnValue(recoverA_rowOffsetPtr, A_rowOffsetPtr, rowNum + 1,
                                                 "check perm A row-offset");
        isPass &= checkAnswerWithReturnValue(recoverA_colIdxPtr, A_colIdxPtr, nnzNum, "check perm A col-indices");
        isPass &= checkAnswerWithReturnValue(recoverA_valuesPtr, A_valuesPtr, nnzNum, "check perm A col-indices");
        if (recoverMat.getRowNum() != rowNum || recoverMat.getColNum() != colNum) {
            SHOW_ERROR("The dim of permuted matrix is not equal to original matrix!")
            return GMSLR_FAILED;
        }
        if (!isPass) {
            SHOW_ERROR("The permutation result is incorrect!")
            return GMSLR_FAILED;
        }
        return GMSLR_SUCCESS;
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::writePermMat(
            const CSRMatrix<HighPrecisionType> &permMat) {
        SHOW_INFO("Write matrix after HID ordering to the file...")
        /* 生成WriteMtxTools */
        char pathBuffer[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", pathBuffer, sizeof(pathBuffer) - 1);
        if (len != -1) {
            pathBuffer[len] = '\0'; // 确保以 null 结尾
        } else {
            TRY_CATCH(THROW_LOGIC_ERROR("Could not get absolute path of current exe file!"))
        }
        SHOW_INFO("File path: ../source/test/testResults/AMSED_perm_mat.mtx")
        WriteMtxTools<HighPrecisionType> writePermMat(pathBuffer, "../source/test/testResults/AMSED_perm_mat.mtx", 1,
                                                      MTX_STORAGE_GENERAL);
        writePermMat.writeMatrix(permMat);
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::setupMSLR() {
        getLastLevelDecomposition();
        // 从倒数第二个水平开始处理
        INT32 level = static_cast<INT32>(m_nlev_used) - 2;
        while (level >= 0) {
            // 对每个水平的B块进行并行ILDLT分解
            getBlockDecomposition(level);
            // 构建低秩修正
            lowRankCorrection(level);
            level--;
        }
        // 在这里计算总的填充
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::getLastLevelDecomposition() {
        // 处理最后一个水平，即整个块执行IC分解
        GMSLRLevelStructure<HighPrecisionType> &lastLevel = *m_levs_all[m_nlev_used - 1];
        std::shared_ptr<CSRMatrix<HighPrecisionType> > lastC = lastLevel.B_mat[0];
        // 生成预条件
        lastLevel.B_precond.realloc(1);
        lastLevel.B_precond[0] = std::make_shared<IncompleteLU<LowPrecisionType, HighPrecisionType> >
                (lastC, GMSLR_LFILL, GMSLR_FACTOR_TOLRANCE);
#ifndef NDEBUG
        // 检查C块划分是否有误
        UINT32 n_start, n_end, n_local, n_remain, colnum;
        n_start = m_lev_ptr_v[m_nlev_used - 1];
        n_end = m_lev_ptr_v[m_nlev_used];
        n_local = n_end - n_start;
        n_remain = this->m_ArowNum - n_end;
        colnum = lastC->getRowNum();
        THROW_EXCEPTION(n_remain != 0 || n_local != colnum,
                        THROW_LOGIC_ERROR("The decomposition of last level was incorrect!"))
#endif
        // 执行IC分解
        lastLevel.B_precond[0]->setup(); // 通过基类指针调用子类方法
        m_incompFactNNZ += lastLevel.B_precond[0]->getPreconditionNonZeroCount();
        this->m_Mnnz += lastLevel.B_precond[0]->getPreconditionNonZeroCount();
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::getBlockDecomposition(INT32 level) {
#ifndef NDEBUG
        THROW_EXCEPTION(level < 0 || level >= m_nlev_used, THROW_LOGIC_ERROR("Input Level number is incorrect!"))
#endif
        GMSLRLevelStructure<HighPrecisionType> &level_str = *m_levs_all[level];
        UINT32 ncomp = level_str.ncomps;
        level_str.B_precond.realloc(ncomp);
#pragma omp parallel for default(none) proc_bind(master) num_threads(THREAD_NUM)  \
        shared(ncomp, level_str, BasePrecondition<HighPrecisionType>::m_Mnnz)
        for (UINT32 i = 0; i < ncomp; ++i) {
            auto localMat = level_str.B_mat[i];
            std::shared_ptr<TriangularPrecondition<HighPrecisionType> > localBlock = std::make_shared<IncompleteLU<
                    LowPrecisionType, HighPrecisionType> >(localMat, GMSLR_LFILL, GMSLR_FACTOR_TOLRANCE);
            localBlock->setup();
            level_str.B_precond[i] = localBlock;
        }
        // 统计M的非零元
        for (UINT32 i = 0; i < ncomp; ++i) {
            UINT32 localMnnz = level_str.B_precond[i]->getPreconditionNonZeroCount();
            m_incompFactNNZ += localMnnz;
            BasePrecondition<HighPrecisionType>::m_Mnnz += localMnnz;
        }
        // for (i = 0; i < ncomp; ++i) {
        //     level_str.B_precond[i] = std::make_shared<IncompleteLDLT<LowPrecisionType, HighPrecisionType> >
        //             (level_str.B_mat[i], AMSED_FACTOR_TOLRANCE);
        //     level_str.B_precond[i]->setup();
        //     // 统计M的非零元
        //     this->m_Mnnz += level_str.B_precond[i]->getPreconditionNonZeroCount();
        // }
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::lowRankCorrection(INT32 level) {
        DenseMatrix<HighPrecisionType> V;
        DenseMatrix<HighPrecisionType> H;
//        AutoAllocateVector<HighPrecisionType> D(m_lanczosSteps, memoryBase);
//        AutoAllocateVector<HighPrecisionType> E(m_lanczosSteps, memoryBase);
//        reorthogonalSecondLanczos(level, *D, *E, V);
//        computeLanczosEigenValuesAndEigenVectors(level, *D, *E, V);
        reorthogonalSecondArnoldi(level, H, V);
        computeArnoldiEigenValuesAndEigenVectors(level, H, V);
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::reorthogonalSecondLanczos(
            INT32 level, HostVector<HighPrecisionType> &D, HostVector<HighPrecisionType> &E,
            DenseMatrix<HighPrecisionType> &V) {
        if (D.getLength() < m_lanczosSteps) D.resize(m_lanczosSteps, RESERVE_NO_DATA);
        if (E.getLength() < m_lanczosSteps) E.resize(m_lanczosSteps, RESERVE_NO_DATA);
        GMSLRLevelStructure<HighPrecisionType> &level_str = *m_levs_all[level];
        CSRMatrix<HighPrecisionType> &currE = level_str.E_mat;
        CSRMatrix<HighPrecisionType> &currC = level_str.C_mat;

        UINT32 n_start = m_lev_ptr_v[level]; // 当前水平B块的起始行号
        UINT32 n_end = m_lev_ptr_v[level + 1]; // 当前水平B块的结束行号
        UINT32 n_local = n_end - n_start; // 当前水平B块的大小
        UINT32 n_remain = this->m_ArowNum - n_end; // 当前水平C块的大小
        UINT32 i, j;

        // 初始化
        HighPrecisionType beta{0.0}, alpha;
        //v1为标准向量，用于迭代初始化
        AutoAllocateVector<HighPrecisionType> v1(n_remain, memoryBase); /// 对应算法中的x_j
        AutoAllocateVector<HighPrecisionType> u1(n_remain, memoryBase); /// 对应算法中的z
        // 选取C-范数下的标准向量x1（v1）
        v1->fillVector(0, n_remain, 1.0);
        // 计算u1 = C * v1
        currC.MatPVec(*v1, *u1);
        //u_tmp=(C*v1,v1)
        HighPrecisionType u_tmp = u1->innerProduct(*v1);
        for (i = 0; i < n_remain; ++i) {
            v1[i] /= sqrt(u_tmp);
        }
        // 初始化z0、z1为全0向量，z0对应迭代过程中的z_{j-1}，z1对应z_j
        AutoAllocateVector<HighPrecisionType> z0(n_remain, memoryBase); // base模式的内存已经清零
        AutoAllocateVector<HighPrecisionType> z1(n_remain, memoryBase);
        //z1=C*v1
        currC.MatPVec(*v1, *z1);
        // 开始迭代
        V.resize(n_remain, m_lanczosSteps, RESERVE_NO_DATA);
        AutoAllocateVector<HighPrecisionType> v(n_remain, memoryBase); /// 对应算法中的x
        AutoAllocateVector<HighPrecisionType> temp_v(n_local, memoryBase);
        // AutoAllocateVector<HighPrecisionType> z1_tmp(n_remain, memoryBase);
        // AutoAllocateVector<HighPrecisionType> out_vec(n_remain, memoryBase);
        AutoAllocateVector<HighPrecisionType> w(n_remain, memoryBase); /// 对应算法中的w

        DenseMatrix<HighPrecisionType> reorth_v, reorth_z; /// 分别记录x、z的正交向量组
        // DenseMatrix<HighPrecisionType> reorth_w;
        reorth_v.resize(n_remain, m_lanczosSteps, RESERVE_NO_DATA);
        reorth_z.resize(n_remain, m_lanczosSteps, RESERVE_NO_DATA);
        // reorth_w.resize(n_remain, m_lanczosSteps, RESERVE_NO_DATA);

        UINT32 count = 0; // 记录实际迭代次数
        for (i = 0; i < m_lanczosSteps; ++i) {
            count++;
            /* 计算 x = E^{T} * B^{-1} * E * x_j - \beta_j * z_{j-1} */
            if (beta != 0.0) z0->scale(beta);
            v->fillVector(0, n_remain, 0.0); // 内存重置，防止后续transMatPVec计算异常
            // 计算v = E^{T} * B^{-1} * E * v1
            currE.MatPVec(*v1, *temp_v); // 计算temp_v = E * v1
            // 计算所有子块对应的temp_v = B^{-1} temp_v
            // levelBlockInvSolve(level, *temp_v, 0);
            levelBlockInvSolveWithCorrection(level, *temp_v, 0);
            currE.transMatPVec(*temp_v, *v); // 计算v = E^{T} * temp_v
            v.add(-1, *z0); // v -= z0
            /* 计算 \alpha_j = (x, x_j) */
            alpha = v->innerProduct(*v1);

            /* 计算 x -= \alpha_j * z_j */
            v.add(-1.0 * alpha, *z1); // 计算 v -= alpha * z1
            /* 计算 x 向量的重正交过程 */
            if (i > 1) {
                // x -= \sum{(x, xj) * zj}
                UINT32 k;
                FLOAT64 tmp_product;
                HighPrecisionType *reorth_v_ptr = reorth_v.getMatValPtr();
                HighPrecisionType *reorth_z_ptr = reorth_z.getMatValPtr();
                for (k = 0; k < i - 1; ++k) {
                    tmp_product = v->innerProduct(0, reorth_v_ptr, n_remain);
                    for (UINT32 idx = 0; idx < n_remain; ++idx) {
                        v[idx] -= reorth_z_ptr[idx] * tmp_product;
                    }
                    // v.add(-1, *out_vec);
                    reorth_v_ptr += n_remain;
                    reorth_z_ptr += n_remain;
                }
            }
            /* 计算 w = C^{-1} * x */
            w.copy(*v);
            // reorth_w.setValsByCol(i, *w);
            // levelCInvRecursiveSolve(level, *w, 0); // 计算 w = C_{l}^(-1) * v = A_{l+1}^{-1} * v
            levelCInvRecursiveSolveWithCorrection(level, *w, 0);

            /* 计算 \beta_{j+1} = sqrt((w, x))*/
            // 当beta小于或等于0，直接退出循环
            HighPrecisionType tmp = w->innerProduct(*v);
            if (tmp <= GMSLR_EXIT_ZERO)
                break;
            beta = sqrt(tmp);
            //将每次更新的alpha，beta填入D,E向量（即对称三对角矩阵的主对角线、副对角线上的值）
            D[i] = alpha; //D中存储了alpha_1,...,alpha_m
            E[i] = beta; //E中存储了beta_2,...,beta_(m+1),但我们只需要beta_2,...,beta_m
            //将v1向量放入V,则V为[v0,v1,v2,...,v(m-1)]
            V.setValsByCol(i, *v1);
            reorth_v.setValsByCol(i, *v1);
            reorth_z.setValsByCol(i, *z1);
            //v1=w/bata,z0=z1,z1=v/beta
            for (j = 0; j < n_remain; ++j) {
                v1[j] = w[j] / beta;
                z0[j] = z1[j]; // 令z_{j-1} = z_j
                z1[j] = v[j] / beta; // 更新z_{j}
            }
        }
        // 将D和E调整为实际的大小
        D.resize(count, RESERVE_DATA);
        E.resize(count, RESERVE_DATA);
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::reorthogonalSecondArnoldi(INT32 level,
                                                                                           DenseMatrix<HighPrecisionType> &H,
                                                                                           DenseMatrix<HighPrecisionType> &V) {
        if (H.getRowNum() < m_lanczosSteps) H.resize(m_lanczosSteps, m_lanczosSteps, RESERVE_NO_DATA);
        GMSLRLevelStructure<HighPrecisionType> &level_str = *m_levs_all[level];
        CSRMatrix<HighPrecisionType> &currE = level_str.E_mat;
        CSRMatrix<HighPrecisionType> &currF = level_str.F_mat;
        CSRMatrix<HighPrecisionType> &currC = level_str.C_mat;

        UINT32 n_start = m_lev_ptr_v[level]; // 当前水平B块的起始行号
        UINT32 n_end = m_lev_ptr_v[level + 1]; // 当前水平B块的结束行号
        UINT32 n_local = n_end - n_start; // 当前水平B块的大小
        UINT32 n_remain = this->m_ArowNum - n_end; // 当前水平C块的大小
        UINT32 i, j;

        // 初始化
        HighPrecisionType beta{0.0}, alpha, temp;
        //v1为标准向量，用于迭代初始化
        AutoAllocateVector<HighPrecisionType> v1(n_remain, memoryBase); /// 对应算法中的x_j
        AutoAllocateVector<HighPrecisionType> u1(n_remain, memoryBase); /// 对应算法中的z
        // 选取C-范数下的标准向量x1（v1）
        // 选取C-范数下的标准向量x1（v1）
        v1->fillVector(0, n_remain, 1.0);
        // 计算u1 = C * v1
        currC.MatPVec(*v1, *u1);
        //u_tmp=(C*v1,v1)
        HighPrecisionType u_tmp = u1->innerProduct(*v1);
        for (i = 0; i < n_remain; ++i) {
            v1[i] /= sqrt(u_tmp);
        }
        // 初始化z0、z1为全0向量，z0对应迭代过程中的z_{j-1}，z1对应z_j
        AutoAllocateVector<HighPrecisionType> z0(n_remain, memoryBase); // base模式的内存已经清零
        AutoAllocateVector<HighPrecisionType> z1(n_remain, memoryBase);
        AutoAllocateVector<HighPrecisionType> w(n_remain, memoryBase);
        AutoAllocateVector<HighPrecisionType> temp_v(n_local, memoryBase);
        //z1=C*v1
        currC.MatPVec(*v1, *z1);
        // 开始迭代
        V.resize(n_remain, m_lanczosSteps, RESERVE_NO_DATA);
        V.setValsByCol(0, *z1);
        UINT32 count = 0; // 记录实际迭代次数
        for (j = 0; j < m_lanczosSteps; ++j) {
            count++;
            // 计算v = E^{T} * B^{-1} * E * v1
            currE.MatPVec(*z1, *temp_v); // 计算temp_v = E * v1
            // 计算所有子块对应的temp_v = B^{-1} temp_v
            // levelBlockInvSolve(level, *temp_v, 0);
            levelBlockInvSolveWithCorrection(level, *temp_v, 0);
            currF.MatPVec(*temp_v, *w); // 计算v = E^{T} * temp_v
            for (i = 0; i <= j; i++) {
                V.getValsByCol(i, *z1);
                temp = z1->innerProduct(*w);
                H.setValue(i, j, temp);
                w.add(-temp, *z1);
            }
            levelCInvRecursiveSolveWithCorrection(level, *w, 0);
            temp = w->norm_2();
            if (j != m_lanczosSteps - 1) H.setValue(j + 1, j, temp);
            if (temp <= 1e-12) break;
            z1 = w;
            z1->scale(1 / temp);
            if (j != m_lanczosSteps - 1) V.setValsByCol(j + 1, *z1);
        }
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::computeLanczosEigenValuesAndEigenVectors(INT32 level,
                                                                                                          HostVector<HighPrecisionType> &D,
                                                                                                          HostVector<HighPrecisionType> &E,
                                                                                                          DenseMatrix<HighPrecisionType> &V) {
        UINT32 dimension = D.getLength();
        UINT32 k = std::min(dimension, static_cast<UINT32>(m_lowRankSize)); // 实际低秩矫正的阶数
#ifndef NDEBUG
        THROW_EXCEPTION(k == 0, THROW_LOGIC_ERROR("The param of low-rank correction is incorrect!"))
#endif
        GMSLRLevelStructure<HighPrecisionType> &level_str = *m_levs_all[level];
        HighPrecisionType lambda;
        int size = static_cast<int>(dimension);
        int ldz = size; //矩阵行数
        int m = k; //找到m个特征值
        HighPrecisionType tmp = std::is_same<HighPrecisionType, FLOAT64>::value
                                ? LAPACKE_dlamch('S')
                                : LAPACKE_slamch('S');
        HighPrecisionType abstol = 2 * tmp; //这个最佳选择为2*dlamch(S)

        AutoAllocateVector<HighPrecisionType> w(size, memoryBase); //前m个元素为升序排列的我们要的特征值
        DenseMatrix<HighPrecisionType> z(DenseMatColumnFirst, ldz, m, memoryBase); //对应的特征向量
        AutoAllocateVector<INT32> ifail(size, memoryBase);
        int info;
        // 传统MSLR低秩校正，不修正特征值，直接使用k个特征值构造校正过程
        // 设置lapack参数，并求特征值
        char JOBZ = 'V';
        char RANGE = 'I';
        HighPrecisionType vl, vu;
        int il = size - k + 1;
        int iu = size;

        if (std::is_same<HighPrecisionType, FLOAT64>::value) {
            info = LAPACKE_dstevx(LAPACK_COL_MAJOR, JOBZ, RANGE, size, (FLOAT64 *) D.getRawValPtr(),
                                  (FLOAT64 *) E.getRawValPtr(),
                                  vl, vu, il, iu, abstol, &m, (FLOAT64 *) w.getRawValPtr(),
                                  (FLOAT64 *) z.getMatValPtr(), ldz,
                                  ifail.getRawValPtr());
        } else {
            info = LAPACKE_sstevx(LAPACK_COL_MAJOR, JOBZ, RANGE, size, (FLOAT32 *) D.getRawValPtr(),
                                  (FLOAT32 *) E.getRawValPtr(),
                                  vl, vu, il, iu, abstol, &m, (FLOAT32 *) w.getRawValPtr(),
                                  (FLOAT32 *) z.getMatValPtr(), ldz,
                                  ifail.getRawValPtr());
        }
        // 生成H和W
        level_str.H_vec.resize(m, RESERVE_NO_DATA);
        level_str.W_mat.resize(V.getRowNum(), m, RESERVE_NO_DATA);
        AutoAllocateVector<HighPrecisionType> tmpvec(ldz, memoryBase);
        AutoAllocateVector<HighPrecisionType> out_vec;
        for (INT32 i = m - 1; i >= 0; --i) {
            lambda = w[i]; //取出最大特征值
            level_str.H_vec[m - i - 1] = lambda / (1.0 - lambda); //从大到小放入对应的位置
            //提取对应的特征向量
            z.getValsByCol(i, *tmpvec);
            V.MatVec(0, dimension - 1, *tmpvec, *out_vec);
            level_str.W_mat.setValsByCol(m - i - 1, *out_vec);
        }
        level_str.rank = m;
        // 更新M的非零元个数
        UINT32 localMnnz = m * V.getRowNum() + m;
        this->m_Mnnz += localMnnz;
        m_lowRankNNZ += localMnnz;
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::computeArnoldiEigenValuesAndEigenVectors(INT32 level,
                                                                                                          DenseMatrix<HighPrecisionType> &H,
                                                                                                          DenseMatrix<HighPrecisionType> &V) {
        UINT32 dimension = H.getRowNum();
        UINT32 k = std::min(dimension, static_cast<UINT32>(m_lowRankSize)); // 实际低秩矫正的阶数
#ifndef NDEBUG
        THROW_EXCEPTION(k == 0, THROW_LOGIC_ERROR("The param of low-rank correction is incorrect!"))
#endif
        GMSLRLevelStructure<HighPrecisionType> &level_str = *m_levs_all[level];
        HighPrecisionType lambda;
        int size = static_cast<int>(dimension);
        int ldz = size; //矩阵行数
        int m = k; //找到m个特征值
        HighPrecisionType tmp = std::is_same<HighPrecisionType, FLOAT64>::value
                                ? LAPACKE_dlamch('S')
                                : LAPACKE_slamch('S');
        HighPrecisionType abstol = 2 * tmp; //这个最佳选择为2*dlamch(S)

        AutoAllocateVector<HighPrecisionType> w(size, memoryBase); //前m个元素为升序排列的我们要的特征值
        AutoAllocateVector<HighPrecisionType> wi(size, memoryBase);
        DenseMatrix<HighPrecisionType> z(DenseMatColumnFirst, ldz, ldz, memoryBase); //对应的特征向量
        AutoAllocateVector<INT32> ifail(size, memoryBase);
        int info;
        // 传统MSLR低秩校正，不修正特征值，直接使用k个特征值构造校正过程
        // 设置lapack参数，并求特征值
        char JOBZ = 'V';
        char RANGE = 'I';
        HighPrecisionType vl, vu;
        int il = size - k + 1;
        int iu = size;

        if (std::is_same<HighPrecisionType, FLOAT64>::value) {
            info = LAPACKE_dhseqr(LAPACK_COL_MAJOR, 'S', JOBZ, size, 1, size, (double *) H.getMatValPtr(), size,
                                  (double *) w->getRawValPtr(), (double *) wi->getRawValPtr(), (double *) z.getMatValPtr(), size);
        } else {
            info = LAPACKE_shseqr(LAPACK_COL_MAJOR, 'S', JOBZ, size, 1, size, (float *) H.getMatValPtr(), size,
                                  (float *) w->getRawValPtr(), (float *) wi->getRawValPtr(), (float *) z.getMatValPtr(), size);
        }
        // 生成H和W
        level_str.H_vec.resize(m, RESERVE_NO_DATA);
        level_str.W_mat.resize(V.getRowNum(), m, RESERVE_NO_DATA);
        AutoAllocateVector<HighPrecisionType> tmpvec((UINT32)ldz, memoryBase);
        AutoAllocateVector<HighPrecisionType> out_vec;
        for (INT32 i = m - 1; i >= 0; --i) {
            lambda = w[i]; //取出最大特征值
            level_str.H_vec[m - i - 1] = lambda / (1.0 - lambda); //从大到小放入对应的位置
            //提取对应的特征向量
            z.getValsByCol(i, *tmpvec);
            V.MatVec(0, dimension - 1, *tmpvec, *out_vec);
            level_str.W_mat.setValsByCol(m - i - 1, *out_vec);
        }
        level_str.rank = m;
        // 更新M的非零元个数
        UINT32 localMnnz = m * V.getRowNum() + m;
        this->m_Mnnz += localMnnz;
        m_lowRankNNZ += localMnnz;
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::levelCInvRecursiveSolve(
            INT32 level, HostVector<HighPrecisionType> &x, UINT32 xLocalStartIdx) {
#ifndef NDEBUG
        // 最后一个水平无C块
        THROW_EXCEPTION(level >= static_cast<INT32>(m_nlev_used) - 1 || level < -1,
                        THROW_LOGIC_ERROR("The level num is out-of-range!"))
#endif
        // 取nextLevel的对象（例如：l = -1 对应A0）
        GMSLRLevelStructure<HighPrecisionType> &nextLevel = *m_levs_all[level + 1];
        if (level == m_nlev_used - 2) {
            // A_{L-1}
            //倒数第二个水平的C存在其C_mat中，也存在最后一水平的B_mat中
            nextLevel.B_precond[0]->MInvSolve(x, xLocalStartIdx);
            return;
        }
        //其他水平，C(level)=A(level+1),按其分块
        //相对于当前水平的下一水平的n_start，n_end
        UINT32 n_start = m_lev_ptr_v[level + 1]; // 下一水平的内点在全局中的起始编号
        UINT32 n_end = m_lev_ptr_v[level + 2]; // 下一水平的内点在全局中的结束边界（不包括当前值）
        UINT32 blockBSize = n_end - n_start; // 下一水平的B块总大小
        UINT32 n_separateIdx = xLocalStartIdx + blockBSize; // 下一水平x1,x2分界点下标
        UINT32 nextCblockSize = this->m_ArowNum - n_end; // 再下一水平C的维数（x2的维数）
        AutoAllocateVector<HighPrecisionType> x_copy(nextCblockSize, memoryBase);
        /* 递归迭代 */
        /*  最外层方程：
         *           L(l+1)              0    x1   x1
         *     E^T*(L^T)^(-1)*D^(-1)     I    x2   x2
         *  中间方程：
         *   D(l+1)   0      x1       x1
         *   0    S(l+1)     y2       x2
         *   其中S(l+1)^(-1)=C(l+1)^(-1)+W(l+1)H(l+1)W(l+1)^(T)
         *
         *   ====> 转换为解下面两个式子：
         *   1. x1 = L^{-T} * D^{-1} * L^{-1} x1;
         *   2. x2 -= E^{T} x1;
         *   然后递归求解下一层对应的x2_new
         * */
        // 计算 x1 = L^{-T} * D^{-1} * L^{-1} x1
        levelBlockInvSolve(level + 1, x, xLocalStartIdx);
        // 计算x_copy = E^{T} x1
//        nextLevel.E_mat.transMatPVec(x, xLocalStartIdx, *x_copy, 0);
        nextLevel.F_mat.MatPVec(x, xLocalStartIdx, *x_copy, 0);
        UINT32 xLastValBound = x.getLength();
        // 计算x2 -= x_copy
        for (UINT32 idx = n_separateIdx, i = 0; idx < xLastValBound; ++idx, ++i) {
            x[idx] -= x_copy[i];
        }
        // x_copy = x2
        x_copy.copy(x, n_separateIdx, 0, nextCblockSize);
        levelCInvRecursiveSolve(level + 1, x, xLocalStartIdx + blockBSize); //x2_new = C(l+1)^(-1) * x2
        /* 最内层方程
         *   L^(T)    D^(-1)*L^(-1)*E    x1    x1
         *     0             I           y2    y2
         *
         *   ====> 转换为解下面两个式子：
         *   x2_new += WHW^{T} * x2
         *   x1 -= L^{-T} * D^{-1} * L^{-1} * E * x2_new
         * */
        AutoAllocateVector<HighPrecisionType> x_temp1, x_temp2;
        DenseMatrix<HighPrecisionType> &currW = nextLevel.W_mat;
        AutoAllocateVector<HighPrecisionType> &currH = nextLevel.H_vec;
        // 计算低秩修正
        m_timer.cpuTimerStart();
        if (currH.getLength() > 0) {
            // CPU_TIMER_FUNC()
            // 特征值修正可能遇到没有修正项的情况，所以要判断
            // 计算x_temp1 = W^{T} * x_copy
            currW.transposeVec(*x_copy, *x_temp1); // x_temp1的大小即为低秩校正的秩
            // 计算x_temp1 = H * x_temp1
            xLastValBound = currH.getLength();
            for (UINT32 i = 0; i < xLastValBound; ++i) {
                x_temp1[i] *= currH[i];
            }
            // x_temp2 = W * x_temp1
            currW.MatVec(*x_temp1, *x_temp2); // x_temp2的大小等于x2的大小，等于nextCblockSize
            // x_new += x_temp2
            xLastValBound = x.getLength();
            for (UINT32 idx = n_separateIdx, i = 0; idx < xLastValBound; ++idx, ++i) {
                x[idx] += x_temp2[i];
            }
        }
        m_timer.cpuTimerEnd();
        m_compLowRankCorrectTime += m_timer.computeCPUtime();
        x_temp2.resize(blockBSize, RESERVE_NO_DATA); // 将辅助变量大小重新调整为整个b块的行数，即E块的行数
        // x_temp2 = E * x_new
        nextLevel.E_mat.MatPVec(x, n_separateIdx, *x_temp2, 0);
        // x_temp2 = L^{-T} * D^{-1} * L^{-1} * x_temp2
        levelBlockInvSolve(level + 1, *x_temp2, 0);
        // x1 -= x_temp2
        for (UINT32 idx = xLocalStartIdx, i = 0; idx < n_separateIdx; ++idx, ++i) {
            x[idx] -= x_temp2[i];
        }
        // 至此，求解结束
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::levelCInvRecursiveSolveWithCorrection(INT32 level,
                                                                                                       HostVector<HighPrecisionType> &x,
                                                                                                       UINT32 xLocalStartIdx) {
        AutoAllocateVector<HighPrecisionType> x_correction(x.getLength() - xLocalStartIdx, memoryBase);
        AutoAllocateVector<HighPrecisionType> x_temp(x_correction.getLength(), memoryBase);
        // x_correction暂存重排后对应的x
        x_correction.copy(x, xLocalStartIdx, 0, x_correction.getLength());
        // 求出不精确的res_inacc = C^{-1} x，其结果直接写入到原来的x中，x_correction保存对应的原来的x
        levelCInvRecursiveSolve(level, x, xLocalStartIdx);
        // 计算x_copy - C * res_inacc，存入到x_correct中
        if (level != -1) {
            m_levs_all[level]->C_mat.MatPVec(x, xLocalStartIdx, *x_temp, 0); // x_temp = C * x
            x_correction.add(-1, *x_temp); // 计算残差 x_residual -= x_temp
        } else {
            // 计算四个分块的乘积
            UINT32 blockBsize = m_lev_ptr_v[1] - m_lev_ptr_v[0];
            levelBlockMatPVec(0, x, 0, *x_temp, 0);
            m_levs_all[0]->E_mat.MatPVecWithoutClearOutVector(x, blockBsize, *x_temp, 0);
            m_levs_all[0]->C_mat.MatPVec(x, blockBsize, *x_temp, blockBsize);
            m_levs_all[0]->F_mat.MatPVec(x, 0, *x_temp, blockBsize);
            x_correction.add(-1, *x_temp); // 计算残差 x_residual -= x_temp
        }
        // 计算 res_correction = C^{-1} * x_correction
        levelCInvRecursiveSolve(level, *x_correction, 0);
        // 给最终答案加上修正值
#ifdef USE_OMP_VECTOR_FUNC
#pragma omp parallel for default(none) num_threads(THREAD_NUM) shared(xLocalStartIdx, x, x_correction)
#endif
        for (UINT32 idx = xLocalStartIdx; idx < x.getLength(); ++idx) {
            x[idx] += x_correction[idx - xLocalStartIdx];
        }
    }


    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::levelBlockInvSolve(INT32 level,
                                                                                    HostVector<HighPrecisionType> &x,
                                                                                    UINT32 xLocalStartIdx) {
#ifndef NDEBUG
        THROW_EXCEPTION(level < 0 || level >= m_levs_all.getDim(),
                        THROW_LOGIC_ERROR("The level index is out-of-range!"));
#endif
        GMSLRLevelStructure<HighPrecisionType> &level_str = *m_levs_all[level];
        SharedPtr1D<CSRMatrix<HighPrecisionType> > &currBlockB = level_str.B_mat;
        SharedPtr1D<TriangularPrecondition<HighPrecisionType> > &currBlockB_precond = level_str.B_precond;
        UINT32 blockBnum = currBlockB.getDim();
        // UINT32 blockOffset = xLocalStartIdx;
        UINT32 globalStartIdx = (*m_dom_ptr_v2[level])[0];

        // 计算所有子块对应的temp_v = B^{-1} temp_v
        m_timer.cpuTimerStart();
#ifdef OPENMP_FOUND
        AutoAllocateVector<UINT32> auxStartIdx(blockBnum + 1, memoryBase);
        UINT32 alignElemNum = 2 * ALIGNED_BYTES / sizeof(HighPrecisionType);

        // 计算每个子块实际存储的起始索引
        for (UINT32 i = 1; i <= blockBnum; ++i) {
            // 每个子块后面插入一段空白区间，空出两个缓存行，避免伪共享
            auxStartIdx[i] = auxStartIdx[i - 1] + currBlockB[i - 1]->getRowNum() + alignElemNum;
        }

        // 启动任务调度
#pragma omp parallel num_threads(THREAD_NUM) proc_bind(master) shared(m_dom_ptr_v2, auxStartIdx, blockBnum, level, globalStartIdx, xLocalStartIdx, currBlockB_precond, currBlockB, x)
        {
#pragma omp for simd schedule(dynamic) nowait
            for (UINT32 bID = 0; bID < blockBnum; ++bID) {
                UINT32 subMatStartRowIdx = (*m_dom_ptr_v2[level])[bID] - globalStartIdx + xLocalStartIdx;
                std::shared_ptr<TriangularPrecondition<HighPrecisionType> > localBlock = currBlockB_precond[bID];
                // 检查越界
#ifndef  NDEBUG
                THROW_EXCEPTION(subMatStartRowIdx + currBlockB[bID]->getRowNum() > x.getLength(),
                                THROW_LOGIC_ERROR("The block offset is out-of-range!"))
#endif

                // 计算时把结果写入到辅助向量里面
                localBlock->MInvSolve(x, subMatStartRowIdx, *m_auxParallelSolveB, auxStartIdx[bID]);
                x.copy(*m_auxParallelSolveB, auxStartIdx[bID], subMatStartRowIdx, currBlockB[bID]->getRowNum());
            }
        }
#else       // 如果不开OpenMP并行，使用下面的串行代码块
        UINT32 blockOffset = xLocalStartIdx;
        for (UINT32 bID = 0; bID < blockBnum; ++bID) {
#ifndef  NDEBUG
                THROW_EXCEPTION(blockOffset + currBlockB[bID]->getRowNum() > x.getLength(),
                                THROW_LOGIC_ERROR("The block offset is out-of-range!"))
#endif
            currBlockB_precond[bID]->MInvSolve(x, blockOffset);
            blockOffset += currBlockB[bID]->getRowNum();
        }
#endif

        m_timer.cpuTimerEnd();
        m_blockBParallelTime += m_timer.computeCPUtime();
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::levelBlockInvSolveWithCorrection(INT32 level,
                                                                                                  HostVector<HighPrecisionType> &x,
                                                                                                  UINT32 xLocalStartIdx) {
        UINT32 blockBsize = m_lev_ptr_v[level + 1] - m_lev_ptr_v[level];
        AutoAllocateVector<HighPrecisionType> x_correction(blockBsize, memoryBase);
        AutoAllocateVector<HighPrecisionType> x_temp(blockBsize, memoryBase);
        // x_correction暂存重排后对应的x
        x_correction.copy(x, xLocalStartIdx, 0, blockBsize);
        // 求出不精确的res_inacc = B^{-1} x，其结果直接写入到原来的x中，x_correction保存对应的原来的x
        levelBlockInvSolve(level, x, xLocalStartIdx);
        // 计算x_copy - B * res_inacc，存入到x_correct中
        levelBlockMatPVec(level, x, xLocalStartIdx, *x_temp, 0);    // x_temp = B * x
        x_correction.add(-1, *x_temp); // 计算残差 x_residual -= x_temp
        // 计算 res_correction = B^{-1} * x_correction
        levelBlockInvSolve(level, *x_correction, 0);
        // 给最终答案加上修正值
#ifdef USE_OMP_VECTOR_FUNC
#pragma omp parallel for default(none) num_threads(THREAD_NUM) shared(xLocalStartIdx, x, x_correction)
#endif
        for (UINT32 idx = xLocalStartIdx; idx < x.getLength(); ++idx) {
            x[idx] += x_correction[idx - xLocalStartIdx];
        }
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::levelBlockMatPVec(INT32 level,
                                                                                   HostVector<HighPrecisionType> &vecIN,
                                                                                   UINT32 inVecStartIdx,
                                                                                   HostVector<HighPrecisionType> &vecOUT,
                                                                                   UINT32 outVecStartIdx) {
#ifndef NDEBUG
        THROW_EXCEPTION(level < 0 || level >= m_levs_all.getDim(),
                        THROW_LOGIC_ERROR("The level index is out-of-range!"));
#endif
        GMSLRLevelStructure<HighPrecisionType> &level_str = *m_levs_all[level];
        SharedPtr1D<CSRMatrix<HighPrecisionType> > &currBlockB = level_str.B_mat;
        UINT32 blockBnum = currBlockB.getDim();
        // UINT32 blockOffset = xLocalStartIdx;
        UINT32 globalStartIdx = (*m_dom_ptr_v2[level])[0];

        // 计算所有子块对应的temp_v = B^{-1} temp_v
        m_timer.cpuTimerStart();
#ifdef OPENMP_FOUND
        AutoAllocateVector<UINT32> auxStartIdx(blockBnum + 1, memoryBase);
        UINT32 alignElemNum = 2 * ALIGNED_BYTES / sizeof(HighPrecisionType);
        // 计算每个子块实际存储的起始索引
        for (UINT32 i = 1; i <= blockBnum; ++i) {
            // 每个子块后面插入一段空白区间，空出两个缓存行，避免伪共享
            auxStartIdx[i] = auxStartIdx[i - 1] + currBlockB[i - 1]->getRowNum() + alignElemNum;
        }
        // 启动任务调度
#pragma omp parallel num_threads(THREAD_NUM) proc_bind(master) \
    shared(m_dom_ptr_v2, auxStartIdx, blockBnum, level, globalStartIdx, inVecStartIdx, outVecStartIdx, currBlockB, vecIN, vecOUT)
        {
#pragma omp for simd schedule(dynamic) nowait
            for (UINT32 bID = 0; bID < blockBnum; ++bID) {
                UINT32 distFromGlobalStartIdx = (*m_dom_ptr_v2[level])[bID] - globalStartIdx;
                UINT32 subInMatStartRowIdx = distFromGlobalStartIdx + inVecStartIdx;
                UINT32 subOutMatStartRowIdx = distFromGlobalStartIdx + outVecStartIdx;
                std::shared_ptr<CSRMatrix<HighPrecisionType> > localBlock = currBlockB[bID];
                // 检查越界
#ifndef  NDEBUG
                THROW_EXCEPTION(subInMatStartRowIdx + currBlockB[bID]->getRowNum() > vecIN.getLength() ||
                                subOutMatStartRowIdx + currBlockB[bID]->getRowNum() > vecOUT.getLength(),
                                THROW_LOGIC_ERROR("The block offset is out-of-range!"))
#endif

                // 计算时把结果写入到辅助向量里面
                localBlock->MatPVec(vecIN, subInMatStartRowIdx, *m_auxParallelSolveB, auxStartIdx[bID]);
                vecOUT.copy(*m_auxParallelSolveB, auxStartIdx[bID], subOutMatStartRowIdx, currBlockB[bID]->getRowNum());
            }
        }
#else       // 如果不开OpenMP并行，使用下面的串行代码块
        UINT32 blockOffsetIN = inVecStartIdx, blockOffsetOUT = outVecStartIdx;
        for (UINT32 bID = 0; bID < blockBnum; ++bID) {
#ifndef  NDEBUG
            THROW_EXCEPTION(blockOffsetIN + currBlockB[bID]->getRowNum() > vecIN.getLength() ||
                            blockOffsetOUT + currBlockB[bID]->getRowNum() > vecOUT.getLength(),
                            THROW_LOGIC_ERROR("The block offset is out-of-range!"))
#endif
        currBlockB[bID]->MatPVec(vecIN, blockOffsetIN, vecOUT, blockOffsetOUT);
            blockOffsetIN += currBlockB[bID]->getRowNum();
            blockOffsetOUT += currBlockB[bID]->getRowNum();
        }
#endif
        m_timer.cpuTimerEnd();
        m_blockBParallelTime += m_timer.computeCPUtime();
    }


    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::setup() {
        if (this->m_isReady == PRECONDITION_READY) return;
        CPU_TIMER_FUNC()
        CPU_TIMER_BEGIN()
        permuteHID();
        CPU_TIMER_END()
        std::cout << " --- HID reordering executes: " << CPU_EXEC_TIME() << " ms." <<
                  std::endl;
#ifndef NDEBUG
        m_pperm.printVector("permutation array after HID ordering");
#endif
        CPU_TIMER_BEGIN()
        setupMSLR();
        CPU_TIMER_END()
        std::cout << " --- setup MSLR structure executes: " << CPU_EXEC_TIME() << " ms." <<
                  std::endl;
#ifndef NDEBUG
        SHOW_INFO("Setup AMSED succeed!")
#endif
        this->m_isReady = PRECONDITION_READY;
        m_blockBParallelTime = 0;
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::MInvSolve(BaseVector<HighPrecisionType> &vec,
                                                                           UINT32 resStartIdx) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_isReady != PRECONDITION_READY,
                        THROW_LOGIC_ERROR("The AMSED precondition is not ready!"))
#endif
        if (m_rhsPermType == GMSLRPermuteRightHandInside) {
            if (m_permRhs.getLength() != this->m_ArowNum) m_permRhs.resize(this->m_ArowNum, RESERVE_NO_DATA);
            prepareRightHand(vec, *m_permRhs);
            levelCInvRecursiveSolve(-1, *m_permRhs, resStartIdx); // 要求A0^{-1} * b，即求C_{-1}^{-1} * b
            recoverRightHand(*m_permRhs, vec);
        } else {
#ifndef NDEBUG
            THROW_EXCEPTION(m_rhsReady == 0,
                            THROW_LOGIC_ERROR(
                                    "The right hand was not permuted with the matrix! Use \"prepareRightHand()\" to prepare for it."
                            ))
            THROW_EXCEPTION(this->m_ArowNum + resStartIdx > vec.getLength(),
                            THROW_LOGIC_ERROR("The permuted right-hand is incorrect!"))
#endif
            m_permRhs->move(vec);
            levelCInvRecursiveSolve(-1, *m_permRhs, resStartIdx); // 要求A0^{-1} * b，即求C_{-1}^{-1} * b
            vec = std::move(*m_permRhs);
        }
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::MInvSolve(BaseVector<HighPrecisionType> &vecIN,
                                                                           BaseVector<HighPrecisionType> &vecOUT) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_isReady != PRECONDITION_READY,
                        THROW_LOGIC_ERROR("The AMSED precondition is not ready!"))
#endif
        if (m_rhsPermType == GMSLRPermuteRightHandInside) {
            if (m_permRhs.getLength() != this->m_ArowNum) m_permRhs.resize(this->m_ArowNum, RESERVE_NO_DATA);
            prepareRightHand(vecIN, *m_permRhs);
            levelCInvRecursiveSolve(-1, *m_permRhs, 0); // 要求A0^{-1} * b，即求C_{-1}^{-1} * b
            recoverRightHand(*m_permRhs, vecOUT);
        } else {
#ifndef NDEBUG
            THROW_EXCEPTION(m_rhsReady == 0,
                            THROW_LOGIC_ERROR(
                                    "The right hand was not permuted with the matrix! Use \"prepareRightHand()\" to prepare for it."
                            ))
            THROW_EXCEPTION(vecIN.getLength() != this->m_ArowNum || vecOUT.getLength() != this->m_ArowNum,
                            THROW_LOGIC_ERROR("The permuted right-hand is incorrect!"))
#endif
            vecOUT.copy(vecIN);
            m_permRhs->move(vecOUT);
            levelCInvRecursiveSolve(-1, *m_permRhs, 0); // 要求A0^{-1} * b，即求C_{-1}^{-1} * b
            vecOUT = std::move(*m_permRhs);
        }
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::MInvSolve(BaseVector<HighPrecisionType> &vecIN,
                                                                           UINT32 inStartIdx,
                                                                           BaseVector<HighPrecisionType> &vecOUT,
                                                                           UINT32 outStartIdx) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_isReady != PRECONDITION_READY,
                        THROW_LOGIC_ERROR("The AMSED precondition is not ready!"))
#endif
        if (m_rhsPermType == GMSLRPermuteRightHandInside) {
            if (m_permRhs.getLength() != this->m_ArowNum) m_permRhs.resize(this->m_ArowNum, RESERVE_NO_DATA);
            prepareRightHand(vecIN, inStartIdx, *m_permRhs, 0);
            levelCInvRecursiveSolve(-1, *m_permRhs, 0); // 要求A0^{-1} * b，即求C_{-1}^{-1} * b
            recoverRightHand(*m_permRhs, 0, vecOUT, outStartIdx);
        } else {
#ifndef NDEBUG
            THROW_EXCEPTION(m_rhsReady == 0,
                            THROW_LOGIC_ERROR(
                                    "The right hand was not permuted with the matrix! Use \"prepareRightHand()\" to prepare for it."
                            ))
            THROW_EXCEPTION(
                    this->m_ArowNum + inStartIdx > vecIN.getLength() ||
                    this->m_ArowNum + outStartIdx > vecOUT.getLength(),
                    THROW_LOGIC_ERROR("The permuted right-hand is incorrect!"))
#endif
            vecOUT.copy(vecIN, inStartIdx, outStartIdx, this->m_ArowNum);
            m_permRhs->move(vecOUT);
            levelCInvRecursiveSolve(-1, *m_permRhs, outStartIdx); // 要求A0^{-1} * b，即求C_{-1}^{-1} * b
            vecOUT = std::move(*m_permRhs);
        }
    }

    template<typename LowPrecisionType, typename HighPrecisionType>
    void GMSLRPrecondition<LowPrecisionType, HighPrecisionType>::MInvSolve(BaseVector<HighPrecisionType> &vec) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_isReady != PRECONDITION_READY,
                        THROW_LOGIC_ERROR("The AMSED precondition is not ready!"))
#endif
        if (m_rhsPermType == GMSLRPermuteRightHandInside) {
            if (m_permRhs.getLength() != this->m_ArowNum) m_permRhs.resize(this->m_ArowNum, RESERVE_NO_DATA);
            prepareRightHand(vec, *m_permRhs);
            levelCInvRecursiveSolve(-1, *m_permRhs, 0); // 要求A0^{-1} * b，即求C_{-1}^{-1} * b
            // levelCInvRecursiveSolveWithCorrection(-1, *m_permRhs, 0);
            recoverRightHand(*m_permRhs, vec);
        } else {
#ifndef NDEBUG
            THROW_EXCEPTION(m_rhsReady == 0,
                            THROW_LOGIC_ERROR(
                                    "The right hand was not permuted with the matrix! Use \"prepareRightHand()\" to prepare for it."
                            ))
            THROW_EXCEPTION(vec.getLength() != this->m_ArowNum,
                            THROW_LOGIC_ERROR("The permuted right-hand is incorrect!"))
#endif
            m_permRhs->move(vec);
            levelCInvRecursiveSolve(-1, *m_permRhs, 0); // 要求A0^{-1} * b，即求C_{-1}^{-1} * b
            vec = std::move(*m_permRhs);
        }
    }
} // HOST
