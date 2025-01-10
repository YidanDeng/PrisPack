/*
 * @author  刘玉琴、邓轶丹
 * @date    2024/11/16
 */
#include "../../include/Preconditioner/AMSEDPrecondition.h"

#include "../../include/utils/ExternalTools/MetisTools.h"

namespace HOST {
    template <typename LowPrecisionType, typename HighPrecisionType>
    AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::AMSEDPrecondition(
        const std::shared_ptr<CSRMatrix<HighPrecisionType>>& matA, INT32 levelNum,
        MatrixReorderOption_t localReorderType, LowRankCorrectType_t lowRankType) {
        this->m_precondType = PreconditionAMSED;
        m_matA = matA;
        this->m_ArowNum = matA->getRowNum();
        this->m_AcolNum = matA->getColNum();
        this->m_Annz = matA->getNNZnum(0, this->m_ArowNum - 1);
        m_nlev_setup = levelNum;
        m_lowRankType = lowRankType;
        m_local_ordering_setup = localReorderType;
        m_pperm.resize(this->m_ArowNum, RESERVE_NO_DATA);
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::AMSEDPrecondition(
        const std::shared_ptr<CSRMatrix<HighPrecisionType>>& matA, INT32 levelNum, INT32 lowRankSize,
        MatrixReorderOption_t localReorderType, LowRankCorrectType_t lowRankType) {
        this->m_precondType = PreconditionAMSED;
        m_matA = matA;
        this->m_ArowNum = matA->getRowNum();
        this->m_AcolNum = matA->getColNum();
        this->m_Annz = matA->getNNZnum(0, this->m_ArowNum - 1);
        m_nlev_setup = levelNum;
        m_lowRankType = lowRankType;
        m_lowRankSize = lowRankSize;
        m_local_ordering_setup = localReorderType;
        m_pperm.resize(this->m_ArowNum, RESERVE_NO_DATA);
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::AMSEDPrecondition(
        const std::shared_ptr<CSRMatrix<HighPrecisionType>>& matA, INT32 levelNum, INT32 lanczosSteps,
        INT32 lowRankSize, MatrixReorderOption_t localReorderType, LowRankCorrectType_t lowRankType,
        LanczosType_t lanczosType) {
        this->m_precondType = PreconditionAMSED;
        m_matA = matA;
        this->m_ArowNum = matA->getRowNum();
        this->m_AcolNum = matA->getColNum();
        this->m_Annz = matA->getNNZnum(0, this->m_ArowNum - 1);
        m_nlev_setup = levelNum;
        m_lowRankType = lowRankType;
        m_lowRankSize = lowRankSize;
        m_lanczosSteps = lanczosSteps;
        m_lanczosType = lanczosType;
        m_local_ordering_setup = localReorderType;
        m_pperm.resize(this->m_ArowNum, RESERVE_NO_DATA);
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::AMSEDPrecondition(
        const std::shared_ptr<CSRMatrix<HighPrecisionType>>& matA, INT32 levelNum, INT32 lanczosSteps,
        INT32 lowRankSize, HighPrecisionType eigCorrectBound, MatrixReorderOption_t localReorderType,
        LowRankCorrectType_t lowRankType, LanczosType_t lanczosType) {
        this->m_precondType = PreconditionAMSED;
        m_matA = matA;
        this->m_ArowNum = matA->getRowNum();
        this->m_AcolNum = matA->getColNum();
        this->m_Annz = matA->getNNZnum(0, this->m_ArowNum - 1);
        m_nlev_setup = levelNum;
        m_lowRankType = lowRankType;
        m_lowRankSize = lowRankSize;
        m_lanczosSteps = lanczosSteps;
        m_lanczosType = lanczosType;
        m_eigValCorrectBound = eigCorrectBound;
        m_local_ordering_setup = localReorderType;
        m_pperm.resize(this->m_ArowNum, RESERVE_NO_DATA);
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::setupPermutationND(HostVector<INT32>& map_v,
        HostVector<INT32>& mapptr_v) {
        UINT32 tlvl, clvl, nrows, i, j, k, size1, size2;
        INT32 domi;
        SharedPtr2D<AutoAllocateVector<UINT32>> level_str;
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

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::buildLevelStructure(HostVector<INT32>& map_v,
        HostVector<INT32>& mapptr_v) {
        INT32 n, i, j, nlev_used, nlev_max, ndom, level, ncomp, maps, mape;
        UINT32 ni, n_start, n_end, n_local, n_remain;
        AutoAllocateVector<UINT32> temp_perm, local_perm;
        CSRMatrix<HighPrecisionType>& A = *m_matA;
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
            LevelStructure<HighPrecisionType>& level_str = *m_levs_all[level];
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
        THROW_EXCEPTION(err != AMSED_SUCCESS, THROW_LOGIC_ERROR("The permutation of HID failed!"))
        writePermMat(Apq);
#endif
        //提取子矩阵
        UINT32 subMatStartRowNo, subMatEndRowNo, subMatStartColNo, subMatEndColNo;
        UINT32 maxSubdomainSize = 0, maxCblockSize = 0;
        for (level = 0; level < nlev_used; ++level) {
            LevelStructure<HighPrecisionType>& level_str = *m_levs_all[level];
            if (level == nlev_used - 1) {
                //最后一个level
                mape = mapptr_v[nlev_max];
                maps = mapptr_v[level];
                ncomp = mape - maps;
                level_str.ncomps = ncomp;

                n_start = domptr_v[maps]; //这一level开始的节点
                n_end = domptr_v[mape]; //这一level结束的节点
                n_local = n_end - n_start; // 这一level的整个B块的大小
                n_remain = n - n_end; // 这个level的C块大小
                maxSubdomainSize = std::max(maxSubdomainSize, n_local);
                maxCblockSize = std::max(maxCblockSize, n_remain);
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
                maxCblockSize = std::max(maxCblockSize, n_remain);

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
                //提取C矩阵
                subMatStartRowNo = n_end;
                subMatEndRowNo = n_end + n_remain - 1;
                subMatStartColNo = n_end;
                subMatEndColNo = n_end + n_remain - 1;
                Apq.getSubMatrix(subMatStartRowNo, subMatEndRowNo, subMatStartColNo, subMatEndColNo, level_str.C_mat);
            }
        }
        // 在这里写入需要的数据到文件中
#ifndef NDEBUG
        char pathBuffer[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", pathBuffer, sizeof(pathBuffer) - 1);
        if (len != -1) {
            pathBuffer[len] = '\0'; // 确保以 null 结尾
        } else {
            TRY_CATCH(THROW_LOGIC_ERROR("Could not get absolute path of current exe file!"))
        }
        std::string prefix = "../source/test/testResults/AMSED_perm_lev_";
        CSRMatrix<HighPrecisionType> fullBmat;
        UINT32 totalLevels = m_nlev_used - 1; // 除了最高水平只有一个块，其余水平均有四个块（B、E、E^T、C）
        for (level = 0; level < totalLevels; ++level) {
            n_start = m_lev_ptr_v[level]; // 当前水平B块的起始行号
            n_end = m_lev_ptr_v[level + 1] - 1; // 当前水平B块的结束行号
            std::string Epath = prefix + std::to_string(level) + "_E.mtx";
            WriteMtxTools<HighPrecisionType> currLevelE(pathBuffer, Epath.data(), 1, MTX_STORAGE_GENERAL);
            currLevelE.writeMatrix(m_levs_all[level]->E_mat);

            std::string Bpath = prefix + std::to_string(level) + "_B.mtx";
            Apq.getSubMatrix(n_start, n_end, n_start, n_end, fullBmat);
            WriteMtxTools<HighPrecisionType> currLevelB(pathBuffer, Bpath.data(), 1, MTX_STORAGE_GENERAL);
            currLevelB.writeMatrix(fullBmat);

            std::string Cpath = prefix + std::to_string(level) + "_C.mtx";
            WriteMtxTools<HighPrecisionType> currLevelC(pathBuffer, Cpath.data(), 1, MTX_STORAGE_GENERAL);
            currLevelC.writeMatrix(m_levs_all[level]->C_mat);
        }
#endif

        // 这里分配辅助变量空间，为后续并行求解B块做准备，分配原则是在不同块之间留够两个缓存行的间隙
        // 这个大小取最大B块行数加两倍最大并行域个缓存行大小，这个大小足以涵盖后续所有有关B逆求解的部分，因此后续无需再重新调整空间大小
        m_auxParallelSolveB.reset(
            maxSubdomainSize + m_levs_all[0]->ncomps * 2 * ALIGNED_BYTES / sizeof(HighPrecisionType), memoryAligned);
#ifdef CUDA_ENABLED
        /* 因为顶层level的C是最大的，所以它对应的辅助空间可以满足各个阶段的计算需求 */
        UINT32 maxLanczosSteps = m_lowRankSize * (AMSED_LANCZOS_MAX_COEFFICIENT + 1);
        m_devEigenvectors.initSpace(maxLanczosSteps * maxLanczosSteps, DEFAULT_GPU);
        m_devLanczosVectorsV.initSpace(maxCblockSize * maxLanczosSteps, DEFAULT_GPU);
        m_devLanczosVectorsZ.initSpace(m_devLanczosVectorsV.getLength(), DEFAULT_GPU);
        m_devAuxLanczosVectorsV.initSpace(m_devLanczosVectorsV.getLength(), DEFAULT_GPU);
        m_devAuxLanczosVectorsZ.initSpace(m_devLanczosVectorsZ.getLength(), DEFAULT_GPU);
#endif
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::setupPermutationNDRecursive(
        CSRMatrix<HighPrecisionType>& A, UINT32 clvl, UINT32& tlvl,
        SharedPtr2D<AutoAllocateVector<UINT32>>& level_str) {
        UINT32 i, j, ncomps, nS, ndom, edgecut, idx, size, k, k2, k1;
        DenseVector<UINT32> map, sep, perm, dom_ptr, row_perm, col_perm;
        SharedPtr1D<AutoAllocateVector<UINT32>> comp_indices;
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
                    continue;
                }
                // 根据标记向量（里面只有0和1）来提取对应的行和列
                getSubMatrixNoPerm(C, sep, sep, row_perm, col_perm, true, B);
                //把seperator移除得到矩阵B,true为提取sep为零元素位置，即不在分割符内的
                //下一水平
                tlvls[i] = tlvl;
                SharedPtr2D<AutoAllocateVector<UINT32>> sub_level_str;
                setupPermutationNDRecursive(B, clvl + 1, tlvls[i], sub_level_str);

                //放入内点
                for (j = 0; j < tlvls[i]; j++) {
                    idx = j;
                    size = sub_level_str[idx].getDim();
                    for (k = 0; k < size; k++) {
                        if (currLevelStrCount[idx] >= level_str[idx].getDim())
                            level_str[idx].realloc(std::max(ncomps, currLevelStrCount[idx]) * 2);
                        level_str[idx][currLevelStrCount[idx]++] = sub_level_str[idx][k];
                        // 这里直接使用shared指针，所以老代码的back就是sub_level_str[idx][k]指向的对象
                        AutoAllocateVector<UINT32>& nodes = *sub_level_str[idx][k];
                        // vec::GeneralVector<UINT32> &nodes = level_str[idx].back();
                        k2 = nodes.getLength();
                        for (k1 = 0; k1 < k2; k1++) {
                            nodes[k1] = (*comp_indices[i])[row_perm[nodes[k1]]];
                        }
                    }
                }
                //放入边分隔符
                idx = tlvls[i];
                if (currLevelStrCount[idx] >= level_str[idx].getDim())
                    level_str[idx].realloc(std::max(ncomps, currLevelStrCount[idx]) * 2);
                level_str[idx].construct(currLevelStrCount[idx], 0, memoryBase);
                // vec::GeneralVector<UINT32> &nodes = level_str[idx].back();
                AutoAllocateVector<UINT32>& nodes = level_str(idx, currLevelStrCount[idx]);
                currLevelStrCount[idx]++;
                nodes.resize(edgecut, RESERVE_NO_DATA);
                UINT32 currIdx = 0;
                for (j = 0; j < nS; j++) {
                    if (sep[j] != 0) {
                        if (currIdx >= nodes.getLength())
                            nodes.resize(std::max(currIdx, (UINT32)1024) * 2,
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
        // 调整level_str为实际大小
        for (i = 0; i < levelStrDim; ++i) {
            level_str[i].realloc(currLevelStrCount[i]);
        }
        tlvl = *std::max_element(tlvls.getRawValPtr(), tlvls.getRawValPtr() + tlvls.getLength());
    }


    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::getConnectedComponents(CSRMatrix<HighPrecisionType>& A,
        SharedPtr1D<AutoAllocateVector<UINT32>>& comp_indices, UINT32& ncomps) {
        UINT32 n, i, j, current_comps, qs, qe, size, idx, idx2, accumcomp, index1, index2;
        n = A.getRowNum();
        AutoAllocateVector<INT32> marker(n, memoryBase); // 记录节点是否被访问
        marker->fillVector(0, n, -1); // 初始化marker向量为-1
        AutoAllocateVector<UINT32> queue(n, memoryBase); // 记录遍历顺序（BFS寻找连通分支）
        AutoAllocateVector<UINT32> comps_size(10, memoryBase); // 记录每个连通分支的size

        // 当前连通分支数，从0开始
        current_comps = 0; // 统计实际写入comps的元素个数
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
                    UINT32* colIdxPtr = A.getColIndicesPtr(0);
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
            }
            for (i = 0; i < n; ++i) {
                idx = comps_map[marker[i]];
                if (currIdxCount[idx] >= comp_indices[idx]->getLength())
                    comp_indices[idx]->resize(
                        std::max(currIdxCount[idx], static_cast<UINT32>(1024)) * 2, RESERVE_DATA);
                (*comp_indices[idx])[currIdxCount[idx]++] = i;
            }
            for (i = 0; i < ncomps; ++i) {
                std::sort(comp_indices[i]->getRawValPtr(), comp_indices[i]->getRawValPtr() + currIdxCount[i],
                          std::less<>());
            }
        } else {
            comp_indices.realloc(current_comps);
            currIdxCount.resize(current_comps, RESERVE_NO_DATA); // 用来统计comp_indices[i]实际写入元素个数
            for (i = 0; i < current_comps; ++i) {
                comp_indices.construct(i, comps_size[i], memoryBase);
            }
            for (i = 0; i < n; ++i) {
                idx = marker[i];
                if (currIdxCount[idx] >= comp_indices[idx]->getLength())
                    comp_indices[idx]->resize(
                        std::max(currIdxCount[idx], static_cast<UINT32>(1024)) * 2, RESERVE_DATA);
                (*comp_indices[idx])[currIdxCount[idx]++] = i;
            }
            for (i = 0; i < current_comps; ++i) {
                std::sort(comp_indices[i]->getRawValPtr(), comp_indices[i]->getRawValPtr() + currIdxCount[i],
                          std::less<>());
            }
            ncomps = current_comps;
        }
        // 将每个向量还原为实际存储大小
        UINT32 comp_indices_dim = comp_indices.getDim();
        for (UINT32 i = 0; i < comp_indices_dim; ++i) {
            comp_indices[i]->resize(currIdxCount[i], RESERVE_DATA);
        }
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::getSubMatrixNoPerm(CSRMatrix<HighPrecisionType>& A,
        HostVector<UINT32>& rows, HostVector<UINT32>& cols, HostVector<UINT32>& row_perm, HostVector<UINT32>& col_perm,
        bool complement, CSRMatrix<HighPrecisionType>& csr_outmat) {
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

    template <typename LowPrecisionType, typename HighPrecisionType>
    INT32 AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::checkPermutation(
        const CSRMatrix<HighPrecisionType>& permA) {
        if (m_pperm.getLength() != m_matA->getRowNum() && m_pperm.getLength() != m_matA->getColNum()) {
            SHOW_ERROR("The permutation array length was incorrect!")
            return AMSED_FAILED;
        }
        AutoAllocateVector<UINT32> checkPerm(m_pperm.getLength(), memoryBase);
        for (UINT32 i = 0; i < m_pperm.getLength(); ++i) {
            if (checkPerm[m_pperm[i]] != 0) {
                SHOW_ERROR("The same row and column of original matrix A was used more than once!")
                return AMSED_FAILED;
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
        UINT32* A_rowOffsetPtr = m_matA->getRowOffsetPtr(0);
        UINT32* A_colIdxPtr = m_matA->getColIndicesPtr(0);
        HighPrecisionType* A_valuesPtr = m_matA->getCSRValuesPtr(0);
        UINT32* recoverA_rowOffsetPtr = recoverMat.getRowOffsetPtr(0);
        UINT32* recoverA_colIdxPtr = recoverMat.getColIndicesPtr(0);
        HighPrecisionType* recoverA_valuesPtr = recoverMat.getCSRValuesPtr(0);
        if (recoverMat.getRowNum() != rowNum || recoverMat.getColNum() != colNum) {
            SHOW_ERROR("The dim of permuted matrix is not equal to original matrix!")
            return AMSED_FAILED;
        }

        bool isPass = checkAnswerWithReturnValue(recoverA_rowOffsetPtr, A_rowOffsetPtr, rowNum + 1,
                                                 "check perm A row-offset");
        isPass &= checkAnswerWithReturnValue(recoverA_colIdxPtr, A_colIdxPtr, nnzNum, "check perm A col-indices");
        isPass &= checkAnswerWithReturnValue(recoverA_valuesPtr, A_valuesPtr, nnzNum, "check perm A col-indices");
        if (recoverMat.getRowNum() != rowNum || recoverMat.getColNum() != colNum) {
            SHOW_ERROR("The dim of permuted matrix is not equal to original matrix!")
            return AMSED_FAILED;
        }
        if (!isPass) {
            SHOW_ERROR("The permutation result is incorrect!")
            return AMSED_FAILED;
        }
        return AMSED_SUCCESS;
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::writePermMat(
        const CSRMatrix<HighPrecisionType>& permMat) {
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

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::setupMSLR() {
        getLastLevelDecomposition();
        INT32 level = static_cast<INT32>(m_nlev_used) - 2;
        // 从倒数第二个水平开始处理
        if (m_lanczosType == UseClassicLanczos) {
            while (level >= 0) {
                // 对每个水平的B块进行并行ILDLT分解
                getBlockDecomposition(level);
                // 构建低秩修正
                classicLanczosForGeneralizedEigenvaluesProblem(level); // 使用一般Lanczos构建低秩修正项
                level--;
            }
        } else if (m_lanczosType == UseRestartLanczos) {
            while (level >= 0) {
                // 对每个水平的B块进行并行ILDLT分解
                getBlockDecomposition(level);
                // 构建低秩修正
                implicitRestartKrylovSchurLanczosForGeneralizedEigenvaluesProblem(level); // 使用重启Lanczos构建低秩修正项
                level--;
            }
        }
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::getLastLevelDecomposition() {
        // 处理最后一个水平，即整个块执行IC分解
        LevelStructure<HighPrecisionType>& lastLevel = *m_levs_all[m_nlev_used - 1];
        std::shared_ptr<CSRMatrix<HighPrecisionType>> lastC = lastLevel.B_mat[0];
        // 生成预条件
        lastLevel.B_precond.realloc(1);
        lastLevel.B_precond[0] = std::make_shared<IncompleteCholesky<LowPrecisionType, HighPrecisionType>>
            (lastC, AMSED_FACTOR_TOLRANCE);
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

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::getBlockDecomposition(INT32 level) {
#ifndef NDEBUG
        THROW_EXCEPTION(level < 0 || level >= m_nlev_used, THROW_LOGIC_ERROR("Input Level number is incorrect!"))
#endif
        LevelStructure<HighPrecisionType>& level_str = *m_levs_all[level];
        UINT32 ncomp = level_str.ncomps;
        level_str.B_precond.realloc(ncomp);
#pragma omp parallel for default(none) proc_bind(master) num_threads(THREAD_NUM)  \
        shared(ncomp, level_str)
        for (UINT32 i = 0; i < ncomp; ++i) {
            auto localMat = level_str.B_mat[i];
            std::shared_ptr<TriangularPrecondition<HighPrecisionType>> localBlock = std::make_shared<IncompleteLDLT<
                LowPrecisionType, HighPrecisionType>>(localMat, AMSED_FACTOR_TOLRANCE);
            localBlock->setup();
            level_str.B_precond[i] = localBlock;
        }
        // 统计M的非零元
        for (UINT32 i = 0; i < ncomp; ++i) {
            UINT32 localMnnz = level_str.B_precond[i]->getPreconditionNonZeroCount();
            m_incompFactNNZ += localMnnz;
            BasePrecondition<HighPrecisionType>::m_Mnnz += localMnnz;
        }
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType,
                           HighPrecisionType>::classicLanczosForGeneralizedEigenvaluesProblem(INT32 level) {
        UINT32 n_end = m_lev_ptr_v[level + 1]; // 当前水平B块的结束行号
        UINT32 n_remain = this->m_ArowNum - n_end; // 当前水平C块的大小
        LevelStructure<HighPrecisionType>& level_str = *m_levs_all[level];

        // 下面的辅助变量都按照最大空间开好，内部就无需重新分配，否则时间开销较大
        AutoAllocateVector<HighPrecisionType> D(m_lanczosSteps, memoryBase); // D中存储了alpha_1,...,alpha_m
        AutoAllocateVector<HighPrecisionType> E(m_lanczosSteps, memoryBase);
        // E中存储了beta_2,...,beta_(m+1),但我们只需要beta_2,...,beta_m
        AutoAllocateVector<HighPrecisionType> V(n_remain * m_lanczosSteps, memoryBase); // Lanczos过程产生的正交向量
        AutoAllocateVector<HighPrecisionType> mu(m_lanczosSteps, memoryBase); // T矩阵分解后的特征值
        AutoAllocateVector<HighPrecisionType> Y(m_lanczosSteps * m_lanczosSteps, memoryBase); // T矩阵的特征向量
        INT32 actualLanczosSteps;

        /* 经典Lanczos过程 */
        reorthogonalGeneralizedLanczos(level, *D, *E, *V, m_lanczosSteps, actualLanczosSteps);
        computeLanczosEigenValuesAndEigenVectors(*D, *E, actualLanczosSteps, *Y, *mu);
        // 寻找收敛的特征值及对应特征向量，即满足beta * Y[idx][idx] < tol的近似特征对
        AutoAllocateVector<HighPrecisionType> convergedEigIndices(actualLanczosSteps, memoryBase); // 收敛特征值对应的下标
        INT32 convergedEigCount = 0;
        UINT32 endIdx = actualLanczosSteps - 1;
        HighPrecisionType betak = E[endIdx];
        for (INT32 i = 0; i < actualLanczosSteps; ++i) {
            if (fabs(betak) * fabs(Y[i * actualLanczosSteps + endIdx]) <= AMSED_EXIT_ZERO) {
                convergedEigIndices[convergedEigCount] = i;
                ++convergedEigCount;
            }
        }
#ifndef NDEBUG
        std::cout << " --- find converged eigen-values, current level: " << level << ", converged eigen-values count: "
            << convergedEigCount << std::endl;
        mu.printVector("eigenValues");
#endif
        if (convergedEigCount == 0) {
#ifndef NDEBUG
            SHOW_WARN("No eigen-values converged, try larger Lanczos step.")
#endif
            return;
        }
        // 构建当前水平的低秩修正项，以当前收敛的近似特征对，构造低秩修正项
        INT32 m = m_lowRankType == AMSEDBasic ? std::min(convergedEigCount, m_lowRankSize) : convergedEigCount;
        //找到m个特征值
        HighPrecisionType lambda; // 当前特征值
        if (m_lowRankType == AMSEDBasic) {
            level_str.H_vec.resize(m, RESERVE_NO_DATA);
            level_str.W_mat.resize(n_remain, m, RESERVE_NO_DATA);
#ifdef OPENMP_FOUND
#pragma omp parallel for default(none) proc_bind(master) num_threads(THREAD_NUM) private(lambda) \
    shared(level_str, mu, m, actualLanczosSteps, n_remain, V, Y, convergedEigIndices, convergedEigCount)
#endif
            for (UINT32 i = 0; i < m; ++i) {
                // 由于原本特征值从小到大排序，所以倒着取前m大特征值，并把特征对按顺序写入当前level的H和W中
                UINT32 currConvergedEigIdx = convergedEigIndices[convergedEigCount - i - 1];
                lambda = mu[currConvergedEigIdx];
                level_str.H_vec[i] = lambda / (1.0 - lambda);
                // 计算 W = V * Y
                HighPrecisionType* localWColsValsPtr = level_str.W_mat.getMatValPtr() + i * n_remain;
                HighPrecisionType* localYColsValsPtr = Y.getRawValPtr() + currConvergedEigIdx * actualLanczosSteps;
                for (UINT32 colIdx = 0; colIdx < actualLanczosSteps; ++colIdx) {
                    UINT32 begin = colIdx * n_remain;
                    for (UINT32 j = 0; j < n_remain; ++j) {
                        localWColsValsPtr[j] += V[begin + j] * localYColsValsPtr[colIdx];
                    }
                }
            }
            level_str.rank = m;
        } else if (m_lowRankType == AMSEDEigenvalueDeflation) {
            // 新方法，修正较小特征值，调节 S^{-1} * S 的特征值往1附近聚集
            HighPrecisionType epsilon = m_eigValCorrectBound;
            HighPrecisionType para = 1.0 - epsilon; //选择大于1-epsilon的特征值
            // 统计几个特征值需要被修正
            INT32 correctEigCount = 0;
            for (UINT32 i = 0; i < m; ++i) {
                UINT32 currConvergedEigIdx = convergedEigIndices[convergedEigCount - i - 1];
                if (mu[currConvergedEigIdx] <= para) break;
                correctEigCount++;
            }
            // 生成H和W
            level_str.H_vec.resize(correctEigCount, RESERVE_NO_DATA);
            level_str.W_mat.resize(n_remain, correctEigCount, RESERVE_NO_DATA);

#ifdef OPENMP_FOUND
#pragma omp parallel for default(none) proc_bind(master) num_threads(THREAD_NUM) private(lambda) \
    shared(level_str, mu, correctEigCount, epsilon, actualLanczosSteps, n_remain, V, Y, convergedEigIndices, convergedEigCount)
#endif
            for (UINT32 i = 0; i < correctEigCount; ++i) {
                // 由于原本特征值从小到大排序，所以倒着取前m大特征值，并把特征对按顺序写入当前level的H和W中
                UINT32 currConvergedEigIdx = convergedEigIndices[convergedEigCount - i - 1];
                lambda = 1.0 - mu[currConvergedEigIdx]; //取出最大特征值
                level_str.H_vec[i] = (epsilon - lambda) / lambda;
                // 计算 W = V * Y
                HighPrecisionType* localWColsValsPtr = level_str.W_mat.getMatValPtr() + i * n_remain;
                HighPrecisionType* localYColsValsPtr = Y.getRawValPtr() + currConvergedEigIdx * actualLanczosSteps;
                for (UINT32 colIdx = 0; colIdx < actualLanczosSteps; ++colIdx) {
                    UINT32 begin = colIdx * n_remain;
                    for (UINT32 j = 0; j < n_remain; ++j) {
                        localWColsValsPtr[j] += V[begin + j] * localYColsValsPtr[colIdx];
                    }
                }
            }
            level_str.rank = correctEigCount;
        }
        // 更新M的非零元个数
        // std::cout << " --- final correction low-rank size: " << level_str.rank << std::endl;
        UINT32 localMnnz = 2 * level_str.rank * n_remain + level_str.rank;
        this->m_Mnnz += localMnnz;
        m_lowRankNNZ += localMnnz;
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::reorthogonalGeneralizedLanczos(
        INT32 level, HostVector<HighPrecisionType>& D, HostVector<HighPrecisionType>& E,
        HostVector<HighPrecisionType>& V, INT32 lanczosSteps, INT32& actualLanczosSteps) {
        if (D.getLength() < lanczosSteps) D.resize(lanczosSteps, RESERVE_NO_DATA);
        if (E.getLength() < lanczosSteps) E.resize(lanczosSteps, RESERVE_NO_DATA);
        // 获得当前level各对象的引用
        LevelStructure<HighPrecisionType>& level_str = *m_levs_all[level];
        CSRMatrix<HighPrecisionType>& currE = level_str.E_mat;
        CSRMatrix<HighPrecisionType>& currC = level_str.C_mat;

        UINT32 n_start = m_lev_ptr_v[level]; // 当前水平B块的起始行号
        UINT32 n_end = m_lev_ptr_v[level + 1]; // 当前水平B块的结束行号
        UINT32 n_local = n_end - n_start; // 当前水平B块的大小
        UINT32 n_remain = this->m_ArowNum - n_end; // 当前水平C块的大小

        // 初始化
        HighPrecisionType beta{0.0}, alpha;
        //v1为C范数下的标准向量，用于迭代初始化
        AutoAllocateVector<HighPrecisionType> v1(n_remain, memoryBase); /// 对应算法中的x_j
        AutoAllocateVector<HighPrecisionType> u1(n_remain, memoryBase);
        // 选取C-范数下的标准向量x1（v1）
        generateArrayRandom1D(v1.getRawValPtr(), n_remain); // 生成随机向量为初始向量，并在C范数下标准化
        // 计算u1 = C * v1
        currC.MatPVec(*v1, *u1);
        //u_tmp=(C*v1,v1)
        HighPrecisionType u_tmp = u1->innerProduct(*v1);
        u_tmp = sqrt(u_tmp);
        u_tmp = 1.0 / u_tmp;
        v1->scale(u_tmp);
        // 初始化z0、z1为全0向量，z0对应迭代过程中的z_{j-1}，z1对应z_j
        AutoAllocateVector<HighPrecisionType> z0(n_remain, memoryBase); // base模式的内存已经清零
        AutoAllocateVector<HighPrecisionType> z1(n_remain, memoryBase);
        //z1=C*v1
        currC.MatPVec(*v1, *z1);
        // 开始迭代
        if (V.getLength() < n_remain * lanczosSteps)
            V.resize(n_remain * lanczosSteps, RESERVE_NO_DATA);

        AutoAllocateVector<HighPrecisionType> v(n_remain, memoryBase); /// 对应算法中的x
        AutoAllocateVector<HighPrecisionType> temp_v(n_local, memoryBase);
        AutoAllocateVector<HighPrecisionType> w(n_remain, memoryBase); /// 对应算法中的w

        DenseMatrix<HighPrecisionType> reorth_v, reorth_z; /// 分别记录x、z的正交向量组
        reorth_v.resize(n_remain, lanczosSteps, RESERVE_NO_DATA);
        reorth_z.resize(n_remain, lanczosSteps, RESERVE_NO_DATA);
        actualLanczosSteps = 0; // 记录实际迭代次数
        UINT32 offsetV = 0; // 记录写入V中的起始位置
        for (UINT32 i = 0; i < lanczosSteps; ++i) {
            actualLanczosSteps++;
            /* 计算 x = E^{T} * B^{-1} * E * x_j - \beta_j * z_{j-1} */
            if (beta != 0.0) z0->scale(beta);
            v->fillVector(0, n_remain, 0.0); // 内存重置，防止后续transMatPVec计算异常
            // 计算v = E^{T} * B^{-1} * E * v1
            currE.MatPVec(*v1, *temp_v); // 计算temp_v = E * v1
            // 计算所有子块对应的temp_v = B^{-1} temp_v
            levelBlockInvSolve(level, *temp_v, 0);
            // levelBlockInvSolveWithCorrection(level, *temp_v, 0);
            currE.transMatPVec(*temp_v, *v); // 计算v = E^{T} * temp_v
            v.add(-1, *z0); // v -= z0
            /* 计算 \alpha_j = (x, x_j) */
            alpha = v->innerProduct(*v1);

            /* 计算 x -= \alpha_j * z_j */
            v.add(-1.0 * alpha, *z1); // 计算 v -= alpha * z1
            /* 计算 x 向量的重正交过程 */
            if (i > 1) {
                /* 每次的内积运算应该用原来v向量的副本，而不是使用更新后的v */
                // x -= \sum{(x, xj) * zj}
                UINT32 k;
                FLOAT64 tmp_product;
                HighPrecisionType* reorth_v_ptr = reorth_v.getMatValPtr();
                HighPrecisionType* reorth_z_ptr = reorth_z.getMatValPtr();
                // w.copy(*v);
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
            levelCInvRecursiveSolve(level, *w, 0); // 计算 w = C_{l}^(-1) * v = A_{l+1}^{-1} * v

            /* 计算 \beta_{j+1} = sqrt((w, x))*/
            // 当beta小于或等于0，直接退出循环
            HighPrecisionType tmp = w->innerProduct(*v);
            if (tmp < 0) // 理论上tmp不为负，但实际计算时会出现负数，这里直接视作接近0并退出
                break;
            beta = sqrt(tmp);
            if (beta <= AMSED_EXIT_ZERO)
                break;
            //将每次更新的alpha，beta填入D,E向量（即对称三对角矩阵的主对角线、副对角线上的值）
            D[i] = alpha;
            E[i] = beta;
            //将v1向量放入V,则V为[v0,v1,v2,...,v(m-1)]
            // V.setValsByCol(i, *v1);
            V.copy(*v1, 0, offsetV, n_remain);
            offsetV += n_remain;
            reorth_v.setValsByCol(i, *v1);
            reorth_z.setValsByCol(i, *z1);
            //v1=w/bata,z0=z1,z1=v/beta
            for (UINT32 j = 0; j < n_remain; ++j) {
                v1[j] = w[j] / beta;
                z0[j] = z1[j]; // 令z_{j-1} = z_j
                z1[j] = v[j] / beta; // 更新z_{j}
            }
        }
        // 将D和E调整为实际的大小
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::computeLanczosEigenValuesAndEigenVectors(
        HostVector<HighPrecisionType>& D, HostVector<HighPrecisionType>& E, INT32 actualLanczosSteps,
        HostVector<HighPrecisionType>& Y, HostVector<HighPrecisionType>& mu) {
        mu.fillVector(0, mu.getLength(), 0);
        Y.fillVector(0, Y.getLength(), 0);

        int m = actualLanczosSteps; //找到m个特征值
        HighPrecisionType tmp = std::is_same_v<HighPrecisionType, FLOAT64> ? LAPACKE_dlamch('S') : LAPACKE_slamch('S');
        HighPrecisionType abstol = 2 * tmp; //这个最佳选择为2*dlamch(S)

        AutoAllocateVector<INT32> ifail(actualLanczosSteps, memoryBase); // 记录错误信息
        int info;
        //设置lapack参数，并求特征值
        char JOBZ = 'V';
        char RANGE = 'A'; // 这个参数表示计算所有特征值
        double vl, vu;
        int il, iu;
        if (std::is_same_v<HighPrecisionType, FLOAT64>) {
            info = LAPACKE_dstevx(LAPACK_COL_MAJOR, JOBZ, RANGE, actualLanczosSteps, (FLOAT64*)D.getRawValPtr(),
                                  (FLOAT64*)E.getRawValPtr(),
                                  vl, vu, il, iu, abstol, &m, (FLOAT64*)mu.getRawValPtr(),
                                  (FLOAT64*)Y.getRawValPtr(), actualLanczosSteps,
                                  ifail.getRawValPtr());
        } else {
            info = LAPACKE_sstevx(LAPACK_COL_MAJOR, JOBZ, RANGE, actualLanczosSteps, (FLOAT32*)D.getRawValPtr(),
                                  (FLOAT32*)E.getRawValPtr(),
                                  vl, vu, il, iu, abstol, &m, (FLOAT32*)mu.getRawValPtr(),
                                  (FLOAT32*)Y.getRawValPtr(), actualLanczosSteps,
                                  ifail.getRawValPtr());
        }
#ifndef NDEBUG
        if (info != 0) {
            ifail.printVector("ifail");
            THROW_LOGIC_ERROR("The computation of eigen-values failed!");
        }
#endif
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType,
                           HighPrecisionType>::implicitRestartKrylovSchurLanczosForGeneralizedEigenvaluesProblem(
        INT32 level) {
        // FLOAT64 startTime{0}, endTime{0}, execTime{0};

        const INT32 stepsForEachRestart = m_lowRankSize * AMSED_LANCZOS_EXPAND_COEFFICIENT; // 每次重启前实际计算的Lanczos迭代步数，即m
        const INT32 expandStepsForRestart = m_lowRankSize * AMSED_LANCZOS_MAX_COEFFICIENT; // 扩展计算的Lanczos迭代步数，即k
        const INT32 restartTimes = 3;
        UINT32 n_start = m_lev_ptr_v[level]; // 当前水平B块的开始行号
        UINT32 n_end = m_lev_ptr_v[level + 1]; // 当前水平B块的结束行号
        UINT32 n_local = n_end - n_start; // 当前水平B块的大小
        UINT32 n_remain = this->m_ArowNum - n_end; // 当前水平C块的大小
        INT32 maxLanczosSteps = m_lowRankSize * (AMSED_LANCZOS_MAX_COEFFICIENT + 1);
        // 最大Lanczos迭代步数不超过设定迭代步数的五倍（此处参考MSLR中的设置）
        LevelStructure<HighPrecisionType>& level_str = *m_levs_all[level];
        CSRMatrix<HighPrecisionType>& currCmat = level_str.C_mat;
        CSRMatrix<HighPrecisionType>& currEmat = level_str.E_mat;
        UINT32 EmatNNZNum = currEmat.getNNZnum(0, currEmat.getRowNum() - 1);
        if (EmatNNZNum == 0) return; // 如果E矩阵为空，则无需计算后续步骤

        /* 初始化Lanczos过程中的变量 */
        // 下面的辅助变量都按照最大空间开好，内部就无需重新分配，否则时间开销较大
        AutoAllocateVector<HighPrecisionType> D(maxLanczosSteps, memoryBase); // D中存储了alpha_1,...,alpha_m
        AutoAllocateVector<HighPrecisionType> E(maxLanczosSteps, memoryBase); // E中存储了beta_1,...,beta_m，但只需用前m-1个

#ifdef CUDA_ENABLED
        AutoAllocateVector<HighPrecisionType> V(n_remain * maxLanczosSteps, memoryPageLocked); // Lanczos过程产生的C-范数下的正交向量
        AutoAllocateVector<HighPrecisionType> auxV(n_remain * maxLanczosSteps, memoryPageLocked);
        // Lanczos过程产生的C-范数下的正交向量
        AutoAllocateVector<HighPrecisionType> Z(n_remain * maxLanczosSteps, memoryPageLocked); // Lanczos过程辅助正交向量
        AutoAllocateVector<HighPrecisionType> auxZ(n_remain * maxLanczosSteps, memoryPageLocked); // Lanczos过程辅助正交向量
        AutoAllocateVector<HighPrecisionType> Y(maxLanczosSteps * maxLanczosSteps, memoryPageLocked); // T矩阵的特征向量
        V->fillVector(0, V.getLength(), 0);
        auxV->fillVector(0, auxV.getLength(), 0);
        Z->fillVector(0, Z.getLength(), 0);
        auxZ->fillVector(0, auxZ.getLength(), 0);
        Y->fillVector(0, Y.getLength(), 0);
#else
        AutoAllocateVector<HighPrecisionType> V(n_remain * maxLanczosSteps, memoryBase); // Lanczos过程产生的C-范数下的正交向量
        AutoAllocateVector<HighPrecisionType> auxV(n_remain * maxLanczosSteps, memoryBase); // Lanczos过程产生的C-范数下的正交向量
        AutoAllocateVector<HighPrecisionType> Z(n_remain * maxLanczosSteps, memoryBase); // Lanczos过程辅助正交向量
        AutoAllocateVector<HighPrecisionType> auxZ(n_remain * maxLanczosSteps, memoryBase); // Lanczos过程辅助正交向量
        AutoAllocateVector<HighPrecisionType> Y(maxLanczosSteps * maxLanczosSteps, memoryBase); // T矩阵的特征向量
#endif

        AutoAllocateVector<HighPrecisionType> mu(maxLanczosSteps, memoryBase); // T矩阵分解后的特征值
        AutoAllocateVector<HighPrecisionType> u(n_remain, memoryBase); // u
        AutoAllocateVector<HighPrecisionType> w(n_remain, memoryBase); // w
        AutoAllocateVector<HighPrecisionType> lastRowY(stepsForEachRestart, memoryBase); // z，即Y的最后一行前m个元素
        AutoAllocateVector<HighPrecisionType> u_temp(n_local, memoryBase); // 对应u的辅助向量
        INT32 actualLanczosSteps{0}; // 记录每次执行Lanczos迭代时实际的迭代步数
        // 计算1步Lanczos迭代
        /* V=[ ]; v=rand(size(C,1),1);  v=v/sqrt(v' * C * v); V=[V,v]; */
        generateArrayRandom1D(V.getRawValPtr(), n_remain); // 生成随机向量，直接放在V的开头处，即v1
        // V->fillVector(0, n_remain, 1);
        FLOAT64 scaleNum = currCmat.getMatrixNorm(*V, 0); // 计算初始随机向量的C-范数
        V->scale(1.0 / scaleNum, 0, n_remain); // 放缩随机向量v1
        /* z = C * v; Z=[Z,z];*/
        currCmat.MatPVec(*V, 0, *Z, 0); // 在Z的首个向量处生成对应的z1=C * v1
        /* u=E'*(B\(E*v)); */
        currEmat.MatPVec(*V, 0, *u_temp, 0); // u_temp = E * v
        levelBlockInvSolve(level, *u_temp, 0); // u_temp = B^{-1} * u_temp
        currEmat.transMatPVec(*u_temp, *u); // u = E^T * u_temp
        /* alpha(1)=u'*v; */
        D[0] = u->innerProduct(*V, 0, 0, n_remain); // alpha(1)=u^T*v1
        /* u=u-alpha(1)*z; */
        u.add(-1.0 * D[0], *Z, 0, 0, n_remain);
        /* u=u-(u'*v)*z; */
        FLOAT64 coefOrthogonal = u->innerProduct(*V, 0, 0, n_remain);
        u.add(-1.0 * coefOrthogonal, *Z, 0, 0, n_remain);
        /* w=C^{-1} * u; */
        w.copy(*u);
        levelCInvRecursiveSolve(level, *w, 0);
        // levelCInvRecursiveSolveWithCorrection(level, *w, 0);
        /* beta(1)=sqrt(w' * u); */
        E[0] = w->innerProduct(*u);
        E[0] = sqrt(E[0]);
        /* v=w/beta(1); V=[V,v]; */
        V.copy(*w, 0, n_remain, n_remain); // 拷贝w的值到v2向量的位置（v1之后，起始位置n_remain，长度n_remain）
        V->scale(1.0 / E[0], n_remain, n_remain);
        /* z = u/beta(1); Z = [Z,z]; */
        Z.copy(*u, 0, n_remain, n_remain);
        Z->scale(1.0 / E[0], n_remain, n_remain);
        // 将1步Lanczos迭代扩展为stepsForEachRestart步
        /* [V,Z,alpha,beta,rk]=expand_lanczos(E, B, C, V, Z, alpha,beta,1,m);
         * m: stepsForEachRestart;
         * rk: actualLanczosSteps;
         * D: alpha;
         * E: beta; */
        expandReorthogonalLanczos(level, *D, *E, *V, *Z, 1, stepsForEachRestart, actualLanczosSteps);
        // 开始重启，matlab模拟大概2次重启就能收敛，最多重启5次，否则计算复杂度过高
        INT32 endEigIdx = actualLanczosSteps - 1; // 记录最后一个近似特征值的索引
        UINT32 restartIdx;
        for (restartIdx = 0; restartIdx < restartTimes; ++restartIdx) {
            if (actualLanczosSteps < stepsForEachRestart) { // 已经提前终止了，就不用再重启
                /*  T=diag(alpha)+diag(beta(1:end-1),1)+diag(beta(1:end-1),-1);
                    [Y,M]=eig(T);  %M的对角元按升序排列，特征向量对应特征值排列
                 说明：计算V=V(:,1:rk)*Y(:,1:rk)的步骤放到后面了 */
                computeLanczosEigenValuesAndEigenVectors(*D, *E, actualLanczosSteps, *Y, *mu);
                break;
            }
            expandReorthogonalLanczos(level, *D, *E, *V, *Z, stepsForEachRestart, expandStepsForRestart,
                                      actualLanczosSteps);

#ifdef CUDA_ENABLED
            UINT32 eigVecLength = actualLanczosSteps * actualLanczosSteps;
            UINT32 originLanczosVecLength = n_remain * actualLanczosSteps;
            m_devLanczosVectorsV.asyncCopyFromHost(*V, 0, 0, originLanczosVecLength, *m_devDataTransStream);
            m_devLanczosVectorsZ.asyncCopyFromHost(*Z, 0, 0, originLanczosVecLength, *m_devDataTransStream);
#endif
            // actualLanczosSteps已更新，更新endEigIdx
            endEigIdx = actualLanczosSteps - 1;
            HighPrecisionType betak = E[endEigIdx]; // 将beta的最后一个分量存起来
            // 计算特征值，升序排列，Y中存对应的特征向量
            computeLanczosEigenValuesAndEigenVectors(*D, *E, actualLanczosSteps, *Y, *mu);
            // 满足收敛条件，退出（最大的那个特征值对应的\beta_k * Y(k,endEigIdx)<=tol）
            if (fabs(betak) * fabs(Y[endEigIdx * actualLanczosSteps + endEigIdx]) <= AMSED_HIGH_PRECISION_EXIT_ZERO
                || restartIdx == restartTimes - 1)
                break;
            // 否则，准备重启
            // 降序排列特征值并调整对应的特征向量
            reverseEigenValuesAndEigenVectors(*mu, *Y, actualLanczosSteps);

            /* 计算auxV = V(:,1:k)*Y(:,1:m), auxZ=Z(:,1:k)*Y(:,1:m) */
            // vk1=V(:,end)就在V第actualLanczosSteps（1-base）列，zk1同理
            // 按列写入最终的结果
#ifdef CUDA_ENABLED
            UINT32 resultLanczosVecLength = n_remain * stepsForEachRestart;
            m_devEigenvectors.asyncCopyFromHost(*Y, 0, 0, eigVecLength, *m_devDataTransStream);
            m_devDataTransStream->synchronize();
            computeLanczosVectorsMultiplyEigenvectorsDEVICE(m_devLanczosVectorsV, m_devAuxLanczosVectorsV,
                                                            m_devEigenvectors,
                                                            n_remain, stepsForEachRestart, actualLanczosSteps);
            computeLanczosVectorsMultiplyEigenvectorsDEVICE(m_devLanczosVectorsZ, m_devAuxLanczosVectorsZ,
                                                            m_devEigenvectors,
                                                            n_remain, stepsForEachRestart, actualLanczosSteps);
#else
            // 因为特征值及对应特征向量已按特征值降序排序，所以将前m（对应变量stepsForEachRestart）个特征值和对应的特征向量保留，其余舍弃
            if (restartIdx > 0) { // 当重启次数大于1次，说明auxV和auxZ已经被使用过，需要将其中的值清零
                auxV->fillVector(0, auxV.getLength(), 0);
                auxZ->fillVector(0, auxZ.getLength(), 0);
            }
            // computeLanczosVectorsMultiplyEigenvectorsHOST(*V, *auxV, *Y, n_remain, stepsForEachRestart,
            //                                               actualLanczosSteps);
            // computeLanczosVectorsMultiplyEigenvectorsHOST(*Z, *auxZ, *Y, n_remain, stepsForEachRestart,
            //                                               actualLanczosSteps);  // 等价于下面的代码
#ifdef OPENMP_FOUND
#pragma omp parallel for default(none) proc_bind(master) num_threads(THREAD_NUM) \
                        shared(stepsForEachRestart, auxV, auxZ, Y, actualLanczosSteps, n_remain, V, Z)
#endif
            for (UINT32 i = 0; i < stepsForEachRestart; ++i) {
                HighPrecisionType* auxVColsValsPtr = auxV.getRawValPtr() + i * n_remain;
                HighPrecisionType* auxZColsValsPtr = auxZ.getRawValPtr() + i * n_remain;
                HighPrecisionType* localYColsValsPtr = Y.getRawValPtr() + i * actualLanczosSteps;
                for (UINT32 colIdx = 0; colIdx < actualLanczosSteps; ++colIdx) { // V的列标，也是Y的行标
                    UINT32 begin = colIdx * n_remain;
                    HighPrecisionType localYval = localYColsValsPtr[colIdx]; // 当前列对应的Y中的值
                    for (UINT32 rowIdx = 0; rowIdx < n_remain; ++rowIdx) {
                        auxVColsValsPtr[rowIdx] += V[begin + rowIdx] * localYval;
                        auxZColsValsPtr[rowIdx] += Z[begin + rowIdx] * localYval;
                    }
                }
            }
#endif

            // 把Y的最后一行对应要保留的部分（前m个）提取出来，并存到z中
            for (UINT32 step = 0, localOffset = endEigIdx; step < stepsForEachRestart; ++step, localOffset +=
                 actualLanczosSteps) {
                lastRowY[step] = Y[localOffset];
            }
            // 将对角矩阵还原为三对角矩阵，并对Lanczos向量组及辅助向量组进行同样的变换
            // 目前auxV存的是有效值，V中前stepsForEachRestart列要被覆盖，所以传参要反着传，Z和auxZ同理
#ifdef CUDA_ENABLED
            convertDiagonal2TriDiagonalDEVICE(*mu, *lastRowY, *D, *E, m_devAuxLanczosVectorsV, m_devLanczosVectorsV,
                                              m_devAuxLanczosVectorsZ, m_devLanczosVectorsZ, stepsForEachRestart,
                                              n_remain);
            m_devComputeStream->synchronize();
            m_devAuxLanczosVectorsV.asyncCopyToHost(*auxV, 0, 0, resultLanczosVecLength, *m_devDataTransStream);
            m_devAuxLanczosVectorsZ.asyncCopyToHost(*auxZ, 0, 0, resultLanczosVecLength, *m_devDataTransStream);
            m_devDataTransStream->synchronize();
#else
            convertDiagonal2TriDiagonal(*mu, *lastRowY, *D, *E, *auxV, *V, *auxZ, *Z, stepsForEachRestart, n_remain);
#endif

            UINT32 retainedLastIdx = stepsForEachRestart - 1;
            E[retainedLastIdx] = lastRowY[retainedLastIdx] * betak; // beta(m)=theta*betak
            // 将vk1移动到第stepsForEachRestart（1-base）列，actualLanczosSteps > stepsForEachRestart，zk1同理
            UINT32 colOffsetOrigin = actualLanczosSteps * n_remain, colOffsetNew = stepsForEachRestart * n_remain;
            V.copy(*V, colOffsetOrigin, colOffsetNew, n_remain);
            Z.copy(*Z, colOffsetOrigin, colOffsetNew, n_remain);
        }
#ifndef NDEBUG
        std::cout << " --- implicit restart Lanczos loops: " << restartIdx + 1 << " times." << std::endl;
#endif
        // 构建当前水平的低秩修正项，以当前收敛的近似特征对，构造低秩修正项
        INT32 m = m_lowRankType == AMSEDBasic ? std::min(actualLanczosSteps, m_lowRankSize) : actualLanczosSteps;
        //找到m个特征值
        HighPrecisionType lambda; // 当前特征值
        if (m_lowRankType == AMSEDBasic) {
            level_str.H_vec.resize(m, RESERVE_NO_DATA);
            level_str.W_mat.resize(n_remain, m, RESERVE_NO_DATA);
#ifdef OPENMP_FOUND
#pragma omp parallel for default(none) proc_bind(master) num_threads(THREAD_NUM) private(lambda) \
    shared(level_str, mu, m, actualLanczosSteps, n_remain, V, Y)
#endif
            for (UINT32 i = 0; i < m; ++i) {
                // 由于原本特征值从小到大排序，所以倒着取前m大特征值，并把特征对按顺序写入当前level的H和W中
                UINT32 currConvergedEigIdx = actualLanczosSteps - i - 1;
                lambda = mu[currConvergedEigIdx];
                level_str.H_vec[i] = lambda / (1.0 - lambda);
                // 计算 W = V * Y
                HighPrecisionType* localWColsValsPtr = level_str.W_mat.getMatValPtr() + i * n_remain;
                HighPrecisionType* localYColsValsPtr = Y.getRawValPtr() + currConvergedEigIdx * actualLanczosSteps;
                for (UINT32 colIdx = 0; colIdx < actualLanczosSteps; ++colIdx) {
                    UINT32 begin = colIdx * n_remain;
                    for (UINT32 j = 0; j < n_remain; ++j) {
                        localWColsValsPtr[j] += V[begin + j] * localYColsValsPtr[colIdx];
                    }
                }
            }
            level_str.rank = m;
        } else if (m_lowRankType == AMSEDEigenvalueDeflation) {
            // 新方法，修正较小特征值，调节 S^{-1} * S 的特征值往1附近聚集
            HighPrecisionType epsilon = m_eigValCorrectBound;
            HighPrecisionType para = 1.0 - epsilon; //选择大于1-epsilon的特征值
            // 统计几个特征值需要被修正
            INT32 correctEigCount = 0;
            for (UINT32 i = 0; i < m; ++i) {
                if (mu[actualLanczosSteps - i - 1] <= para || correctEigCount >= m_lanczosSteps) break;
                correctEigCount++;
            }
            // 生成H和W
            level_str.H_vec.resize(correctEigCount, RESERVE_NO_DATA);
            level_str.W_mat.resize(n_remain, correctEigCount, RESERVE_NO_DATA);

#ifdef OPENMP_FOUND
#pragma omp parallel for default(none) proc_bind(master) num_threads(THREAD_NUM) private(lambda) \
    shared(level_str, mu, correctEigCount, epsilon, actualLanczosSteps, n_remain, V, Y)
#endif
            for (UINT32 i = 0; i < correctEigCount; ++i) {
                // 由于原本特征值从小到大排序，所以倒着取前m大特征值，并把特征对按顺序写入当前level的H和W中
                UINT32 currConvergedEigIdx = actualLanczosSteps - i - 1;
                lambda = 1.0 - mu[currConvergedEigIdx]; //取出最大特征值
                level_str.H_vec[i] = (epsilon - lambda) / lambda;
                // 计算 W = V * Y
                HighPrecisionType* localWColsValsPtr = level_str.W_mat.getMatValPtr() + i * n_remain;
                HighPrecisionType* localYColsValsPtr = Y.getRawValPtr() + currConvergedEigIdx * actualLanczosSteps;
                for (UINT32 colIdx = 0; colIdx < actualLanczosSteps; ++colIdx) {
                    UINT32 begin = colIdx * n_remain;
                    for (UINT32 j = 0; j < n_remain; ++j) {
                        localWColsValsPtr[j] += V[begin + j] * localYColsValsPtr[colIdx];
                    }
                }
            }
            level_str.rank = correctEigCount;
        }

        // 更新M的非零元个数
        // std::cout << " --- final correction low-rank size: " << level_str.rank << std::endl;
        UINT32 localMnnz = 2 * level_str.rank * n_remain + level_str.rank;
        this->m_Mnnz += localMnnz;
        m_lowRankNNZ += localMnnz;
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::expandReorthogonalLanczos(INT32 level,
        HostVector<HighPrecisionType>& D, HostVector<HighPrecisionType>& E, HostVector<HighPrecisionType>& V,
        HostVector<HighPrecisionType>& Z, INT32 originalLanczosSteps, INT32 expandSteps, INT32& actualLanczosSteps) {
#ifndef NDEBUG
        THROW_EXCEPTION(originalLanczosSteps >= expandSteps,
                        THROW_LOGIC_ERROR("The expand steps should larger than orginal steps."))
#endif

        if (D.getLength() < expandSteps) D.resize(expandSteps, RESERVE_DATA);
        if (E.getLength() < expandSteps) E.resize(expandSteps, RESERVE_DATA);
        // 获得当前level各对象的引用
        LevelStructure<HighPrecisionType>& level_str = *m_levs_all[level];
        CSRMatrix<HighPrecisionType>& currEmat = level_str.E_mat;

        UINT32 n_start = m_lev_ptr_v[level]; // 当前水平B块的开始行号
        UINT32 n_end = m_lev_ptr_v[level + 1]; // 当前水平B块的结束行号
        UINT32 n_local = n_end - n_start; // 当前水平B块的大小
        UINT32 n_remain = this->m_ArowNum - n_end; // 当前水平C块的大小
        UINT32 maxSpaceSizeForLanczosVectors = (expandSteps + 1) * n_remain;
        // Lanczos向量个数始终比alpha（beta）的个数多一个
        if (V.getLength() < maxSpaceSizeForLanczosVectors) V.resize(maxSpaceSizeForLanczosVectors, RESERVE_DATA);
        if (Z.getLength() < maxSpaceSizeForLanczosVectors) Z.resize(maxSpaceSizeForLanczosVectors, RESERVE_DATA);

        AutoAllocateVector<HighPrecisionType> u(n_remain, memoryBase); // u
        AutoAllocateVector<HighPrecisionType> w(n_remain, memoryBase); // w
        AutoAllocateVector<HighPrecisionType> u_temp(n_local, memoryBase); // 对应u的辅助向量

        // 初始化
        actualLanczosSteps = originalLanczosSteps; // 记录实际执行的Lanczos迭代步数，因为有提前退出的情况
        UINT32 vStartPos = originalLanczosSteps * n_remain; // 表示V(:,i)向量的实际起始位置
        for (UINT32 stepIdx = originalLanczosSteps; stepIdx < expandSteps; ++stepIdx) {
            if (E[stepIdx - 1] <= AMSED_EXIT_ZERO) return;
            // for i=m+1:k(1-base), for i=m:k-1(0-base)
            actualLanczosSteps++;
            /* u=E'*(B \ (E*V(:,i)))-beta(i-1)*Z(:,i-1) */
            currEmat.MatPVec(V, vStartPos, *u_temp, 0); // u_temp = E*V(:,i)
            levelBlockInvSolve(level, *u_temp, 0); // u_temp = B^{-1} * u_temp
            u->fillVector(0, n_remain, 0);
            currEmat.transMatPVec(*u_temp, 0, *u, 0); // u = E^T * u_temp
            u.add(-1.0 * E[stepIdx - 1], Z, vStartPos - n_remain, 0, n_remain); // u=u-beta(i-1)*Z(:,i-1)
            /* alpha(i)=u'*V(:,i); */
            D[stepIdx] = u->innerProduct(V, vStartPos, 0, n_remain); // alpha(i)=u^T*V(:,i)
            /* u=u-alpha(i)*Z(:,i) */
            u.add(-1.0 * D[stepIdx], Z, vStartPos, 0, n_remain);
            /* 完全正交化 u=u-Z(:,1:i)*(V(:,1:i)'*u) */
            // w.copy(*u); // 拷贝u的值到辅助向量中，记录正交化之前的值，此时的w相当于u
            UINT32 reorthOffset = 0;
            for (UINT32 reorthIdx = 0; reorthIdx <= stepIdx; ++reorthIdx) {
                HighPrecisionType reorthCoef = V.innerProduct(*u, 0, reorthOffset, n_remain);
                u.add(-1.0 * reorthCoef, Z, reorthOffset, 0, n_remain);
                reorthOffset += n_remain;
            }
            /* w = C\u */
            w.copy(*u);
            levelCInvRecursiveSolve(level, *w, 0); // w=C^{-1} * u
            // levelCInvRecursiveSolveWithCorrection(level, *w, 0);
            /* beta(i)=sqrt(w' * u) */
            E[stepIdx] = w->innerProduct(*u);
            if (E[stepIdx] < 0) { // 实际计算中由于B逆和C逆是近似计算，可能导致这里出现负数，直接停止计算
                E[stepIdx] = AMSED_EXIT_ZERO;
                break;
            }
            E[stepIdx] = sqrt(E[stepIdx]);
            if (E[stepIdx] <= AMSED_EXIT_ZERO) break; // if (beta(i)==0) return

            UINT32 nextStartPos = vStartPos + n_remain; // 更新的v在V中存储的起始位置
            HighPrecisionType nextCoef = 1.0 / E[stepIdx];
            V.copy(*w, 0, nextStartPos, n_remain); // 拷贝w的值到向量的V(:,i+1)位置
            V.scale(nextCoef, nextStartPos, n_remain); // v=w/beta(i); V=[V,v]
            Z.copy(*u, 0, nextStartPos, n_remain); // 拷贝w的值到向量的Z(:,i+1)位置
            Z.scale(nextCoef, nextStartPos, n_remain); // z = u/beta(i); Z = [Z,z]

            vStartPos += n_remain;
        }
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::reverseEigenValuesAndEigenVectors(
        HostVector<HighPrecisionType>& mu, HostVector<HighPrecisionType>& Y, INT32 eigenvaluesCount) {
        if (eigenvaluesCount == 0) return;
        UINT32 upperBoundIdx = eigenvaluesCount / 2;
#ifdef OPENMP_FOUND
#pragma omp parallel for default(none) proc_bind(master) num_threads(THREAD_NUM) \
shared(upperBoundIdx, mu, eigenvaluesCount, Y)
#endif
        for (UINT32 eigenvalueIdx = 0; eigenvalueIdx < upperBoundIdx; ++eigenvalueIdx) {
            UINT32 pairIdx = eigenvaluesCount - 1 - eigenvalueIdx;
            HighPrecisionType tempVal;
            // 翻转特征值
            tempVal = mu[eigenvalueIdx];
            mu[eigenvalueIdx] = mu[pairIdx];
            mu[pairIdx] = tempVal;
            UINT32 vecStartPos = eigenvalueIdx * eigenvaluesCount;
            UINT32 vecPairStartPos = pairIdx * eigenvaluesCount;
            // 翻转特征向量
            for (UINT32 vecIdx = 0; vecIdx < eigenvaluesCount; ++vecIdx) {
                UINT32 vecLocalStartPos = vecStartPos + vecIdx;
                UINT32 vecLocalPairStartPos = vecPairStartPos + vecIdx;
                tempVal = Y[vecLocalStartPos];
                Y[vecLocalStartPos] = Y[vecLocalPairStartPos];
                Y[vecLocalPairStartPos] = tempVal;
            }
        }
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::convertDiagonal2TriDiagonal(
        HostVector<HighPrecisionType>& mu, HostVector<HighPrecisionType>& lastRowY, HostVector<HighPrecisionType>& D,
        HostVector<HighPrecisionType>& E, HostVector<HighPrecisionType>& V, HostVector<HighPrecisionType>& auxV,
        HostVector<HighPrecisionType>& Z, HostVector<HighPrecisionType>& auxZ, INT32 retainedLanczosSteps,
        INT32 vectorLanczosLength) {
        /*  m=retainedLanczosSteps;
            for i=1:m
                alpha(i)=M(i,i);
            end */
        D.copy(mu, 0, 0, retainedLanczosSteps);
        // beta=zeros(m-1,1)
        E.fillVector(0, retainedLanczosSteps - 1, 0);
        // 实际运算时正交变换矩阵Q不用实际存储，而是直接将对应的变换应用到Lanczos向量组上
        UINT32 lastIdx = retainedLanczosSteps - 1;
        for (UINT32 i = 0; i < lastIdx; ++i) { // for i=1:m-1(1-base) => for i=0:m-2(0-base)
            /* delta=sqrt(z(i)^2+z(i+1)^2); */
            HighPrecisionType z_i, z_iplus1; // z[i], z[i+1]
            z_i = lastRowY[i];
            z_iplus1 = lastRowY[i + 1];
            HighPrecisionType delta = sqrt(z_i * z_i + z_iplus1 * z_iplus1);
            HighPrecisionType c, s; // Givens变换参数
            /* c=z(i+1)/delta; s=-z(i)/delta; */
            c = z_iplus1 * 1.0 / delta;
            s = -1.0 * z_i / delta;
            /* z(i+1)=delta; z(i)=0; */
            lastRowY[i + 1] = delta;
            lastRowY[i] = 0;
            HighPrecisionType square_c = c * c;
            HighPrecisionType square_s = s * s;
            HighPrecisionType c_dot_s = c * s;
            /* a=[c,s;-s,c]*[alpha(i),beta(i); beta(i), alpha(i+1)]*[c,s;-s,c]^T
             * 转换为：
             * a11 = c^2 * alpha(i) + 2 * c * s * beta(i) + s^2 * alpha(i+1)
             * a12 = -c * s * alpha(i) + (c^2 - s^2) * beta(i) + c * s * alpha(i+1)
             * a21 = -c * s * alpha(i) + (c^2 - s^2) * beta(i) + c * s * alpha(i+1) = a12
             * a22 = s^2 * alpha(i) - 2 * c * s * beta(i) + c^2 * alpha(i+1)
             * 最终：
             * alpha(i)=a11; beta(i)=a12; alpha(i+1)=a22; */
            HighPrecisionType alpha_i = D[i], beta_i = E[i], alpha_iplus1 = D[i + 1], tempRes = 2 * c_dot_s * beta_i;
            D[i] = square_c * alpha_i + tempRes + square_s * alpha_iplus1;
            E[i] = -1.0 * c_dot_s * alpha_i + (square_c - square_s) * beta_i + c_dot_s * alpha_iplus1;
            D[i + 1] = square_s * alpha_i - tempRes + square_c * alpha_iplus1;
            /* Q(i:i+1,:)=[c,s;-s,c]*Q(i:i+1,:); => V(:,i:i+1)=V(:,i:i+1) * [c,s;-s,c]^T; Z和V做同样的变换。
             * 注意：这里的V已经是Vk * Y之后的了 */
            transposedGivensApplied2Columns(c, s, i, i + 1, V, auxV, Z, auxZ, vectorLanczosLength);
            if (i > 0) { // if i>1(1-base) => if i>0(0-base)
                /* gamma=-s*beta(i-1);  beta(i-1)=c*beta(i-1); */
                HighPrecisionType gamma = -1.0 * s * E[i - 1];
                E[i - 1] *= c;
                for (UINT32 r = i; r >= 1; --r) { // for r=i:-1:2(1-base) => for r=i:-1:1(0-base)
                    /* sigma=sqrt(gamma^2+beta(r)^2); */
                    HighPrecisionType alpha_rsubi = D[r - 1], beta_rsubi = E[r - 1], beta_r = E[r], alpha_r = D[r];
                    HighPrecisionType sigma = sqrt(gamma * gamma + beta_r * beta_r);
                    /* c=beta(r)/sigma; s=-gamma/sigma; */
                    c = beta_r * 1.0 / sigma;
                    s = -1.0 * gamma / sigma;
                    square_c = c * c;
                    square_s = s * s;
                    c_dot_s = c * s;
                    /* a=[c,s;-s,c]*[alpha(r-1),beta(r-1); beta(r-1), alpha(r)]*[c,s;-s,c]^T
                     * 转换为：
                     * a11 = c^2 * alpha(r-1) + 2 * c * s * beta(r-1) + s^2 * alpha(r)
                     * a12 = -c * s * alpha(r-1) + (c^2 - s^2) * beta(r-1) + c * s * alpha(r)
                     * a21 = -c * s * alpha(r-1) + (c^2 - s^2) * beta(r-1) + c * s * alpha(r) = a12
                     * a22 = s^2 * alpha(r-1) - 2 * c * s * beta(r-1) + c^2 * alpha(r)
                     * 最终：
                     * alpha(r-1)=a11; beta(r-1)=a12; alpha(r)=a22; */
                    tempRes = 2 * c_dot_s * beta_rsubi;
                    D[r - 1] = square_c * alpha_rsubi + tempRes + square_s * alpha_r;
                    E[r - 1] = -1.0 * c_dot_s * alpha_rsubi + (square_c - square_s) * beta_rsubi + c_dot_s * alpha_r;
                    D[r] = square_s * alpha_rsubi - tempRes + square_c * alpha_r;
                    // beta(r)=sigma
                    E[r] = sigma;
                    /*  if r>2
                            gamma=-s*beta(r-2);  beta(r-2)=c*beta(r-2);
                        end */
                    if (r > 1) {
                        gamma = -1.0 * s * E[r - 2];
                        E[r - 2] *= c;
                    }
                    /* Q(r-1:r,:)=[c,s;-s,c]*Q(r-1:r,:); => V(:,r-1:r)=V(:,r-1:r) * [c,s;-s,c]^T; Z和V做同样的变换。
                    * 注意：这里的V已经是Vk * Y之后的了，auxV中已保留有效的值，现在要写回到V中，所以这里和函数接口顺序是反的 */
                    transposedGivensApplied2Columns(c, s, r - 1, r, V, auxV, Z, auxZ, vectorLanczosLength);
                }
            }
        }
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::transposedGivensApplied2Columns(HighPrecisionType c,
        HighPrecisionType s, UINT32 colIdx1, UINT32 colIdx2, HostVector<HighPrecisionType>& V,
        HostVector<HighPrecisionType>& auxV, HostVector<HighPrecisionType>& Z, HostVector<HighPrecisionType>& auxZ,
        INT32 vectorLanczosLength) {
        // 将Givens变换（转置）右乘到Lanczos向量组的列上，相当于colIdx1和colIdx2提取出来的两列[a_i, b_i] => [a_i * c + b_i * s, -a_i * s + b_i * c]
        UINT32 actualStartPos1 = colIdx1 * vectorLanczosLength, actualStartPos2 = colIdx2 * vectorLanczosLength;
#ifdef OPENMP_FOUND
#pragma omp parallel for default(none) proc_bind(master) num_threads(THREAD_NUM) \
shared(vectorLanczosLength, V, Z, auxV, auxZ, c, s, actualStartPos1, actualStartPos2)
#endif
        for (UINT32 rowIdx = 0; rowIdx < vectorLanczosLength; ++rowIdx) {
            UINT32 localIdx1 = actualStartPos1 + rowIdx;
            UINT32 localIdx2 = actualStartPos2 + rowIdx;
            // 更新a_i
            auxV[localIdx1] = V[localIdx1] * c + V[localIdx2] * s;
            auxZ[localIdx1] = Z[localIdx1] * c + Z[localIdx2] * s;
            // 更新b_i
            auxV[localIdx2] = -1.0 * V[localIdx1] * s + V[localIdx2] * c;
            auxZ[localIdx2] = -1.0 * Z[localIdx1] * s + Z[localIdx2] * c;
        }
    }


    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::levelCInvRecursiveSolve(
        INT32 level, HostVector<HighPrecisionType>& x, UINT32 xLocalStartIdx) {
#ifndef NDEBUG
        // 最后一个水平无C块
        THROW_EXCEPTION(level >= static_cast<INT32>(m_nlev_used) - 1 || level < -1,
                        THROW_LOGIC_ERROR("The level num is out-of-range!"))
#endif
        // 取nextLevel的对象（例如：l = -1 对应A0）
        LevelStructure<HighPrecisionType>& nextLevel = *m_levs_all[level + 1];
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
        nextLevel.E_mat.transMatPVec(x, xLocalStartIdx, *x_copy, 0);
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
        DenseMatrix<HighPrecisionType>& currW = nextLevel.W_mat;
        AutoAllocateVector<HighPrecisionType>& currH = nextLevel.H_vec;
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

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::levelCInvRecursiveSolveWithCorrection(INT32 level,
        HostVector<HighPrecisionType>& x, UINT32 xLocalStartIdx) {
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
            // // 注意一种特殊情况，对于level为-1时，C即为A0，此时的A0是按未重排存储的，x_temp计算的结果也是未重排之后的情况
            //             AutoAllocateVector<HighPrecisionType> x_recover(this->m_ArowNum, memoryBase);
            //             for (UINT32 i = 0; i < this->m_ArowNum; ++i) {
            //                 x_recover[m_pperm[i]] = x[i];
            //             }
            //             m_matA->MatPVec(*x_recover, 0, *x_temp, 0); // x_temp = C * x
            //             // x_correction对应的是重排后的结果
            // #ifdef USE_OMP_VECTOR_FUNC
            // #pragma omp parallel for default(none) num_threads(THREAD_NUM) shared(x_correction, x_temp, m_pperm)
            // #endif
            //             for (UINT32 i = 0; i < this->m_ArowNum; ++i) {
            //                 x_correction[i] -= x_temp[m_pperm[i]];
            //             }

            // 计算四个分块的乘积
            UINT32 blockBsize = m_lev_ptr_v[1] - m_lev_ptr_v[0];
            levelBlockMatPVec(0, x, 0, *x_temp, 0);
            m_levs_all[0]->E_mat.MatPVecWithoutClearOutVector(x, blockBsize, *x_temp, 0);
            m_levs_all[0]->C_mat.MatPVec(x, blockBsize, *x_temp, blockBsize);
            m_levs_all[0]->E_mat.transMatPVec(x, 0, *x_temp, blockBsize);
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

    //     template<typename LowPrecisionType, typename HighPrecisionType>
    //     void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::levelBlockInvSolve(INT32 level,
    //         HostVector<HighPrecisionType> &x, UINT32 xLocalStartIdx) {
    //         LevelStructure<HighPrecisionType> &level_str = *m_levs_all[level];
    //         SharedPtr1D<CSRMatrix<HighPrecisionType> > &currBlockB = level_str.B_mat;
    //         SharedPtr1D<TriangularPrecondition<HighPrecisionType> > &currBlockB_precond = level_str.B_precond;
    //         UINT32 blockBnum = currBlockB.getDim();
    //         UINT32 blockOffset = xLocalStartIdx;
    //         UINT32 globalStartIdx = (*m_dom_ptr_v2[level])[0];
    //         // 计算所有子块对应的temp_v = B^{-1} temp_v
    //         m_timer.cpuTimerStart();
    // #ifdef OPENMP_FOUND
    //         AutoAllocateVector<UINT32> auxStartIdx(blockBnum + 1, memoryBase);
    //         UINT32 alignElemNum = 2 * ALIGNED_BYTES / sizeof(HighPrecisionType);
    //         // 计算每个子块实际存储的起始索引
    //         for (UINT32 i = 1; i <= blockBnum; ++i) {
    //             // 每个子块后面插入一段空白区间，空出两个缓存行，避免伪共享
    //             auxStartIdx[i] = auxStartIdx[i - 1] + currBlockB[i - 1]->getRowNum() + alignElemNum;
    //         }
    //
    // #pragma omp parallel for default(none) proc_bind(master) num_threads(THREAD_NUM)  \
    //     shared(m_dom_ptr_v2, auxStartIdx, blockBnum, level, globalStartIdx, xLocalStartIdx, currBlockB_precond, currBlockB, x)
    //         for (UINT32 bID = 0; bID < blockBnum; ++bID) {
    //             UINT32 subMatStartRowIdx = (*m_dom_ptr_v2[level])[bID] - globalStartIdx + xLocalStartIdx;
    //             std::shared_ptr<TriangularPrecondition<HighPrecisionType> > localBlock = currBlockB_precond[bID];
    // #ifndef  NDEBUG
    //             THROW_EXCEPTION(subMatStartRowIdx + currBlockB[bID]->getRowNum() > x.getLength(),
    //                             THROW_LOGIC_ERROR("The block offset is out-of-range!"))
    // #endif
    //             // 计算时把结果写入到辅助向量里面
    //             localBlock->MInvSolve(x, subMatStartRowIdx, *m_auxParallelSolveB, auxStartIdx[bID]);
    //         }
    //         for (UINT32 bID = 0; bID < blockBnum; ++bID) {
    //             UINT32 currBlockSize = currBlockB[bID]->getRowNum();
    //             // 最后再写回到原向量中
    //             x.copy(*m_auxParallelSolveB, auxStartIdx[bID], blockOffset, currBlockSize);
    //             blockOffset += currBlockSize;
    //         }
    //
    // #else       // 如果不开OpenMP并行，使用下面的串行代码块
    //                 for (UINT32 bID = 0; bID < blockBnum; ++bID) {
    // #ifndef  NDEBUG
    //                     THROW_EXCEPTION(blockOffset + currBlockB[bID]->getRowNum() > x.getLength(),
    //                                     THROW_LOGIC_ERROR("The block offset is out-of-range!"))
    // #endif
    //                     currBlockB_precond[bID]->MInvSolve(x, blockOffset);
    //                     blockOffset += currBlockB[bID]->getRowNum();
    //                 }
    // #endif
    //         m_timer.cpuTimerEnd();
    //         m_blockBParallelTime += m_timer.computeCPUtime();
    //     }


    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::levelBlockInvSolve(INT32 level,
        HostVector<HighPrecisionType>& x, UINT32 xLocalStartIdx) {
#ifndef NDEBUG
        THROW_EXCEPTION(level < 0 || level >= m_levs_all.getDim(),
                        THROW_LOGIC_ERROR("The level index is out-of-range!"));
#endif
        LevelStructure<HighPrecisionType>& level_str = *m_levs_all[level];
        SharedPtr1D<CSRMatrix<HighPrecisionType>>& currBlockB = level_str.B_mat;
        SharedPtr1D<TriangularPrecondition<HighPrecisionType>>& currBlockB_precond = level_str.B_precond;
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
                std::shared_ptr<TriangularPrecondition<HighPrecisionType>> localBlock = currBlockB_precond[bID];
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
        // for (UINT32 bID = 0; bID < blockBnum; ++bID) {
        //     UINT32 currBlockSize = currBlockB[bID]->getRowNum();
        //     // 最后再写回到原向量中
        //     x.copy(*m_auxParallelSolveB, auxStartIdx[bID], blockOffset, currBlockSize);
        //     blockOffset += currBlockSize;
        // }
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

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::levelBlockInvSolveWithCorrection(INT32 level,
        HostVector<HighPrecisionType>& x, UINT32 xLocalStartIdx) {
        UINT32 blockBsize = m_lev_ptr_v[level + 1] - m_lev_ptr_v[level];
        AutoAllocateVector<HighPrecisionType> x_correction(blockBsize, memoryBase);
        AutoAllocateVector<HighPrecisionType> x_temp(blockBsize, memoryBase);
        // x_correction暂存重排后对应的x
        x_correction.copy(x, xLocalStartIdx, 0, blockBsize);
        // 求出不精确的res_inacc = B^{-1} x，其结果直接写入到原来的x中，x_correction保存对应的原来的x
        levelBlockInvSolve(level, x, xLocalStartIdx);
        // 计算x_copy - B * res_inacc，存入到x_correct中
        levelBlockMatPVec(level, x, xLocalStartIdx, *x_temp, 0); // x_temp = B * x
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

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::levelBlockMatPVec(INT32 level,
        HostVector<HighPrecisionType>& vecIN, UINT32 inVecStartIdx, HostVector<HighPrecisionType>& vecOUT,
        UINT32 outVecStartIdx) {
#ifndef NDEBUG
        THROW_EXCEPTION(level < 0 || level >= m_levs_all.getDim(),
                        THROW_LOGIC_ERROR("The level index is out-of-range!"));
#endif
        LevelStructure<HighPrecisionType>& level_str = *m_levs_all[level];
        SharedPtr1D<CSRMatrix<HighPrecisionType>>& currBlockB = level_str.B_mat;
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
    shared(m_dom_ptr_v2, auxStartIdx, blockBnum, level, globalStartIdx, inVecStartIdx, outVecStartIdx , currBlockB, vecIN, vecOUT)
        {
#pragma omp for simd schedule(dynamic) nowait
            for (UINT32 bID = 0; bID < blockBnum; ++bID) {
                UINT32 distFromGlobalStartIdx = (*m_dom_ptr_v2[level])[bID] - globalStartIdx;
                UINT32 subInMatStartRowIdx = distFromGlobalStartIdx + inVecStartIdx;
                UINT32 subOutMatStartRowIdx = distFromGlobalStartIdx + outVecStartIdx;
                std::shared_ptr<CSRMatrix<HighPrecisionType>> localBlock = currBlockB[bID];
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

#ifdef CUDA_ENABLED
    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::computeLanczosVectorsMultiplyEigenvectorsDEVICE(
        DeviceVector<HighPrecisionType>& devV, DeviceVector<HighPrecisionType>& devAuxV,
        DeviceVector<HighPrecisionType>& devY, INT32 n_remain, INT32 stepsForEachRestart, INT32 actualLanczosSteps) {
        // 数据传输在外部进行，使用Data流异步传输，因此需要记录Data流的传输事件，并根据这个进行同步
        // m_devComputeStream->waitEvent(*cudaEvent);
        // cudaStream.synchronize(); // 确保所有数据已经传输完毕
        m_cublasTools.cublasMatMatMul(1.0, CUBLAS_OP_N, devV, CUBLAS_OP_N, devY, 0.0, devAuxV, n_remain,
                                      actualLanczosSteps, stepsForEachRestart);
    }


    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::convertDiagonal2TriDiagonalDEVICE(
        HostVector<HighPrecisionType>& mu, HostVector<HighPrecisionType>& lastRowY, HostVector<HighPrecisionType>& D,
        HostVector<HighPrecisionType>& E, DeviceVector<HighPrecisionType>& devV,
        DeviceVector<HighPrecisionType>& devAuxV,
        DeviceVector<HighPrecisionType>& devZ, DeviceVector<HighPrecisionType>& devAuxZ, INT32 retainedLanczosSteps,
        INT32 vectorLanczosLength) {
        /*  m=retainedLanczosSteps;
            for i=1:m
                alpha(i)=M(i,i);
            end */
        D.copy(mu, 0, 0, retainedLanczosSteps);
        // beta=zeros(m-1,1)
        E.fillVector(0, retainedLanczosSteps - 1, 0);
        // 实际运算时正交变换矩阵Q不用实际存储，而是直接将对应的变换应用到Lanczos向量组上
        UINT32 lastIdx = retainedLanczosSteps - 1;
        for (UINT32 i = 0; i < lastIdx; ++i) { // for i=1:m-1(1-base) => for i=0:m-2(0-base)
            /* delta=sqrt(z(i)^2+z(i+1)^2); */
            HighPrecisionType z_i, z_iplus1; // z[i], z[i+1]
            z_i = lastRowY[i];
            z_iplus1 = lastRowY[i + 1];
            HighPrecisionType delta = sqrt(z_i * z_i + z_iplus1 * z_iplus1);
            HighPrecisionType c, s; // Givens变换参数
            /* c=z(i+1)/delta; s=-z(i)/delta; */
            c = z_iplus1 * 1.0 / delta;
            s = -1.0 * z_i / delta;
            /* z(i+1)=delta; z(i)=0; */
            lastRowY[i + 1] = delta;
            lastRowY[i] = 0;
            HighPrecisionType square_c = c * c;
            HighPrecisionType square_s = s * s;
            HighPrecisionType c_dot_s = c * s;
            /* a=[c,s;-s,c]*[alpha(i),beta(i); beta(i), alpha(i+1)]*[c,s;-s,c]^T
             * 转换为：
             * a11 = c^2 * alpha(i) + 2 * c * s * beta(i) + s^2 * alpha(i+1)
             * a12 = -c * s * alpha(i) + (c^2 - s^2) * beta(i) + c * s * alpha(i+1)
             * a21 = -c * s * alpha(i) + (c^2 - s^2) * beta(i) + c * s * alpha(i+1) = a12
             * a22 = s^2 * alpha(i) - 2 * c * s * beta(i) + c^2 * alpha(i+1)
             * 最终：
             * alpha(i)=a11; beta(i)=a12; alpha(i+1)=a22; */
            HighPrecisionType alpha_i = D[i], beta_i = E[i], alpha_iplus1 = D[i + 1], tempRes = 2 * c_dot_s * beta_i;
            D[i] = square_c * alpha_i + tempRes + square_s * alpha_iplus1;
            E[i] = -1.0 * c_dot_s * alpha_i + (square_c - square_s) * beta_i + c_dot_s * alpha_iplus1;
            D[i + 1] = square_s * alpha_i - tempRes + square_c * alpha_iplus1;
            /* Q(i:i+1,:)=[c,s;-s,c]*Q(i:i+1,:); => V(:,i:i+1)=V(:,i:i+1) * [c,s;-s,c]^T; Z和V做同样的变换。
             * 注意：这里的V已经是Vk * Y之后的了 */
            transposedGivensApplied2ColumnsDEVICE(c, s, i, i + 1, devV, devAuxV, devZ, devAuxZ, vectorLanczosLength,
                                                  *m_devComputeStream);
            if (i > 0) { // if i>1(1-base) => if i>0(0-base)
                /* gamma=-s*beta(i-1);  beta(i-1)=c*beta(i-1); */
                HighPrecisionType gamma = -1.0 * s * E[i - 1];
                E[i - 1] *= c;
                for (UINT32 r = i; r >= 1; --r) { // for r=i:-1:2(1-base) => for r=i:-1:1(0-base)
                    /* sigma=sqrt(gamma^2+beta(r)^2); */
                    HighPrecisionType alpha_rsubi = D[r - 1], beta_rsubi = E[r - 1], beta_r = E[r], alpha_r = D[r];
                    HighPrecisionType sigma = sqrt(gamma * gamma + beta_r * beta_r);
                    /* c=beta(r)/sigma; s=-gamma/sigma; */
                    c = beta_r * 1.0 / sigma;
                    s = -1.0 * gamma / sigma;
                    square_c = c * c;
                    square_s = s * s;
                    c_dot_s = c * s;
                    /* a=[c,s;-s,c]*[alpha(r-1),beta(r-1); beta(r-1), alpha(r)]*[c,s;-s,c]^T
                     * 转换为：
                     * a11 = c^2 * alpha(r-1) + 2 * c * s * beta(r-1) + s^2 * alpha(r)
                     * a12 = -c * s * alpha(r-1) + (c^2 - s^2) * beta(r-1) + c * s * alpha(r)
                     * a21 = -c * s * alpha(r-1) + (c^2 - s^2) * beta(r-1) + c * s * alpha(r) = a12
                     * a22 = s^2 * alpha(r-1) - 2 * c * s * beta(r-1) + c^2 * alpha(r)
                     * 最终：
                     * alpha(r-1)=a11; beta(r-1)=a12; alpha(r)=a22; */
                    tempRes = 2 * c_dot_s * beta_rsubi;
                    D[r - 1] = square_c * alpha_rsubi + tempRes + square_s * alpha_r;
                    E[r - 1] = -1.0 * c_dot_s * alpha_rsubi + (square_c - square_s) * beta_rsubi + c_dot_s * alpha_r;
                    D[r] = square_s * alpha_rsubi - tempRes + square_c * alpha_r;
                    // beta(r)=sigma
                    E[r] = sigma;
                    /*  if r>2
                            gamma=-s*beta(r-2);  beta(r-2)=c*beta(r-2);
                        end */
                    if (r > 1) {
                        gamma = -1.0 * s * E[r - 2];
                        E[r - 2] *= c;
                    }
                    /* Q(r-1:r,:)=[c,s;-s,c]*Q(r-1:r,:); => V(:,r-1:r)=V(:,r-1:r) * [c,s;-s,c]^T; Z和V做同样的变换。
                    * 注意：这里的V已经是Vk * Y之后的了，auxV中已保留有效的值，现在要写回到V中，所以这里和函数接口顺序是反的 */
                    transposedGivensApplied2ColumnsDEVICE(c, s, r - 1, r, devV, devAuxV, devZ, devAuxZ,
                                                          vectorLanczosLength, *m_devComputeStream);
                }
            }
        }
    }
#endif


    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::setup() {
        if (this->m_isReady == PRECONDITION_READY) return;
        permuteHID();
#ifndef NDEBUG
        // 存储边界信息到指定文件中
        std::stringstream ss;
        getBuildFolderAbsolutePath(ss);
        writeVectorToFile(m_lev_ptr_v.getRawValPtr(), m_lev_ptr_v.getLength(), ss.str().c_str(),
                          "../source/test/testResults/AMSED_B_grid.txt");

        m_pperm.printVector("permutation array after HID ordering");
#endif
        FLOAT64 startTime = omp_get_wtime();
        setupMSLR();
        FLOAT64 endTime = omp_get_wtime();
        FLOAT64 execTime = endTime - startTime;
        std::cout << " --- setup MSLR structure executes: " << execTime * 1000 << " ms." <<
            std::endl;
#ifndef NDEBUG
        SHOW_INFO("Setup AMSED succeed!")
#endif
        this->m_isReady = PRECONDITION_READY;
        m_blockBParallelTime = 0;
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::MInvSolve(BaseVector<HighPrecisionType>& vec,
                                                                           UINT32 resStartIdx) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_isReady != PRECONDITION_READY,
                        THROW_LOGIC_ERROR("The AMSED precondition is not ready!"))
#endif
        if (m_rhsPermType == AMSEDPermuteRightHandInside) {
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

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::MInvSolve(BaseVector<HighPrecisionType>& vecIN,
                                                                           BaseVector<HighPrecisionType>& vecOUT) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_isReady != PRECONDITION_READY,
                        THROW_LOGIC_ERROR("The AMSED precondition is not ready!"))
#endif
        if (m_rhsPermType == AMSEDPermuteRightHandInside) {
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

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::MInvSolve(BaseVector<HighPrecisionType>& vecIN,
                                                                           UINT32 inStartIdx,
                                                                           BaseVector<HighPrecisionType>& vecOUT,
                                                                           UINT32 outStartIdx) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_isReady != PRECONDITION_READY,
                        THROW_LOGIC_ERROR("The AMSED precondition is not ready!"))
#endif
        if (m_rhsPermType == AMSEDPermuteRightHandInside) {
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
                this->m_ArowNum + inStartIdx > vecIN.getLength() || this->m_ArowNum + outStartIdx > vecOUT.getLength(),
                THROW_LOGIC_ERROR("The permuted right-hand is incorrect!"))
#endif
            vecOUT.copy(vecIN, inStartIdx, outStartIdx, this->m_ArowNum);
            m_permRhs->move(vecOUT);
            levelCInvRecursiveSolve(-1, *m_permRhs, outStartIdx); // 要求A0^{-1} * b，即求C_{-1}^{-1} * b
            vecOUT = std::move(*m_permRhs);
        }
    }

    template <typename LowPrecisionType, typename HighPrecisionType>
    void AMSEDPrecondition<LowPrecisionType, HighPrecisionType>::MInvSolve(BaseVector<HighPrecisionType>& vec) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_isReady != PRECONDITION_READY,
                        THROW_LOGIC_ERROR("The AMSED precondition is not ready!"))
#endif
        if (m_rhsPermType == AMSEDPermuteRightHandInside) {
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
