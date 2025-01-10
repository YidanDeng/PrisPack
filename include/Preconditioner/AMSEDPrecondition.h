/*
 * @author  刘玉琴、邓轶丹
 * @date    2024/11/16
 * @details
 *          论文标题：基于特征值放缩的代数多水平Schur补预条件子
 *          Title：  An Algebraic Multilevel Schur complement preconditioner based on Eigenvalue Deflation
 */

#ifndef AMSED_PRECONDITION_H
#define AMSED_PRECONDITION_H

#include "AMSEDPrecondition.h"
#include "BasePreconditon.h"
#include "IncompleteCholesky.h"
#include "IncompleteLDLT.h"
#include "../utils/MemoryTools/SharedPtrTools.h"
#include "../MatrixClass/CSRMatrix.h"
#include "../MatrixClass/DenseMatrix.h"
#include "../utils/ExternalTools/MatrixReorderTools.h"
#include "../utils/ExternalTools/MetisTools.h"
#include "../utils/TimerTools/CPUtimer.hpp"
#include "../utils/TestTools/WriteMtxTools.h"
#include "../utils/TestTools/checkTools.hpp"
#include "../utils/TestTools/generateTools.hpp"
#include "../utils/TestTools/WriteDataTools.hpp"

/* 第三方库头文件 */
#include "metis.h"
#include "lapacke.h"

#ifdef CUDA_ENABLED
#include "../CUDA/BLAS/CUBLAStools.cuh"
#include "../CUDA/StreamController.cuh"
#include "../CUDA/EventController.cuh"
#include "../VectorClass/SyncDeviceVector.cuh"
#include "AMSEDDeviceTools.cuh"
#endif

#define AMSED_SUCCESS 0
#define AMSED_FAILED (-1)

//mslr中的默认参数
#define AMSED_EXIT_ZERO (1e-15)
#define AMSED_HIGH_PRECISION_EXIT_ZERO (1e-30)
#define AMSED_FACTOR_TOLRANCE 1e-3 //ic分解和ILDLT分解公用的参数
#define AMSED_LANCZOS_MAX_COEFFICIENT 4  // 最大Lanczos迭代次数的倍数（相较于原始的low-rank size）
#define AMSED_LANCZOS_EXPAND_COEFFICIENT 2  // 扩展Lanczos的倍数（相较于原始的low-rank size）
#define AMSED_MSTEPS  30  //广义lanczos算法迭代步数
#define AMSED_RANK  16   //低秩修正的秩
#define AMSED_MINSEP 3  //边（点）分隔符的最小size
#define AMSED_EIGVAL_CORRECTION 0.8     // 特征值修正下界参数epsilon

enum LowRankCorrectType {
    AMSEDBasic,
    AMSEDEigenvalueDeflation
};

typedef enum LowRankCorrectType LowRankCorrectType_t;

enum LanczosType {
    UseClassicLanczos,
    UseRestartLanczos
};

typedef enum LanczosType LanczosType_t;

enum AMSEDPermType {
    AMSEDPermuteRightHandOutside,
    AMSEDPermuteRightHandInside
};

typedef enum AMSEDPermType AMSEDPermType_t;

namespace HOST {
    template <typename RightHandPrecisionType>
    struct LevelStructure {
        UINT32 rank{0}; ///< 这一level低秩修正的size
        UINT32 ncomps{0}; ///< 这一level的区域数
        SharedPtr1D<CSRMatrix<RightHandPrecisionType>> B_mat; ///< 当前区域提取的所有B子块
        SharedPtr1D<TriangularPrecondition<RightHandPrecisionType>> B_precond; ///< 当前区域所有B子块对应的不完全分解（最后一层用IC分解）
        CSRMatrix<RightHandPrecisionType> E_mat; ///< 当前区域提取的E子块
        CSRMatrix<RightHandPrecisionType> C_mat; ///< 当前区域提取的C子块
        AutoAllocateVector<RightHandPrecisionType> H_vec; ///< 低秩修正项中的H，为一个对角矩阵，这里用向量存
        DenseMatrix<RightHandPrecisionType> W_mat; ///< 低秩修正项中的W，对应广义特征值问题的特征向量
    };


    template <typename LowPrecisionType, typename HighPrecisionType>
    class AMSEDPrecondition : public BasePrecondition<HighPrecisionType> {
    private:
        UINT32 m_minsep{AMSED_MINSEP};
        UINT32 m_nlev_max{0};
        UINT32 m_nlev_used{0};
        HighPrecisionType m_eigValCorrectBound{AMSED_EIGVAL_CORRECTION};
        // FLOAT64 m_tolerance{AMSED_PRECISION_CONVERGENCE};
        INT32 m_nlev_setup{0};
        UINT32 m_ncomp_setup{0};

        MatrixReorderOption_t m_local_ordering_setup{MatrixReorderAMD};
        std::shared_ptr<CSRMatrix<HighPrecisionType>> m_matA; ///< 原始矩阵，转存变量
        SharedPtr1D<LevelStructure<HighPrecisionType>> m_levs_all;

        AutoAllocateVector<UINT32> m_pperm; ///< pperm为重排序向量，下标为现在的节点编号，值为对应的节点原始的编号
        AutoAllocateVector<UINT32> m_lev_ptr_v; ///< 每个level开始的节点（类似子域的rowOffset）
        SharedPtr1D<AutoAllocateVector<UINT32>> m_dom_ptr_v2; ///< 每个level每个区域开始的节点
        AutoAllocateVector<HighPrecisionType> m_permRhs; ///< 用于求解过程的一个辅助变量，存储排序后的值
        AMSEDPermType_t m_rhsPermType{AMSEDPermuteRightHandInside};
        ///< 如果选outside这个参数，就代表排序过程是在外部处理好的，MInvSolve仅关注求解过程
        LowRankCorrectType_t m_lowRankType{AMSEDBasic}; ///< 选择低秩矫正的类型，如果是basic就是一般的MSLR，是EigenvalueDeflation就是新方法
        INT32 m_lowRankSize{AMSED_RANK};
        UINT32 m_lanczosSteps{AMSED_MSTEPS}; ///< Lanczos迭代上界
        LanczosType_t m_lanczosType{UseRestartLanczos}; ///< Lanczos迭代类型

        UINT8 m_rhsReady{0}; ///< 标记变量，表示右端项是否已经随着系数矩阵A一起重排，只有重排后才能进行计算，否则结果是对不上的

        CPUtimer m_timer; ///< 计时器，用来记录模块运行的时间
        FLOAT64 m_blockBParallelTime{0}; ///< 统计对角块并行求解耗时
        FLOAT64 m_compLowRankCorrectTime{0}; ///< 统计求逆过程中低秩修正的时间
        FLOAT64 m_LanczosTime{0}; ///< 统计Lanczos的时间
        FLOAT64 m_blockDecompositionTime{0}; ///< 统计不完全分解的时间

        /* 用于并行的辅助空间 */
        AutoAllocateVector<HighPrecisionType> m_auxParallelSolveB; ///< 用于并行求分块B逆的辅助空间，避免伪共享
#ifdef CUDA_ENABLED     // 创建用于GPU并行的辅助变量
        SharedObject<DEVICE::StreamController> m_devComputeStream;
        SharedObject<DEVICE::StreamController> m_devDataTransStream;
        DEVICE::CUBLAStools<HighPrecisionType> m_cublasTools;

        DEVICE::SyncDeviceVector<HighPrecisionType> m_devLanczosVectorsV;
        DEVICE::SyncDeviceVector<HighPrecisionType> m_devAuxLanczosVectorsV;
        DEVICE::SyncDeviceVector<HighPrecisionType> m_devLanczosVectorsZ;
        DEVICE::SyncDeviceVector<HighPrecisionType> m_devAuxLanczosVectorsZ;
        DEVICE::SyncDeviceVector<HighPrecisionType> m_devEigenvectors;
#endif

        /* 设置各部分fill的统计量 */
        UINT32 m_incompFactNNZ{0};
        UINT32 m_lowRankNNZ{0};


        /* ===================== HID（Hierarchical Interface Decomposition）排序及其工具接口 ===================== */
        /** @brief 获得整个矩阵的重排序*/
        void permuteHID() {
            AutoAllocateVector<INT32> map_v(0, memoryBase), mapptr_v(0, memoryBase);
            setupPermutationND(*map_v, *mapptr_v);
            buildLevelStructure(*map_v, *mapptr_v);
#ifdef CUDA_ENABLED
            // 创建CUDA流，当前只有一个GPU，所以在默认GPU上创建
            m_devComputeStream.construct(DEFAULT_GPU);
            m_devDataTransStream.construct(DEFAULT_GPU);
            // 将cublas绑定到计算流上
            m_cublasTools.resetStream(m_devComputeStream.get());
            // warm-up
            warmUp(m_levs_all[0]->C_mat.getRowNum(), m_lowRankSize * AMSED_LANCZOS_EXPAND_COEFFICIENT,
                   m_lowRankSize * AMSED_LANCZOS_MAX_COEFFICIENT);
#endif
        }

        /** @brief   建立多水平结构
         * @param [in]  map_v      每个节点所在区域，长度为节点个数，对应每个节点所在区域的编号
         * @param [in]  mapptr_v   长度为tlvl+1，例如：当tlvl=4，每个level包含几个区域[0, 8, 12, 14,15] */
        void buildLevelStructure(HostVector<INT32>& map_v, HostVector<INT32>& mapptr_v);

        /** @brief ND排序，找到每个节点所在区域，和用来索引区域的map向量（类似于子域的rowOffset）
        * @param map_v      每个节点所在区域，长度为节点个数，对应每个节点所在区域的编号
        * @param mapptr_v   长度为tlvl+1，每个level包含几个区域[0, 8, 12, 14,15]
        *                   例如：level 1有8个区域mapptr_v[1]-mapptr_v[0] */
        void setupPermutationND(HostVector<INT32>& map_v, HostVector<INT32>& mapptr_v);


        /** @brief ND划分（递归处理）
         * @param [in]                                                                                                                                                                                                                                                                                                                               clvl 当前level
         * @param [in,out] tlvl 总level
         * @param level_str 存储多水平结构 level_str[i]为第i水平，level_str[i][j]为第i水平第j个划分
         * @return 返回0则运行成功 */
        void setupPermutationNDRecursive(CSRMatrix<HighPrecisionType>& A, UINT32 clvl, UINT32& tlvl,
                                         SharedPtr2D<AutoAllocateVector<UINT32>>& level_str);


        /** @brief 获取矩阵的连通分支
         * @param [in] A: 输入矩阵
         * @param[out]   comp_indices vector容器，每一个位置又是一个数组，存储某个连通分支的所有元素索引。
         *                            comp_indices[i]存放连通分支i包含的节点编号
         * @param[in,out] ncomps      人为规定的连通分支数,如果分出来的连通分支数小于事先规定的，则缩小ncomps；如果分出来的连通分支数大于规定的，则合并连通分支 */
        static void getConnectedComponents(CSRMatrix<HighPrecisionType>& A,
                                           SharedPtr1D<AutoAllocateVector<UINT32>>& comp_indices, UINT32& ncomps);

        /** @brief 根据标记向量提取子矩阵
         * @param [in]  rows 值为0或1的标记向量，长度为m_rowNum
         * @param [in]  cols 值为0或1的标记向量，长度为m_colNum
         * @param [out] row_perm 从原矩阵到新矩阵要提取哪些行,函数外只需要声明就好
         * @param [out] col_perm 从原矩阵到新矩阵要提取哪些列,函数外只需要声明就好
         * @param [in]  complement false：提取rows和cols中非零位置的元素，true：提取rows和cols中零位置的元素
         * @param [out] csr_outmat 提取的子矩阵 */
        static void getSubMatrixNoPerm(CSRMatrix<HighPrecisionType>& A, HostVector<UINT32>& rows,
                                       HostVector<UINT32>& cols,
                                       HostVector<UINT32>& row_perm, HostVector<UINT32>& col_perm, bool complement,
                                       CSRMatrix<HighPrecisionType>& csr_outmat);

        /** @brief 一个简单的check，检查重排序向量是否合理 */
        INT32 checkPermutation(const CSRMatrix<HighPrecisionType>& permA);

        /** @brief 一个用于展示HID排序结果的函数，将排序后的完整矩阵输入mtx文件，用于可视化 */
        static void writePermMat(const CSRMatrix<HighPrecisionType>& permMat);

        /* =================================== 初步构建MSLR框架下所有子块 =================================== */
        /** @brief 构建MSLR预条件 */
        void setupMSLR();

        /** @brief 计算最后一个水平的IC分解 */
        void getLastLevelDecomposition();

        /** @brief 对除最后一个水平外每个水平的B并行地进行ILDL分解*/
        void getBlockDecomposition(INT32 level);

        /* ======================================== 低秩矫正模块 ======================================== */
        /** @brief 得到当前水平的低秩校正 */
        void classicLanczosForGeneralizedEigenvaluesProblem(INT32 level);

        /** @brief 完全重正交化的SecondLanczos算法，求解E^(T)*B^(-1)*Ex=lamda*C*x这样的广义特征值问题，得到三对角和V矩阵
         * @param [in] level 水平编号
         * @param [in, out] D 向量，存储对称三对角矩阵的对角线元素，因为有提前退出的情况，所以长度未必等于Lanczos算法迭代步数
         * @param [in, out] E 向量，存储对称三对角矩阵的次对角线元素，因为有提前退出的情况，所以长度未必等于Lanczos算法迭代步数
         * @param [in, out] V denseMatrix，存储正交向量，列数为m_steps,V为[v0,v1,v2,...,v(m-1)],不需要提前初始化，声明即可
         * @param [in] lanczosSteps 设定的Lanczos迭代步数
         * @param [in, out] actualLanczosSteps 实际的Lanczos迭代步数（收敛的特征值个数） */
        void reorthogonalGeneralizedLanczos(INT32 level, HostVector<HighPrecisionType>& D,
                                            HostVector<HighPrecisionType>& E,
                                            HostVector<HighPrecisionType>& V, INT32 lanczosSteps,
                                            INT32& actualLanczosSteps);

        /** @brief 求对称三对角矩阵的前k个特征值lamda以及对应的特征向量y,生成H和W(W=V*y)
         * @details 调用lapack中函数接口LAPACKE_dstevx
         * @param [in] D 向量，存储对称三对角矩阵的对角线元素，即为Lanczos过程计算出来的对应结果
         * @param [in] E 向量，存储对称三对角矩阵的次对角线元素，即为Lanczos过程计算出来的对应结果
         * @param [in] actualLanczosSteps Lanczos实际执行步数，对应D和E中实际存储的有效元素长度，和V中实际存储的有效向量列数
         * @param [in, out] Y denseMatrix，存储T矩阵的特征向量（与特征值一一对应）
         * @param [in, out] mu 存储T矩阵的特征值（默认升序排列） */
        void computeLanczosEigenValuesAndEigenVectors(HostVector<HighPrecisionType>& D,
                                                      HostVector<HighPrecisionType>& E,
                                                      INT32 actualLanczosSteps,
                                                      HostVector<HighPrecisionType>& Y,
                                                      HostVector<HighPrecisionType>& mu);

        /** @brief 隐式重启Lanczos方法 */
        void implicitRestartKrylovSchurLanczosForGeneralizedEigenvaluesProblem(INT32 level);

        /** @brief 用来计算扩展的Lanczos过程，把m步Lanczos扩展到k步
         * @attention 实际会产生actualLanczosSteps+1个有效向量，因为存了初始随机向量
         * @param [in] level: 当前水平编号
         * @param [in, out] D:  Lanczos过程求出的alpha，包含actualLanczosSteps个有效值
         * @param [in, out] E:  Lanczos过程求出的beta，包含actualLanczosSteps个有效值
         * @param [in, out] V:  Lanczos正交向量（C-范数下），有actualLanczosSteps+1个有效向量
         * @param [in, out] Z:  Lanczos正交向量（非C-范数下），有actualLanczosSteps+1个有效向量
         * @param [in] originalLanczosSteps:    原先执行的Lanczos步数，即m
         * @param [in] expandSteps: 要扩展的Lanczos步数，即k
         * @param [in, out] actualLanczosSteps:  实际执行的Lanczos步数 */
        void expandReorthogonalLanczos(INT32 level, HostVector<HighPrecisionType>& D, HostVector<HighPrecisionType>& E,
                                       HostVector<HighPrecisionType>& V, HostVector<HighPrecisionType>& Z,
                                       INT32 originalLanczosSteps, INT32 expandSteps, INT32& actualLanczosSteps);

        /** @brief 将特征值及其对应特征向量反序 */
        void reverseEigenValuesAndEigenVectors(HostVector<HighPrecisionType>& mu, HostVector<HighPrecisionType>& Y,
                                               INT32 eigenvaluesCount);

        /** @brief 构造一个正交变换，使得QM = TQ^{T}
         * @note 将特征值构成的对角矩阵复原为三对角形状，并对V应用对应的变换以构建新的V（Lanczos向量组）
         * @param [in] mu: 原Lanczos过程求出的特征值向量（对应Tk矩阵谱分解后求得的对角矩阵M）
         * @param [in] lastRowY: Tk谱分解后求得的特征向量组Y的最后一行，长度为retainedLanczosSteps
         * @param [in, out] D: Lanczos过程求出的alpha，包含retainedLanczosSteps个有效值
         * @param [in, out] E: Lanczos过程求出的beta，包含retainedLanczosSteps个有效值
         * @param [in] V: Lanczos向量，包含retainedLanczosSteps个有效C-正交向量
         * @param [in, out] auxV: V对应的辅助向量组
         * @param [in] Z: Lanczos向量，包含retainedLanczosSteps个有效正交向量
         * @param [in, out] auxZ: Z对应的辅助向量组
         * @param [in] retainedLanczosSteps: 保留的有效Lanczos步数
         * @param [in] vectorLanczosLength */
        void convertDiagonal2TriDiagonal(HostVector<HighPrecisionType>& mu, HostVector<HighPrecisionType>& lastRowY,
                                         HostVector<HighPrecisionType>& D, HostVector<HighPrecisionType>& E,
                                         HostVector<HighPrecisionType>& V, HostVector<HighPrecisionType>& auxV,
                                         HostVector<HighPrecisionType>& Z, HostVector<HighPrecisionType>& auxZ,
                                         INT32 retainedLanczosSteps, INT32 vectorLanczosLength);

        /** @brief 用于求V_k * Y * Q^T */
        static void transposedGivensApplied2Columns(HighPrecisionType c, HighPrecisionType s, UINT32 colIdx1,
                                                    UINT32 colIdx2,
                                                    HostVector<HighPrecisionType>& V,
                                                    HostVector<HighPrecisionType>& auxV,
                                                    HostVector<HighPrecisionType>& Z,
                                                    HostVector<HighPrecisionType>& auxZ,
                                                    INT32 vectorLanczosLength);

        /* ======================================== 不同水平子模块 ======================================== */
        /** @brief 递归求y=C_{l}^(-1)* x，C_{l} = A_{l+1}，所以这个可以用来求指定level的y=A_{l+1}^(-1)* x，
         *        初始值可以设置level为-1，来进行A0^(-1)*x的计算
         * @param [in] level 当前水平，从-1开始对应A0（c_{-1}），0对应A1（c0），以此类推
         * @param [in, out] x 向量
         * @param [in] xLocalStartIdx C子块对应当前解向量的起始下标 */
        void levelCInvRecursiveSolve(INT32 level, HostVector<HighPrecisionType>& x, UINT32 xLocalStartIdx);

        /** @brief 递归求y=C_{l}^(-1)* x，C_{l} = A_{l+1}，所以这个可以用来求指定level的y=A_{l+1}^(-1)* x，
         *        初始值可以设置level为-1，来进行A0^(-1)*x的计算
         * @note 这个是带修正的，可以用来求精度要求更高的y=C_{l}^(-1) * x
         * @param [in] level 当前水平，从-1开始对应A0（c_{-1}），0对应A1（c0），以此类推
         * @param [in, out] x 向量
         * @param [in] xLocalStartIdx C子块对应当前解向量的起始下标*/
        void levelCInvRecursiveSolveWithCorrection(INT32 level, HostVector<HighPrecisionType>& x,
                                                   UINT32 xLocalStartIdx);

        /** @brief 用来求B块对应的y=B_{l}^(-1)* x
         * @param [in] level 当前水平，从0开始
         * @param [in, out] x 向量
         * @param [in] xLocalStartIdx C子块对应当前解向量的起始下标 */
        void levelBlockInvSolve(INT32 level, HostVector<HighPrecisionType>& x, UINT32 xLocalStartIdx);

        /** @brief 用来求B块对应的y=B_{l}^(-1)* x
         * @note 这个是带修正的，可以用来求精度要求更高的y=C_{l}^(-1) * x
         * @param [in] level 当前水平，从0开始
         * @param [in, out] x 向量
         * @param [in] xLocalStartIdx C子块对应当前解向量的起始下标 */
        void levelBlockInvSolveWithCorrection(INT32 level, HostVector<HighPrecisionType>& x, UINT32 xLocalStartIdx);

        /** @brief 用来求B块对应的y=B_{l} * x
         * @param [in] level 当前水平，从0开始
         * @param [in] vecIN 向量
         * @param [in] inVecStartIdx C子块对应当前解向量的起始下标
         * @param [out] vecOUT
         * @param [in] outVecStartIdx */
        void levelBlockMatPVec(INT32 level, HostVector<HighPrecisionType>& vecIN, UINT32 inVecStartIdx,
                               HostVector<HighPrecisionType>& vecOUT, UINT32 outVecStartIdx);

        /* ======================================== CUDA 辅助函数 ======================================== */

#ifdef CUDA_ENABLED
        void computeLanczosVectorsMultiplyEigenvectorsDEVICE(DeviceVector<HighPrecisionType>& devV,
                                                             DeviceVector<HighPrecisionType>& devAuxV,
                                                             DeviceVector<HighPrecisionType>& devY,
                                                             INT32 n_remain, INT32 stepsForEachRestart,
                                                             INT32 actualLanczosSteps);


        void convertDiagonal2TriDiagonalDEVICE(HostVector<HighPrecisionType>& mu,
                                               HostVector<HighPrecisionType>& lastRowY,
                                               HostVector<HighPrecisionType>& D, HostVector<HighPrecisionType>& E,
                                               DeviceVector<HighPrecisionType>& devV,
                                               DeviceVector<HighPrecisionType>& devAuxV,
                                               DeviceVector<HighPrecisionType>& devZ,
                                               DeviceVector<HighPrecisionType>& devAuxZ,
                                               INT32 retainedLanczosSteps, INT32 vectorLanczosLength);

        inline void warmUp(INT32 n_remain, INT32 stepsForEachRestart, INT32 actualLanczosSteps) {
            for (UINT32 i = 0; i < 5; ++i) {
                computeLanczosVectorsMultiplyEigenvectorsDEVICE(m_devLanczosVectorsV, m_devAuxLanczosVectorsV,
                                                                m_devEigenvectors, n_remain,
                                                                stepsForEachRestart, actualLanczosSteps);
                transposedGivensApplied2ColumnsDEVICE<HighPrecisionType>(0.0, 0.0, 0, 1, m_devAuxLanczosVectorsV,
                                                                         m_devLanczosVectorsV,
                                                                         m_devAuxLanczosVectorsZ, m_devLanczosVectorsZ,
                                                                         n_remain,
                                                                         *m_devComputeStream);
            }
            m_devComputeStream->synchronize();
        }


#endif

    public:
        AMSEDPrecondition() = default;

        /** @brief AMSED的构造函数
         * @param [in] matA: 原始矩阵A，传入共享指针
         * @param [in] levelNum: MSLR结构的水平数
         * @param [in] localReorderType: 选择HID排序过程中，每个水平的B子块的重排序方法，目前只有三种：1. 不排序内点；2. 使用AMD排序重排内点；3.使用RCM排序内点
         * @param [in] lowRankType: 低秩校正模式，目前有两种：1. AMSEDBasic，使用经典MSLR低秩校正方法；2. AMSEDEigenvalueDeflation，使用本论文新方法。*/
        AMSEDPrecondition(const std::shared_ptr<CSRMatrix<HighPrecisionType>>& matA, INT32 levelNum,
                          MatrixReorderOption_t localReorderType, LowRankCorrectType_t lowRankType);

        /** @brief AMSED的构造函数，部分预设参数无需手动输入，均使用默认值
         * @param [in] matA: 原始矩阵A，传入共享指针
         * @param [in] levelNum: MSLR结构的水平数
         * @param [in] lowRankSize: MSLR低秩修正大小，原则上不能超过预设Lanczos迭代步数，如果超过则自动调整为Lanczos步数的大小
         * @param [in] localReorderType: 选择HID排序过程中，每个水平的B子块的重排序方法，目前只有三种：1. 不排序内点；2. 使用AMD排序重排内点；3.使用RCM排序内点
         * @param [in] lowRankType: 低秩校正模式，目前有两种：1. AMSEDBasic，使用经典MSLR低秩校正方法；2. AMSEDEigenvalueDeflation，使用本论文新方法。*/
        AMSEDPrecondition(const std::shared_ptr<CSRMatrix<HighPrecisionType>>& matA, INT32 levelNum, INT32 lowRankSize,
                          MatrixReorderOption_t localReorderType, LowRankCorrectType_t lowRankType);

        /** @brief AMSED的构造函数，部分预设参数无需手动输入，均使用默认值
         * @note eigCorrectBound仅对新方法有用，使用MSLR老方法可随意填入该值。
         * @param [in] matA: 原始矩阵A，传入共享指针
         * @param [in] levelNum: MSLR结构的水平数
         * @param [in] lanczosSteps: 计算Lanczos算法的迭代步数
         * @param [in] lowRankSize: MSLR低秩修正大小(上界)
         * @param [in] localReorderType: 选择HID排序过程中，每个水平的B子块的重排序方法，目前只有三种：1. 不排序内点；2. 使用AMD排序重排内点；3.使用RCM排序内点
         * @param [in] lowRankType: 低秩校正模式，目前有两种：1. AMSEDBasic，使用经典MSLR低秩校正方法；2. AMSEDEigenvalueDeflation，使用本论文新方法。 */
        AMSEDPrecondition(const std::shared_ptr<CSRMatrix<HighPrecisionType>>& matA, INT32 levelNum,
                          INT32 lanczosSteps, INT32 lowRankSize, MatrixReorderOption_t localReorderType,
                          LowRankCorrectType_t lowRankType, LanczosType_t lanczosType);

        /** @brief AMSED的构造函数，可选择使用基本MSLR还是新方法，提供完整的参数列表设置，可以根据具体情况任意调整每个参数。
         * @note eigCorrectBound仅对新方法有用，使用MSLR老方法可随意填入该值。
         * @param [in] matA: 原始矩阵A，传入共享指针
         * @param [in] levelNum: MSLR结构的水平数
         * @param [in] lanczosSteps: 计算Lanczos算法的迭代步数
         * @param [in] lowRankSize: MSLR低秩修正大小(上界)
         * @param [in] eigCorrectBound: AMSED方法特征值校正下界
         * @param [in] localReorderType: 选择HID排序过程中，每个水平的B子块的重排序方法，目前只有三种：1. 不排序内点；2. 使用AMD排序重排内点；3.使用RCM排序内点
         * @param [in] lowRankType: 低秩校正模式，目前有两种：1. AMSEDBasic，使用经典MSLR低秩校正方法；2. AMSEDEigenvalueDeflation，使用本论文新方法。 */
        AMSEDPrecondition(const std::shared_ptr<CSRMatrix<HighPrecisionType>>& matA, INT32 levelNum,
                          INT32 lanczosSteps, INT32 lowRankSize, HighPrecisionType eigCorrectBound,
                          MatrixReorderOption_t localReorderType, LowRankCorrectType_t lowRankType,
                          LanczosType_t lanczosType);

        ~AMSEDPrecondition() override = default;

        /** @brief 获取LRC的fill */
        inline FLOAT64 getLowRankCorrectionRatio() {
            return 1.0 * m_lowRankNNZ / this->m_Annz;
        }

        /** @brief 获取不完全分解的fill*/
        inline FLOAT64 getIncompleteFactorRatio() {
            return 1.0 * m_incompFactNNZ / this->m_Annz;
        }

        /** @brief 预条件的创建，核心接口 */
        void setup() override;

        /** @brief 让右端项的排序过程在MInvSolve中执行，否则在外部自己处理*/
        void setRightHandPermutationType(const AMSEDPermType_t& setType) {
            m_rhsPermType = setType;
        }

        /** @brief 将原始向量根据系数矩阵的重排规则排列成新的向量。
         * @param [in] originRhs: 原始向量，一般为原始右端项。
         * @param [in, out] newRhs: 重排后的新向量 */
        void prepareRightHand(BaseVector<HighPrecisionType>& originRhs, BaseVector<HighPrecisionType>& newRhs) {
#ifndef NDEBUG
            THROW_EXCEPTION(newRhs.getLength() != this->m_ArowNum || originRhs.getLength() != this->m_ArowNum,
                            THROW_LOGIC_ERROR("The dim of new right-hand or original right-hand is incorrect!"));
#endif
            if (m_rhsReady == 1) return;
            // 将当前vec映射到内部辅助数组中
            for (UINT32 i = 0; i < this->m_ArowNum; ++i) {
                newRhs[i] = originRhs[m_pperm[i]];
            }
            m_rhsReady = 1;
        }

        /** @brief 将原始向量根据系数矩阵的重排规则排列成新的向量。
         * @param [in] originRhs: 原始向量，一般为原始右端项。
         * @param [in] originStartIdx: 待排序区间在原始向量中的起始位置。
         * @param [in] newRhs: 重排后的新向量。
         * @param [in] newStartIdx: 重排后的区间在新向量中的起始位置。*/
        void prepareRightHand(BaseVector<HighPrecisionType>& originRhs, UINT32 originStartIdx,
                              BaseVector<HighPrecisionType>& newRhs, UINT32 newStartIdx) {
#ifndef NDEBUG
            THROW_EXCEPTION(
                newRhs.getLength() < this->m_ArowNum + newStartIdx || originRhs.getLength() < this->m_ArowNum +
                originStartIdx,
                THROW_LOGIC_ERROR("The dim of new right-hand or original right-hand is incorrect!"));
#endif
            if (m_rhsReady == 1) return;
            // 将当前vec映射到内部辅助数组中
            for (UINT32 i = 0; i < this->m_ArowNum; ++i) {
                newRhs[i + newStartIdx] = originRhs[m_pperm[i] + originStartIdx];
            }
            m_rhsReady = 1;
        }

        /** @brief 将重排后的向量还原为原始向量。
         * @param [in] newRhs: 重排后的新向量
         * @param [in, out] originRhs: 还原后的向量，一般对应原始右端项。*/
        void recoverRightHand(BaseVector<HighPrecisionType>& newRhs, BaseVector<HighPrecisionType>& originRhs) {
#ifndef NDEBUG
            THROW_EXCEPTION(newRhs.getLength() != this->m_ArowNum || originRhs.getLength() != this->m_ArowNum,
                            THROW_LOGIC_ERROR("The dim of new right-hand or original right-hand is incorrect!"));
#endif
            if (m_rhsReady == 0) return;
            // 将辅助数组中的结果还原回原来的向量中
            for (UINT32 i = 0; i < this->m_ArowNum; ++i) {
                originRhs[m_pperm[i]] = newRhs[i];
            }
            m_rhsReady = 0;
        }

        /** @brief 将重排后的矩阵还原为原始向量。
         * @param [in] newRhs: 重排后的新向量。
         * @param [in] newStartIdx: 重排后的区间在新向量中的起始位置。
         * @param [in] originRhs: 原始向量，一般为原始右端项。
         * @param [in] originStartIdx: 已排序区间在原始向量中的起始位置。*/
        void recoverRightHand(BaseVector<HighPrecisionType>& newRhs, UINT32 newStartIdx,
                              BaseVector<HighPrecisionType>& originRhs, UINT32 originStartIdx) {
#ifndef NDEBUG
            THROW_EXCEPTION(
                newRhs.getLength() < this->m_ArowNum + newStartIdx || originRhs.getLength() < this->m_ArowNum +
                originStartIdx,
                THROW_LOGIC_ERROR("The dim of new right-hand or original right-hand is incorrect!"));
#endif
            if (m_rhsReady == 0) return;
            // 将辅助数组中的结果还原回原来的向量中
            for (UINT32 i = 0; i < this->m_ArowNum; ++i) {
                originRhs[m_pperm[i] + originStartIdx] = newRhs[i + newStartIdx];
            }
            m_rhsReady = 0;
        }

        inline FLOAT64 getParallelBInvSolveTime() const {
            return m_blockBParallelTime;
        }

        inline FLOAT64 getCompLowRankCorrectTime() const {
            return m_compLowRankCorrectTime;
        }

        inline FLOAT64 getCompLanczosTime() const {
            return m_LanczosTime;
        }

        inline FLOAT64 getBlockDecompositionTime() const {
            return m_blockDecompositionTime;
        }

        /* ================================= 求解器接口，具体说明见父类接口注释 ================================= */
        void MInvSolve(BaseVector<HighPrecisionType>& vec, UINT32 resStartIdx) override;

        void MInvSolve(BaseVector<HighPrecisionType>& vecIN, BaseVector<HighPrecisionType>& vecOUT) override;

        void MInvSolve(BaseVector<HighPrecisionType>& vecIN, UINT32 inStartIdx, BaseVector<HighPrecisionType>& vecOUT,
                       UINT32 outStartIdx) override;

        void MInvSolve(BaseVector<HighPrecisionType>& vec) override;
    };

    template class AMSEDPrecondition<FLOAT32, FLOAT32>;
    template class AMSEDPrecondition<FLOAT64, FLOAT64>;
    template class AMSEDPrecondition<FLOAT32, FLOAT64>;
} // HOST

#endif //AMSED_PRECONDITION_H
