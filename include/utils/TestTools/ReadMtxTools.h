/*
 * @author  邓轶丹
 * @date    2024/8/9
 * @brief   参考了mmio的写法，实现读入c++风格的mtx文件读入工具，暂时未加入复数数据或特殊结构矩阵的处理
 * @version 0.1
 * @details
 *      标准格式mtx文件头带“%%MatrixMarket”的那一行字符串记录了该文件中矩阵的属性（一般是文件首行），主要涉及四个参数：
 *      object      sparse/dense        data-type       storage-scheme
 *      所以有专门处理文件头的函数来解析这四个数据，并判断当前数据文件是否符合要求。
 *      当前版本仅支持实数域上的矩阵和右端项，矩阵的存储格式必须是coo格式，右端项必须是有标准文件头的mtx文件且以array形式存储。
 *      例如，对于矩阵的四个属性：
 *      matrix      coordinate          real/complex            general/symmetric
 *      对于右端项的四个属性：
 *      matrix      array               real/complex            general
 *      只有符合以上属性的系数矩阵A和右端项b文件才能使用本读取工具。
 *
 *      特别说明：
 *          读入复数矩阵时只能把实部和虚部分离存储，计算过程中也会通过扩展矩阵的方式将复数问题转换为实数问题。
 *
 *      补充特殊情况：
 *          当矩阵为对称阵时，一般情况下MatrixMarket文件只存一半的矩阵（上三角区或下三角区），本工具默认将这种只存一半的矩阵
 *          扩展为完整矩阵（可以通过相应接口取消这种设定）
 */

#ifndef PMSLS_NEW_READMTXTOOLS_H
#define PMSLS_NEW_READMTXTOOLS_H

#include <libgen.h>
#include <unistd.h>
#include "../../../include/MatrixClass/CSRMatrix.h"
#include "../../../include/MatrixClass/COOMatrix.h"
#include "../../../include/MatrixClass/DenseMatrix.h"
#include "../../../include/MatrixClass/MatrixTools.h"
#include "../../../include/utils/MemoryTools/SharedPtrTools.h"
#include "../../../include/utils/ErrorHandler.h"
#include "../BaseUtils.h"
#include "generateTools.hpp"
#include "mmio.h"

namespace HOST {
    template<typename ValType>
    class ReadMtxTools {
    private:
        std::ifstream m_AStream;
        std::ifstream m_BStream;
        UINT32 m_rdim{0}; ///< 系数矩阵行数
        UINT32 m_cdim{0}; ///< 系数矩阵列数
        UINT32 m_nnzNum{0}; ///< 系数矩阵非0元个数（文件中给出的数值）
        UINT32 m_bCols{1}; ///< 右端项个数
        UINT32 m_actRealNNZidx{0}; ///< 过滤掉文件中可能误入的0元后的实际非零元个数
        UINT32 m_actImageNNZidx{0}; ///< 过滤掉文件中0元后的实际非零元个数
        mtx_structure_type_t m_matStructure{MTX_STRUCTURE_COO};
        mtx_data_type_t m_matDataType{MTX_DATA_REAL};
        mtx_storage_scheme_t m_matStoreScheme{MTX_STORAGE_GENERAL};
        UINT8 m_base{USE_ONE_BASED_INDEX}; ///< 下标默认从1开始
        UINT8 m_isSymmFullSize{1};          ///< 如果遇到对称阵，是否要将其展开为完整矩阵
        size_t m_bufferSize{10 * 1024 * 1024}; ///< 文件预取缓冲区，对于超大文件比较有用，默认10MB，不建议太小（必须>=1024），否则会导致数据读取异常

        COOMatrix<ValType> m_auxRealCOO;        ///< 用来存储实部对应的CSR矩阵
        COOMatrix<ValType> m_auxImageCOO;       ///< 用来存储虚部对应的CSR矩阵
        UINT32 m_auxRealZeroElementCount{0};
        UINT32 m_auxImageZeroElementCount{0};

        void processMatHeader();

        void processRhsHeader();

        void processMatLine(const std::string_view &line);

    public:
        /** @brief 读取mtx文件（不带右端项）
         * @attention 这里的相对路径都是相对于最终编译好的二进制执行文件
         * @param [in] binaryPath: 编译好的可执行文件所在路径（直接传argv[0]）
         * @param [in] Apath: 系数矩阵数据文件的相对路径
         * @param [in] base: 下标从几开始（MatrixMarket默认为1，C++标准为0）*/
        ReadMtxTools(char *binaryPath, const char *Apath, UINT8 base);


        /** @brief 读取mtx文件（矩阵+右端项）
         * @attention 这里的相对路径都是相对于最终编译好的二进制执行文件
         * @param [in] binaryPath: 编译好的可执行文件所在路径（直接传argv[0]）
         * @param [in] Apath: 系数矩阵数据文件的相对路径
         * @param [in] Bpath: 右端项文件（mtx或rhs文件）
         * @param [in] base: 下标从几开始（MatrixMarket默认为1，C++标准为0）*/
        ReadMtxTools(char *binaryPath, const char *Apath, const char *Bpath, UINT8 base);

        void setFullStoredSymmertricMat() {
            this->m_isSymmFullSize = 1;
        }

        void setHalfStoredSymmertricMat() {
            this->m_isSymmFullSize = 0;
        }

        /** @brief 加载实数域上的系数矩阵
         * @attention 原始文件必须以coo格式存储 */
        void loadMatrix(CSRMatrix<ValType> &outMat);

        /** @brief 加载复数域上的系数矩阵，最后返回实部和虚部分开存储的CSR矩阵
         * @attention 原始文件必须以coo格式存储 */
        void loadMatrix(CSRMatrix<ValType> &rMat, CSRMatrix<ValType> &iMat);

        /** @brief 加载实数域上的右端项，并存储为稠密矩阵（列主序）
         * @attention 不管是单右端项还是多右端项，都必须以标准mtx的array格式存储 */
        void loadRightHand(DenseMatrix<ValType> &rhs);

        /** @brief 加载实数域上的右端项，并存储为向量（以列主序的形式）
         * @attention 不管是单右端项还是多右端项，都必须以标准mtx的array格式存储 */
        void loadRightHand(HostVector<ValType> &rhs);

        /** @brief 加载复数域上的右端项，并存储为向量（以列主序的形式）
         * @attention 不管是单右端项还是多右端项，都必须以标准mtx的array格式存储 */
        void loadRightHand(HostVector<ValType> &rhsR, HostVector<ValType> &rhsI);

        /** @brief （只有实数矩阵能用）针对没有右端项的问题，生成随机标准答案x和对应的右端项Ax = B*/
        void loadRightHand(HostVector<ValType> &res, HostVector<ValType> &rhs, UINT32 rhsNum);


        ~ReadMtxTools() {
            m_AStream.close();
            m_BStream.close();
        }
    };

    template
    class ReadMtxTools<FLOAT32>;

    template
    class ReadMtxTools<FLOAT64>;
}


#endif //PMSLS_NEW_READMTXTOOLS_H
