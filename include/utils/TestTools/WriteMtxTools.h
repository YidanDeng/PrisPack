/*
 * @author  邓轶丹
 * @date    2024/11/18
 * @brief   参考了mmio的写法，实现读入c++风格的mtx文件写入工具，暂时未加入复数数据或特殊结构矩阵的处理
 * @version 0.1
 * @details
 *      标准格式mtx文件头带“%%MatrixMarket”的那一行字符串记录了该文件中矩阵的属性（一般是文件首行），主要涉及四个参数：
 *      object      sparse/dense        data-type       storage-scheme
 *      所以有专门处理文件头的函数来写入这四个部分，根据当前数据参数写入对应的类型。
 *      当前版本仅支持实数域上的矩阵和右端项，待写入矩阵的存储格式必须是csr格式，右端项必须是向量或稠密矩阵。
 *
 *      例如，对于矩阵的四个属性：
 *      matrix      coordinate          real            general/symmetric
 *      对于右端项的四个属性：
 *      matrix      array               real            general
 *      只有符合以上属性的系数矩阵A和右端项b文件才能使用本写入工具。
 *
 *      补充特殊情况：
 *          当矩阵为对称阵时，一般情况下MatrixMarket文件只存一半（上三角区或下三角区），如果将实际上的对称矩阵
 *          已经扩展为完整的矩阵，那么写入矩阵类型应该为general而非symmetric
 */

#ifndef WRITEMTXTOOLS_H
#define WRITEMTXTOOLS_H

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

namespace HOST {
    template<typename ValType>
    class WriteMtxTools {
    private:
        std::ofstream m_AStream; ///< 矩阵A的写入流
        std::ofstream m_BStream; ///< 右端项b的写入流
        mtx_data_type_t m_matDataType{MTX_DATA_REAL};
        mtx_storage_scheme_t m_matStoreScheme{MTX_STORAGE_GENERAL};
        UINT8 m_base{USE_ONE_BASED_INDEX}; ///< 下标默认从1开始
        UINT8 m_isSymmFullSize{1}; ///< 如果遇到对称阵，是否要将其展开为完整矩阵
        std::unique_ptr<char[]> m_bufferPtr; ///< 写入缓冲区
        size_t m_bufferSize{10 * 1024 * 1024}; ///< 文件预取缓冲区，对于超大文件比较有用，默认10MB，不建议太小（必须>=1024），否则会导致数据读取异常
        size_t m_currentBufferSize{0}; ///< 当前缓冲区已使用大小

        /** @brief 刷新缓冲区，将缓冲区中的内容写入文件 */
        void flushBuffer(std::ofstream &outputStream) {
            if (m_currentBufferSize > 0) {
                outputStream.write(m_bufferPtr.get(), m_currentBufferSize);
                m_currentBufferSize = 0;
            }
        }

        /** @brief 向缓冲区添加数据
         * @param data: 要写入的数据 */
        void addToBuffer(const std::string &data, std::ofstream &outputStream) {
            char *currBufferPtr = m_bufferPtr.get() + m_currentBufferSize;
            const char *currDataPtr = data.c_str();
            size_t currDataSize = data.size(), copySize;
            while (currDataSize > 0) {
                // 如果缓冲区目前还未填满，先尝试填满缓冲区
                if (m_currentBufferSize < m_bufferSize) {
                    copySize = m_bufferSize - m_currentBufferSize;
                    copySize = copySize < currDataSize ? copySize : currDataSize;
                    std::memcpy(currBufferPtr, currDataPtr, copySize);
                    m_currentBufferSize += copySize;
                    currBufferPtr += copySize;
                    currDataPtr += copySize;
                    currDataSize -= copySize;
                }
                // 如果缓冲区已装满，直接写入文件
                if (m_currentBufferSize == m_bufferSize) {
                    flushBuffer(outputStream);
                    currBufferPtr = m_bufferPtr.get();
                }
            }
        }

        /** @brief 写入文件头 */
        void writeHeader(UINT32 rows, UINT32 cols, UINT32 nnz, std::ofstream &outputStream) {
            std::ostringstream header;
            header << "%%MatrixMarket matrix coordinate real "
                    << (m_matStoreScheme != MTX_STORAGE_GENERAL ? "symmetric" : "general") << "\n";
            header << rows << " " << cols << " " << nnz << "\n";
            addToBuffer(header.str(), outputStream);
        }

    public:
        /** @brief 写入mtx文件（不带右端项）
         * @attention 这里的相对路径都是相对于最终编译好的二进制执行文件
         * @param [in] binaryPath: 编译好的可执行文件所在路径（直接传argv[0]）
         * @param [in] Apath: 系数矩阵数据文件的相对路径（必须带上后缀mtx）
         * @param [in] base: 每个元素编号下标从几开始（MatrixMarket默认为1，C++标准为0）
         * @param [in] matStoreScheme: 将要写入的矩阵存储的类型（这里表示的意思和实际这个矩阵是不是对称的无关，
         *             如果一个对称矩阵在写入时就被展开为一般矩阵，那么这里还是传入MTX_STORAGE_GENERAL这个参数）
         */
        WriteMtxTools(char *binaryPath, const char *Apath, UINT8 base, mtx_storage_scheme_t matStoreScheme);

        ~WriteMtxTools() {
            if (m_currentBufferSize > 0)
                SHOW_WARN("Some data was forgot to write into file, please check it again!")
            m_AStream.close();
            m_BStream.close();
        }

        void writeMatrix(const CSRMatrix<ValType> &csrMat);
    };

    template class WriteMtxTools<FLOAT32>;
    template class WriteMtxTools<FLOAT64>;
}


#endif //WRITEMTXTOOLS_H
