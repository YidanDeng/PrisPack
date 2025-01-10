/*
 * @author  邓轶丹
 * @date    2024/8/9
 * @details 仿照mmio中的函数，编写C++版本的MTX写入工具
 */

#include "../../../include/utils/TestTools/WriteMtxTools.h"

namespace HOST {
    template<typename ValType>
    WriteMtxTools<ValType>::WriteMtxTools(char *binaryPath, const char *Apath, UINT8 base,
                                          mtx_storage_scheme_t matStoreScheme) {
        // 创建路径的副本，否则会导致后续文件打开异常
        char *pathCopy = strdup(binaryPath);
#ifndef NDEBUG
        THROW_EXCEPTION(pathCopy == nullptr, THROW_INVALID_ARGUMENT("Failed to duplicate path"))
#endif
        INT32 status = chdir(dirname(pathCopy));
        THROW_EXCEPTION(status == -1, THROW_INVALID_ARGUMENT("Changing work dir failed!"))
        m_base = base;
        m_matStoreScheme = matStoreScheme;
        m_AStream.open(Apath, std::ios::out | std::ios::binary);
        m_bufferPtr = std::make_unique<char[]>(m_bufferSize);
#ifndef NDEBUG
        SHOW_INFO("Prepare Matrix writing file: " << Apath)
        THROW_EXCEPTION(!m_AStream.is_open(), THROW_INVALID_ARGUMENT("Create Matrix file failed!"))
        SHOW_INFO("There is no right-hand need to be created.")
#endif
    }

    template<typename ValType>
    void WriteMtxTools<ValType>::writeMatrix(const CSRMatrix<ValType> &csrMat) {
        const UINT32 *rowPtr = csrMat.getRowOffsetPtr(0);
        if (rowPtr == nullptr || rowPtr[csrMat.getRowNum()] - rowPtr[0] == 0) return;
        const UINT32 *colIdx = csrMat.getColIndicesPtr(0);
        const ValType *values = csrMat.getCSRValuesPtr(0);
        UINT32 rows = csrMat.getRowNum();
        UINT32 cols = csrMat.getColNum();
        UINT32 nnz = csrMat.getNNZnum(0, rows - 1);

        writeHeader(rows, cols, nnz, m_AStream);
        // 使用 std::numeric_limits 动态获取精度
        // std::numeric_limits<float>::digits10 通常是 6。
        // std::numeric_limits<double>::digits10 通常是 15。
        // 再加 1 是为了保留一点额外的精度。
        INT32 precision = std::numeric_limits<ValType>::digits10 + 1;
        for (UINT32 i = 0; i < rows; ++i) {
            std::ostringstream line;
            for (UINT32 j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
                line << i + m_base << " " // 根据当前base类型选择是1-base还是0-base
                        << colIdx[j] + m_base << " "
                        << std::setprecision(precision) << values[j] << "\n";
            }
            addToBuffer(line.str(), m_AStream);
        }

        // 把剩余的数据刷新进缓冲区
        flushBuffer(m_AStream);
        m_AStream.close();
    }
}
