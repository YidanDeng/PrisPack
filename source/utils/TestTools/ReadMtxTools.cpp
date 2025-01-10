/*
 * @author  邓轶丹
 * @date    2024/8/9
 * @details 仿照mmio中的函数，编写C++版本的MTX读取工具
 */

#include "../../../include/utils/TestTools/ReadMtxTools.h"

namespace HOST {
    template<typename ValType>
    ReadMtxTools<ValType>::ReadMtxTools(char *binaryPath, const char *Apath, const char *Bpath, UINT8 base) {
        // 创建路径的副本，否则会导致后续文件打开异常
        char *pathCopy = strdup(binaryPath);
#ifndef NDEBUG
        THROW_EXCEPTION(pathCopy == nullptr, THROW_INVALID_ARGUMENT("Failed to duplicate path"))
#endif
        INT32 status = chdir(dirname(pathCopy));
        THROW_EXCEPTION(status == -1, THROW_INVALID_ARGUMENT("Changing work dir failed!"))
        m_base = base;
        // 使用二进制流的形式读入文件
        m_AStream = std::ifstream(Apath, std::ios::binary);
        m_BStream = std::ifstream(Bpath, std::ios::binary);
#ifndef NDEBUG
        SHOW_INFO("Open Matrix file: " << Apath)
        THROW_EXCEPTION(!m_AStream.is_open(), THROW_INVALID_ARGUMENT("Open Matrix file failed!"))
        SHOW_INFO("Open Right Hand file: " << Bpath)
        THROW_EXCEPTION(!m_BStream.is_open(), THROW_INVALID_ARGUMENT("Open Right Hand file failed!"))
#endif
    }

    template<typename ValType>
    ReadMtxTools<ValType>::ReadMtxTools(char *binaryPath, const char *Apath, UINT8 base) {
        // 创建路径的副本，否则会导致后续文件打开异常
        char *pathCopy = strdup(binaryPath);
#ifndef NDEBUG
        THROW_EXCEPTION(pathCopy == nullptr, THROW_INVALID_ARGUMENT("Failed to duplicate path"))
#endif
        INT32 status = chdir(dirname(pathCopy));
        THROW_EXCEPTION(status == -1, THROW_INVALID_ARGUMENT("Changing work dir failed!"))
        m_base = base;
        // 使用二进制流的形式读入文件
        m_AStream = std::ifstream(Apath, std::ios::binary);
#ifndef NDEBUG
        SHOW_INFO("Open Matrix file: " << Apath)
        THROW_EXCEPTION(!m_AStream.is_open(), THROW_INVALID_ARGUMENT("Open Matrix file failed!"))
        SHOW_INFO("There is no right-hand file, please generate your test right-hand and result.")
#endif
    }


    template<typename ValType>
    void ReadMtxTools<ValType>::processMatHeader() {
        THROW_EXCEPTION(!m_AStream.is_open(), THROW_INVALID_ARGUMENT("Open Matrix file failed!"))
        std::string matFileLine;
        std::string temp;
        bool headerExist = false;
        std::string mtx, crd, data_type, storage_scheme;
        while (std::getline(m_AStream, matFileLine)) {
            if (matFileLine[0] != '%') break;
            std::istringstream iss(matFileLine);
            iss >> temp;
            if (temp == MatrixMarketBanner) {
                headerExist = true;
                // 处理第一个参数
                iss >> mtx;
                transStrToLower(mtx);
                THROW_EXCEPTION(mtx != MM_MTX_STR, THROW_INVALID_ARGUMENT("Unsupportable mtx file!"))
                // 处理第二个参数
                iss >> crd;
                transStrToLower(crd);
                if (crd == MM_SPARSE_STR) {
                    m_matStructure = MTX_STRUCTURE_COO;
                } else if (crd == MM_DENSE_STR) {
                    m_matStructure = MTX_STRUCTURE_ARRAY;
                } else {
                    THROW_INVALID_ARGUMENT("Unsupportable mtx file!");
                }
                // 处理第三个参数
                iss >> data_type;
                transStrToLower(data_type);
                if (data_type == MM_REAL_STR) {
                    m_matDataType = MTX_DATA_REAL;
                } else if (data_type == MM_COMPLEX_STR) {
                    m_matDataType = MTX_DATA_COMPLEX;
                } else if (data_type == MM_PATTERN_STR) {
                    m_matDataType = MTX_DATA_PATTERN;
                } else if (data_type == MM_INT_STR) {
                    m_matDataType = MTX_DATA_INTEGER;
                } else {
                    THROW_INVALID_ARGUMENT("Unsupportable mtx file!");
                }
                // 处理第四个参数
                iss >> storage_scheme;
                transStrToLower(storage_scheme);
                if (storage_scheme == MM_GENERAL_STR) {
                    m_matStoreScheme = MTX_STORAGE_GENERAL;
                } else if (storage_scheme == MM_SYMM_STR) {
                    m_matStoreScheme = MTX_STORAGE_SYMMETRIC;
                } else if (storage_scheme == MM_HERM_STR) {
                    m_matStoreScheme = MTX_STORAGE_HERMITIAN;
                } else if (storage_scheme == MM_SKEW_STR) {
                    m_matStoreScheme = MTX_STORAGE_SKEW;
                } else {
                    THROW_INVALID_ARGUMENT("Unsupportable mtx file!");
                }
            }
        }
        THROW_EXCEPTION(!headerExist, THROW_INVALID_ARGUMENT("No header in mtx file!"))
        std::istringstream iss(matFileLine);
        iss >> m_rdim >> m_cdim;
        if (m_matStructure == MTX_STRUCTURE_COO) {
            iss >> m_nnzNum;
            SHOW_INFO("COO mat row: " << m_rdim << ", mat col: " << m_cdim << ", mat nnz: " << m_nnzNum)
        } else {
            SHOW_INFO("Dense mat row: " << m_rdim << ", mat col: " << m_cdim)
        }
        if ((m_matStoreScheme == MTX_STORAGE_SYMMETRIC || m_matStoreScheme == MTX_STORAGE_SKEW) && m_isSymmFullSize) {
            m_nnzNum *= 2;
            m_nnzNum -= m_rdim; // 对角线上元素算了两遍
            SHOW_INFO("Restore symmetric matrix to full size: " << m_nnzNum)
        }
    }

    template<typename ValType>
    void ReadMtxTools<ValType>::processRhsHeader() {
        THROW_EXCEPTION(!m_BStream.is_open(), THROW_INVALID_ARGUMENT("Open Matrix file failed!"))
        std::string rhsFileLine;
        std::string temp;
        bool headerExist = false;
        std::string mtx, crd, data_type, storage_scheme;
        while (std::getline(m_BStream, rhsFileLine)) {
            if (rhsFileLine[0] != '%') break;
            std::istringstream iss(rhsFileLine);
            iss >> temp;
            if (temp == MatrixMarketBanner) {
                headerExist = true;
                // 处理第一个参数
                iss >> mtx;
                transStrToLower(mtx);
                THROW_EXCEPTION(mtx != MM_MTX_STR, THROW_INVALID_ARGUMENT("Unsupportable mtx file!"))
                // 处理第二个参数
                iss >> crd;
                transStrToLower(crd);
                if (crd != MM_DENSE_STR) {
                    THROW_INVALID_ARGUMENT("Unsupportable mtx file!");
                }
                // 处理第三个参数
                iss >> data_type;
                transStrToLower(data_type);
                mtx_data_type_t bType;
                if (data_type == MM_REAL_STR) {
                    bType = MTX_DATA_REAL;
                } else if (data_type == MM_COMPLEX_STR) {
                    bType = MTX_DATA_COMPLEX;
                } else if (data_type == MM_PATTERN_STR) {
                    bType = MTX_DATA_PATTERN;
                } else if (data_type == MM_INT_STR) {
                    bType = MTX_DATA_INTEGER;
                } else {
                    THROW_INVALID_ARGUMENT("Unsupportable mtx file!");
                }
                THROW_EXCEPTION(bType != m_matDataType,
                                THROW_INVALID_ARGUMENT("The data-types of rhs and mat are not matched!"))
                // 处理第四个参数
                iss >> storage_scheme;
                transStrToLower(storage_scheme);
                if (storage_scheme != MM_GENERAL_STR && storage_scheme != MM_SYMM_STR &&
                    storage_scheme != MM_HERM_STR && storage_scheme != MM_SKEW_STR) {
                    THROW_INVALID_ARGUMENT("Unsupportable mtx file!");
                }
            }
        }
        THROW_EXCEPTION(!headerExist, THROW_INVALID_ARGUMENT("No header in mtx file!"))
        std::istringstream iss(rhsFileLine);
        UINT32 bRows, bCols;
        iss >> bRows >> bCols;
        THROW_EXCEPTION(bRows != m_cdim, THROW_INVALID_ARGUMENT("The dims of rhs and mat are not matched!"))
        m_bCols = bCols;
        SHOW_INFO("Right hand row: " << bRows << ", mat col: " << bCols)
    }


    template<typename ValType>
    void ReadMtxTools<ValType>::processMatLine(const std::string_view &line) {
        if (line.empty() || line[0] == '%') return; // 跳过空行和注释行
#ifndef NDEBUG
        THROW_EXCEPTION(m_actRealNNZidx >= m_nnzNum, THROW_OUT_OF_RANGE("The nnz idx is out-of-rang!"))
#endif
        std::istringstream lineStream((std::string(line)));
        UINT32 rowIdx, colIdx;
        ValType rVal, iVal;
        lineStream >> rowIdx >> colIdx >> rVal;
        if (m_matDataType == MTX_DATA_COMPLEX) lineStream >> iVal; // 若存在虚部，则读入
        if (m_matDataType == MTX_DATA_REAL) {
            if ((std::is_same<ValType, FLOAT32>::value && IS_FLOAT_NEAR_ZERO(rVal)) ||
                (std::is_same<ValType, FLOAT64>::value && IS_DOUBLE_NEAR_ZERO(rVal)) ||
                (std::is_same<ValType, long double>::value && IS_LONG_DOUBLE_NEAR_ZERO(rVal))) {
                // 如果原数据文件中存储了接近于0的元素，则略过不处理
#ifndef NDEBUG
                m_auxRealZeroElementCount++;
                if (m_auxRealZeroElementCount < 10)
                    SHOW_WARN("The zero element should not be stored in sparse Mat, row: " << rowIdx << ", col: " <<
                    colIdx << ", val: " << rVal)
#endif
                return;
            }
            if (m_base != 0) {
                rowIdx -= m_base;
                colIdx -= m_base;
            }
            // mtx文件数组索引从1开始，要转换为0开始
            m_auxRealCOO.pushBack(rowIdx, colIdx, rVal);
            if (m_matStoreScheme == MTX_STORAGE_SYMMETRIC && rowIdx != colIdx && m_isSymmFullSize) {
                // 如果是对称矩阵，则必须把对应的另一个元素也加入最终的矩阵中
                m_auxRealCOO.pushBack(colIdx, rowIdx, rVal);
                m_actRealNNZidx += 1;
            } else if (m_matStoreScheme == MTX_STORAGE_SKEW && rowIdx != colIdx && m_isSymmFullSize) {
                // 反对称矩阵同理
                m_auxRealCOO.pushBack(colIdx, rowIdx, -1.0 * rVal);
                m_actRealNNZidx += 1;
            }
            m_actRealNNZidx += 1;
        } else if (m_matDataType == MTX_DATA_COMPLEX) {
            UINT8 isRealPartZero{0}, isImagPartZero{0};
            isRealPartZero = (std::is_same<ValType, FLOAT32>::value && IS_FLOAT_NEAR_ZERO(rVal)) ||
                             (std::is_same<ValType, FLOAT64>::value && IS_DOUBLE_NEAR_ZERO(rVal)) ||
                             (std::is_same<ValType, long double>::value && IS_LONG_DOUBLE_NEAR_ZERO(rVal));
            isImagPartZero = (std::is_same<ValType, FLOAT32>::value && IS_FLOAT_NEAR_ZERO(iVal)) ||
                             (std::is_same<ValType, FLOAT64>::value && IS_DOUBLE_NEAR_ZERO(iVal)) ||
                             (std::is_same<ValType, long double>::value && IS_LONG_DOUBLE_NEAR_ZERO(iVal));
            if (m_base != 0) {
                rowIdx -= m_base;
                colIdx -= m_base;
            }
            if (isRealPartZero == 0) {
                // mtx文件数组索引从1开始，要转换为0开始
                m_auxRealCOO.pushBack(rowIdx, colIdx, rVal);
                if (m_matStoreScheme == MTX_STORAGE_SYMMETRIC && rowIdx != colIdx && m_isSymmFullSize) {
                    // 如果是对称矩阵，则必须把对应的另一个元素也加入最终的矩阵中
                    m_auxRealCOO.pushBack(colIdx, rowIdx, rVal);
                    m_actRealNNZidx += 1;
                } else if (m_matStoreScheme == MTX_STORAGE_SKEW && rowIdx != colIdx && m_isSymmFullSize) {
                    // 反对称矩阵同理
                    m_auxRealCOO.pushBack(colIdx, rowIdx, -1.0 * rVal);
                    m_actRealNNZidx += 1;
                }
                m_actRealNNZidx += 1;
            }
            if (isImagPartZero == 0) {
                // mtx文件数组索引从1开始，要转换为0开始
                m_auxImageCOO.pushBack(rowIdx, colIdx, iVal);
                if (m_matStoreScheme == MTX_STORAGE_SYMMETRIC && rowIdx != colIdx && m_isSymmFullSize) {
                    // 如果是对称矩阵，则必须把对应的另一个元素也加入最终的矩阵中
                    m_auxImageCOO.pushBack(colIdx, rowIdx, iVal);
                    m_actImageNNZidx += 1;
                } else if (m_matStoreScheme == MTX_STORAGE_SKEW && rowIdx != colIdx && m_isSymmFullSize) {
                    // 反对称矩阵同理
                    m_auxImageCOO.pushBack(colIdx, rowIdx, -1.0 * iVal);
                    m_actImageNNZidx += 1;
                }
                m_actImageNNZidx += 1;
            }
        } else {
            THROW_LOGIC_ERROR("Unsupportable data version");
        }
    }


    template<typename ValType>
    void ReadMtxTools<ValType>::loadRightHand(DenseMatrix<ValType> &rhs) {
        processRhsHeader();
        if (rhs.getStorageType() != DenseMatColumnFirst) rhs.resetStorageType(DenseMatColumnFirst, RESERVE_NO_DATA);
        rhs.resize(m_cdim, m_bCols, RESERVE_NO_DATA);
        if (m_matDataType == MTX_DATA_REAL) {
            ValType *valsPtr = rhs.getMatValPtr();
            UINT32 actual_length = m_cdim * m_bCols;
            for (UINT32 i = 0; i < actual_length; ++i) {
                m_BStream >> valsPtr[i];
            }
        } else {
            THROW_LOGIC_ERROR("Unsupportable data version");
        }
    }

    template<typename ValType>
    void ReadMtxTools<ValType>::loadRightHand(HostVector<ValType> &rhs) {
        processRhsHeader();
        if (m_matDataType == MTX_DATA_REAL) {
            rhs.resize(m_cdim * m_bCols, RESERVE_NO_DATA);
            UINT32 actual_length = m_cdim * m_bCols;
            for (UINT32 i = 0; i < actual_length; ++i) {
                m_BStream >> rhs[i];
            }
        } else {
            THROW_LOGIC_ERROR("Unsupportable data version");
        }
    }

    template<typename ValType>
    void ReadMtxTools<ValType>::loadRightHand(HostVector<ValType> &rhsR, HostVector<ValType> &rhsI) {
        processRhsHeader();
        if (m_matDataType == MTX_DATA_COMPLEX) {
            rhsR.resize(m_cdim * m_bCols, RESERVE_NO_DATA);
            rhsI.resize(m_cdim * m_bCols, RESERVE_NO_DATA);
            UINT32 actual_length = m_cdim * m_bCols;
            for (UINT32 i = 0; i < actual_length; ++i) {
                m_BStream >> rhsR[i];
                m_BStream >> rhsI[i];
            }
        } else {
            THROW_LOGIC_ERROR("Unsupportable data version");
        }
    }

    template<typename ValType>
    void ReadMtxTools<ValType>::loadRightHand(HostVector<ValType> &res, HostVector<ValType> &rhs, UINT32 rhsNum) {
#ifndef NDEBUG
        THROW_EXCEPTION(m_BStream.is_open(), THROW_INVALID_ARGUMENT("The file is already opened!"))
#endif
        m_bCols = rhsNum;
        res.resize(m_cdim * m_bCols, RESERVE_NO_DATA);
        rhs.resize(m_cdim * m_bCols, RESERVE_NO_DATA);
        if (m_matDataType == MTX_DATA_REAL) {
            UINT32 matNNZNum = m_auxRealCOO.getNNZnum();
#ifndef NDEBUG
            THROW_EXCEPTION(matNNZNum == 0,
                            THROW_LOGIC_ERROR("The matrix file was not loaded! Matrix data is missing!"))
#endif
            const UINT32 *RowIdxPtr = m_auxRealCOO.getRowIndicesPtr();
            const UINT32 *ColIdxPtr = m_auxRealCOO.getColIndicesPtr();
            const ValType *matValuesPtr = m_auxRealCOO.getCOOValuesPtr();

            UINT32 actualLength = res.getLength();
            if (rhs.getMemoryType() != memoryBase) rhs.fillVector(0, actualLength, 0);
            generateArrayRandom1D(res.getRawValPtr(), actualLength);
            for (UINT32 j = 0; j < rhsNum; ++j) {
                UINT32 offset = j * m_cdim;
                for (UINT32 i = 0; i < matNNZNum; ++i) {
                    rhs[RowIdxPtr[i] + offset] += matValuesPtr[i] * res[ColIdxPtr[i] + offset];
                }
            }
        } else {
            THROW_LOGIC_ERROR("Unsupportable data version");
        }
    }

    template<typename ValType>
    void ReadMtxTools<ValType>::loadMatrix(CSRMatrix<ValType> &outMat) {
        processMatHeader();
        std::vector<char> buffer(m_bufferSize);
        std::string leftover; // 保存上次读取的剩余部分
        if (m_matStructure == MTX_STRUCTURE_COO) {
            // 原始数据文件中，矩阵以COO格式存储
            m_auxRealCOO.resize(m_rdim, m_cdim, m_nnzNum, 0);
            std::streamsize bytesRead;
            size_t lastPos, pos;
            std::string_view currentChunk;
            while (m_AStream) {
                // 如果 leftover 有内容，将它移动到 buffer 的前端
                if (!leftover.empty()) {
                    std::memmove(buffer.data(), leftover.data(), leftover.size());
                }
                m_AStream.read(buffer.data() + leftover.size(), m_bufferSize - leftover.size());
                bytesRead = m_AStream.gcount();
                if (bytesRead <= 0) break;
                // 调整总的有效字节数
                bytesRead += leftover.size();
                // 获取当前数据块
                currentChunk = std::string_view(buffer.data(), bytesRead);
                lastPos = 0;
                while ((pos = currentChunk.find('\n', lastPos)) != std::string_view::npos) {
                    processMatLine(currentChunk.substr(lastPos, pos - lastPos));
                    lastPos = pos + 1;
                }
                // 保存不完整的行
                leftover = currentChunk.substr(lastPos);
            }
            // 处理最后一块不完整的数据
            if (!leftover.empty()) {
                processMatLine(std::string_view(leftover));
            }
#ifndef NDEBUG
            if (m_auxRealZeroElementCount > 0) {
                SHOW_WARN(
                    "There exists " << m_auxRealZeroElementCount <<
                    " zero element(s) in original data file. actual non-zero element count: " << m_actRealNNZidx);
            }
#endif
            // 将存储为COO格式的临时数据转换为CSR数据
            transCOO2CSR(m_auxRealCOO, outMat);
        } else {
            THROW_LOGIC_ERROR("Unsupportable matrix type!");
        }
        SHOW_INFO("Reading matrix file finished!")
    }

    template<typename ValType>
    void ReadMtxTools<ValType>::loadMatrix(CSRMatrix<ValType> &rMat, CSRMatrix<ValType> &iMat) {
        processMatHeader();
#ifndef NDEBUG
        if (m_matDataType != MTX_DATA_COMPLEX) {
            SHOW_WARN("This function is designed for complex matrix and right-hand, but your original data is real.")
        }
#endif
        std::vector<char> buffer(m_bufferSize);
        std::string leftover; // 保存上次读取的剩余部分
        if (m_matStructure == MTX_STRUCTURE_COO) {
            // 原始数据文件中，矩阵以COO格式存储
            m_auxRealCOO.resize(m_rdim, m_cdim, m_nnzNum, 0);
            m_auxImageCOO.resize(m_rdim, m_cdim, m_nnzNum, 0);
            std::streamsize bytesRead;
            size_t lastPos, pos;
            std::string_view currentChunk;
            while (m_AStream) {
                // 如果 leftover 有内容，将它移动到 buffer 的前端
                if (!leftover.empty()) {
                    std::memmove(buffer.data(), leftover.data(), leftover.size());
                }
                m_AStream.read(buffer.data() + leftover.size(), m_bufferSize - leftover.size());
                bytesRead = m_AStream.gcount();
                if (bytesRead <= 0) break;
                // 调整总的有效字节数
                bytesRead += leftover.size();
                // 获取当前数据块
                currentChunk = std::string_view(buffer.data(), bytesRead);
                lastPos = 0;
                while ((pos = currentChunk.find('\n', lastPos)) != std::string_view::npos) {
                    processMatLine(currentChunk.substr(lastPos, pos - lastPos));
                    lastPos = pos + 1;
                }
                // 保存不完整的行
                leftover = currentChunk.substr(lastPos);
            }
            // 处理最后一块不完整的数据
            if (!leftover.empty()) {
                processMatLine(std::string_view(leftover));
            }
#ifndef NDEBUG
            if (m_auxRealZeroElementCount > 0) {
                SHOW_WARN(
                    "There exists " << m_auxRealZeroElementCount <<
                    " zero element(s) in original data file. actual non-zero element count: " << m_actRealNNZidx);
            }
#endif
            // 将存储为COO格式的临时数据转换为CSR数据
            transCOO2CSR(m_auxRealCOO, rMat);
            transCOO2CSR(m_auxImageCOO, iMat);
        } else {
            THROW_LOGIC_ERROR("Unsupportable matrix type!");
        }
        SHOW_INFO("Reading matrix file finished!")
    }
} // namespace HOST
