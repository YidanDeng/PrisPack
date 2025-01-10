/*
 * @author  邓轶丹
 * @date    2024/6/12
 * @details 实现基于稠密矩阵的相关操作函数
 */

#include "../../include/MatrixClass/DenseMatrix.h"

namespace HOST {
    template <typename ValType>
    DenseMatrix<ValType>::DenseMatrix(DenseMatrix<ValType>&& pre) noexcept {
        m_rowNum = pre.m_rowNum;
        m_colNum = pre.m_colNum;
        m_data_vec = std::move(pre.m_data_vec);
        m_storageType = pre.m_storageType;
    }

    template <typename ValType>
    DenseMatrix<ValType>::DenseMatrix(const DenseMatrix<ValType>& pre) {
        m_rowNum = pre.m_rowNum;
        m_colNum = pre.m_colNum;
        m_data_vec = pre.m_data_vec;
        m_storageType = pre.m_storageType;
    }

    template <typename ValType>
    DenseMatrix<ValType>& DenseMatrix<ValType>::operator=(DenseMatrix<ValType>&& pre_mat) noexcept {
        if (&pre_mat == this) return *this;
        m_rowNum = pre_mat.m_rowNum;
        m_colNum = pre_mat.m_colNum;
        m_data_vec = std::move(pre_mat.m_data_vec);
        m_storageType = pre_mat.m_storageType;
        return *this;
    }

    template <typename ValType>
    DenseMatrix<ValType>& DenseMatrix<ValType>::operator=(const DenseMatrix<ValType>& pre_mat) {
        if (&pre_mat == this) return *this;
        m_rowNum = pre_mat.m_rowNum;
        m_colNum = pre_mat.m_colNum;
        m_data_vec = pre_mat.m_data_vec;
        m_storageType = pre_mat.m_storageType;
        return *this;
    }

    template <typename ValType>
    DenseMatrix<ValType>::DenseMatrix(const DenseMatStoreType_t& denseMatStoreType, const UINT32& rowNum,
                                      const UINT32& colNum, const memoryType_t& memoryTypeHost) {
        m_storageType = denseMatStoreType;
        m_rowNum = rowNum;
        m_colNum = colNum;
        UINT32 totalLength = rowNum * colNum;
        m_data_vec.reset(totalLength, memoryTypeHost);
    }


    template <typename ValType>
    void DenseMatrix<ValType>::getValsByCol(const UINT32& colNo, HostVector<ValType>& vec) {
#ifndef NDEBUG
        THROW_EXCEPTION(vec.getLength() < m_rowNum,
                        THROW_RANGE_ERROR("The dim of out-vec is not equal to current mat row num!"))
        THROW_EXCEPTION(colNo >= m_colNum, THROW_OUT_OF_RANGE("The col num is out-of-range!"))
#endif
        UINT32 offset;
        if (m_storageType == DenseMatRowFirst) {
            offset = colNo;
            for (UINT32 i = 0; i < m_rowNum; ++i) {
                vec[i] = m_data_vec[offset];
                offset += m_colNum;
            }
        } else {
            offset = colNo * m_rowNum;
            vec.copy(*m_data_vec, offset, 0, m_rowNum);
        }
    }

    template <typename ValType>
    void DenseMatrix<ValType>::setValsByCol(const UINT32& colNo, const HostVector<ValType>& vec) {
#ifndef NDEBUG
        THROW_EXCEPTION(vec.getLength() != m_rowNum,
                        THROW_RANGE_ERROR("The dim of out-vec is not equal to current mat row num!"))
        THROW_EXCEPTION(colNo >= m_colNum, THROW_OUT_OF_RANGE("The col num is out-of-range!"))
#endif
        UINT32 offset;
        if (m_storageType == DenseMatRowFirst) {
            offset = colNo;
            for (UINT32 i = 0; i < m_rowNum; ++i) {
                m_data_vec[offset] = vec[i];
                offset += m_colNum;
            }
        } else {
            offset = colNo * m_rowNum;
            m_data_vec.copy(vec, 0, offset, m_rowNum);
        }
    }


    template <typename ValType>
    void DenseMatrix<ValType>::transpose() {
        UINT32 offset = 0, newIdx, oldIdx; ///< newIdx用来遍历存储意义上的列，oldIdx用来遍历存储意义上的行
        if (m_rowNum == m_colNum) {
            UINT32 dim = m_rowNum;
            ValType temp;
            for (UINT32 i = 0; i < dim; ++i) {
                newIdx = offset + dim + i; // 待交换元素所在的索引
                for (UINT32 j = i + 1; j < dim; ++j) {
                    oldIdx = offset + j;
                    temp = m_data_vec[oldIdx];
                    m_data_vec[oldIdx] = m_data_vec[newIdx];
                    m_data_vec[newIdx] = temp;
                    newIdx += dim;
                }
                offset += dim;
            }
        } else {
            AutoAllocateVector<ValType> transData(m_data_vec.getLength(), m_data_vec->getMemoryType());
            UINT32 newIdxShift, oldIdxShift, iBound, jBound;
            if (m_storageType == DenseMatRowFirst) {
                newIdxShift = m_rowNum;
                oldIdxShift = m_colNum;
                iBound = m_rowNum;
                jBound = m_colNum;
            } else {
                newIdxShift = m_colNum;
                oldIdxShift = m_rowNum;
                iBound = m_colNum;
                jBound = m_rowNum;
            }
            for (UINT32 i = 0; i < iBound; ++i) {
                newIdx = i; // 待交换元素所在的索引
                for (UINT32 j = 0; j < jBound; ++j) {
                    oldIdx = offset + j;
                    transData[newIdx] = this->m_data_vec[oldIdx];
                    newIdx += newIdxShift;
                }
                offset += oldIdxShift;
            }
            m_data_vec = std::move(transData);
            UINT32 tempRowNum = m_rowNum;
            m_rowNum = m_colNum;
            m_colNum = tempRowNum;
        }
    }

    template <typename ValType>
    void DenseMatrix<ValType>::printMatrix(const char* message) {
#ifndef NINFO
        SHOW_INFO("Print CSR matrix.")
#endif
        std::string type = m_data_vec->getMemoryType() == memoryBase
                               ? "Base"
                               : m_data_vec->getMemoryType() == memoryAligned
                               ? "Aligned"
                               : "Page-locked";
        std::cout << L_GREEN << "[Dense matrix: " << type << "] " << L_BLUE << message << " --- "
            << "row:" << m_rowNum << ", col:" << m_colNum << COLOR_NONE << std::endl;
        UINT32 actRowNum = m_rowNum < 20 ? m_rowNum : 20;
        for (UINT32 i = 0; i < actRowNum; ++i) {
            std::cout << L_GREEN << "row[" << i << "]:" << COLOR_NONE;
            UINT32 actColNum = m_colNum < 20 ? m_colNum : 20;
            for (UINT32 j = 0; j < actColNum; ++j) {
                ValType val;
                getValue(i, j, val);
                std::cout << " " << val;
            }
            if (m_colNum > 20) std::cout << "... (rest columns were fold)";
            std::cout << std::endl;
        }
        if (m_rowNum > 20) std::cout << "... (rest rows were fold)" << std::endl;
    }


    template <typename ValType>
    void DenseMatrix<ValType>::resize(const UINT32& new_rowNum, const UINT32& new_colNum, const UINT32& new_nnz,
                                      UINT8 need_reserve) {
        if (new_rowNum == m_rowNum && new_colNum == m_colNum && m_data_vec.getLength() == new_nnz &&
            need_reserve)
            return;
#ifndef NDEBUG
        THROW_EXCEPTION(new_rowNum * new_colNum != new_nnz, THROW_LOGIC_ERROR("The nnz size is not correct!"))
#endif
        if (!need_reserve) {
            m_data_vec.resize(new_nnz, RESERVE_NO_DATA);
        } else {
            AutoAllocateVector<ValType> tempData(new_nnz, m_data_vec->getMemoryType());
            UINT32 oldOffset = 0, newOffset = 0, eachCopiedLength;
            if (m_storageType == DenseMatRowFirst) {
                eachCopiedLength = m_colNum < new_colNum ? m_colNum : new_colNum;
                for (UINT32 i = 0; i < m_rowNum; ++i) {
                    tempData.copy(*m_data_vec, oldOffset, newOffset, eachCopiedLength);
                    oldOffset += m_colNum;
                    newOffset += new_colNum;
                }
            } else {
                eachCopiedLength = m_rowNum < new_rowNum ? m_rowNum : new_rowNum;
                for (UINT32 i = 0; i < m_colNum; ++i) {
                    tempData.copy(*m_data_vec, oldOffset, newOffset, eachCopiedLength);
                    oldOffset += m_rowNum;
                    newOffset += new_rowNum;
                }
            }
            m_data_vec = std::move(tempData);
        }
        m_rowNum = new_rowNum;
        m_colNum = new_colNum;
    }

    template <typename ValType>
    void DenseMatrix<ValType>::MatVec(UINT32 start, UINT32 end, const HostVector<ValType>& vec,
                                      HostVector<ValType>& out_vec) {
        UINT32 n = vec.getLength();
#ifndef NDEBUG
        THROW_EXCEPTION(n < end - start + 1, THROW_LOGIC_ERROR("The vector length is incorrect!"))
#endif
        UINT32 rowNum = this->m_rowNum, begin;
        out_vec.resize(rowNum, 0);
        if (this->m_storageType == DenseMatColumnFirst) {
            for (UINT32 i = start; i <= end; ++i) {
                begin = i * rowNum;
                UINT32 colIdx = i - start;
                for (UINT32 j = 0; j < rowNum; ++j) {
                    out_vec[j] += this->m_data_vec[begin + j] * vec[colIdx];
                }
            }
        } else {
#ifdef USE_OMP_MATRIX_FUNC
#pragma omp parallel for default(none) num_threads(THREAD_NUM)  private(begin) \
                shared(start, end, rowNum, out_vec, vec, m_data_vec)
#endif
            for (UINT32 i = 0; i < rowNum; ++i) {
                begin = i * m_colNum;
                out_vec[i] = 0;
                for (UINT32 j = start; j <= end; ++j) {
                    out_vec[i] += this->m_data_vec[begin + j] * vec[j - start];
                }
            }
        }
    }

    template <typename ValType>
    void DenseMatrix<ValType>::MatVec(const HostVector<ValType>& vec, HostVector<ValType>& out_vec) {
        return this->MatVec(0, this->m_colNum - 1, vec, out_vec);
    }

    template <typename ValType>
    void DenseMatrix<ValType>::transposeVec(const HostVector<ValType>& vec, HostVector<ValType>& out_vec) {
        UINT32 m, n, i, j, begin;
        m = this->m_rowNum;
        n = this->m_colNum;
        if (out_vec.getLength() != n) out_vec.resize(n, RESERVE_NO_DATA);
#ifndef NDEBUG
        THROW_EXCEPTION(vec.getLength() != m, THROW_LOGIC_ERROR("Dense transpose MatVec Input is not match!"))
#endif
        if (m_storageType == DenseMatColumnFirst) {
            for (i = 0; i < n; ++i) {
                begin = i * m;
                for (j = 0; j < m; ++j)
                    out_vec[i] += this->m_data_vec[begin + j] * vec[j];
            }
        } else {
            for (i = 0; i < m; ++i) {
                begin = i * n;
                for (j = 0; j < n; ++j)
                    out_vec[j] += this->m_data_vec[begin + i] * vec[i];
            }
        }
    }

    template <typename ValType>
    void DenseMatrix<ValType>::MatMatMul(const DenseMatrix<ValType>& inMat, DenseMatrix<ValType>& outMat) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_colNum != inMat.getRowNum()|| outMat.getRowNum() != this->m_rowNum ||
                        outMat.getColNum() != inMat.getColNum(),
                        THROW_LOGIC_ERROR("The dimension of Matrices are not match!"))
#endif

        if (this->m_storageType == DenseMatColumnFirst) {
            UINT32 m, k, n;
            m = this->m_rowNum;
            k = this->m_colNum;
            n = inMat.getColNum();
            const ValType* inMatPtr = inMat.getMatValPtr();

#pragma omp parallel for default(none) proc_bind(master) num_threads(THREAD_NUM) shared(m, n, k, m_data_vec, outMat, inMatPtr)
            for (UINT32 i = 0; i < n; ++i) {
                ValType* localOutMatPtr = outMat.getMatValPtr() + i * m;
                const ValType* localInMatPtr = inMatPtr + i * k;
                for (UINT32 colIdx = 0; colIdx < k; ++colIdx) { // V的列标，也是Y的行标
                    UINT32 begin = colIdx * m;
                    ValType localInMatVal = localInMatPtr[colIdx]; // 当前列对应的Y中的值
                    for (UINT32 rowIdx = 0; rowIdx < m; ++rowIdx) {
                        localOutMatPtr[rowIdx] += this->m_data_vec[begin + rowIdx] * localInMatVal;
                    }
                }
            }
        } else if (this->m_storageType == DenseMatRowFirst) {
            UINT32 m = this->m_rowNum;
            UINT32 k = this->m_colNum;
            UINT32 n = inMat.getColNum();
            const ValType* inMatPtr = inMat.getMatValPtr();
            ValType* outMatPtr = outMat.getMatValPtr();

            // 初始化结果矩阵
            for (UINT32 i = 0; i < m * n; ++i) {
                outMatPtr[i] = 0;
            }

            for (UINT32 rowIdx = 0; rowIdx < m; ++rowIdx) { // 遍历每一行
                for (UINT32 colIdx = 0; colIdx < n; ++colIdx) { // 遍历每一列
                    for (UINT32 kIdx = 0; kIdx < k; ++kIdx) { // 遍历内积
                        outMatPtr[rowIdx * n + colIdx] += this->m_data_vec[rowIdx * k + kIdx] * inMatPtr[kIdx * n +
                            colIdx];
                    }
                }
            }
        }
    }
} // HOST
