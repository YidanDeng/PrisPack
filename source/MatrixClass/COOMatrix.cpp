/*
 * @author  邓轶丹
 * @date    2024/5/10
 * @details 实现基于COO的相关操作函数
 */

#include "../../include/MatrixClass/COOMatrix.h"

namespace HOST {
    template<typename ValType>
    COOMatrix<ValType>::COOMatrix() {
#ifndef NINFO
        SHOW_INFO("Default constructor for COO matrix begin!")
#endif
        BaseMatrix<ValType>::m_matType = matrixCOO;     // 标记当前数组存储类型为COO格式
    }

    template<typename ValType>
    COOMatrix<ValType>::COOMatrix(const UINT32 &rowNum, const UINT32 &colNum, UINT32 nnz,
                                  const memoryType_t &memoryTypeHost) {

#ifndef NINFO
        SHOW_INFO("Constructor with parameters for COO matrix begin!")
#endif

#ifndef NDEBUG
        ERROR_CHECK(rowNum + 1 >= UINT32_MAX, DEBUG_MESSEGE_OPTION,
                    "The row number of current CSR matrix overflowed! It should not be more than 0xffffffffU "
                    "(UINT32_MAX)!");

        ERROR_CHECK(colNum + 1 >= UINT32_MAX, DEBUG_MESSEGE_OPTION,
                    "The column number of current CSR matrix overflowed! It should not be more than 0xffffffffU "
                    "(UINT32_MAX)!");

        ERROR_CHECK(nnz >= UINT32_MAX, DEBUG_MESSEGE_OPTION,
                    "The total none-zero number of current CSR matrix overflowed! It should not be more than 0xffffffffU "
                    "(UINT32_MAX)!");
#endif
        BaseMatrix<ValType>::m_matType = matrixCOO;     // 标记当前数组存储类型为COO格式
        m_rows = rowNum;
        m_cols = colNum;
        m_nnz = nnz;
        if (memoryTypeHost != memoryBase) {
            i_vec.reset(nnz, memoryTypeHost);
            j_vec.reset(nnz, memoryTypeHost);
            data_vec.reset(nnz, memoryTypeHost);
        } else {
            i_vec->resize(nnz, RESERVE_NO_DATA);
            j_vec->resize(nnz, RESERVE_NO_DATA);
            data_vec->resize(nnz, RESERVE_NO_DATA);
        }
        m_memoryType = memoryTypeHost;

    }

    template<typename ValType>
    COOMatrix<ValType>::COOMatrix(const memoryType_t &memoryTypeHost) {
        BaseMatrix<ValType>::m_matType = matrixCOO;     // 标记当前数组存储类型为COO格式
        if (memoryTypeHost != memoryBase) {
            i_vec.reset(0, memoryTypeHost);
            j_vec.reset(0, memoryTypeHost);
            data_vec.reset(0, memoryTypeHost);
            m_memoryType = memoryTypeHost;
        }
    }


    template<typename ValType>
    COOMatrix<ValType>::COOMatrix(const COOMatrix<ValType> &pre) {
        BaseMatrix<ValType>::m_matType = matrixCOO;     // 标记当前数组存储类型为COO格式
        m_rows = pre.m_rows;
        m_cols = pre.m_cols;
        m_nnz = pre.m_nnz;
        m_actNNZ = pre.m_actNNZ;
        if (pre.m_memoryType != memoryBase) {
            i_vec.reset(m_nnz, pre.m_memoryType);
            j_vec.reset(m_nnz, pre.m_memoryType);
            data_vec.reset(m_nnz, pre.m_memoryType);
            m_memoryType = pre.m_memoryType;
        } else {
            i_vec->resize(pre.m_nnz, RESERVE_NO_DATA);
            j_vec->resize(pre.m_nnz, RESERVE_NO_DATA);
            data_vec->resize(pre.m_nnz, RESERVE_NO_DATA);
        }
        i_vec->copy(*pre.i_vec);
        j_vec->copy(*pre.j_vec);
        data_vec->copy(*pre.data_vec);
    }

    template<typename ValType>
    COOMatrix<ValType>::COOMatrix(COOMatrix<ValType> &&pre) noexcept {
        BaseMatrix<ValType>::m_matType = matrixCOO;     // 标记当前数组存储类型为COO格式
        m_rows = pre.m_rows;
        m_cols = pre.m_cols;
        m_nnz = pre.m_nnz;
        m_actNNZ = pre.m_actNNZ;
        m_memoryType = pre.m_memoryType;
        i_vec = std::move(pre.i_vec);
        j_vec = std::move(pre.j_vec);
        data_vec = std::move(pre.data_vec);
        pre.m_rows = 0;
        pre.m_cols = 0;
        pre.m_nnz = 0;
        pre.m_actNNZ = 0;
    }


    template<typename ValType>
    COOMatrix<ValType> &COOMatrix<ValType>::operator=(COOMatrix<ValType> &&pre) noexcept {
        if (&pre == this)
            return *this;
        m_rows = pre.m_rows;
        m_cols = pre.m_cols;
        m_nnz = pre.m_nnz;
        m_actNNZ = pre.m_actNNZ;
        m_memoryType = pre.m_memoryType;
        i_vec = std::move(pre.i_vec);
        j_vec = std::move(pre.j_vec);
        data_vec = std::move(pre.data_vec);
        pre.m_rows = 0;
        pre.m_cols = 0;
        pre.m_nnz = 0;
        pre.m_actNNZ = 0;
        return *this;
    }

    template<typename ValType>
    COOMatrix<ValType> &COOMatrix<ValType>::operator=(const COOMatrix<ValType> &pre) {
        if (&pre == this)
            return *this;
        m_rows = pre.m_rows;
        m_cols = pre.m_cols;
        m_nnz = pre.m_nnz;
        m_actNNZ = pre.m_actNNZ;
        if (m_memoryType != pre.m_memoryType) {
            i_vec.reset(m_nnz, pre.m_memoryType);
            j_vec.reset(m_nnz, pre.m_memoryType);
            data_vec.reset(m_nnz, pre.m_memoryType);
            m_memoryType = pre.m_memoryType;
        } else {
            i_vec->resize(m_nnz, RESERVE_NO_DATA);
            j_vec->resize(m_nnz, RESERVE_NO_DATA);
            data_vec->resize(m_nnz, RESERVE_NO_DATA);
        }
        i_vec->copy(*pre.i_vec);
        j_vec->copy(*pre.j_vec);
        data_vec->copy(*pre.data_vec);
        return *this;
    }


    template<typename ValType>
    void COOMatrix<ValType>::clear() {
        m_rows = 0;
        m_cols = 0;
        m_nnz = 0;
        m_actNNZ = 0;
        // 只清除申请的内存空间，不删除AutoAllocateVector对象
        i_vec->clear();
        j_vec->clear();
        data_vec->clear();
    }

    template<typename ValType>
    void COOMatrix<ValType>::resize(const UINT32 &new_rowNum, const UINT32 &new_colNum, const UINT32 &new_nnz,
                                    UINT8 need_reserve) {
        if (need_reserve == RESERVE_NO_DATA) {
            i_vec->resize(new_nnz, RESERVE_NO_DATA);
            j_vec->resize(new_nnz, RESERVE_NO_DATA);
            data_vec->resize(new_nnz, RESERVE_NO_DATA);
            m_actNNZ = 0;
        } else {
            UINT8 isReserved;
            UINT32 currRowIdx, currColIdx;
            ValType currVal;
            AutoAllocateVector<UINT32> tempIvec(m_nnz, m_memoryType);
            AutoAllocateVector<UINT32> tempJvec(m_nnz, m_memoryType);
            AutoAllocateVector<ValType> tempData(m_nnz, m_memoryType);
            UINT32 newNNZcount = 0;
            for (UINT32 idx = 0; idx < m_nnz; ++idx) {
                isReserved = 1;
                currRowIdx = i_vec->getValue(idx);
                currColIdx = j_vec->getValue(idx);
                currVal = data_vec->getValue(idx);
                isReserved &= currRowIdx < new_rowNum;
                isReserved &= currColIdx < new_colNum;
                if (isReserved) {
                    tempIvec->setValue(newNNZcount, currRowIdx);
                    tempJvec->setValue(newNNZcount, currColIdx);
                    tempData->setValue(newNNZcount, currVal);
                    newNNZcount++;
                }
            }
#ifndef NDEBUG
            ERROR_CHECK(new_nnz < newNNZcount, DEBUG_MESSEGE_OPTION,
                        "The new none-zero element size is less than the actual size!");
#endif
            m_actNNZ = newNNZcount;
            i_vec->move(*tempIvec);
            j_vec->move(*tempJvec);
            data_vec->move(*tempData);
            i_vec->resize(new_nnz, RESERVE_DATA);
            j_vec->resize(new_nnz, RESERVE_DATA);
            data_vec->resize(new_nnz, RESERVE_DATA);
        }
        m_rows = new_rowNum;
        m_cols = new_colNum;
        m_nnz = new_nnz;
    }


} // HOST