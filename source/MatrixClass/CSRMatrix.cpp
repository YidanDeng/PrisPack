/*
 * @author  邓轶丹
 * @date    2024/4/29
 * @details 实现基于CSR的相关操作函数
 */
#include "../../include/MatrixClass/CSRMatrix.h"


namespace HOST {
    template<typename ValType>
    CSRMatrix<ValType>::CSRMatrix() {
#ifndef NINFO
        SHOW_INFO("Default constructor for CSR matrix begin!")
#endif
        BaseMatrix<ValType>::m_matType = matrixCSR; // 标记当前数组存储类型为CSR格式
        m_rowOffset->resize(1, RESERVE_NO_DATA);
        m_rowOffset[0] = 0;
    }

    template<typename ValType>
    CSRMatrix<ValType>::CSRMatrix(const memoryType_t &memoryType) {
        BaseMatrix<ValType>::m_matType = matrixCSR; // 标记当前数组存储类型为CSR格式
        // 由于默认内存模式是base，只有当前内存模式是其他形式，才需要重置内存空间类型
        if (memoryType != memoryBase) {
            m_rowOffset.reset(1, memoryType);
            m_colIndices.reset(0, memoryType);
            m_values.reset(0, memoryType);
        } else {
            m_rowOffset->resize(1, RESERVE_NO_DATA);
        }
        m_memoryType = memoryType;
        m_rowOffset[0] = 0;
    }

    template<typename ValType>
    CSRMatrix<ValType>::CSRMatrix(const UINT32 &rowNum, const UINT32 &colNum, const UINT32 &nnzNum,
                                  memoryType_t memoryType) {
#ifndef NINFO
        SHOW_INFO("Constructor with parameters for CSR matrix begin!")
#endif

#ifndef NDEBUG
        ERROR_CHECK(rowNum + 1 >= UINT32_MAX, DEBUG_MESSEGE_OPTION,
                    "The row number of current CSR matrix overflowed! It should not be more than 0xffffffffU "
                    "(UINT32_MAX)!");

        ERROR_CHECK(colNum + 1 >= UINT32_MAX, DEBUG_MESSEGE_OPTION,
                    "The column number of current CSR matrix overflowed! It should not be more than 0xffffffffU "
                    "(UINT32_MAX)!");

        ERROR_CHECK(nnzNum >= UINT32_MAX, DEBUG_MESSEGE_OPTION,
                    "The total none-zero number of current CSR matrix overflowed! It should not be more than 0xffffffffU "
                    "(UINT32_MAX)!");
#endif
        BaseMatrix<ValType>::m_matType = matrixCSR; // 标记当前数组存储类型为CSR格式
        this->m_rowNum = rowNum;
        this->m_colNum = colNum;
        this->m_nnzCapacity = nnzNum;
        if (memoryType != memoryBase) {
            m_rowOffset.reset(rowNum + 1, memoryType);
            m_colIndices.reset(nnzNum, memoryType);
            m_values.reset(nnzNum, memoryType);
            m_rowOffset->fillVector(0, m_rowOffset->getLength(), 0);
        } else {
            m_rowOffset->resize(rowNum + 1, RESERVE_NO_DATA);
            m_colIndices->resize(nnzNum, RESERVE_NO_DATA);
            m_values->resize(nnzNum, RESERVE_NO_DATA);
        }
        m_memoryType = memoryType;
#ifndef NDEBUG
        m_rowOffset->fillVector(1, m_rowOffset->getLength() - 1, UINT32_MAX);
#endif
    }

    template<typename ValType>
    CSRMatrix<ValType>::CSRMatrix(CSRMatrix<ValType> &&pre_mat) noexcept {
#ifndef NINFO
        SHOW_INFO("Move constructor for CSR matrix begin!")
#endif
        m_rowNum = pre_mat.m_rowNum;
        m_colNum = pre_mat.m_colNum;
        m_nnzCapacity = pre_mat.m_nnzCapacity;
        m_memoryType = pre_mat.m_memoryType;
        m_isFormated = pre_mat.m_isFormated;
        m_rowOffset = std::move(pre_mat.m_rowOffset);
        m_colIndices = std::move(pre_mat.m_colIndices);
        m_values = std::move(pre_mat.m_values);
        // 由于现有向量对象剥夺了原向量对象中的各存储变量的控制权，则原向量对象必须有相关重置操作
        pre_mat.m_rowNum = 0;
        pre_mat.m_colNum = 0;
        pre_mat.m_nnzCapacity = 0;
    }

    template<typename ValType>
    CSRMatrix<ValType>::CSRMatrix(const CSRMatrix<ValType> &pre_mat) {
#ifndef NINFO
        SHOW_INFO("Copy constructor for CSR matrix begin!")
#endif
        BaseMatrix<ValType>::m_matType = matrixCSR; // 标记当前数组存储类型为CSR格式
        m_memoryType = pre_mat.m_memoryType;
        m_rowOffset->resize(pre_mat.m_rowNum + 1, RESERVE_NO_DATA);
        m_colIndices->resize(pre_mat.m_nnzCapacity, RESERVE_NO_DATA);
        m_values->resize(pre_mat.m_nnzCapacity, RESERVE_NO_DATA);
        m_rowNum = pre_mat.m_rowNum;
        m_colNum = pre_mat.m_colNum;
        m_nnzCapacity = pre_mat.m_nnzCapacity;
        m_isFormated = pre_mat.m_isFormated;
        m_rowOffset->copy(*pre_mat.m_rowOffset);
        m_colIndices->copy(*pre_mat.m_colIndices);
        m_values->copy(*pre_mat.m_values);
    }


    template<typename ValType>
    void CSRMatrix<ValType>::transpose() {
        UINT32 oldRowNum = m_rowNum;
        UINT32 oldColNum = m_colNum;
        // 记录转置后的每一行有多少元素，后面可以改一改直接用来做row offset
        AutoAllocateVector<UINT32> newRowCount(m_colNum + 1, m_memoryType);
        if (m_memoryType != memoryBase) newRowCount->fillVector(0, m_colNum + 1, 0);

        // 记录转置后矩阵每一行已成功放置的元素个数
        AutoAllocateVector<UINT32> newRowFixed(m_colNum, memoryBase);
        UINT32 actualNNZnum = m_rowOffset[m_rowNum + 1];
        for (UINT32 i = 0; i < actualNNZnum; ++i) {
            newRowCount[m_colIndices[i] + 1]++; // 统计转置后每行元素个数
        }
        for (UINT32 i = 1; i <= m_colNum; ++i) {
            // 重新调整new_row_count，使其成为转置后的row offset模式
            newRowCount[i] += newRowCount[i - 1];
        }
        AutoAllocateVector<ValType> newValue(m_nnzCapacity, m_memoryType);
        AutoAllocateVector<UINT32> newColIndices(m_nnzCapacity, m_memoryType);
#ifndef NDEBUG
        newValue->fillVector(0, m_nnzCapacity, 0);
        newColIndices->fillVector(0, m_nnzCapacity, 0);
#endif
        for (UINT32 i = 0; i < m_rowNum; ++i) {
            UINT32 preColNo, newRowStart, insertPos;
            for (UINT32 j = m_rowOffset[i]; j < m_rowOffset[i + 1]; ++j) {
                preColNo = m_colIndices->getValue(j); // 记录原先值对应的列号，这个列号等价于转置后的行号
                newRowStart = newRowCount[preColNo]; // 记录转置后待写入值的那一行的起始位置
                insertPos = newRowStart + newRowFixed[preColNo]; // 记录即将写入值的位置
#ifndef NDEBUG
                THROW_EXCEPTION(std::fabs(newValue[insertPos]) != 0 || newColIndices[insertPos] != 0,
                                THROW_LOGIC_ERROR("The same position have been changed more than once! "
                                    "Please make sure your original data is correct."))
#endif
                newColIndices[insertPos] = i;
                newValue[insertPos] = m_values->getValue(j);
                newRowFixed[preColNo]++;
            }
        }
        m_rowNum = oldColNum;
        m_colNum = oldRowNum;
        m_rowOffset->move(*newRowCount);
        m_colIndices->move(*newColIndices);
        m_values->move(*newValue);
    }


    template<typename ValType>
    void CSRMatrix<ValType>::resetMemoryType(const memoryType_t &memoryTypeHost, UINT8 needReserveData) {
        if (memoryTypeHost == m_memoryType) return;
        if (!needReserveData) {
            m_rowOffset.reset(m_rowOffset->getLength(), memoryTypeHost);
            m_colIndices.reset(m_nnzCapacity, memoryTypeHost);
            m_values.reset(m_nnzCapacity, memoryTypeHost);
        } else {
            AutoAllocateVector<UINT32> tempOffset(m_rowOffset->getLength(), memoryTypeHost);
            tempOffset->copy(*m_rowOffset);
            m_rowOffset = std::move(tempOffset);
            AutoAllocateVector<UINT32> tempColIdx(m_nnzCapacity, memoryTypeHost);
            tempColIdx->copy(*m_colIndices);
            m_colIndices = std::move(tempColIdx);
            AutoAllocateVector<ValType> tempValues(m_nnzCapacity, memoryTypeHost);
            tempValues->copy(*m_values);
            m_values = std::move(tempValues);
        }
        m_memoryType = memoryTypeHost;
    }

    template<typename ValType>
    void CSRMatrix<ValType>::setValue(const UINT32 &rowNo, const UINT32 &colNo, const ValType &val) {
#ifndef NDEBUG
        THROW_EXCEPTION(rowNo > m_rowNum - 1, THROW_OUT_OF_RANGE("The parameter \"rowNo\" out of range!"))
        THROW_EXCEPTION(colNo > m_colNum - 1, THROW_OUT_OF_RANGE("The parameter \"colNo\" out of range!"))
#endif
        // 二分查找
        UINT32 left_ptr = m_rowOffset[rowNo],
                right_ptr = m_rowOffset[rowNo + 1] - 1;
        while (left_ptr <= right_ptr) {
            if (m_colIndices->getValue(left_ptr) == colNo) {
                // 如果左指针所指位置就是要写入值的位置，则写入后直接返回
                m_values[left_ptr] = val;
                return;
            }
            if (m_colIndices->getValue(right_ptr) == colNo) {
                // 如果右指针所指位置就是要写入值的位置，则写入后直接返回
                m_values[right_ptr] = val;
                return;
            }
            UINT32 mid_ptr = left_ptr + (right_ptr - left_ptr) / 2; // 直接计算right + left可能会有溢出
            UINT32 curr_mid_index = m_colIndices->getValue(mid_ptr);
            if (curr_mid_index == colNo) {
                // 如果中位指针所指位置就是要写入值的位置，则写入后直接返回
                m_values[mid_ptr] = val;
                return;
            } else if (curr_mid_index > colNo) {
                right_ptr = mid_ptr - 1;
            } else {
                left_ptr = mid_ptr + 1;
            }
        } // end while
        // 如果找不到元素
#ifndef NDEBUG
        std::cerr << YELLOW << __func__ << ": " << L_RED << "[ERROR] row: " << rowNo
                << "col: " << colNo << ", position does not exist!" << COLOR_NONE
                << std::endl;
#endif
        exit(EXIT_FAILURE);
    }

    template<typename ValType>
    void CSRMatrix<ValType>::setColsValsByRow(const UINT32 &rowNo, const UINT32 *const &colIndices,
                                              const ValType *const &vals, const UINT32 &valNum) {
#ifndef NDEBUG
        //现在在debug模式下，没填充过的行对应的rowOffset[rowNo+1]的值等于UINT32_MAX
        THROW_EXCEPTION(rowNo > 0 && (*m_rowOffset)[rowNo] == UINT32_MAX,
                        THROW_LOGIC_ERROR("The last step of setting values was not executed correctly! "
                            "Pleas make sure that you have correct row-offset of CSR matrix or have done the k-th "
                            "step before setting the k+1-th step of values."))
        ERROR_CHECK(rowNo >= this->m_rowNum, DEBUG_MESSEGE_OPTION, "The parameter \"rowNo\" out of range!");
        ERROR_CHECK(
            (*m_rowOffset)[rowNo + 1] < UINT32_MAX && (valNum != ((*m_rowOffset)[rowNo + 1] - (*m_rowOffset)[rowNo])),
            DEBUG_MESSEGE_OPTION, "The current memory length is not equal to data length!");
#endif
        UINT32 start = (*m_rowOffset)[rowNo];
        UINT32 end = start + valNum;
        (*m_rowOffset)[rowNo + 1] = end; // 更新当前行的rowOffset
#ifndef NWARN
        if (end > m_nnz) {
            SHOW_WARN("The current non-zero values\'s number is greater than the storage size. Try to resize it.")
            m_colIndices->resize(end, RESERVE_DATA);
            m_values->resize(end, RESERVE_DATA);
            m_nnz = end;
        }
#endif
        for (UINT32 i = 0, currIdx = start; i < valNum; ++i, ++currIdx) {
            (*m_colIndices)[currIdx] = colIndices[i]; // 将具体的列索引值拷贝进当前的colIndices数组
            (*m_values)[currIdx] = vals[i]; // 将具体的矩阵值拷贝进当前的values数组
        }
    }

    template<typename ValType>
    void CSRMatrix<ValType>::getValue(const UINT32 &rowNo, const UINT32 &colNo, ValType &val) {
#ifndef NDEBUG
        ERROR_CHECK(rowNo > m_rowNum - 1, DEBUG_MESSEGE_OPTION, "The parameter \"rowNo\" out of range!");
        ERROR_CHECK(colNo > m_colNum - 1, DEBUG_MESSEGE_OPTION, "The parameter \"colNo\" out of range!");
#endif
        // 二分查找
        UINT32 left_ptr = m_rowOffset[rowNo],
                right_ptr = m_rowOffset[rowNo + 1] - 1;
        while (left_ptr <= right_ptr) {
            if (m_colIndices->getValue(left_ptr) == colNo) {
                // 如果左指针所指位置就是要找的值，则找到后直接返回
                val = m_values->getValue(left_ptr);
                return;
            }
            if (m_colIndices->getValue(right_ptr) == colNo) {
                // 如果右指针所指位置就是要找的值，则找到后直接返回
                val = m_values->getValue(right_ptr);
                return;
            }
            UINT32 mid_ptr = left_ptr + (right_ptr - left_ptr) / 2; // 直接计算right + left可能会有溢出
            UINT32 curr_mid_index = m_colIndices->getValue(mid_ptr);
            if (curr_mid_index == colNo) {
                // 如果中位指针所指位置就是要找的数，则读出后直接返回
                val = m_values->getValue(mid_ptr);
                return;
            } else if (curr_mid_index > colNo) {
                right_ptr = mid_ptr - 1;
            } else {
                left_ptr = mid_ptr + 1;
            }
        } // end while
#ifndef NDEBUG
        std::cerr << YELLOW << __func__ << ": " << L_RED << "[ERROR] row: " << rowNo
                << "col: " << colNo << ", position does not exist!" << COLOR_NONE
                << std::endl;
#endif
        exit(-1);
    }

    template<typename ValType>
    void CSRMatrix<ValType>::getColsValsByRow(const UINT32 &rowNo, UINT32 *&colIndices, ValType *&vals,
                                              UINT32 &valNum) {
#ifndef NDEBUG
        ERROR_CHECK(vals != nullptr || colIndices != nullptr, DEBUG_MESSEGE_OPTION,
                    "The parameter \"vals\" must be a NULL pointer! Otherwise memory leak will happen.");
        ERROR_CHECK(rowNo > this->m_rowNum - 1, DEBUG_MESSEGE_OPTION,
                    "The parameter \"rowNo\" out of range!");
#endif
        UINT32 start = m_rowOffset->getValue(rowNo), end = m_rowOffset->getValue(rowNo + 1);
        valNum = end - start;
        vals = &m_values[start];
        colIndices = &m_colIndices[start];
    }

    template<typename ValType>
    CSRMatrix<ValType> &CSRMatrix<ValType>::operator=(CSRMatrix<ValType> &&pre_mat) noexcept {
        if (this == &pre_mat) // default操作，保证自我复制安全
            return *this;
        // 当前矩阵对象接管mat中的所有资源
        m_rowNum = pre_mat.m_rowNum;
        m_colNum = pre_mat.m_colNum;
        m_nnzCapacity = pre_mat.m_nnzCapacity;
        m_isFormated = pre_mat.m_isFormated;
        m_memoryType = pre_mat.m_memoryType;
        m_rowOffset = std::move(pre_mat.m_rowOffset);
        m_colIndices = std::move(pre_mat.m_colIndices);
        m_values = std::move(pre_mat.m_values);
        // 由于现有mat对象剥夺了原mat对象中的各存储变量的控制权，则原mat对象必须有相关重置操作
        pre_mat.m_rowNum = 0;
        pre_mat.m_colNum = 0;
        pre_mat.m_nnzCapacity = 0;
        return *this;
    }


    template<typename ValType>
    CSRMatrix<ValType> &CSRMatrix<ValType>::operator=(const CSRMatrix<ValType> &pre_mat) {
        if (&pre_mat == this)
            return *this;
        if (m_rowNum != pre_mat.m_rowNum) m_rowOffset->resize(pre_mat.m_rowNum + 1, RESERVE_NO_DATA);
        if (m_nnzCapacity != pre_mat.m_nnzCapacity) {
            m_colIndices->resize(pre_mat.m_nnzCapacity, RESERVE_NO_DATA);
            m_values->resize(pre_mat.m_nnzCapacity, RESERVE_NO_DATA);
        }
        m_rowNum = pre_mat.m_rowNum;
        m_colNum = pre_mat.m_colNum;
        m_nnzCapacity = pre_mat.m_nnzCapacity;
        m_isFormated = pre_mat.m_isFormated;
        m_rowOffset->copy(*pre_mat.m_rowOffset);
        m_colIndices->copy(*pre_mat.m_colIndices);
        m_values->copy(*pre_mat.m_values);
        return *this;
    }


    template<typename ValType>
    void CSRMatrix<ValType>::clear() {
#ifndef NINFO
        SHOW_INFO("Current space will be released.")
#endif
        m_rowNum = 0;
        m_colNum = 0;
        m_nnzCapacity = 0;
        // 只清除申请的内存空间，不删除AutoAllocateVector对象
        m_rowOffset->clear();
        m_colIndices->clear();
        m_values->clear();
    }

    template<typename ValType>
    void CSRMatrix<ValType>::resize(const UINT32 &new_rowNum, const UINT32 &new_colNum, const UINT32 &new_nnz,
                                    UINT8 need_reserve) {
        if (new_rowNum == m_rowNum && new_colNum == m_colNum && new_nnz == m_nnzCapacity && need_reserve) return;
        if (need_reserve == RESERVE_NO_DATA) {
            m_rowOffset->resize(new_rowNum + 1, RESERVE_NO_DATA);
            m_colIndices->resize(new_nnz, RESERVE_NO_DATA);
            m_values->resize(new_nnz, RESERVE_NO_DATA);
            if (m_rowOffset->getMemoryType() != memoryBase) m_rowOffset->fillVector(0, m_rowOffset->getLength(), 0);
#ifndef NDEBUG
            m_rowOffset->fillVector(1, m_rowOffset->getLength() - 1, UINT32_MAX);
#endif
        } else {
            UINT32 actual_rowNum = m_rowNum > new_rowNum ? new_rowNum : m_rowNum; ///< 实际有效的行数
#ifndef NWARN
            if (actual_rowNum == 0) SHOW_WARN("The actual row num is zero, but you choose the resize option is"
                                              "\"RESERVE_DATA\" which is useless.")
#endif
            m_rowOffset->resize(new_rowNum + 1, RESERVE_DATA);
            m_rowOffset->setValue(0, 0); // resize操作可能会丢弃原空间，所以rowOffset第一个数不一定为0，必须重新赋值
            if (new_colNum < m_colNum) {
                // 如果新的列数少于原列数，说明有非0元需要舍弃，可能需要调整rowOffset和colIndices
                UINT32 guessNum = m_rowOffset->getValue(actual_rowNum);
                AutoAllocateVector<UINT32> new_colIndices(guessNum, m_memoryType);
                AutoAllocateVector<ValType> new_values(guessNum, m_memoryType);
                // 遍历所有行，找出列索引超出范围的元素，并删除它们
                UINT32 j = 0, lower_bound = 0, upper_bound;
                for (UINT32 i = 0; i < actual_rowNum; ++i) {
                    upper_bound = m_rowOffset->getValue(i + 1);
                    UINT32 k = lower_bound;
                    // 如果当前行有元素的列索引值超出列边界，则直接舍去
                    while (k < upper_bound) {
                        UINT32 val_index = m_colIndices[k];
                        if (val_index < new_colNum) {
                            new_colIndices[j] = val_index;
                            new_values[j] = m_values[k];
                            j++;
                        }
                        k++;
                    }
                    lower_bound = upper_bound;
                    m_rowOffset[i + 1] = j; // 根据实际保留的每行非0元个数，重新更新row_offset
                }
                m_colIndices->move(*new_colIndices);
                m_values->move(*new_values);
            }
#ifndef NDEBUG
            ERROR_CHECK(
                new_nnz < m_rowOffset->getValue(actual_rowNum) && m_rowOffset->getValue(actual_rowNum) != UINT32_MAX,
                DEBUG_MESSEGE_OPTION, "The new none-zero number size is incorrect!");
#endif
            m_colIndices->resize(new_nnz, RESERVE_DATA);
            m_values->resize(new_nnz, RESERVE_DATA);
            if (new_rowNum > actual_rowNum) {
                // 如果是扩充row，新行对应的所有rowOffset值即为原rowOffset最后一个值
                m_rowOffset->fillVector(actual_rowNum + 1, new_rowNum - actual_rowNum,
                                        m_rowOffset->getValue(actual_rowNum));
            }
        }
        this->m_rowNum = new_rowNum;
        this->m_colNum = new_colNum;
        this->m_nnzCapacity = new_nnz;
    }

    template<typename ValType>
    void CSRMatrix<ValType>::getSubMatrixWithoutCopy(const UINT32 &startRowNo, const UINT32 &endRowNo,
                                                     const UINT32 &startColNo, const UINT32 &endColNo,
                                                     HostVector<UINT32> &nnzStartPosForEachRow,
                                                     HostVector<UINT32> &nnzNumForEachRow) {
        // 一些debug处理
#ifndef NDEUBG
        ERROR_CHECK(startRowNo > endRowNo || startColNo > endColNo, DEBUG_MESSEGE_OPTION,
                    "The range of sub-matrix is incorrect!");
        ERROR_CHECK(startRowNo > m_rowNum || startColNo > m_colNum, DEBUG_MESSEGE_OPTION,
                    "The range of sub-matrix is out-of-range!");
#endif
        if (m_isFormated == 0) formatStructure();
        // 调整标志向量的大小到合适的范围
        UINT32 subMatRowNum = endRowNo - startRowNo + 1;
        nnzStartPosForEachRow.resize(subMatRowNum, RESERVE_NO_DATA);
        nnzNumForEachRow.resize(subMatRowNum, RESERVE_NO_DATA);
        nnzNumForEachRow.fillVector(0, subMatRowNum, 0); // 计数变量清零
        // 先提取子矩阵元素所在的大概区间，随后再根据列号剔除不在范围内的值
        UINT32 subStart, subEnd, colIdx;
        for (UINT32 rowIdx = startRowNo, rowNo = 0; rowIdx <= endRowNo; ++rowIdx, ++rowNo) {
            subStart = m_rowOffset->getValue(rowIdx);
            subEnd = m_rowOffset->getValue(rowIdx + 1) - 1;
            // 先正序遍历当前行，发现第一个符合范围的非零元，记录当前下标
            while (subStart <= subEnd) {
                colIdx = m_colIndices->getValue(subStart);
                if (colIdx >= startColNo && colIdx <= endColNo) {
                    break;
                }
                subStart++;
            }
            nnzStartPosForEachRow[rowNo] = subStart;
            // 再反序遍历当前非零元，找到第一个符合范围的非零元，记录下标，用于统计当前行中属于子矩阵的非零元数量
            while (subEnd >= subStart) {
                colIdx = m_colIndices->getValue(subEnd);
                if (colIdx >= startColNo && colIdx <= endColNo) {
                    break;
                }
                subEnd--;
            }
            nnzNumForEachRow[rowNo] = subStart > subEnd ? 0 : subEnd - subStart + 1;
        }
    }


    template<typename ValType>
    void CSRMatrix<ValType>::getSubMatrix(const UINT32 &startRowNo, const UINT32 &endRowNo, const UINT32 &startColNo,
                                          const UINT32 &endColNo, CSRMatrix<ValType> &outMat) {
        // 一些debug处理
#ifndef NDEUBG
        ERROR_CHECK(startRowNo > endRowNo || startColNo > endColNo, DEBUG_MESSEGE_OPTION,
                    "The range of sub-matrix is incorrect!");
        ERROR_CHECK(startRowNo > m_rowNum || startColNo > m_colNum, DEBUG_MESSEGE_OPTION,
                    "The range of sub-matrix is out-of-range!");
#endif
        DenseVector<UINT32> nnzStartPosForEachRow, nnzNumForEachRow;
        getSubMatrixWithoutCopy(startRowNo, endRowNo, startColNo, endColNo, nnzStartPosForEachRow, nnzNumForEachRow);
        UINT32 totalNNZnum = nnzNumForEachRow.sum(0, nnzNumForEachRow.getLength());
        outMat.resize(endRowNo - startRowNo + 1, endColNo - startColNo + 1, totalNNZnum, RESERVE_NO_DATA);
        // 拷贝子矩阵的colIndices和values，在拷贝过程中生成子矩阵的rowOffset
        AutoAllocateVector<UINT32> tempColIdx(endColNo - startColNo + 1, memoryBase);
        for (UINT32 rowIdx = 0; rowIdx < outMat.m_rowNum; ++rowIdx) {
            UINT32 pos = nnzStartPosForEachRow[rowIdx], num = nnzNumForEachRow[rowIdx];
            const UINT32 *cols = &m_colIndices[pos];
            const ValType *vals = &m_values[pos];
            // 将原始矩阵的列索引转换为子矩阵的列索引
            if (startColNo == 0) {
                // 如果子矩阵列范围从0开始，那就无需做列索引转换
                outMat.setColsValsByRow(rowIdx, cols, vals, num);
            } else {
                // 如果子矩阵在原始矩阵中的范围不是从0开始，就需要转换，不能将原列索引复制到子矩阵中
                UINT32 bound = std::min(tempColIdx->getLength(), num);
                for (UINT32 colIdx = 0; colIdx < bound; ++colIdx) {
                    tempColIdx[colIdx] = cols[colIdx] - startColNo;
                }
                outMat.setColsValsByRow(rowIdx, &tempColIdx[0], vals, num);
            }
        }
    }


    template<typename ValType>
    void CSRMatrix<ValType>::getSubMatrix(const HostVector<UINT32> &rowPerm, const HostVector<UINT32> &colPerm,
                                          CSRMatrix<ValType> &outMat) const {
        outMat.m_isFormated = 0;
        UINT32 subMatRowNum = rowPerm.getLength();
        UINT32 subMatColNum = colPerm.getLength();
#ifndef NDBUG
        ERROR_CHECK(subMatRowNum > m_rowNum || subMatColNum > m_colNum, DEBUG_MESSEGE_OPTION,
                    "The range of permutation is incorrect!");
#endif
        UINT32 maxNNZnum = 0;
        for (UINT32 i = 0; i < rowPerm.getLength(); ++i) {
            maxNNZnum += m_rowOffset[rowPerm[i] + 1] - m_rowOffset[rowPerm[i]];
        }
        outMat.resize(subMatRowNum, subMatColNum, maxNNZnum, RESERVE_NO_DATA);

        // 根据colPerm，计算子矩阵非零元列号对照表
        std::unique_ptr<UINT32[]> colIdxTable = std::make_unique<UINT32[]>(m_colNum);
        std::fill_n(colIdxTable.get(), m_colNum, UINT_MAX);
        for (UINT32 idx = 0; idx < subMatColNum; ++idx) colIdxTable[colPerm[idx]] = idx;
        UINT32 rowIdx, newColIdx, start, end, nnzCount;
        nnzCount = 0;
        for (UINT32 idx = 0; idx < subMatRowNum; ++idx) {
            rowIdx = rowPerm[idx];
#ifndef NDEBUG
            ERROR_CHECK(rowIdx >= m_rowNum, DEBUG_MESSEGE_OPTION, "The row index is out-of-range!");
#endif
            start = m_rowOffset->getValue(rowIdx);
            end = m_rowOffset->getValue(rowIdx + 1);
            for (UINT32 colIdx = start; colIdx < end; ++colIdx) {
                newColIdx = colIdxTable[m_colIndices->getValue(colIdx)];
                if (newColIdx == UINT_MAX) continue;
                (*outMat.m_colIndices)[nnzCount] = newColIdx;
                (*outMat.m_values)[nnzCount] = m_values->getValue(colIdx);
                nnzCount++;
            }
            (*outMat.m_rowOffset)[idx + 1] = nnzCount;
        }
    }

    template<typename ValType>
    void CSRMatrix<ValType>::getSubMatrix(const UINT32 *rowPermPtr, UINT32 rowPermLength, const UINT32 *colPermPtr,
                                          UINT32 colPermLength, CSRMatrix<ValType> &outMat) const {
        outMat.m_isFormated = 0;
#ifndef NDBUG
        ERROR_CHECK(rowPermLength > m_rowNum || colPermLength > m_colNum, DEBUG_MESSEGE_OPTION,
                    "The range of permutation is incorrect!");
#endif
        UINT32 maxNNZnum = 0;
        for (UINT32 i = 0; i < rowPermLength; ++i) {
            maxNNZnum += m_rowOffset[rowPermPtr[i] + 1] - m_rowOffset[rowPermPtr[i]];
        }
        outMat.resize(rowPermLength, colPermLength, maxNNZnum, RESERVE_NO_DATA);

        // 根据colPerm，计算子矩阵非零元列号对照表
        std::unique_ptr<UINT32[]> colIdxTable = std::make_unique<UINT32[]>(m_colNum);
        std::fill_n(colIdxTable.get(), m_colNum, UINT_MAX);
        for (UINT32 idx = 0; idx < colPermLength; ++idx) colIdxTable[colPermPtr[idx]] = idx;
        UINT32 rowIdx, newColIdx, start, end, nnzCount;
        nnzCount = 0;
        for (UINT32 idx = 0; idx < rowPermLength; ++idx) {
            rowIdx = rowPermPtr[idx];
#ifndef NDEBUG
            ERROR_CHECK(rowIdx >= m_rowNum, DEBUG_MESSEGE_OPTION, "The row index is out-of-range!");
#endif
            start = m_rowOffset->getValue(rowIdx);
            end = m_rowOffset->getValue(rowIdx + 1);
            for (UINT32 colIdx = start; colIdx < end; ++colIdx) {
                newColIdx = colIdxTable[m_colIndices->getValue(colIdx)];
                if (newColIdx == UINT_MAX) continue;
                (*outMat.m_colIndices)[nnzCount] = newColIdx;
                (*outMat.m_values)[nnzCount] = m_values->getValue(colIdx);
                nnzCount++;
            }
            (*outMat.m_rowOffset)[idx + 1] = nnzCount;
        }
    }

    template<typename ValType>
    void CSRMatrix<ValType>::getSubMatrix(const UINT32 &startRowNo, const UINT32 &endRowNo,
                                          const HostVector<UINT32> &colOffset, CSRMatrix<ValType> &outMat) {
#ifndef NDEBUG
        ERROR_CHECK(startRowNo > endRowNo, DEBUG_MESSEGE_OPTION,
                    "The start number of rows was greater than the end of rows!");
        ERROR_CHECK(startRowNo >= m_rowNum || endRowNo >= m_rowNum, DEBUG_MESSEGE_OPTION,
                    "The start row number was out-of-range!");
        ERROR_CHECK(colOffset.getLength() <= 1 || colOffset.getLength() > m_colNum + 1, DEBUG_MESSEGE_OPTION,
                    "The length of col-offset was incorrect!");
        bool isAscending = true;
        for (UINT32 i = 1; i < colOffset.getLength(); ++i)
            if (colOffset[i] < colOffset[i - 1]) {
                isAscending = false;
                break;
            }
        // 保证列分割点是从小到大排列的
        ERROR_CHECK(!isAscending, DEBUG_MESSEGE_OPTION, "The col-offset vector should be in ascending order!");
#endif //NDEBUG
        // 如果当前矩阵不是规范形式的CSR矩阵，则先调整为规范形式（每行非零元按列索引从小到大排列），方便后续切块
        if (m_isFormated == 0) formatStructure();
        UINT32 colOffsetSize = colOffset.getLength();
        UINT32 maxSubMatNNZnum = m_rowOffset[endRowNo + 1] - m_rowOffset[startRowNo];
        UINT32 subMatRowNum = endRowNo - startRowNo + 1;

        // 开始划分分区
        outMat.resize(subMatRowNum * (colOffsetSize - 1), m_colNum, maxSubMatNNZnum, RESERVE_NO_DATA);
        AutoAllocateVector<UINT32> startPos(subMatRowNum, memoryBase); ///< 保存子块当前分区非零元实际存储起始下标
        for (UINT32 rowIdx = 0; rowIdx < subMatRowNum; ++rowIdx) {
            startPos[rowIdx] = m_rowOffset[startRowNo + rowIdx];
        }
        UINT32 currStartPos, ///< 记录当前访问的位于原始矩阵中的非零元实际存储下标
                currUpperBound, ///< 当前子矩阵列范围上界
                currLowerBound = colOffset[0], ///< 当前子矩阵列范围下界
                currPos, ///< 用于扫描当前行非零元的一个访问下标
                currSubRowIdx = 0; ///< 实际存储子矩阵中已成功写入非零元的行数
        UINT32 nnzIdx = 0;
        UINT32 nextRowStartPos;
        // 遍历每一个分割符（子块），并将值写入到最终的存储矩阵中
        for (UINT32 separatorIdx = 1; separatorIdx < colOffsetSize; ++separatorIdx) {
            // 记录子块逻辑上的结束边界
            currUpperBound = colOffset[separatorIdx];
            for (UINT32 rowIdx = 0; rowIdx < subMatRowNum; ++rowIdx) {
                currStartPos = startPos[rowIdx];
                // 找到列索引位于当前分割区间之间的所有非零元
                nextRowStartPos = m_rowOffset[startRowNo + rowIdx + 1];
                for (currPos = currStartPos; currPos < nextRowStartPos; ++currPos) {
                    if (m_colIndices[currPos] >= currUpperBound) break;
                    outMat.m_colIndices[nnzIdx] = m_colIndices[currPos] - currLowerBound;
                    outMat.m_values[nnzIdx] = m_values[currPos];
                    nnzIdx++;
                }
                // 更新输出矩阵的行偏移
                outMat.m_rowOffset[currSubRowIdx + 1] = nnzIdx;
                currSubRowIdx++;
                startPos[rowIdx] = currPos; // 更新下一个子块的实际存储起始边界
            }
            // 更新子块的起始边界
            currLowerBound = currUpperBound;
        }
    }

    template<typename ValType>
    void CSRMatrix<ValType>::getSubMatrix(const UINT32 &startRowNo, const UINT32 &endRowNo,
                                          const HostVector<UINT32> &colOffset,
                                          UniquePtr1D<HOST::CSRMatrix<ValType> > &outMat,
                                          const memoryType_t &outMatType) {
#ifndef NDEBUG
        THROW_EXCEPTION(startRowNo > endRowNo,
                        THROW_LOGIC_ERROR("The start number of rows was greater than the end of rows!"))
        THROW_EXCEPTION(startRowNo >= m_rowNum || endRowNo >= m_rowNum,
                        THROW_OUT_OF_RANGE("The start row number was out-of-range!"))
        THROW_EXCEPTION(colOffset.getLength() <= 1 || colOffset.getLength() > m_colNum + 1,
                        THROW_LOGIC_ERROR("The length of col-offset was incorrect!"))
        bool isAscending = true;
        for (UINT32 i = 1; i < colOffset.getLength(); ++i)
            if (colOffset[i] < colOffset[i - 1]) {
                isAscending = false;
                break;
            }
        // 保证列分割点是从小到大排列的
        THROW_EXCEPTION(!isAscending, THROW_LOGIC_ERROR("The col-offset vector should be in ascending order!"))
#endif //NDEBUG
        // 如果当前矩阵不是规范形式的CSR矩阵，则先调整为规范形式（每行非零元按列索引从小到大排列），方便后续切块
        if (m_isFormated == 0) formatStructure();
        UINT32 colOffsetSize = colOffset.getLength();
        UINT32 maxSubMatNNZnum = m_rowOffset[endRowNo + 1] - m_rowOffset[startRowNo];
        UINT32 subMatRowNum = endRowNo - startRowNo + 1;

        // 开始划分分区，调整块CSR大小，预分配内存
        outMat.realloc(colOffsetSize - 1);
        UINT32 tempColDim, tempNNZnum = (maxSubMatNNZnum + outMat.getDim() - 1) / outMat.getDim(); // 均分时向上取整
        // 由于所有子块的非零元个数暂时是一个预估值，后续可能需要扩大空间，则这里需要计算一个扩容参数
        UINT32 expandSize = (tempNNZnum + 1) / 2;
        expandSize = expandSize < 1 ? 1 : expandSize;
        for (INT32 idx = 0; idx < outMat.getDim(); ++idx) {
            tempColDim = colOffset[idx + 1] - colOffset[idx];
            outMat[idx] = std::make_unique<CSRMatrix<ValType> >(subMatRowNum, tempColDim, tempNNZnum, outMatType);
        }
        AutoAllocateVector<UINT32> startPos(subMatRowNum, memoryBase); ///< 保存子块当前分区非零元实际存储起始下标
        for (UINT32 rowIdx = 0; rowIdx < subMatRowNum; ++rowIdx) {
            startPos[rowIdx] = m_rowOffset[startRowNo + rowIdx];
        }
        UINT32 currStartPos, ///< 记录当前访问的位于原始矩阵中的非零元实际存储下标
                currUpperBound, ///< 当前子矩阵列范围上界
                currLowerBound = colOffset[0], ///< 当前子矩阵列范围下界
                currPos, ///< 用于扫描当前行非零元的一个访问下标
                currNNZnum, ///< 当前子块中的非零元个数
                nnzIdx; ///< 当前待插入子矩阵的非零元在子矩阵实际存储向量中的下标
        UINT32 nextRowStartPos, blockIdx;
        // 遍历每一个分割符（子块），并将值写入到最终的存储矩阵中
        for (UINT32 separatorIdx = 1; separatorIdx < colOffsetSize; ++separatorIdx) {
            // 记录子块逻辑上的结束边界
            nnzIdx = 0;
            currUpperBound = colOffset[separatorIdx];
            for (UINT32 rowIdx = 0; rowIdx < subMatRowNum; ++rowIdx) {
                currStartPos = startPos[rowIdx];
                // 找到列索引位于当前分割区间之间的所有非零元
                nextRowStartPos = m_rowOffset[startRowNo + rowIdx + 1];
                blockIdx = separatorIdx - 1;
                UINT32 currValColIdx;
                for (currPos = currStartPos; currPos < nextRowStartPos; ++currPos) {
                    currValColIdx = m_colIndices[currPos];
                    if (currValColIdx >= currUpperBound) break;
                    if (currValColIdx >= currLowerBound) {
                        // 当前非零元在子区间范围内时，加入子矩阵
                        currNNZnum = outMat[blockIdx]->getValNum();
                        if (nnzIdx >= currNNZnum) {
                            outMat[blockIdx]->resize(subMatRowNum, outMat[blockIdx]->getColNum(),
                                                     currNNZnum + expandSize,
                                                     RESERVE_DATA);
                        }
                        outMat[blockIdx]->m_colIndices[nnzIdx] = m_colIndices[currPos] - currLowerBound;
                        outMat[blockIdx]->m_values[nnzIdx] = m_values[currPos];
                        nnzIdx++;
                    }
                }
                // 更新输出矩阵的行偏移
                outMat[blockIdx]->m_rowOffset[rowIdx + 1] = nnzIdx;
                startPos[rowIdx] = currPos; // 更新下一个子块的实际存储起始边界
            }
            // 更新子块的起始边界
            currLowerBound = currUpperBound;
        }
    }

    template<typename ValType>
    void CSRMatrix<ValType>::formatStructure() {
        if (m_isFormated) return;
        if (m_rowNum == 0 && m_colNum == 0 && m_nnzCapacity == 0) return;
        UINT32 *sortListPtr1 = &m_colIndices[0];
        ValType *sortListPtr2 = &m_values[0];
        UINT32 currNNZ;
        for (UINT32 rowIdx = 0; rowIdx < m_rowNum; ++rowIdx) {
            currNNZ = m_rowOffset->getValue(rowIdx + 1) - m_rowOffset->getValue(rowIdx);
            sortVectorPair(sortListPtr1, sortListPtr2, currNNZ);
            sortListPtr1 += currNNZ;
            sortListPtr2 += currNNZ;
        }
        m_isFormated = 1;
    }

    template<typename ValType>
    void CSRMatrix<ValType>::pivotReorderByRow(HostVector<UINT32> &colPerm) {
        colPerm.resize(m_colNum, RESERVE_NO_DATA);
        /* 遍历所有行，将当前行中绝对值较大的非零元交换到对角线上 */
        UINT32 currRowStartIdx, currRowEndIdx, currMaxNNZPos;
        ValType currMaxNNZ;
        // 记录哪些列已经交换到正确的位置，0表示未匹配，1表示匹配
        std::unique_ptr<UINT8[]> isFixed = std::make_unique<UINT8[]>(m_colNum);
        std::fill_n(isFixed.get(), m_colNum, 0);

        // 先遍历一遍所有非零元，记录交换情况
        UINT8 isDiagFound;
        FLOAT64 tnorm;
        FLOAT64 tol = 1e-8;
        ValType diagVal;
        for (UINT32 i = 0; i < m_rowNum; ++i) {
            currRowStartIdx = m_rowOffset[i];
            currRowEndIdx = m_rowOffset[i + 1];
            currMaxNNZ = 0;
            isDiagFound = 0;
            tnorm = 0.0;
            for (UINT32 j = currRowStartIdx; j < currRowEndIdx; ++j) {
                tnorm += fabs(m_values[j]);
                if (m_colIndices[j] == i) {
                    isDiagFound = 1;
                    diagVal = m_values[j];
                }
                if (fabs(m_values[j]) >= fabs(currMaxNNZ) && !isFixed[m_colIndices[j]]) {
                    // 如果当前的非零元是最大元，且该非零元所在的列还未交换过，则更新对应变量
                    currMaxNNZPos = j;
                    currMaxNNZ = m_values[j];
                }
            }
#ifndef NDEBUG
            THROW_EXCEPTION(isFixed[m_colIndices[currMaxNNZPos]], THROW_LOGIC_ERROR("The curr col was fixed twice!"))
#endif
            // 如果当前行的主元不满足要求，交换主元所在列和最大元所在列，并更新置换数组
            if (isDiagFound && fabs(diagVal) >= tnorm * tol) {
                colPerm[i] = i;
                isFixed[i] = 1;
            } else {
                colPerm[m_colIndices[currMaxNNZPos]] = i;
                // 更新列放置状态
                isFixed[m_colIndices[currMaxNNZPos]] = 1;
            }
        }
        // 更新置换结果
        for (UINT32 i = 0; i < m_rowNum; ++i) {
            currRowStartIdx = m_rowOffset[i];
            currRowEndIdx = m_rowOffset[i + 1];
            for (UINT32 j = currRowStartIdx; j < currRowEndIdx; ++j) {
                m_colIndices[j] = colPerm[m_colIndices[j]];
            }
        }
        UINT8 isDiagNonZero;
        // 检查主元
        for (UINT32 i = 0; i < m_rowNum; ++i) {
            currRowStartIdx = m_rowOffset[i];
            currRowEndIdx = m_rowOffset[i + 1];
            isDiagNonZero = 0;
            for (UINT32 j = currRowStartIdx; j < currRowEndIdx; ++j) {
                if (m_colIndices[j] == i && fabs(m_values[j]) > 0) {
                    isDiagNonZero = 1;
                    break;
                }
            }
            if (!isDiagNonZero) {
                SHOW_ERROR("Zero diagonal element occurred in mat at row: " << i)
                SHOW_ERROR("Curr nnz num: " << currRowEndIdx - currRowStartIdx)
                std::cout << "[INFO] curr line:";
                for (UINT32 j = currRowStartIdx; j < currRowEndIdx; ++j) {
                    std::cout << " " << m_values[j] << "(" << m_colIndices[j] << ")";
                }
            }
            THROW_EXCEPTION(!isDiagNonZero, THROW_LOGIC_ERROR("Zero diagonal encountered!"))
        }
    }


    template<typename ValType>
    void CSRMatrix<ValType>::pivotReorder(const HostVector<UINT32> &rowPerm, HostVector<UINT32> &colPerm) {
        colPerm.resize(m_colNum, RESERVE_NO_DATA);
        UINT32 subMatNNZnum = 0;
        for (UINT32 i = 0; i < rowPerm.getLength(); ++i) {
            subMatNNZnum += m_rowOffset[rowPerm[i] + 1] - m_rowOffset[rowPerm[i]];
        }
        AutoAllocateVector<UINT32> tempOffset(rowPerm.getLength() + 1, m_memoryType);
        AutoAllocateVector<UINT32> tempColIdx(subMatNNZnum, m_memoryType);
        AutoAllocateVector<ValType> tempVal(subMatNNZnum, m_memoryType);
        UINT32 nnzCount = 0, rowIdx, start, end;
        for (UINT32 idx = 0; idx < rowPerm.getLength(); ++idx) {
            rowIdx = rowPerm[idx];
#ifndef NDEBUG
            ERROR_CHECK(rowIdx >= m_rowNum, DEBUG_MESSEGE_OPTION, "The row index is out-of-range!");
#endif
            start = m_rowOffset->getValue(rowIdx);
            end = m_rowOffset->getValue(rowIdx + 1);
            for (UINT32 colIdx = start; colIdx < end; ++colIdx) {
                if (fabs(m_values.getValue(colIdx)) == 0) {
                    SHOW_ERROR("Curr val is zero! row: " << idx << ", col: " << m_colIndices->getValue(colIdx))
                    exit(-1);
                }
                tempColIdx[nnzCount] = m_colIndices->getValue(colIdx);
                tempVal[nnzCount] = m_values->getValue(colIdx);
                nnzCount++;
            }
            tempOffset[idx + 1] = nnzCount;
        }
        m_rowOffset = std::move(tempOffset);
        m_colIndices = std::move(tempColIdx);
        m_values = std::move(tempVal);
        m_rowNum = rowPerm.getLength();
        pivotReorderByRow(colPerm);
        m_isFormated = 0;
    }
}
