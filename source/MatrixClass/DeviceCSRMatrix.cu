/*
 * @author  邓轶丹
 * @date    2024/5/4
 * @details 实现在GPU上的基于CSR的相关操作函数
 */

#include "../../include/MatrixClass/DeviceCSRMatrix.cuh"

namespace DEVICE {
    template<typename ValType>
    DeviceCSRMatrix<ValType>::DeviceCSRMatrix(const UINT32 &rowNum, const UINT32 &colNum, const UINT32 &nnz,
                                              const INT32 &deviceID) : m_rowNum(rowNum), m_colNum(colNum), m_nnz(nnz),
                                                                       m_deviceID(deviceID) {
#ifndef NINFO
        SHOW_INFO("Allocating Device CSR matrix...")
#endif
        INT32 status = 0;
        status |= m_row_offset.initSpace(rowNum + 1, deviceID);
        status |= m_col_indices.initSpace(nnz, deviceID);
        status |= m_val.initSpace(nnz, deviceID);
#ifndef NDEBUG
        if (status != CUDA_SUCCESS) {
            SHOW_ERROR("The allocation for device CSR matrix failed!")
            exit(memoryAllocationFailed);
        }
#endif
#ifndef NINFO
        SHOW_INFO("Device CSR matrix allocation success!")
#endif
    }

    template<typename ValType>
    DeviceCSRMatrix<ValType>::DeviceCSRMatrix(const INT32 &deviceID) {
        m_deviceID = deviceID;
        CHECK_CUDA(m_row_offset.initSpace(1, deviceID))
        CHECK_CUDA(m_col_indices.initSpace(0, deviceID))
        CHECK_CUDA(m_val.initSpace(0, deviceID))
    }


    template<typename ValType>
    void DeviceCSRMatrix<ValType>::resetAndClearMembers(const UINT32 &new_row, const UINT32 &new_col,
                                                        const UINT32 &new_nnz) {
        if (m_rowNum != new_row) m_rowNum = new_row;
        if (m_colNum != new_col) m_colNum = new_col;
        if (m_nnz != new_nnz) m_nnz = new_nnz;
        if (m_row_offset.getLength() != m_rowNum + 1) {
            m_row_offset.clear();
            m_row_offset.initSpace(m_rowNum + 1, m_deviceID);
        }
        if (m_col_indices.getLength() != m_nnz) {
            m_col_indices.clear();
            m_col_indices.initSpace(new_nnz, m_deviceID);
        }
        if (m_val.getLength() != m_nnz) {
            m_val.clear();
            m_val.initSpace(new_nnz, m_deviceID);
        }
    }


    template<typename ValType>
    void DeviceCSRMatrix<ValType>::copyMatFromHost(HOST::CSRMatrix<ValType> &host_mat) {
        if (host_mat.getRowNum() == 0) {
            resetAndClearMembers(0, 0, 0);
            return;
        }
        UINT32 currNNZnum = host_mat.getNNZnum(0, host_mat.getRowNum() - 1);
        resetAndClearMembers(host_mat.getRowNum(), host_mat.getColNum(), currNNZnum);

        const UINT32 *host_rowOffset_ptr = host_mat.getRowOffsetPtr(0);
        const UINT32 *host_colIndices_ptr = host_mat.getColIndicesPtr(0);
        const ValType *host_val_ptr = host_mat.getCSRValuesPtr(0);

#ifndef NTIME
        std::cout << L_BLUE << "[INFO] Copy CSR Matrix from Host to Device:" << COLOR_NONE << std::endl;
        GPU_TIMER_FUNC(copyMatFromHost)
        GPU_TIMER_BEGIN(copyMatFromHost)
#endif
        m_row_offset.copyFromHost(host_rowOffset_ptr, 0, m_rowNum + 1);
#ifndef NTIME
        GPU_TIMER_END(copyMatFromHost)
        std::cout << L_GREEN << "--- copy row_offset executes: " << COLOR_NONE << GPU_EXEC_TIME(copyMatFromHost)
                << " ms." << std::endl;

        GPU_TIMER_BEGIN(copyMatFromHost)
#endif
        m_col_indices.copyFromHost(host_colIndices_ptr, 0, m_nnz);
#ifndef NTIME
        GPU_TIMER_END(copyMatFromHost)
        std::cout << L_GREEN << "--- copy col_indices executes: " << COLOR_NONE << GPU_EXEC_TIME(copyMatFromHost)
                << " ms." << std::endl;

        GPU_TIMER_BEGIN(copyMatFromHost)
#endif
        m_val.copyFromHost(host_val_ptr, 0, m_nnz);
#ifndef NTIME
        GPU_TIMER_END(copyMatFromHost)
        std::cout << L_GREEN << "--- copy mat_val executes: " << COLOR_NONE << GPU_EXEC_TIME(copyMatFromHost)
                << " ms." << std::endl;
#endif
    }


    template<typename ValType>
    void DeviceCSRMatrix<ValType>::copyMatToHost(HOST::CSRMatrix<ValType> &host_mat) {
#ifndef NWARN
        if (m_row_offset.getLength() == 0 || m_col_indices.getLength() == 0 || m_val.getLength() == 0) {
            SHOW_WARN("The matrix memory space was not allocated, no data copied.")
            return;
        }
#endif
        if (host_mat.getRowNum() != m_rowNum || host_mat.getColNum() != m_colNum || host_mat.getValNum() != m_nnz)
            host_mat.resize(m_rowNum, m_colNum, m_nnz, RESERVE_NO_DATA);

        // 将outMat中已分配好的内部空间转移到外部，方便赋值
        HOST::MovePrivateVector<CSR_MATRIX(ValType), UINT32> temp_offsetCSR(host_mat,
                                                                            &CSR_MATRIX(ValType)::moveRowOffsetTo,
                                                                            &CSR_MATRIX(ValType)::moveRowOffsetFrom);
        HOST::MovePrivateVector<CSR_MATRIX(ValType), UINT32> temp_colIndCSR(host_mat,
                                                                            &CSR_MATRIX(ValType)::moveColIndicesTo,
                                                                            &CSR_MATRIX(
                                                                                ValType)::moveColIndicesFrom);
        HOST::MovePrivateVector<CSR_MATRIX(ValType), ValType> temp_valCSR(host_mat,
                                                                          &CSR_MATRIX(ValType)::moveValuesTo,
                                                                          &CSR_MATRIX(ValType)::moveValuesFrom);

        UINT32 *host_rowOffset_ptr = &temp_offsetCSR[0];
        UINT32 *host_colIndices_ptr = &temp_colIndCSR[0];
        ValType *host_val_ptr = &temp_valCSR[0];
        std::cout << L_BLUE << "[INFO] Copy CSR Matrix from Device to Host:" << COLOR_NONE << std::endl;
        GPU_TIMER_FUNC(copyVecToHost)
        GPU_TIMER_BEGIN(copyVecToHost)
        m_row_offset.copyToHost(host_rowOffset_ptr, 0, m_rowNum + 1);
        GPU_TIMER_END(copyVecToHost)
        std::cout << L_GREEN << "--- copy row_offset executes: " << COLOR_NONE << GPU_EXEC_TIME(copyVecToHost)
                << " ms." << std::endl;
        GPU_TIMER_BEGIN(copyVecToHost)
        m_col_indices.copyToHost(host_colIndices_ptr, 0, m_nnz);
        GPU_TIMER_END(copyVecToHost)
        std::cout << L_GREEN << "--- copy col_indices executes: " << COLOR_NONE << GPU_EXEC_TIME(copyVecToHost)
                << " ms." << std::endl;
        GPU_TIMER_BEGIN(copyVecToHost)
        m_val.copyToHost(host_val_ptr, 0, m_nnz);
        GPU_TIMER_END(copyVecToHost)
        std::cout << L_GREEN << "--- copy mat_val executes: " << COLOR_NONE << GPU_EXEC_TIME(copyVecToHost)
                << " ms." << std::endl;
    }


    template<typename ValType>
    void DeviceCSRMatrix<ValType>::asyncCopyFromHost(HOST::CSRMatrix<ValType> &host_mat,
                                                     const StreamController &stream) {
#ifndef NWARN
        if (host_mat.getMemoryType() != memoryPageLocked) {
            host_mat.resetMemoryType(memoryPageLocked, RESERVE_DATA);
        }
#endif
        UINT32 currNNZnum = host_mat.getNNZnum(0, host_mat.getRowNum() - 1);
        if (host_mat.getRowNum() == 0 || currNNZnum == 0) {
            // 如果没有有效值需要拷贝，则直接返回
            resetAndClearMembers(0, 0, 0);
            return;
        }
        resetAndClearMembers(host_mat.getRowNum(), host_mat.getColNum(), currNNZnum);
        HOST::MovePrivateVector<HOST::CSRMatrix<ValType>, UINT32> tempOffset(host_mat,
                                                                             &HOST::CSRMatrix<ValType>::moveRowOffsetTo,
                                                                             &HOST::CSRMatrix<
                                                                                 ValType>::moveRowOffsetFrom);
        HOST::MovePrivateVector<HOST::CSRMatrix<ValType>, UINT32> tempColIndices(host_mat,
            &HOST::CSRMatrix<ValType>::moveColIndicesTo,
            &HOST::CSRMatrix<ValType>::moveColIndicesFrom);
        HOST::MovePrivateVector<HOST::CSRMatrix<ValType>, ValType> tempVal(host_mat,
                                                                           &HOST::CSRMatrix<ValType>::moveValuesTo,
                                                                           &HOST::CSRMatrix<ValType>::moveValuesFrom);
        m_row_offset.asyncCopyFromHost(*tempOffset, 0, 0, m_row_offset.getLength(), stream);
        m_col_indices.asyncCopyFromHost(*tempColIndices, 0, 0, m_nnz, stream);
        m_val.asyncCopyFromHost(*tempVal, 0, 0, m_nnz, stream);
    }

    template<typename ValType>
    void DeviceCSRMatrix<ValType>::asyncCopyToHost(HOST::CSRMatrix<ValType> &host_mat, const StreamController &stream) {
#ifndef NWARN
        if (host_mat.getMemoryType() != memoryPageLocked) {
            SHOW_WARN("Async-copy need host matrix to be stored in page-locked type!")
            host_mat.resetMemoryType(memoryPageLocked, RESERVE_NO_DATA);
        }
#endif
        host_mat.resize(m_rowNum, m_colNum, m_nnz, RESERVE_NO_DATA);
        if (m_rowNum == 0 || m_nnz == 0) return; // 没有需要拷贝的数据，直接返回
        HOST::MovePrivateVector<HOST::CSRMatrix<ValType>, UINT32> tempOffset(host_mat,
                                                                             &HOST::CSRMatrix<ValType>::moveRowOffsetTo,
                                                                             &HOST::CSRMatrix<
                                                                                 ValType>::moveRowOffsetFrom);
        HOST::MovePrivateVector<HOST::CSRMatrix<ValType>, UINT32> tempColIndices(host_mat,
            &HOST::CSRMatrix<ValType>::moveColIndicesTo,
            &HOST::CSRMatrix<ValType>::moveColIndicesFrom);
        HOST::MovePrivateVector<HOST::CSRMatrix<ValType>, ValType> tempVal(host_mat,
                                                                           &HOST::CSRMatrix<ValType>::moveValuesTo,
                                                                           &HOST::CSRMatrix<ValType>::moveValuesFrom);
        m_row_offset.asyncCopyToHost(*tempOffset, 0, 0, tempOffset->getLength(), stream);
        m_col_indices.asyncCopyToHost(*tempColIndices, 0, 0, tempColIndices->getLength(), stream);
        m_val.asyncCopyToHost(*tempVal, 0, 0, tempVal->getLength(), stream);
    }

    template<typename ValType>
    void
    DeviceCSRMatrix<ValType>::asyncCopyFrom(const DeviceCSRMatrix<ValType> &devMat, const StreamController &stream) {
        if (devMat.m_deviceID != m_deviceID) return;
        resetAndClearMembers(devMat.m_rowNum, devMat.m_colNum, devMat.m_nnz);
        m_row_offset.asyncCopyFromCurrentDevice(devMat.m_row_offset, 0, 0, m_row_offset.getLength(), stream);
        m_col_indices.asyncCopyFromCurrentDevice(devMat.m_col_indices, 0, 0, m_nnz, stream);
        m_val.asyncCopyFromCurrentDevice(devMat.m_val, 0, 0, m_nnz, stream);
    }


    template<typename ValType>
    void
    DeviceCSRMatrix<ValType>::asyncCopyFromPeer(const DeviceCSRMatrix<ValType> &devMat, const StreamController &stream,
                                                const EventController &event) {
        if (devMat.m_deviceID == m_deviceID) {
            asyncCopyFrom(devMat, stream);
            return;
        }
        resetAndClearMembers(devMat.m_rowNum, devMat.m_colNum, devMat.m_nnz);
        m_row_offset.asyncCopyFromPeer(devMat.m_row_offset, event, 0, 0, m_row_offset.getLength(), stream);
        m_col_indices.asyncCopyFromPeer(devMat.m_col_indices, event, 0, 0, m_nnz, stream);
        m_val.asyncCopyFromPeer(devMat.m_val, event, 0, 0, m_nnz, stream);
    }


    template<typename ValType>
    void DeviceCSRMatrix<ValType>::printMatrix(const char *message) {
        std::cout << L_GREEN << "[CSR matrix: Device " << m_deviceID << "] " << L_BLUE << message << " --- "
                << "row:" << m_rowNum << ", col:" << m_colNum << ", nnz:"
                << m_row_offset.getThrustPtr()[m_rowNum] << "(max:" << m_nnz << ")" << COLOR_NONE << std::endl;
        m_row_offset.printVector("row offset");
        m_col_indices.printVector("col indices");
        m_val.printVector("values");
    }
} // DEVICE
