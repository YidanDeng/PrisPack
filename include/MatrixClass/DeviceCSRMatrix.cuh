/*
 * @author  邓轶丹
 * @date    2024/5/4
 * @details GPU上的稀疏压缩行矩阵类
 */

#ifndef PMSLS_NEW_DEVICECSRMATRIX_CUH
#define PMSLS_NEW_DEVICECSRMATRIX_CUH

//#include "../utils/MemoryTools/DeviceMemoryController.cuh"
#include "../utils/TimerTools/GPUtimer.cuh"
#include "../VectorClass/SyncDeviceVector.cuh"
#include "CSRMatrix.h"
#include "MatrixTools.h"
#include "../VectorClass/VectorTools.h"

namespace DEVICE {

    template<typename ValType>
    class DeviceCSRMatrix {
    private:
        SyncDeviceVector<UINT32> m_row_offset;
        SyncDeviceVector<UINT32> m_col_indices;
        SyncDeviceVector<ValType> m_val;

        UINT32 m_nnz{0};
        UINT32 m_colNum{0};
        UINT32 m_rowNum{0};
        INT32 m_deviceID{DEFAULT_GPU};

    public:
        DeviceCSRMatrix() = default;

        /* 构造函数与析构函数 */
        explicit DeviceCSRMatrix(const INT32 &deviceID);

        DeviceCSRMatrix(const UINT32 &rowNum, const UINT32 &colNum, const UINT32 &nnz, const INT32 &deviceID);

        DeviceCSRMatrix(const DeviceCSRMatrix<ValType> &pre) = delete;

        ~DeviceCSRMatrix() = default;

        /* 各种操作函数 */
        inline UINT32 *getRowOffsetPtr() const {
            return m_row_offset.getRawValPtr();
        }


        inline UINT32 *getColIndicesPtr() const {
            return m_col_indices.getRawValPtr();
        }


        inline ValType *getValuesPtr() const {
            return m_val.getRawValPtr();
        }


        inline UINT32 getNNZ() const {
            return m_nnz;
        }

        inline UINT32 getRowNum() const {
            return m_rowNum;
        }

        inline UINT32 getColNum() const {
            return m_colNum;
        }

        inline INT32 getDeviceID() const {
            return m_deviceID;
        }

        void copyMatFromHost(HOST::CSRMatrix<ValType> &host_mat);

        void copyMatToHost(HOST::CSRMatrix<ValType> &host_mat);

        void asyncCopyFrom(const DeviceCSRMatrix<ValType> &devMat, const StreamController &stream);

        void asyncCopyFromPeer(const DeviceCSRMatrix<ValType> &devMat, const StreamController &stream,
                               const EventController &event);

        void asyncCopyFromHost(HOST::CSRMatrix<ValType> &host_mat, const StreamController &stream);

        void asyncCopyToHost(HOST::CSRMatrix<ValType> &host_mat, const StreamController &stream);


        /** @brief 根据新的参数重新分配底层空间并设置相关成员变量，不保留原有值。*/
        void resetAndClearMembers(const UINT32 &new_row, const UINT32 &new_col, const UINT32 &new_nnz);

        void printMatrix(const char *message);
    };

    template class DeviceCSRMatrix<FLOAT32>;
    template class DeviceCSRMatrix<FLOAT64>;

} // DEVICE

#endif //PMSLS_NEW_DEVICECSRMATRIX_CUH
