/*
 * @author  邓轶丹
 * @date    2024/6/12
 * @details 稠密矩阵类，支持行主序或列主序存储，可根据实际情况选择存储模式（一般、对齐内存、锁页内存）
 */

#ifndef PMSLS_NEW_DENSEMATRIX_H
#define PMSLS_NEW_DENSEMATRIX_H

#include "BaseMatrix.h"
#include "../VectorClass/AutoAllocateVector.h"
#include "../utils/ErrorHandler.h"

namespace HOST {
    enum DenseMatStoreType {
        DenseMatRowFirst, ///< 元素存储时按行优先排列
        DenseMatColumnFirst ///< 元素存储时按列优先排列
    };

    typedef enum DenseMatStoreType DenseMatStoreType_t;

    template<typename ValType>
    class DenseMatrix : public BaseMatrix<ValType> {
    private:
        UINT32 m_rowNum{0}; ///< 矩阵行数
        UINT32 m_colNum{0}; ///< 矩阵列数
        AutoAllocateVector<ValType> m_data_vec; ///< 存储数据一维数组
        DenseMatStoreType_t m_storageType{DenseMatColumnFirst}; ///< 是否按列优先存储，1：表示按列优先；0：表示按行优先

    public:
        DenseMatrix() = default;

        DenseMatrix(const DenseMatrix<ValType> &pre);

        DenseMatrix(DenseMatrix<ValType> &&pre) noexcept ;

        /** @brief 子类有参构造函数，根据行数和列数初始化稠密矩阵
         * @param [in] rowNum: 新建的矩阵的行数
         * @param [in] colNum: 新建的矩阵的列数 */
        DenseMatrix(const DenseMatStoreType_t &denseMatStoreType, const UINT32 &rowNum, const UINT32 &colNum,
                    const memoryType_t &memoryTypeHost);

        ~DenseMatrix() override = default;

        /** @brief 赋值运算符重载（拷贝赋值）
         * @param [in] pre_mat: 原矩阵
         * */
        DenseMatrix &operator=(const DenseMatrix<ValType> &pre_mat);

        /** @brief 赋值运算符重载（移动赋值） */
        DenseMatrix &operator=(DenseMatrix<ValType> &&pre_mat) noexcept;

        ValType operator()(UINT32 rowIdx, UINT32 colIdx) {
            return m_storageType == DenseMatColumnFirst ? m_data_vec[colIdx * m_rowNum + rowIdx] : m_data_vec[rowIdx * m_colNum + colIdx];
        }

        inline void getValue(const UINT32 &rowNo, const UINT32 &colNo, ValType &val) override {
#ifndef NDEBUG
            THROW_EXCEPTION(rowNo >= m_rowNum || colNo >= m_colNum,
                            THROW_OUT_OF_RANGE("The idx of row num or col num is out-of-range!"))
#endif
            if (m_storageType == DenseMatColumnFirst) val = m_data_vec[colNo * m_rowNum + rowNo];
            else val = m_data_vec[rowNo * m_colNum + colNo];
        }


        inline UINT32 getRowNum() const {
            return m_rowNum;
        }

        inline UINT32 getColNum() const {
            return m_colNum;
        }

        void getValsByCol(const UINT32 &colNo, HostVector<ValType> &vec);

        void setValsByCol(const UINT32 &colNo, const HostVector<ValType> &vec);


        inline void setValue(const UINT32 &rowNo, const UINT32 &colNo, const ValType &val) override {
#ifndef NDEBUG
            THROW_EXCEPTION(rowNo >= m_rowNum || colNo >= m_colNum,
                            THROW_OUT_OF_RANGE("The idx of row num or col num is out-of-range!"))
#endif
            if (m_storageType == DenseMatColumnFirst) m_data_vec[colNo * m_rowNum + rowNo] = val;
            else m_data_vec[rowNo * m_colNum + colNo] = val;
        }

        void transpose() override;

        void printMatrix(const char *message) override;

        inline void clear() override {
            m_rowNum = 0;
            m_colNum = 0;
            m_data_vec->clear();
        }

        inline void resetStorageType(const DenseMatStoreType_t &newType, const UINT8 &need_reserve) {
            if (newType != m_storageType) {
                if (need_reserve) {
                    transpose();
                    std::swap(m_rowNum, m_colNum);
                    m_storageType = newType;
                } else {
                    m_storageType = newType;
                }
            }
        }

        void resize(const UINT32 &new_rowNum, const UINT32 &new_colNum, const UINT32 &new_nnz,
                    UINT8 need_reserve) override;


        inline void resize(const UINT32 &new_rowNum, const UINT32 &new_colNum, UINT8 need_reserve) {
            resize(new_rowNum, new_colNum, new_rowNum * new_colNum, need_reserve);
        }

        /** @brief 获取实际存储矩阵的空间首指针（只读） */
        inline const ValType *getMatValPtr() const {
            return &m_data_vec[0];
        }

        /** @brief 获取实际存储矩阵的空间首指针 */
        inline ValType *getMatValPtr() {
            return &m_data_vec[0];
        }

        inline const DenseMatStoreType_t &getStorageType() const {
            return m_storageType;
        }

        /** @brief 稠密矩阵提取[start, end]列构成子矩阵，子矩阵进行矩阵向量乘
         * @param start 开始列
         * @param end 结束列（包括当前结束列）
         * @param vec 待乘向量
         * @param out_vec 输出向量，只需声明即可，无需初始化 */
        void MatVec(UINT32 start, UINT32 end, const HostVector<ValType> &vec, HostVector<ValType> &out_vec);

        void MatVec(const HostVector<ValType> &vec, HostVector<ValType> &out_vec);

        /**
        * @brief 稠密矩阵的转置乘以某个向量
        * @param vec 输入向量
        * @param out_vec 输出向量，只需声明即可
        */
        void transposeVec(const HostVector<ValType> &vec, HostVector<ValType> &out_vec);

        void fillMatrixWithValues(ValType value) {
            m_data_vec->fillVector(0, m_rowNum * m_colNum, value);
        }

        void MatMatMul(const DenseMatrix<ValType> & inMat, DenseMatrix<ValType> &outMat);
    };

    template class DenseMatrix<INT32>;
    template class DenseMatrix<FLOAT32>;
    template class DenseMatrix<FLOAT64>;
} // HOST

#endif //PMSLS_NEW_DENSEMATRIX_H
