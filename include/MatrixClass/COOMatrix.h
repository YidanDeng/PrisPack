/*
 * @author  邓轶丹
 * @date    2024/5/10
 * @details COO矩阵类，可根据实际情况选择存储模式（一般、对齐内存、锁页内存）
 */

#ifndef PMSLS_NEW_COOMATRIX_H
#define PMSLS_NEW_COOMATRIX_H

#define COO_MATRIX(VAL_TYPE) HOST::COOMatrix<VAL_TYPE>

#include "BaseMatrix.h"
#include "../VectorClass/AutoAllocateVector.h"


namespace HOST {
    /** @brief 以COO格式存储的稀疏矩阵类
     * @attention   该矩阵填充值只能用pushBack函数从底层向量尾部插入值，构造函数只负责分配足够的空间，若要随机访问底层存储向量，
     *              必须使用相关move函数将私有成员变量转移到外部操作，但必须保证移动前后存储的有效非零元个数一致。*/
    template<typename ValType>
    class COOMatrix final : public HOST::BaseMatrix<ValType> {
    private:
        UINT32 m_rows{0};                       ///< 行数
        UINT32 m_cols{0};                       ///< 列数
        UINT32 m_nnz{0};                        ///< 可存放非零元个数
        UINT32 m_actNNZ{0};                     ///< 实际有效非零元个数
        AutoAllocateVector<UINT32> i_vec;       ///< 行向量
        AutoAllocateVector<UINT32> j_vec;       ///< 列向量
        AutoAllocateVector<ValType> data_vec;   ///< 值向量
        memoryType_t m_memoryType{memoryBase};

    public:
        COOMatrix();

        explicit COOMatrix(const memoryType_t &memoryTypeHost);

        /** @brief COO矩阵有参构造函数，根据指定行数、列数、非零元个数生成矩阵。
        * @param rowNum 行数
        * @param colNum 列数
        * @param nnz 非零元个数 */
        COOMatrix(const UINT32 &rowNum, const UINT32 &colNum, UINT32 nnz, const memoryType_t &memoryTypeHost);

        COOMatrix(const COOMatrix<ValType> &pre);

        COOMatrix(COOMatrix<ValType> &&pre) noexcept;

        ~COOMatrix() override = default;

        COOMatrix &operator=(const COOMatrix<ValType> &pre);

        COOMatrix &operator=(COOMatrix<ValType> &&pre) noexcept;

        void getValue(const UINT32 &rowNo, const UINT32 &colNo, ValType &val) override {
#ifndef NDEBUG
            ERROR_CHECK(rowNo >= m_rows || colNo >= m_cols, DEBUG_MESSEGE_OPTION,
                        "The row-ID or col-ID is out-of-range!");
#endif
            for (UINT32 i = 0; i < m_actNNZ; i++) {
                if (i_vec->getValue(i) == rowNo && j_vec->getValue(i) == colNo) {
                    val = data_vec->getValue(i);
                    break;
                }
            }
        }

        inline void
        setup(const UINT32 &rowNum, const UINT32 &colNum, const UINT32 &actNNZNum, const HostVector<UINT32> &rowIdx,
              const HostVector<UINT32> &colIdx, const HostVector<ValType> &nnz) {
//            SHOW_INFO("setup begin")
            resize(rowNum, colNum, actNNZNum, RESERVE_NO_DATA);
            i_vec.copy(rowIdx);
            j_vec.copy(colIdx);
            data_vec.copy(nnz);
            m_actNNZ = actNNZNum;
//            SHOW_INFO("setup finish")
        }

        inline UINT32 getRowNum() const {
            return m_rows;
        }

        inline UINT32 getColNum() const {
            return m_cols;
        }

        /** @brief 该函数记录的是值向量可存储元素容量，不一定等于实际非零元个数 */
        inline UINT32 getValNum() const {
            return m_nnz;
        }

        /** @brief 获得实际存储非零元个数。*/
        inline UINT32 getNNZnum() {
            return m_actNNZ;
        }


        /** @brief 获取rowIndices的首地址
         * @attention: 返回const（只读）指针 */
        inline const UINT32 *getRowIndicesPtr() const {
            return &i_vec[0];
        }

        /** @brief 获取列索引向量的首地址
         * @attention: 返回const（只读）指针 */
        inline const UINT32 *getColIndicesPtr() const {
            return &j_vec[0];
        }

        /** @brief 返回值向量的首地址
         * @attention: 返回const（只读）指针 */
        inline const ValType *getCOOValuesPtr() const {
            return &data_vec[0];
        }

        inline void setValue(const UINT32 &rowNo, const UINT32 &colNo, const ValType &val) override {
#ifndef NDEBUG
            ERROR_CHECK(rowNo >= m_rows || colNo >= m_cols, DEBUG_MESSEGE_OPTION,
                        "The row-ID or col-ID is out-of-range!");
#endif
            for (UINT32 i = 0; i < m_actNNZ; i++) {
                if (i_vec->getValue(i) == rowNo && j_vec->getValue(i) == colNo) {
                    data_vec[i] = val;
                    break;
                }
            }
        }

        inline void pushBack(const UINT32 &rowIdx, const UINT32 &colIdx, const ValType &val) {
            if (m_actNNZ >= m_nnz) {
                m_nnz = std::max(m_nnz, static_cast<UINT32>(256));
                m_nnz *= 2;
                i_vec->resize(m_nnz, RESERVE_DATA);
                j_vec->resize(m_nnz, RESERVE_DATA);
                data_vec->resize(m_nnz, RESERVE_DATA);
            }
            i_vec[m_actNNZ] = rowIdx;
            j_vec[m_actNNZ] = colIdx;
            data_vec[m_actNNZ] = val;
            m_actNNZ++;
        }

        inline void transpose() override {
            std::swap(m_rows, m_cols);
            AutoAllocateVector<UINT32> temp;
            temp = std::move(j_vec);
            j_vec = std::move(i_vec);
            i_vec = std::move(temp);
        }

        inline void printMatrix(const char *message) override {
            std::string type = m_memoryType == memoryBase ? "Base" :
                               m_memoryType == memoryAligned ? "Aligned" : "Page-locked";
            std::cout << L_GREEN << "[COO matrix: " << type << "] " << L_BLUE << message << " --- "
                      << "row:" << m_rows << ", col:" << m_cols << ", nnz:"
                      << m_actNNZ << "(max:" << m_nnz << ")" << COLOR_NONE << std::endl;
            i_vec->printVector("row idx");
            j_vec->printVector("column idx");
            data_vec->printVector("values");
        }

        void clear() override;

        void resize(const UINT32 &new_rowNum, const UINT32 &new_colNum, const UINT32 &new_nnz,
                    UINT8 need_reserve) override;


        inline void moveRowIndicesTo(HostVector<UINT32> &vec) {
            if (vec.getMemoryType() != m_memoryType) {
#ifndef NWARN
                SHOW_WARN("The row-indices has different memory type with original COO row-offset. The move operation "
                          "will be transformed to copy operation.")
#endif
            }
            vec.move(*i_vec);
        }

        inline void moveColIndicesTo(HostVector<UINT32> &vec) {
            if (vec.getMemoryType() != m_memoryType) {
#ifndef NWARN
                SHOW_WARN("The col-indices has different memory type with original COO col-indices. The move operation "
                          "will be transformed to copy operation.")
#endif
            }
            vec.move(*j_vec);
        }

        inline void moveValuesTo(HostVector<ValType> &vec) {
            if (vec.getMemoryType() != m_memoryType) {
#ifndef NWARN
                SHOW_WARN("The values has different memory type with original COO values. The move operation "
                          "will be transformed to copy operation.")
#endif
            }
            vec.move(*data_vec);
        }

        inline void moveRowIndicesFrom(HostVector<UINT32> &vec) {
            if (vec.getLength() != m_nnz) {
#ifndef NWARN
                SHOW_WARN(
                        "The length of row-offset is not equal to original COO row-offset! Move operation was denied.")
#endif
                return;
            }
            if (vec.getMemoryType() != m_memoryType) {
#ifndef NWARN
                SHOW_WARN("The row-indices has different memory type with original COO row-offset. The move operation "
                          "will be transformed to copy operation.")
#endif
            }
            i_vec->move(vec);
        }


        inline void moveColIndicesFrom(HostVector<UINT32> &vec) {
            if (vec.getLength() != m_nnz) {
#ifndef NWARN
                SHOW_WARN(
                        "The length of col-indices is not equal to original COO col-indices! Move operation was denied.")
#endif
                return;
            }
            if (vec.getMemoryType() != m_memoryType) {
#ifndef NWARN
                SHOW_WARN("The col-indices has different memory type with original COO col-indices. The move operation "
                          "will be transformed to copy operation.")
#endif
            }
            j_vec->move(vec);
        }


        /** @brief 将被移动到外部的value向量移动回矩阵内部 */
        inline void moveValuesFrom(HostVector<ValType> &vec) {
            if (vec.getLength() != m_nnz) {
#ifndef NWARN
                SHOW_WARN(
                        "The length of values is not equal to original COO values! Move operation was denied.")
#endif
                return;
            }
            if (vec.getMemoryType() != m_memoryType) {
#ifndef NWARN
                SHOW_WARN("The values has different memory type with original COO values. The move operation "
                          "will be transformed to copy operation.")
#endif
            }
            data_vec->move(vec);
        }

        inline void setActNNZ(const UINT32 &actNNZ) {
            m_actNNZ = actNNZ;
        }


    };


    template
    class COOMatrix<INT32>;

    template
    class COOMatrix<FLOAT32>;

    template
    class COOMatrix<FLOAT64>;

//    template
//    class COOMatrix<std::complex<FLOAT32>>;
//
//    template
//    class COOMatrix<std::complex<FLOAT64>>;

} // HOST

#endif //PMSLS_NEW_COOMATRIX_H
