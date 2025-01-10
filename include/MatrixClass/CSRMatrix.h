/*
 * @author  邓轶丹
 * @date    2024/4/29
 * @details 稀疏压缩行矩阵类，可根据实际情况选择存储模式（一般、对齐内存、锁页内存）
 */

#ifndef PMSLS_NEW_CSRMATRIX_H
#define PMSLS_NEW_CSRMATRIX_H

#define CSR_MATRIX(VAL_TYPE) HOST::CSRMatrix<VAL_TYPE>

#include "BaseMatrix.h"
#include "../VectorClass/VectorTools.h"
#include "../VectorClass/AutoAllocateVector.h"
#include "../utils/ErrorHandler.h"
#include "../utils/MemoryTools/UniquePtrTools.h"

namespace HOST {
    template <typename ValType>
    class CSRMatrix final : public HOST::BaseMatrix<ValType> {
    private:
        UINT32 m_rowNum{0}; ///<矩阵行数
        UINT32 m_colNum{0}; ///<矩阵列数
        UINT32 m_nnzCapacity{0}; ///<矩阵可存储非零元的个数，该值不一定等于实际有效存储的非0元个数
        AutoAllocateVector<UINT32> m_rowOffset; ///<行偏移，实际存储有效非零元个数由此检索
        AutoAllocateVector<UINT32> m_colIndices; ///<列索引向量
        AutoAllocateVector<ValType> m_values; ///<值向量，长度等于m_nnz
        memoryType_t m_memoryType{memoryBase}; ///<数据存储类型
        // 标准CSR要求每行非零元按列索引由小到大排列
        UINT8 m_isFormated{1}; ///<表示矩阵是否以标准CSR格式存储

    public:
        /** @brief 子类无参构造函数。*/
        CSRMatrix();

        /** @brief 子类构造函数，根据指定数据存储类型创建底层存储对象。
         * @attention 默认为空对象，可通过resize函数重新调整实际存储空间大小。*/
        explicit CSRMatrix(const memoryType_t& memoryType);

        /** @brief 子类复制构造函数。*/
        CSRMatrix(const CSRMatrix<ValType>& pre_mat);


        /** @brief 子类移动构造函数 */
        CSRMatrix(CSRMatrix<ValType>&& pre_mat) noexcept;


        /** @brief 子类析构函数 */
        ~CSRMatrix() override = default;


        /** @brief 子类有参构造函数。
         * @details 精确分配实际空间。
         * @param [in] rowNum: 行数
         * @param [in] colNum: 列数
         * @param [in] nnzNum: 矩阵中所有非0元个数 */
        CSRMatrix(const UINT32& rowNum, const UINT32& colNum, const UINT32& nnzNum, memoryType_t memoryType);

        /** @brief 赋值运算符重载（拷贝赋值）
         * @param [in] pre_mat: 原矩阵 */
        CSRMatrix& operator=(const CSRMatrix<ValType>& pre_mat);

        /** @brief 赋值运算符重载（移动赋值）*/
        CSRMatrix& operator=(CSRMatrix<ValType>&& pre_mat) noexcept;

        /** @brief 根据行号获取rowOffset的地址
         * @attention: 返回const（只读）指针 */
        inline const UINT32* getRowOffsetPtr(const UINT32& rowNo) const {
            return &m_rowOffset[rowNo];
        }

        inline UINT32* getRowOffsetPtr(const UINT32& rowNo) {
            return &m_rowOffset[rowNo];
        }

        /** @brief 根据行号获得该行元素起始地址
         * @attention: 返回const（只读）指针 */
        inline const UINT32* getColIndicesPtr(const UINT32& rowNo) const {
            return &m_colIndices[m_rowOffset[rowNo]];
        }

        inline UINT32* getColIndicesPtr(const UINT32& rowNo) {
            return &m_colIndices[m_rowOffset[rowNo]];
        }

        /** @brief 根据行号获取该行元素的起始地址
         * @attention: 返回const（只读）指针 */
        inline const ValType* getCSRValuesPtr(const UINT32& rowNo) const {
            return &m_values[m_rowOffset[rowNo]];
        }

        inline ValType* getCSRValuesPtr(const UINT32& rowNo) {
            return &m_values[m_rowOffset[rowNo]];
        }

        /** @brief 获取CSR矩阵存储空间类型 */
        inline memoryType_t getMemoryType() const {
            return m_memoryType;
        }

        inline UINT8 getFormatStatus() const {
            return m_isFormated;
        }

        /** @brief 将rowOffset的底层空间转移给外部变量
         * @param [in,out] vec: 外部向量对象，用来承接转移出的rowOffset空间 */
        inline void moveRowOffsetTo(HostVector<UINT32>& vec) {
#ifndef NWARN
            if (vec.getMemoryType() != m_memoryType) {
                SHOW_WARN("The row-offset has different memory type with original CSR row-offset. The move operation "
                    "will be transformed to copy operation.")
            }
#endif
            vec.move(*m_rowOffset);
        }

        /** @brief 将colIndices的底层空间转移给外部变量
         * @param [in,out] vec: 外部向量对象，用来承接转移出的colIndices空间 */
        inline void moveColIndicesTo(HostVector<UINT32>& vec) {
#ifndef NWARN
            if (vec.getMemoryType() != m_memoryType) {
                SHOW_WARN("The col-indices has different memory type with original CSR col-indices. The move operation "
                    "will be transformed to copy operation.")
            }
#endif
            vec.move(*m_colIndices);
        }

        /** @brief 将values的底层空间转移给外部变量
         * @param [in,out] vec: 外部向量对象，用来承接转移出的values空间 */
        inline void moveValuesTo(HostVector<ValType>& vec) {
#ifndef NWARN
            if (vec.getMemoryType() != m_memoryType) {
                SHOW_WARN("The values has different memory type with original CSR values. The move operation "
                    "will be transformed to copy operation.")
            }
#endif
            vec.move(*m_values);
        }

        /** @brief 将外部变量的底层空间归还给内部rowOffset
         * @param [in,out] vec: 外部向量对象，用来承接转移出的rowOffset空间 */
        inline void moveRowOffsetFrom(HostVector<UINT32>& vec) {
#ifndef NWARN
            if (vec.getLength() != m_rowNum + 1) {
                SHOW_WARN(
                    "The length of row-offset is not equal to original CSR row-offset! Move operation was denied.")
                return;
            }
            if (vec.getMemoryType() != m_memoryType) {
                SHOW_WARN("The row-offset has different memory type with original CSR row-offset. The move operation "
                    "will be transformed to copy operation.")
            }
#endif
            m_rowOffset->move(vec);
        }

        /** @brief 将外部变量的底层空间归还给内部colIndices
         * @param [in,out] vec: 外部向量对象，用来承接转移出的colIndices空间 */
        inline void moveColIndicesFrom(HostVector<UINT32>& vec) {
#ifndef NWARN
            if (vec.getLength() != m_nnz) {
                SHOW_WARN("The length of col-indices is not equal to original CSR col-indices! "
                    "Move operation was denied.")
                return;
            }
            if (vec.getMemoryType() != m_memoryType) {
                SHOW_WARN("The col-indices has different memory type with original CSR col-indices. The move operation "
                    "will be transformed to copy operation.")
            }
#endif
            m_colIndices->move(vec);
        }

        /** @brief 将外部变量的底层空间归还给内部values
         * @param [in,out] vec: 外部向量对象，用来承接转移出的values空间 */
        inline void moveValuesFrom(HostVector<ValType>& vec) {
#ifndef NWARN
            if (vec.getLength() != m_nnz) {
                SHOW_WARN("The length of values is not equal to original CSR values! "
                    "Move operation was denied.")
                return;
            }
            if (vec.getMemoryType() != m_memoryType) {
                SHOW_WARN("The values has different memory type with original CSR values. The move operation "
                    "will be transformed to copy operation.")
            }
#endif
            m_values->move(vec);
        }


        /** @brief 根据行号和列号获取具体的值
         * @param [in] rowNo: 行号（从0开始）
         * @param [in] colNo: 列号（从0开始）
         * @param [in, out] val: 获取的值 */
        void getValue(const UINT32& rowNo, const UINT32& colNo, ValType& val) override;

        /** @brief 根据行号获取某一行的值
         * @attention 该函数直接返回指针，可用于读取数据，如非必要不要通过指针修改对应的值。
         * @param [in] rowNo: 行号（从0开始）；
         * @param [in,out] colIndices: 根据行号获取到的某一行列索引指针，该变量必须在外部声明并且初始值为nullptr；
         * @param [in,out] vals: 根据行号获取到的某一行值指针，该变量必须在外部声明并且初始值为nullptr；
         * @param [out] valNum: 根据行号获取到的某一行非0元个数。 */
        void getColsValsByRow(const UINT32& rowNo, UINT32*& colIndices, ValType*& vals, UINT32& valNum);


        inline UINT32 getRowNum() const {
            return m_rowNum;
        }

        inline UINT32 getColNum() const {
            return m_colNum;
        }

        /** @brief 该函数记录的是值向量可存储元素容量，不一定等于实际CSR矩阵非零元个数 */
        inline UINT32 getValNum() const {
            return m_nnzCapacity;
        }

        /** @brief 记录从startRowNo行到endRowNo行（包括第endRowNo行）的实际存储非零元个数。*/
        inline UINT32 getNNZnum(const UINT32& startRowNo, const UINT32& endRowNo) const {
#ifndef NDEBUG
            ERROR_CHECK(startRowNo > endRowNo, DEBUG_MESSEGE_OPTION,
                        "The range of start row number and end row number was incorrect!");
            ERROR_CHECK(endRowNo >= m_rowNum || startRowNo >= m_rowNum, DEBUG_MESSEGE_OPTION,
                        "The start row number or end row number was out-of-range!");
#endif
            return m_rowOffset->getValue(endRowNo + 1) - m_rowOffset->getValue(startRowNo);
        }

        /** @brief 通过行号和列号写入值。
         * @attention 这个函数使用前必须确保有正确的colIndices向量和rowOffset向量。
         * @param [in] rowNo: 行号（从0开始）；
         * @param [in] colNo: 列号（从0开始）；
         * @param [in] val: 要写入的值。 */
        void setValue(const UINT32& rowNo, const UINT32& colNo, const ValType& val) override;

        // set函数
        /** @brief 按行号写入行值和对应的列索引
         * @attention colIndices和vals必须在外部初始化并赋值，随后通过该函数将其拷贝到csr对应的变量中。
         * @param [in] rowNo: 行号（从0开始，0代表第1行）
         * @param [in] colIndices: 列索引
         * @param [in] vals: 待插入行具体的非0元
         * @param [in] valNum: 非0元个数 */
        void setColsValsByRow(const UINT32& rowNo, const UINT32* const & colIndices, const ValType* const & vals,
                              const UINT32& valNum);


        void resetMemoryType(const memoryType_t& memoryTypeHost, UINT8 needReserveData);


        /** @brief 打印矩阵。 */
        inline void printMatrix(const char* message) override {
#ifndef NINFO
            SHOW_INFO("Print CSR matrix.")
#endif
            std::string type = m_memoryType == memoryBase
                                   ? "Base"
                                   : m_memoryType == memoryAligned
                                   ? "Aligned"
                                   : "Page-locked";
            std::cout << L_GREEN << "[CSR matrix: " << type << "] " << L_BLUE << message << " --- "
                << "row:" << m_rowNum << ", col:" << m_colNum << ", nnz:"
                << m_rowOffset[m_rowNum] << "(max:" << m_nnzCapacity << ")" << COLOR_NONE << std::endl;
            m_rowOffset->printVector("row offset");
            m_colIndices->printVector("column indices");
            m_values->printVector("values");
        }

        inline void printMatrixRow(const char* message, const UINT32& rowNum) {
            std::string type = m_memoryType == memoryBase
                                   ? "Base"
                                   : m_memoryType == memoryAligned
                                   ? "Aligned"
                                   : "Page-locked";
            std::cout << L_GREEN << "[CSR matrix: " << type << "] " << L_BLUE << message << " --- "
                << "row:" << m_rowNum << ", col:" << m_colNum << ", nnz:"
                << m_rowOffset[m_rowNum] << "(max:" << m_nnzCapacity << ")" << COLOR_NONE << std::endl;
            std::cout << L_BLUE << "[INFO] row-" << rowNum << ":" << COLOR_NONE;
            for (UINT32 i = m_rowOffset[rowNum]; i < m_rowOffset[rowNum + 1]; ++i) {
                std::cout << " " << m_values[i] << "(" << m_colIndices[i] << ")";
            }
            std::cout << std::endl;
        }


        /** @brief 实现CSR矩阵的转置，转置后仍为CSR矩阵（在原位上修改） */
        void transpose() override;

        /** @brief 用来清空本地全部内存空间和变量，并保证当前对象是可析构的。*/
        void clear() override;


        /** @brief 用于重新调整当前CSR矩阵的大小。
         * @param [in] new_rowNum:      新矩阵在逻辑上的行数；
         * @param [in] new_colNum:      新矩阵在逻辑上的列数；
         * @param [in] new_nnz:         新矩阵拥有的非零元总数，如果比原nnz大，则只开辟空间不实际填值，如果比原来的小，
                                        则根据具体情况舍弃非零元；
         * @param [in] need_reserve:    一个0-1变量，宏定义在config文件中，不需要保存值则传入“RESERVE_NO_DATA”（对应值为0），
                                        反之传入"RESERVE_DATA"（对应值为1）*/
        void resize(const UINT32& new_rowNum, const UINT32& new_colNum, const UINT32& new_nnz,
                    UINT8 need_reserve) override;


        /** @brief 从原始csr矩阵中分离出子矩阵，行范围[startRowNo, endRowNo]，列范围[startColNo, endColNo]。
         * @attention 当前CSR矩阵必须是标准形式，即每一行中的非0元按列标号由小到大排列,
         *              该方法不拷贝具体的子矩阵，而是返回可以确定子矩阵存储范围的相关标志性变量。
         *              因此，这种方法只是从原矩阵中切出对应的子矩阵，列索引的变化需在外部处理。
         * @param [in] startRowNo:  子矩阵起始行号；
         * @param [in] endRowNo:    子矩阵结束行号；
         * @param [in] startColNo:  子矩阵开始列号；
         * @param [in] endColNo:    子矩阵结束列号；
         * @param [in,out] nnzStartPosForEachRow:       找到子矩阵后，记录子矩阵每一行非零元在原始矩阵中的起始下标；
         * @param [in,out] nnzNumForEachRow:            找到子矩阵后，记录子矩阵每一行实际的非零元个数。
         * */
        void getSubMatrixWithoutCopy(const UINT32& startRowNo, const UINT32& endRowNo, const UINT32& startColNo,
                                     const UINT32& endColNo, HostVector<UINT32>& nnzStartPosForEachRow,
                                     HostVector<UINT32>& nnzNumForEachRow);


        /** @brief 根据指定行范围和列范围获取子矩阵。行范围[startRowNo, endRowNo]，列范围[startColNo, endColNo]。
         * @attention   当前CSR矩阵必须是标准形式，即每一行中的非0元按列标号由小到大排列。
         *              最后生成的子矩阵列索引以子矩阵为准。
         * @param [in] startRowNo:  子矩阵实际起始行下标（包括第startRowNo行）
         * @param [in] endRowNo:    子矩阵实际结束行下标（包括第endRowNo行）
         * @param [in] startColNo:  子矩阵实际起始列下标（包括第startColNo行）
         * @param [in] endColNo:    子矩阵实际结束列下标（包括第endColNo行）
         * @param [in,out] outMat:  输出矩阵，只需在外部声明一个默认对象即可 */
        void getSubMatrix(const UINT32& startRowNo, const UINT32& endRowNo, const UINT32& startColNo,
                          const UINT32& endColNo, CSRMatrix<ValType>& outMat);

        /** @brief 根据指定行重排和列重排向量获得子矩阵。
         * @details 重排后的子矩阵严格遵守CSR标准，即每一行中的非0元按列标号由小到大排列
         * @attention 子矩阵列索引以子矩阵为准
         * @param [in] rowPerm:     子矩阵行映射向量
         * @param [in] colPerm:     子矩阵列映射向量
         * @param [in,out] outMat:  输出矩阵，只需在外部声明一个默认对象即可 */
        void getSubMatrix(const HostVector<UINT32>& rowPerm, const HostVector<UINT32>& colPerm,
                          CSRMatrix<ValType>& outMat) const;

        void getSubMatrix(const UINT32* rowPermPtr, UINT32 rowPermLength, const UINT32* colPermPtr,
                          UINT32 colPermLength,
                          CSRMatrix<ValType>& outMat) const;

        /** @brief 根据行范围提取矩阵，再将其按列分割点划分为若干子矩阵，并将它们放在同一个CSR矩阵中
         * @details 后续提取对应的子矩阵只需提取outMat对应的行就行 */
        void getSubMatrix(const UINT32& startRowNo, const UINT32& endRowNo, const HostVector<UINT32>& colOffset,
                          CSRMatrix<ValType>& outMat);


        void getSubMatrix(const UINT32& startRowNo, const UINT32& endRowNo, const HostVector<UINT32>& colOffset,
                          UniquePtr1D<HOST::CSRMatrix<ValType>>& outMat, const memoryType_t& outMatType);

        /** @brief 用来检测当前矩阵是否为标准CSR格式，如果不是则进行调整。
         * @details 该方法主要针对那些列索引被打乱的CSR矩阵，将其调整至标准形式
         * @attention 由于排序算法采用的是简单插入排序，若矩阵宽度较大，则会导致比较严重的性能损失 */
        void formatStructure();

        void MatPVecWithRawPtr(const ValType* in_vec, ValType* out_vec) const {
            UINT32 k1, k2, col, i, j, rowNum;
            ValType val;
            rowNum = this->m_rowNum;
#ifdef USE_OMP_MATRIX_FUNC
#pragma omp parallel for default(none) num_threads(THREAD_NUM) private(k1, k2, j, col, val) shared(rowNum, in_vec, out_vec)
#endif
            for (i = 0; i < rowNum; ++i) {
                k1 = this->m_rowOffset[i];
                k2 = this->m_rowOffset[i + 1];
                out_vec[i] = 0;
                for (j = k1; j < k2; ++j) {
                    col = m_colIndices[j];
                    val = m_values[j];
                    out_vec[i] += in_vec[col] * val;
                }
            }
        }

        void MatPVec(const HostVector<ValType>& in_vec, HostVector<ValType>& out_vec) const {
            if (this->m_rowNum == 0 || this->m_colNum == 0 || in_vec.getLength() == 0) return;
#ifndef NDEBUG
            THROW_EXCEPTION(in_vec.getLength() != m_colNum || out_vec.getLength() != m_rowNum,
                            THROW_RANGE_ERROR("Input dim is not match with matrix"))
#endif
            MatPVecWithRawPtr(&in_vec[0], &out_vec[0]);
        }

        void MatPVec(const HostVector<ValType>& in_vec, UINT32 inStartIdx, HostVector<ValType>& out_vec,
                     INT32 outStartIdx) const {
#ifndef NDEBUG
            THROW_EXCEPTION(m_colNum + inStartIdx > in_vec.getLength() || m_rowNum + outStartIdx > out_vec.getLength(),
                            THROW_RANGE_ERROR("Input dim is not match with matrix"))
#endif
            MatPVecWithRawPtr(&in_vec[inStartIdx], &out_vec[outStartIdx]);
        }

        void MatPVecWithoutClearOutVector(const HostVector<ValType>& in_vec, UINT32 inStartIdx,
                                          HostVector<ValType>& out_vec,
                                          INT32 outStartIdx) const {
#ifndef NDEBUG
            THROW_EXCEPTION(m_colNum + inStartIdx > in_vec.getLength() || m_rowNum + outStartIdx > out_vec.getLength(),
                            THROW_RANGE_ERROR("Input dim is not match with matrix"))
#endif
            UINT32 k1, k2, col, i, j, rowNum;
            ValType val;
            rowNum = this->m_rowNum;
#ifdef USE_OMP_MATRIX_FUNC
#pragma omp parallel for default(none) num_threads(THREAD_NUM) private(k1, k2, j, col, val) \
    shared(rowNum, in_vec, out_vec, outStartIdx, inStartIdx)
#endif
            for (i = 0; i < rowNum; ++i) {
                k1 = this->m_rowOffset[i];
                k2 = this->m_rowOffset[i + 1];
                for (j = k1; j < k2; ++j) {
                    col = m_colIndices[j];
                    val = m_values[j];
                    out_vec[i + outStartIdx] += in_vec[col + inStartIdx] * val;
                }
            }
        }

        /** @brief 计算 y = A * alpha * x + beta * y */
        void MatPVec(ValType alpha, const HostVector<ValType>& in_vec, UINT32 inStartIdx,
                     ValType beta, HostVector<ValType>& out_vec, INT32 outStartIdx) {
#ifndef NDEBUG
            THROW_EXCEPTION(m_colNum + inStartIdx > in_vec.getLength() || m_rowNum + outStartIdx > out_vec.getLength(),
                            THROW_RANGE_ERROR("Input dim is not match with matrix"))
#endif
            UINT32 k1, k2, col, i, j, rowNum, outLocalStartIdx;
            ValType val;
            rowNum = this->m_rowNum;
#ifdef USE_OMP_MATRIX_FUNC
#pragma omp parallel for default(none) num_threads(THREAD_NUM) private(k1, k2, j, col, val, outLocalStartIdx) \
shared(rowNum, in_vec, out_vec, outStartIdx, inStartIdx, beta, alpha)
#endif
            for (i = 0; i < rowNum; ++i) {
                k1 = this->m_rowOffset[i];
                k2 = this->m_rowOffset[i + 1];
                outLocalStartIdx = i + outStartIdx;
                out_vec[outLocalStartIdx] *= beta;
                for (j = k1; j < k2; ++j) {
                    col = m_colIndices[j];
                    val = m_values[j];
                    out_vec[outLocalStartIdx] += alpha * in_vec[col + inStartIdx] * val;
                }
            }
        }


        /** @brief  实现A的转置乘一个向量
         * @attention 因为转置的特殊性，out_vec无法在内部实现内存清零的操作，所以必须人为处理out_vec的初始化操作，否则乘积操作结果可能是错误的
         */
        void transMatPVec(const ValType* in_vec, ValType* out_vec) const {
            UINT32 m, k1, k2, i, j, tmp_index;
            ValType tmp_value;
            m = this->getRowNum();
            // #ifdef USE_OMP_MATRIX_FUNC
            // #pragma omp parallel for default(none) private(k1, k2, j, tmp_index, tmp_value) shared(m, in_vec, out_vec)
            // #endif
            for (i = 0; i < m; ++i) {
                k1 = m_rowOffset[i];
                k2 = m_rowOffset[i + 1];
                for (j = k1; j < k2; ++j) {
                    tmp_index = m_colIndices[j];
                    tmp_value = m_values[j];
                    out_vec[tmp_index] += tmp_value * in_vec[i];
                }
            }
        }

        inline void transMatPVec(const HostVector<ValType>& in_vec, HostVector<ValType>& out_vec) const {
#ifndef NDEBUG
            THROW_EXCEPTION(in_vec.getLength() != m_rowNum || out_vec.getLength() != m_colNum,
                            THROW_RANGE_ERROR("Input dim is not match with matrix"))
#endif
            transMatPVec(&in_vec[0], &out_vec[0]);
        }

        inline void transMatPVec(const HostVector<ValType>& in_vec, UINT32 inStartIdx, HostVector<ValType>& out_vec,
                                 INT32 outStartIdx) const {
#ifndef NDEBUG
            THROW_EXCEPTION(m_rowNum + inStartIdx > in_vec.getLength() || m_colNum + outStartIdx > out_vec.getLength(),
                            THROW_RANGE_ERROR("Input dim is not match with matrix"))
#endif
            transMatPVec(&in_vec[inStartIdx], &out_vec[outStartIdx]);
        }


        inline void unit(const UINT32& dim) {
            m_rowNum = dim;
            m_colNum = dim;
            m_nnzCapacity = dim;
            m_rowOffset.resize(dim + 1, RESERVE_NO_DATA);
            m_colIndices.resize(dim, RESERVE_NO_DATA);
            m_values.resize(dim, RESERVE_NO_DATA);
            std::iota(&m_rowOffset[0], &m_rowOffset[0] + dim + 1, 0);
            std::iota(&m_colIndices[0], &m_colIndices[0] + dim, 0);
            m_values->fillVector(0, dim, 1);
        }

        /** @brief 计算某个向量在当前矩阵下的范数，res = \sqrt(v^T * A * v) */
        FLOAT64 getMatrixNorm(const HostVector<ValType>& vec, UINT32 vecStartIdx) const {
#ifndef NDEBUG
            THROW_EXCEPTION(vecStartIdx + m_colNum > vec.getLength(), THROW_OUT_OF_RANGE("The vec range is out-of-range!"))
#endif

            FLOAT64 sum = 0.0;
#pragma omp parallel for default(none) num_threads(THREAD_NUM) reduction(+:sum) schedule(dynamic) shared(vec, vecStartIdx, m_rowNum, m_rowOffset, m_colIndices, m_values)
            for (UINT32 rowIdx = 0; rowIdx < m_rowNum; ++rowIdx) {
                UINT32 startIdx = m_rowOffset[rowIdx];
                UINT32 endIdx = m_rowOffset[rowIdx + 1];
                FLOAT64 currSum = 0.0;
                for (UINT32 elemIdx = startIdx; elemIdx < endIdx; ++elemIdx) {
                    currSum += m_values[elemIdx] * vec[vecStartIdx + m_colIndices[elemIdx]];
                }
                sum += vec[vecStartIdx + rowIdx] * currSum;
            }
            return sqrt(sum);
        }

        void pivotReorderByRow(HostVector<UINT32>& colPerm);

        void pivotReorder(const HostVector<UINT32>& rowPerm, HostVector<UINT32>& colPerm);
    };


    template class CSRMatrix<INT32>;
    template class CSRMatrix<FLOAT32>;
    template class CSRMatrix<FLOAT64>;
}


#endif //PMSLS_NEW_CSRMATRIX_H
