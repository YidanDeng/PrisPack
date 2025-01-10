#ifndef MATRIX_BASE_H
#define MATRIX_BASE_H

#include "../VectorClass/BaseVector.h"
#include "../VectorClass/VectorTools.h"

namespace HOST {
    enum matrixType {
        matrixDense,       ///< 表示一般的稠密矩阵
        matrixCSR,         ///< 表示以稀疏压缩行格式存储的矩阵
        matrixCOO          ///< 表示以COO格式存储的矩阵
    };

    typedef enum matrixType matrixType_t;

    template<typename ValType>
    class BaseMatrix {
    protected:
        matrixType_t m_matType{matrixDense};

    public:
        /** @brief 基类无参构造函数。 */
        BaseMatrix() = default;

        /** @brief 基类拷贝构造函数。 */
        BaseMatrix(const BaseMatrix<ValType> &mat) = default;

        /** @brief 基类移动构造函数。 */
        BaseMatrix(BaseMatrix<ValType> &&mat) noexcept = default;

        /** @brief 基类析构函数。 */
        virtual ~BaseMatrix() = default;

        // get函数
        /** @brief 通过行号和列号获取值。
         * @param [in] rowNo: 行号
         * @param [in] colNo: 列号
         * @param [in,out] val: 取到的值 */
        virtual void getValue(const UINT32 &rowNo, const UINT32 &colNo, ValType &val) = 0;

        // set函数
        /** @brief 通过行号和列号写入值。
         * @param [in] rowNo: 行号
         * @param [in] colNo: 列号
         * @param [in] val: 要写入的值 */
        virtual void setValue(const UINT32 &rowNo, const UINT32 &colNo, const ValType &val) = 0;

        // 其他操作函数
        /** @brief 矩阵转置。 */
        virtual void transpose() = 0;

        /** @brief 打印矩阵。 */
        virtual void printMatrix(const char * message) = 0;

        virtual void clear() = 0;

        virtual void resize(const UINT32 &new_rowNum, const UINT32 &new_colNum, const UINT32 &new_nnz,
                            UINT8 need_reserve) = 0;

        // 异型矩阵拷贝，暂时不实现，矩阵之间的转换比较复杂，后续可能会有相关的需求
        // virtual void copy(const BaseMatrix<ValType> &pre) = 0;

    };

    enum privateMemberStatus {
        privateMemberFixed = 0,
        privateMemberNotFixed = 1
    };

    typedef enum privateMemberStatus privateMemberStatus_t;

    /** @brief 用于转移矩阵类中的private向量对象到外部空间，方便直接进行操作，最后自动归还到原对象内部
     * @attention 转移函数必须在外部写好，确认内部不会发生错误，且最大程度保证数据在转移过程中的完整性 */
    template<typename MatrixType, typename VectorValType>
    class MovePrivateVector {
    private:
        void (MatrixType::* m_moveToPtr)(HostVector<VectorValType> &);      ///< 函数指针，用于指向矩阵类中的用于移动私有向量成员的函数（内部到外部）

        void (MatrixType::* m_moveFromPtr)(HostVector<VectorValType> &);    ///< 函数指针，用于指向矩阵类中的用于移动私有向量成员的函数（外部到内部）

        std::unique_ptr<HostVector<VectorValType>> m_vectorPtr;      ///< 用来存放临时转出的vector对象

        MatrixType *m_matPtr{nullptr};          ///< 转存矩阵对象地址，并不实际分配空间
        privateMemberStatus_t fixedStatus{privateMemberNotFixed}; ///< 状态变量，用于记录取出的变量是否已归还

    public:
        /** @brief 用于转移某些类中的private向量对象到外部空间，方便直接进行操作，最后自动归还到原对象内部
         * @attention 转移函数必须在外部写好，确认内部不会发生错误，且最大程度保证数据在转移过程中的完整性
         * @param [in] mat:                     矩阵对象
         * @param [in] moveToFuncPtr:           用来转移private对象到外部的函数指针
         * @param [in] moveFromFuncPtr:         用来将被转移的private对象归还到对象内部的函数指针 */
        MovePrivateVector(MatrixType &mat, void (MatrixType::* moveToFuncPtr)(HostVector<VectorValType> &),
                          void (MatrixType::* moveFromFuncPtr)(HostVector<VectorValType> &)) {
            initializeVector(m_vectorPtr, 0, mat.getMemoryType());
            m_matPtr = &mat;
            m_moveToPtr = moveToFuncPtr;
            m_moveFromPtr = moveFromFuncPtr;
            (m_matPtr->*m_moveToPtr)(*m_vectorPtr);
#ifndef NINFO
            SHOW_INFO("Move private vector begin!")
#endif
        }

        inline HostVector<VectorValType> &operator*() {
            return *m_vectorPtr;
        }

        inline std::unique_ptr<HostVector<VectorValType>> &operator->() {
            return m_vectorPtr;
        }

        inline VectorValType &operator[](const UINT32 &idx) {
            return (*m_vectorPtr)[idx];
        }

        /** @brief 用于获得当前对象的转移状态 */
        inline privateMemberStatus_t getStatus() const {
            return fixedStatus;
        };

        /** @brief 用于提前归还被转移的向量，而不用强行等待当前对象生存周期结束 */
        inline void moveBack() {
            if (fixedStatus == privateMemberFixed) return;          // 若当前对象已经被归还至原矩阵内部，则不进行移动操作
            (m_matPtr->*m_moveFromPtr)(*m_vectorPtr);
            fixedStatus = privateMemberFixed;
        }

        /** @brief 用于重新转移向量对象到外部空间 */
        inline void move() {
            if (fixedStatus == privateMemberNotFixed) return;       // 当前向量已经被转移过，则不进行移动操作
            (m_matPtr->*m_moveToPtr)(*m_vectorPtr);
            fixedStatus = privateMemberNotFixed;
        }

        ~MovePrivateVector() {
            if (fixedStatus == privateMemberFixed) return;
            (m_matPtr->*m_moveFromPtr)(*m_vectorPtr);
#ifndef NINFO
            SHOW_INFO("Move private vector finished.")
#endif
        }
    };
}


#endif /* MATRIX_BASE_H */
