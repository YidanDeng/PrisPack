/*
 * @author  邓轶丹
 * @date    2024/6/10
 * @details 预条件基类，用于实现预条件的多态
 */

#ifndef PMSLS_NEW_BASEPRECONDITON_H
#define PMSLS_NEW_BASEPRECONDITON_H

#include "../VectorClass/BaseVector.h"

enum PreconditionType {
    PreconditionNone,
    PreconditionIC,
    PreconditionILUT,
    PreconditionILDLT,
    PreconditionGMSLR,
    PreconditionAMSED
};
typedef PreconditionType PreconditionType_t;


enum SymmetricType {
    SymmetricPositiveDefiniteMatrix,
    SymmetricMatrix,
    AsymmetricMatrix,
};

typedef enum SymmetricType SymmetricType_t;

#define PRECONDITION_READY 1
#define PRECONDITION_NOT_READY 0

template<typename RightHandValType>
class BasePrecondition {
protected:
    PreconditionType_t m_precondType{PreconditionNone};
    UINT8 m_isReady{PRECONDITION_NOT_READY};
    UINT32 m_ArowNum{0}; ///< 原系数矩阵行数
    UINT32 m_AcolNum{0}; ///< 原系数矩阵列数
    UINT32 m_Annz{0}; ///< 原系数矩阵非零元个数
    UINT32 m_Mnnz{0}; ///< 统计预条件的所有非零元个数

public:
    BasePrecondition() = default;

    virtual ~BasePrecondition() = default;

    BasePrecondition(const BasePrecondition &other)
        : m_precondType(other.m_precondType),
          m_isReady(other.m_isReady),
          m_ArowNum(other.m_ArowNum),
          m_AcolNum(other.m_AcolNum),
          m_Annz(other.m_Annz),
          m_Mnnz(other.m_Mnnz) {
    }

    BasePrecondition(BasePrecondition &&other) noexcept
        : m_precondType(other.m_precondType),
          m_isReady(other.m_isReady),
          m_ArowNum(other.m_ArowNum),
          m_AcolNum(other.m_AcolNum),
          m_Annz(other.m_Annz),
          m_Mnnz(other.m_Mnnz) {
        other.m_isReady = PRECONDITION_NOT_READY;
        other.m_ArowNum = 0;
        other.m_AcolNum = 0;
        other.m_Annz = 0;
        other.m_Mnnz = 0;
    }

    BasePrecondition & operator=(const BasePrecondition &other) {
        if (this == &other)
            return *this;
        m_precondType = other.m_precondType;
        m_isReady = other.m_isReady;
        m_ArowNum = other.m_ArowNum;
        m_AcolNum = other.m_AcolNum;
        m_Annz = other.m_Annz;
        m_Mnnz = other.m_Mnnz;
        return *this;
    }

    BasePrecondition & operator=(BasePrecondition &&other) noexcept {
        if (this == &other)
            return *this;
        m_precondType = other.m_precondType;
        m_isReady = other.m_isReady;
        m_ArowNum = other.m_ArowNum;
        m_AcolNum = other.m_AcolNum;
        m_Annz = other.m_Annz;
        m_Mnnz = other.m_Mnnz;
        other.m_isReady = PRECONDITION_NOT_READY;
        other.m_ArowNum = 0;
        other.m_AcolNum = 0;
        other.m_Annz = 0;
        other.m_Mnnz = 0;
        return *this;
    }

    /** @brief 预条件的构建 */
    virtual void setup() = 0;

    inline UINT8 isReady() const {
        return m_isReady;
    }

    /** @brief 求解 vec = M^{-1} * vec，最后的结果覆盖原有值
     * @param [in,out] vec: 输入向量 */
    virtual void MInvSolve(BaseVector<RightHandValType> &vec) = 0;

    /** @brief 求解 vec = M^{-1} * vec，最后的结果覆盖原有值，但是只转换x的一部分值（vec.length > M.dim）
     * @param [in,out] vec: 输入向量
     * @param [in] resStartIdx: 要转换的片段在x中的起始位置 */
    virtual void MInvSolve(BaseVector<RightHandValType> &vec, UINT32 resStartIdx) = 0;

    /** @brief 求解 vecOUT = M^{-1} * vecIN，不更改输入向量的值，但是会更改输出向量的值，并调整vec.length == M.dim
     * @param [in] vecIN: 输入向量
     * @param [in, out] vecOUT: 输出向量 */
    virtual void MInvSolve(BaseVector<RightHandValType> &vecIN, BaseVector<RightHandValType> &vecOUT) = 0;

    /** @brief 求解 vecOUT = M^{-1} * vecIN，不更改输入向量的值，但是会更改输出向量的值
     * @param [in] vecIN: 输入向量
     * @param [in] inStartIdx:  要求的片段在输入向量的起始位置
     * @param [in, out] vecOUT: 输出向量
     * @param [in] outStartIdx: 要输出的片段在输出向量的起始位置 */
    virtual void MInvSolve(BaseVector<RightHandValType> &vecIN, UINT32 inStartIdx, BaseVector<RightHandValType> &vecOUT,
                           UINT32 outStartIdx) = 0;

    inline const PreconditionType_t &getPreconditionType() const {
        return m_precondType;
    }

    UINT32 getPreconditionNonZeroCount() const {
        return m_Mnnz;
    }

    UINT32 getOriginalNonZeroCount() const {
        return m_Annz;
    }

    FLOAT64 getPreconditionFillinRatio() const {
        return m_Mnnz / static_cast<FLOAT64>(m_Annz);
    }
};


/** @brief 针对三角分解类型的预条件设计基类及其公共接口，主要针对ILU、IC、ILDLT这三类方法
 * @note
 *      ILU：分解形式为A = L * U，L表示下三角矩阵，U表示上三角矩阵（后续出现的相同符号的含义不变）；
 *      IC：分解形式为A = L * L^{T} = L * U；
 *      ILDLT：分解形式A = L * D * L^{T} = (L * D) * L^{T} = L * U；
 *      因此，针对三角分解形式的预条件方法，要求A^{-1} * x（x表示任意和A同维度的向量），即求两个子式：
 *          1. L * y = x；
 *          2. U * z = y。
 *      最后，z 即为 A^{-1} * x。
 *      对于这类方法，我们只需要在子类实现三角求解的具体细节，在顶层调用对应求解接口即可。*/
template<typename RightHandValType>
class TriangularPrecondition : public BasePrecondition<RightHandValType> {
protected:
    /* protected下面这几个接口必须在子类中实现 */
    /** @brief 计算预条件矩阵上三角部分对应的稀疏三角方程组，即vec = U^{-1} * vec */
    virtual void MSolveUpperUsePtr(RightHandValType *vec) = 0;

    /** @brief 计算对角线对应的简单线性方程组，即vec = D^{-1} * vec
     * @attention 目前只有ILDLT需要实例化这个步骤，其他不用 */
    virtual void MSolveDiagonalUsePtr(RightHandValType *vec) = 0;

    /** @brief 计算预条件矩阵下三角部分对应的稀疏三角方程组，即vec = L^{-1} * vec */
    virtual void MSolveLowerUsePtr(RightHandValType *vec) = 0;

public:
    TriangularPrecondition() = default;

    ~TriangularPrecondition() override = default;

    TriangularPrecondition(const TriangularPrecondition &other)
        : BasePrecondition<RightHandValType>(other) {
    }

    TriangularPrecondition(TriangularPrecondition &&other) noexcept
        : BasePrecondition<RightHandValType>(std::move(other)) {
    }

    TriangularPrecondition & operator=(const TriangularPrecondition &other) {
        if (this == &other)
            return *this;
        BasePrecondition<RightHandValType>::operator =(other);
        return *this;
    }

    TriangularPrecondition & operator=(TriangularPrecondition &&other) noexcept {
        if (this == &other)
            return *this;
        BasePrecondition<RightHandValType>::operator =(std::move(other));
        return *this;
    }

    /** @brief 计算预条件矩阵上三角部分对应的稀疏三角方程组，即vec = L^{-T} vec */
    void MSolveUpper(BaseVector<RightHandValType> &vec) {
#ifndef NDEBUG
        THROW_EXCEPTION(vec.getLength() != this->m_ArowNum, THROW_LOGIC_ERROR("The dim of in-vec is incompatible!"))
#endif
        RightHandValType *dataPtr = vec.getRawValPtr();
        MSolveUpperUsePtr(dataPtr);
    }

    void MSolveUpper(BaseVector<RightHandValType> &vec, UINT32 resStartIdx) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_ArowNum + resStartIdx > vec.getLength(),
                        THROW_LOGIC_ERROR("The dim of in-vec is incompatible!"))
#endif
        RightHandValType *dataPtr = vec.getRawValPtr() + resStartIdx;
        MSolveUpperUsePtr(dataPtr);
    }

    void MSolveUpper(const BaseVector<RightHandValType> &vecIN, BaseVector<RightHandValType> &vecOUT) {
#ifndef NDEBUG
        THROW_EXCEPTION(vecIN.getLength() != this->m_ArowNum, THROW_LOGIC_ERROR("The dim of in-vec is incompatible!"))
#endif
        if (vecOUT.getLength() < this->m_ArowNum) vecOUT.resize(this->m_ArowNum, RESERVE_NO_DATA);
        vecOUT.copy(vecIN);
        RightHandValType *dataPtr = vecOUT.getRawValPtr();
        MSolveUpperUsePtr(dataPtr);
    }

    void MSolveUpper(BaseVector<RightHandValType> &vecIN, UINT32 inStartIdx, BaseVector<RightHandValType> &vecOUT,
                     UINT32 outStartIdx) {
        // copy里面有越界检查，这里省去
        vecOUT.copy(vecIN, inStartIdx, outStartIdx, this->m_ArowNum);
        RightHandValType *outVecPtr = vecOUT.getRawValPtr() + outStartIdx;
        MSolveUpperUsePtr(outVecPtr);
    }

    void MSolveDiagonal(BaseVector<RightHandValType> &vec) {
#ifndef NDEBUG
        THROW_EXCEPTION(vec.getLength() != this->m_ArowNum, THROW_LOGIC_ERROR("The dim of in-vec is incompatible!"))
#endif
        RightHandValType *dataPtr = vec.getRawValPtr();
        MSolveDiagonalUsePtr(dataPtr);
    }

    void MSolveDiagonal(BaseVector<RightHandValType> &vec, UINT32 resStartIdx) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_ArowNum + resStartIdx > vec.getLength(),
                        THROW_LOGIC_ERROR("The dim of in-vec is incompatible!"))
#endif
        RightHandValType *dataPtr = vec.getRawValPtr() + resStartIdx;
        MSolveDiagonalUsePtr(dataPtr);
    }

    void MSolveDiagonal(const BaseVector<RightHandValType> &vecIN, BaseVector<RightHandValType> &vecOUT) {
#ifndef NDEBUG
        THROW_EXCEPTION(vecIN.getLength() != this->m_ArowNum, THROW_LOGIC_ERROR("The dim of in-vec is incompatible!"))
#endif
        if (vecOUT.getLength() < this->m_ArowNum) vecOUT.resize(this->m_ArowNum, RESERVE_NO_DATA);
        vecOUT.copy(vecIN);
        RightHandValType *dataPtr = vecOUT.getRawValPtr();
        MSolveDiagonalUsePtr(dataPtr);
    }

    void MSolveDiagonal(BaseVector<RightHandValType> &vecIN, UINT32 inStartIdx, BaseVector<RightHandValType> &vecOUT,
                        UINT32 outStartIdx) {
        // copy里面有越界检查，这里省去
        vecOUT.copy(vecIN, inStartIdx, outStartIdx, this->m_ArowNum);
        RightHandValType *outVecPtr = vecOUT.getRawValPtr() + outStartIdx;
        MSolveDiagonalUsePtr(outVecPtr);
    }

    void MSolveLower(BaseVector<RightHandValType> &vec) {
#ifndef NDEBUG
        THROW_EXCEPTION(vec.getLength() != this->m_ArowNum, THROW_LOGIC_ERROR("The dim of in-vec is incompatible!"))
#endif
        RightHandValType *dataPtr = vec.getRawValPtr();
        MSolveLowerUsePtr(dataPtr);
    }

    void MSolveLower(BaseVector<RightHandValType> &vec, UINT32 resStartIdx) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_ArowNum + resStartIdx > vec.getLength(),
                        THROW_LOGIC_ERROR("The dim of in-vec is incompatible!"))
#endif
        RightHandValType *dataPtr = vec.getRawValPtr() + resStartIdx;
        MSolveLowerUsePtr(dataPtr);
    }

    void MSolveLower(const BaseVector<RightHandValType> &vecIN, BaseVector<RightHandValType> &vecOUT) {
#ifndef NDEBUG
        THROW_EXCEPTION(vecIN.getLength() != this->m_ArowNum, THROW_LOGIC_ERROR("The dim of in-vec is incompatible!"))
#endif
        if (vecOUT.getLength() < this->m_ArowNum) vecOUT.resize(this->m_ArowNum, RESERVE_NO_DATA);
        vecOUT.copy(vecIN);
        RightHandValType *dataPtr = vecOUT.getRawValPtr();
        MSolveLowerUsePtr(dataPtr);
    }

    void MSolveLower(BaseVector<RightHandValType> &vecIN, UINT32 inStartIdx, BaseVector<RightHandValType> &vecOUT,
                     UINT32 outStartIdx) {
        // copy里面有越界检查，这里省去
        vecOUT.copy(vecIN, inStartIdx, outStartIdx, this->m_ArowNum);
        RightHandValType *outVecPtr = vecOUT.getRawValPtr() + outStartIdx;
        MSolveLowerUsePtr(outVecPtr);
    }

    /** @brief 求解 vec = (L * U)^{-1} * vec，最后的结果覆盖原有值
     * @param [in,out] vec: 输入向量 */
    void MInvSolve(BaseVector<RightHandValType> &vec) override {
#ifndef NDEBUG
        THROW_EXCEPTION(vec.getLength() != this->m_ArowNum, THROW_LOGIC_ERROR("The dim of in-vec is incompatible!"))
#endif
        RightHandValType *dataPtr = vec.getRawValPtr();
        MSolveLowerUsePtr(dataPtr);
        if (this->m_precondType == PreconditionILDLT) MSolveDiagonalUsePtr(dataPtr);
        MSolveUpperUsePtr(dataPtr);
    }

    /** @brief 求解 vec = (L * U)^{-1} * vec，最后的结果覆盖原有值，但是只转换x的一部分值（vec.length > M.dim）
     * @param [in,out] vec: 输入向量
     * @param [in] resStartIdx: 要转换的片段在x中的起始位置 */
    void MInvSolve(BaseVector<RightHandValType> &vec, UINT32 resStartIdx) override {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_ArowNum + resStartIdx > vec.getLength(),
                        THROW_LOGIC_ERROR("The dim of in-vec is incompatible!"))
#endif
        RightHandValType *dataPtr = vec.getRawValPtr() + resStartIdx;
        MSolveLowerUsePtr(dataPtr);
        if (this->m_precondType == PreconditionILDLT) MSolveDiagonalUsePtr(dataPtr);
        MSolveUpperUsePtr(dataPtr);
    }

    /** @brief 求解 vecOUT = (L * U)^{-1} * vecIN，不更改输入向量的值，但是会更改输出向量的值，并调整vec.length == M.dim
     * @param [in] vecIN: 输入向量
     * @param [in, out] vecOUT: 输出向量 */
    void MInvSolve(BaseVector<RightHandValType> &vecIN, BaseVector<RightHandValType> &vecOUT) override {
#ifndef NDEBUG
        THROW_EXCEPTION(vecIN.getLength() != this->m_ArowNum, THROW_LOGIC_ERROR("The dim of in-vec is incompatible!"))
#endif
        if (vecOUT.getLength() < this->m_ArowNum) vecOUT.resize(this->m_ArowNum, RESERVE_NO_DATA);
        vecOUT.copy(vecIN);
        RightHandValType *dataPtr = vecOUT.getRawValPtr();
        MSolveLowerUsePtr(dataPtr);
        if (this->m_precondType == PreconditionILDLT) MSolveDiagonalUsePtr(dataPtr);
        MSolveUpperUsePtr(dataPtr);
    }

    /** @brief 求解 vecOUT = (L * U)^{-1} * vecIN，不更改输入向量的值，但是会更改输出向量的值
     * @param [in] vecIN: 输入向量，一般是个比较长的右端项，当前答案只占其中某个片段
     * @param [in] inStartIdx:  要求的片段在输入向量的起始位置
     * @param [in, out] vecOUT: 输出向量，一般是个比较长的右端项，当前答案只占其中某个片段
     * @param [in] outStartIdx: 要输出的片段在输出向量的起始位置 */
    void MInvSolve(BaseVector<RightHandValType> &vecIN, UINT32 inStartIdx, BaseVector<RightHandValType> &vecOUT,
                   UINT32 outStartIdx) override {
        vecOUT.copy(vecIN, inStartIdx, outStartIdx, this->m_ArowNum);
        RightHandValType *dataPtr = vecOUT.getRawValPtr() + outStartIdx;
        MSolveLowerUsePtr(dataPtr);
        if (this->m_precondType == PreconditionILDLT) MSolveDiagonalUsePtr(dataPtr);
        MSolveUpperUsePtr(dataPtr);
    }
};


#endif //PMSLS_NEW_BASEPRECONDITON_H
