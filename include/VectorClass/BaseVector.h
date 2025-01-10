#ifndef VECTOR_BASE_H
#define VECTOR_BASE_H

#include "../../config/config.h"
#include "../../config/debug.h"
#include "../../config/headers.h"
#include "../../include/utils/ErrorHandler.h"

#if  CUDA_ENABLED

#include "../../config/CUDAheaders.cuh"

#endif


#define RESERVE_DATA 1
#define RESERVE_NO_DATA 0
#define HOST_SPACE (-1)

enum memoryType {
    memoryBase, ///< CPU上内存，表示使用一般的内存（直接通过malloc或者new申请的数组）
    memoryAligned, ///< CPU上内存，表示对齐的内存
    memoryPageLocked, ///< CPU上内存，表示锁页内存（主要是在异构并行时使用）
    memoryDeviceSync ///< GPU上的内存
};

typedef enum memoryType memoryType_t;

/** @brief 向量基类，定义公有方法的接口 */
template <class ValType>
class BaseVector {
protected:
    INT32 m_location{HOST_SPACE}; ///< 内存空间所在的位置，-1表示CPU，非负整数对应GPU编号
    UINT32 m_length{0};
    size_t m_byteSize{0};
    ValType* m_valuesPtr{nullptr};
    memoryType_t m_memoryType{memoryBase};

public:
    BaseVector(const BaseVector& other) = delete;

    BaseVector& operator=(const BaseVector& other) = delete;

    BaseVector(BaseVector&& other) noexcept
        : m_location(other.m_location),
          m_length(other.m_length),
          m_byteSize(other.m_byteSize),
          m_valuesPtr(other.m_valuesPtr),
          m_memoryType(other.m_memoryType) {
        other.m_length = 0;
        other.m_byteSize = 0;
        other.m_valuesPtr = nullptr;
    }

    BaseVector& operator=(BaseVector&& other) noexcept {
        if (this == &other)
            return *this;
#ifndef NDEBUG
        if ((other.m_location == HOST_SPACE || m_location == HOST_SPACE) && m_location != other.m_location) {
            SHOW_WARN("The current two vectors are on different devices, move operation is denied!")
            return *this;
        }
#endif
        m_location = other.m_location;
        m_length = other.m_length;
        m_byteSize = other.m_byteSize;
        m_valuesPtr = other.m_valuesPtr;
        m_memoryType = other.m_memoryType;
        other.m_length = 0;
        other.m_byteSize = 0;
        other.m_valuesPtr = nullptr;
        return *this;
    }

    BaseVector() = default;

    virtual ~BaseVector() = default; // 因为子类有不同的内存释放方法，所以这里设置为default，后续子类会用override覆盖

    inline UINT32 getLength() const {
        return m_length;
    }

    /** @brief 获取当前向量的总字节数。 */
    inline size_t getByteSize() const {
        return m_byteSize;
    }

    /** @brief 获取当前内存类型。 */
    inline memoryType_t getMemoryType() const {
        return m_memoryType;
    }

    inline ValType* getRawValPtr() const {
        return m_valuesPtr;
    }

    inline INT32 getLocation() const {
        return m_location;
    }

    inline ValType& operator[](const UINT32& idx) {
        return m_valuesPtr[idx];
    }

    inline const ValType& operator[](const UINT32& idx) const {
        return m_valuesPtr[idx];
    }


    virtual void printVector(const char* message) const = 0;

    // 其他函数
    /** @brief 调整当前向量的长度
     * @attention 各子类在实现resize时务必更新基类相关变量
     * @param [in] newLen: 调整后的新长度
     * @param [in] needReserve: 是否需要保留原先存储的值 */
    virtual void resize(const UINT32& newLen, UINT8 needReserve) = 0;

    /** @brief 清空当前向量的存储空间，并把所有标量清空为默认状态（指针置为nullptr，其他变量置为0）*/
    virtual void clear() = 0;


    virtual void copy(const BaseVector<ValType>& vec) = 0;

    virtual void copy(const BaseVector<ValType>& vec, UINT32 src_start, UINT32 dst_start,
                      UINT32 length) = 0;
};


template <class ValType>
class HostVector : public BaseVector<ValType> {
protected:
    inline void setup(UINT32 length) {
        this->resize(length, RESERVE_NO_DATA);
    }

public:
    // 构造函数与析构函数
    /** @brief 基类无参构造函数。 */
    HostVector() = default;

    /** @brief 基类拷贝构造函数。 */
    HostVector(const HostVector<ValType>& pre_vec) {
        setup(pre_vec.m_length);
        copy(pre_vec);
    };

    /** @brief 基类移动构造函数。 */
    HostVector(HostVector<ValType>&& pre_vec) noexcept {
        move(pre_vec);
    }

    /** @brief 基类析构函数。 */
    ~HostVector() override = default;

    // get 函数
    /** @brief 获取第index位置的值。 */
    inline ValType getValue(const UINT32& index) const {
        return this->m_valuesPtr[index];
    }

    // set 函数
    /** @brief 向第index个位置赋值。
     * @param [in] index: 需要写入的元素位置（下标从0开始）
     * @param [in] val: 具体写入的值 */
    inline void setValue(const UINT32& index, const ValType& val) {
        this->m_valuesPtr[index] = val;
    }

    // 运算符重载
    HostVector& operator=(const HostVector<ValType>& pre) {
        if (&pre == this)
            return *this;
        if (BaseVector<ValType>::m_length != pre.m_length) {
            this->resize(pre.m_length, RESERVE_NO_DATA);
        }
        memcpy(BaseVector<ValType>::m_valuesPtr, pre.m_valuesPtr, pre.m_byteSize);
        return *this;
    }

    HostVector& operator=(HostVector<ValType>&& pre) noexcept {
        if (&pre == this)
            return *this;
        move(pre);
        return *this;
    }


    /** @brief 各派生类的顶层调用空间移动函数，支持异型空间移动（但必须都在CPU上）
     * @attention 需要注意不同内存类型的空间不能直接移动，必须拷贝其值再清空，直接移动空间可能会导致析构出现异常 */
    void move(HostVector<ValType>& vec) {
        if (BaseVector<ValType>::m_memoryType != vec.m_memoryType) {
            if (BaseVector<ValType>::m_length != vec.m_length) this->resize(vec.m_length, RESERVE_NO_DATA);
            memcpy(BaseVector<ValType>::m_valuesPtr, vec.m_valuesPtr, vec.m_byteSize);
            vec.clear();
        } else {
            if (BaseVector<ValType>::m_valuesPtr) this->clear(); // 因为要移动pre的空间到本地，如果本地空间不清空则会造成内存泄漏
            BaseVector<ValType>::m_valuesPtr = vec.m_valuesPtr;
            BaseVector<ValType>::m_length = vec.m_length;
            BaseVector<ValType>::m_byteSize = vec.m_byteSize;
            vec.m_byteSize = 0;
            vec.m_valuesPtr = nullptr;
            vec.m_length = 0;
        }
    }

    /** @brief 各派生类的顶层调用空间移动函数，支持异型空间移动（但必须都在CPU上）
     * @attention 需要注意不同内存类型的空间不能直接移动，必须拷贝其值再清空，直接移动空间可能会导致析构出现异常 */
    void move(BaseVector<ValType>& vec) {
        if (vec.getLocation() != HOST_SPACE) { return; }
        if (BaseVector<ValType>::m_memoryType != vec.getMemoryType()) {
            if (BaseVector<ValType>::m_length != vec.getLength()) this->resize(vec.getLength(), RESERVE_NO_DATA);
            memcpy(BaseVector<ValType>::m_valuesPtr, vec.getRawValPtr(), vec.getByteSize());
            vec.clear();
        } else {
            if (BaseVector<ValType>::m_valuesPtr) this->clear(); // 因为要移动pre的空间到本地，如果本地空间不清空则会造成内存泄漏
            BaseVector<ValType>::operator=(std::move(vec));
        }
    }

    /** @brief 用来拷贝向量的函数，和move函数一样是一个顶层调用，主要在多态情况下使用
     * @attention 拷贝操作以最短的向量为准 */
    inline void copy(const BaseVector<ValType>& vec) override {
        size_t actual_byte = this->m_byteSize <= vec.getByteSize() ? this->m_byteSize : vec.getByteSize();
#if CUDA_ENABLED
        if (vec.getMemoryType() == memoryDeviceSync) {
            CHECK_CUDA(cudaMemcpy(this->m_valuesPtr, vec.getRawValPtr(), actual_byte, cudaMemcpyDeviceToHost))
            return;
        }
#endif
        memcpy(this->m_valuesPtr, vec.getRawValPtr(), actual_byte);
    }

    /** @brief 用来拷贝向量的函数，和move函数一样是一个顶层调用，主要在多态情况下使用
     * @attention 拷贝操作以最短的向量为准
     * @param [in] vec: 被拷贝的向量；
     * @param [in] src_start: 被拷贝部分的起始下标；
     * @param [in] dst_start: 拷贝至当前向量的起始下标；
     * @param [in] length: 拷贝的元素个数 */
    inline void copy(const BaseVector<ValType>& vec, UINT32 src_start, UINT32 dst_start,
                     UINT32 length) override {
#ifndef NDEBUG
        if (src_start >= vec.getLength() || src_start + length > vec.getLength() ||
            dst_start >= this->m_length || dst_start + length > this->m_length) {
            SHOW_ERROR("The start position or copied length is out-of-range!")
            std::cerr << " --- dst_start: " << dst_start << ", dst_length: " << this->m_length << ", src_start: "
                << src_start << ", src_length: " << vec.getLength() << ", copied length: " << length
                << std::endl;
            exit(EXIT_FAILURE);
        }
#endif
#if CUDA_ENABLED
        if (vec.getMemoryType() == memoryDeviceSync) {
            CHECK_CUDA(cudaMemcpy(this->m_valuesPtr, vec.getRawValPtr(), length * sizeof(ValType),
                                  cudaMemcpyDeviceToHost))
            return;
        }
#endif
        memcpy(&this->m_valuesPtr[dst_start], &vec[src_start], length * sizeof(ValType));
    }

    inline void printVector(const char* message) const override {
        std::string type = BaseVector<ValType>::m_memoryType == memoryBase
                               ? "Base"
                               : BaseVector<ValType>::m_memoryType == memoryAligned
                               ? "Aligned"
                               : "Page-locked";
        std::cout << L_GREEN << "[memory-type: " << type << "]" << L_PURPLE << "(" << message << ")"
            << COLOR_NONE;
        UINT32 maxLength = 30 >= BaseVector<ValType>::m_length ? BaseVector<ValType>::m_length : 30;
        for (UINT32 i = 0; i < maxLength; ++i) {
            std::cout << "  " << getValue(i);
        }
        if (BaseVector<ValType>::m_length > maxLength)
            std::cout << L_CYAN << " ... (the rest values were folded)" << COLOR_NONE;
        std::cout << std::endl;
    }

    /** @brief 实现vec3 = a * vec1 + b * vec2 */
    void add(ValType coef1, const HostVector<ValType>& vec1, ValType coef2, const HostVector<ValType>& vec2) {
#ifndef NDEBUG
        if (vec1.m_length != this->m_length || vec2.m_length != this->m_length) {
            SHOW_ERROR("Function add -> \"vec3 = a * vec1 + b * vec2\" may be applied incompatible vectors! "
                "Check length again!")
            return;
        }
#endif
        ValType sum;
#ifdef USE_OMP_VECTOR_FUNC
        if (this->m_length >= 1e4) {
#pragma omp parallel for default(none) num_threads(THREAD_NUM) shared(BaseVector<ValType>::m_length, vec1, vec2, coef1, coef2) private(sum)
            for (INT32 i = 0; i < BaseVector<ValType>::m_length; ++i) {
                sum = coef1 * vec1[i] + coef2 * vec2[i];
                setValue(i, sum);
            }
            return;
        }
#endif
        for (INT32 i = 0; i < BaseVector<ValType>::m_length; ++i) {
            sum = coef1 * vec1[i] + coef2 * vec2[i];
            setValue(i, sum);
        }
    }

    /** @brief 实现vec3 = a * vec1 + b * vec2 */
    void add(ValType coef1, const HostVector<ValType>& vec1, UINT32 srcStartIdx1,
             ValType coef2, const HostVector<ValType>& vec2, UINT32 srcStartIdx2,
             UINT32 dstStartIdx, UINT32 length) {
#ifndef NDEBUG
        THROW_EXCEPTION(srcStartIdx1 + length > vec1.getLength() || dstStartIdx + length > this->m_length ||
                        srcStartIdx2 + length > vec2.getLength(),
                        THROW_OUT_OF_RANGE("The length or start idx was incorrect!"))
#endif

#ifdef USE_OMP_VECTOR_FUNC
        if (length >= 1e4) {
#pragma omp parallel for default(none) num_threads(THREAD_NUM) \
    shared(length, BaseVector<ValType>::m_valuesPtr, vec1, vec2, coef1, coef2, srcStartIdx1, srcStartIdx2, dstStartIdx)
            for (INT32 i = 0; i < length; ++i) {
                this->m_valuesPtr[dstStartIdx + i] = coef1 * vec1[srcStartIdx1 + i] + coef2 * vec2[srcStartIdx2 + i];
            }
            return;
        }
#endif
        for (INT32 i = 0; i < length; ++i) {
            this->m_valuesPtr[dstStartIdx + i] = coef1 * vec1[srcStartIdx1 + i] + coef2 * vec2[srcStartIdx2 + i];
        }
    }

    /** @brief 实现vec2 += a * vec1 */
    void add(ValType coef, const HostVector<ValType>& vec) {
#ifndef NDEBUG
        if (vec.m_length != this->m_length) {
            SHOW_ERROR("Function add -> \"vec3 = a * vec1 + b * vec2\" may be applied incompatible vectors! "
                "Check length again!")
            return;
        }
#endif

        ValType sum;
#ifdef USE_OMP_VECTOR_FUNC
        if (this->m_length >= 1e4) {
#pragma omp parallel for default(none) num_threads(THREAD_NUM)  shared(BaseVector<ValType>::m_length, coef, vec) private(sum)
            for (INT32 i = 0; i < BaseVector<ValType>::m_length; ++i) {
                sum = getValue(i);
                sum += coef * vec.getValue(i);
                setValue(i, sum);
            }
            return;
        }
#endif
        for (INT32 i = 0; i < BaseVector<ValType>::m_length; ++i) {
            sum = getValue(i);
            sum += coef * vec.getValue(i);
            setValue(i, sum);
        }
    }

    /** @brief 实现vec2[dstStartIdx:dstStartIdx+length-1] += coef * vec1[srcStartIdx:srcStartIdx+length-1] */
    void add(ValType coef, const HostVector<ValType>& vec, UINT32 srcStartIdx, UINT32 dstStartIdx, UINT32 length) {
#ifndef NDEBUG
        THROW_EXCEPTION(srcStartIdx + length > vec.getLength() || dstStartIdx + length > this->m_length,
                        THROW_OUT_OF_RANGE("The length or start idx was incorrect!"))
#endif

#ifdef USE_OMP_VECTOR_FUNC
        if (length >= 1e4) {
#pragma omp parallel for default(none) num_threads(THREAD_NUM)  \
    shared(length, BaseVector<ValType>::m_valuesPtr, dstStartIdx, srcStartIdx, coef, vec)
            for (INT32 i = 0; i < length; ++i) {
                this->m_valuesPtr[dstStartIdx + i] += coef * vec[srcStartIdx + i];
            }
            return;
        }
#endif
        for (INT32 i = 0; i < length; ++i) {
            this->m_valuesPtr[dstStartIdx + i] += coef * vec[srcStartIdx + i];
        }
    }

    inline void fillVector(UINT32 start, UINT32 length, ValType value) {
#ifdef NDEBUG
        ERROR_CHECK(start + length > BaseVector<ValType>::m_length, DEBUG_MESSEGE_OPTION,
                    "The length is out-of-range!");
#endif
        std::fill_n(&(*this)[start], length, value);
    }

    inline void scale(const ValType& coef) {
        UINT32 vecLength = this->m_length;
        ValType temp;
#ifdef USE_OMP_VECTOR_FUNC
        if (this->m_length >= 1e4) {
#pragma omp parallel for default(none) num_threads(THREAD_NUM)  shared(vecLength, coef) private(temp)
            for (UINT32 i = 0; i < vecLength; ++i) {
                temp = getValue(i);
                temp *= coef;
                setValue(i, temp);
            }
            return;
        }
#endif
        for (UINT32 i = 0; i < vecLength; ++i) {
            temp = getValue(i);
            temp *= coef;
            setValue(i, temp);
        }
    }

    inline void scale(const ValType& coef, UINT32 startIdx, UINT32 length) {
        UINT32 endIdx = startIdx + length;
#ifndef NDEBUG
        THROW_EXCEPTION(endIdx > this->m_length, THROW_OUT_OF_RANGE("The end bound is out-of-range!"))
#endif
        ValType temp;
#ifdef USE_OMP_VECTOR_FUNC
        if (length >= 1e4) {
#pragma omp parallel for default(none) num_threads(THREAD_NUM) shared(BaseVector<ValType>::m_valuesPtr, length, coef, startIdx, endIdx)
            for (UINT32 i = startIdx; i < endIdx; ++i) {
                BaseVector<ValType>::m_valuesPtr[i] *= coef;
            }
            return;
        }
#endif
        for (UINT32 i = startIdx; i < endIdx; ++i) {
            this->m_valuesPtr[i] *= coef;
        }
    }

    inline FLOAT64 innerProduct(const HostVector<ValType>& vec) {
#ifndef NDEBUG
        THROW_EXCEPTION(this->m_length != vec.getLength(),
                        THROW_RANGE_ERROR("The dim of two vector are not compatible!"))
#endif
        FLOAT64 out_num = 0.0;
        UINT32 i, len;
        len = this->m_length;
#ifdef USE_OMP_VECTOR_FUNC
        if (len >= 1e4) {
#pragma omp parallel for default(none) num_threads(THREAD_NUM)  shared(vec, len) reduction(+:out_num)
            for (i = 0; i < len; ++i) {
                out_num += getValue(i) * vec[i];
            }
            return out_num;
        }
#endif
        for (i = 0; i < len; ++i) {
            out_num += getValue(i) * vec[i];
        }
        return out_num;
    }

    inline FLOAT64 innerProduct(UINT32 localStartIdx, const ValType* vecStartPtr, UINT32 localLength) {
        FLOAT64 out_num = 0;
        ValType* localStartPtr = this->m_valuesPtr + localStartIdx;
#ifdef USE_OMP_VECTOR_FUNC
        if (localLength >= 1e4) {
#pragma omp parallel for default(none) num_threads(THREAD_NUM)  shared(localLength, localStartPtr, vecStartPtr) reduction(+:out_num)
            for (UINT32 i = 0; i < localLength; ++i) {
                out_num += localStartPtr[i] * vecStartPtr[i];
            }
            return out_num;
        }
#endif
        for (UINT32 i = 0; i < localLength; ++i) {
            out_num += localStartPtr[i] * vecStartPtr[i];
        }
        return out_num;
    }

    inline FLOAT64 innerProduct(const HostVector<ValType>& vec, UINT32 vecStartIdx, UINT32 localStartIdx,
                                UINT32 localLength) {
#ifndef NDEBUG
        THROW_EXCEPTION((localStartIdx + localLength > this->m_length) || (vecStartIdx + localLength > vec.getLength()),
                        THROW_RANGE_ERROR("The dim of two vector are not compatible!"))
#endif
        FLOAT64 out_num = 0;
        ValType* localStartPtr = this->m_valuesPtr + localStartIdx;
        ValType* vecStartPtr = vec.getRawValPtr() + vecStartIdx;
#ifdef USE_OMP_VECTOR_FUNC
        if (localLength >= 1e4) {
#pragma omp parallel for default(none) num_threads(THREAD_NUM)  shared(localLength, localStartPtr, vecStartPtr) reduction(+:out_num)
            for (UINT32 i = 0; i < localLength; ++i) {
                out_num += localStartPtr[i] * vecStartPtr[i];
            }
            return out_num;
        }
#endif
        for (UINT32 i = 0; i < localLength; ++i) {
            out_num += localStartPtr[i] * vecStartPtr[i];
        }
        return out_num;
    }

    inline FLOAT64 innerProductHighPrecision(const HostVector<ValType>& vec) {
        FLOAT64 local_sum{0.0};
        FLOAT64 local_correction{0.0};
        FLOAT64 corrected_next_term{0.0}, new_sum{0.0};
        for (UINT32 i = 0; i < this->m_length; ++i) {
            corrected_next_term = this->m_valuesPtr[i] * vec[i] + local_correction;
            new_sum = local_sum + corrected_next_term;
            // 更新局部校正值，以进行下一次迭代
            local_correction = corrected_next_term - (new_sum - local_sum);
            local_sum = new_sum;
        }
        return local_sum + local_correction;
    }

    FLOAT64 norm_2() const {
        FLOAT64 out_num = 0;
        UINT32 i, len;
        ValType value;
        len = BaseVector<ValType>::m_length;
#ifdef USE_OMP_VECTOR_FUNC
        if (BaseVector<ValType>::m_length >= 1e4) {
#pragma omp parallel for default(none) num_threads(THREAD_NUM)  private(value) shared(len) reduction(+:out_num)
            for (i = 0; i < len; ++i) {
                value = getValue(i);
                out_num += value * value;
            }
            out_num = sqrt(out_num);
            return out_num;
        }
#endif
        for (i = 0; i < len; ++i) {
            value = getValue(i);
            out_num += value * value;
        }
        out_num = sqrt(out_num);
        return out_num;
    }

    ValType norm_1() const {
        ValType out_num = 0;
        UINT32 curr_len, i;
        ValType curr_val;
        curr_len = BaseVector<ValType>::m_length;
#ifdef USE_OMP_VECTOR_FUNC
        if (curr_len >= 1e4) {
#pragma omp parallel for default(none) num_threads(THREAD_NUM)  private(curr_val) shared(curr_len) reduction(+:out_num)
            for (i = 0; i < curr_len; ++i) {
                curr_val = getValue(i);
                out_num += curr_val < 0 ? curr_val * -1.0 : curr_val;
            }
            return out_num;
        }
#endif
        for (i = 0; i < curr_len; ++i) {
            curr_val = getValue(i);
            out_num += curr_val < 0 ? curr_val * -1.0 : curr_val;
        }
        return out_num;
    }

    ValType norm_inf() const {
        ValType max_value;
        UINT32 curr_len = BaseVector<ValType>::m_length;
        ValType curr_val;
#ifdef USE_OMP_VECTOR_FUNC
        if (curr_len >= 1e4) {
#pragma omp parallel for default(none) num_threads(THREAD_NUM)  private(curr_val) shared(curr_len) reduction(max:max_value)
            for (UINT32 i = 0; i < curr_len; ++i) {
                curr_val = getValue(i);
                curr_val = curr_val < 0 ? -1.0 * curr_val : curr_val;
                if (i == 0 || curr_val > max_value) {
                    max_value = curr_val;
                }
            }
            return max_value;
        }
#endif
        for (UINT32 i = 0; i < curr_len; ++i) {
            curr_val = getValue(i);
            curr_val = curr_val < 0 ? -1.0 * curr_val : curr_val;
            if (i == 0 || curr_val > max_value) {
                max_value = curr_val;
            }
        }
        return max_value;
    }

    ValType min() {
        ValType* min_val;
        ValType *start = &(*this)[0], *end = start + BaseVector<ValType>::m_length;
        min_val = std::min_element(start, end);
        return *min_val;
    }

    ValType max() {
        ValType* min_val;
        ValType* start = &(*this)[0];
        ValType* end = start + BaseVector<ValType>::m_length;
        min_val = std::max_element(start, end);
        return *min_val;
    }

    /** @brief 向量中的所有元素求和 */
    ValType sum(const UINT32& startIdx, const UINT32& length) {
        FLOAT64 out_num = 0;
        UINT32 i;
        ValType value;
        UINT32 maxIdx = startIdx + length;
#ifndef NDEBUG
        THROW_EXCEPTION(maxIdx > this->m_length, THROW_OUT_OF_RANGE("The sum function is out-of-range!"))
#endif
#pragma omp parallel for default(none) num_threads(THREAD_NUM)  private(value) shared(startIdx, maxIdx) reduction(+:out_num)
        for (i = startIdx; i < maxIdx; ++i) {
            value = getValue(i);
            out_num += value;
        }
        return out_num;
    }


    /** @brief 向量中的所有元素求和 */
    inline ValType sum() {
        return sum(0, BaseVector<ValType>::m_length);
    }

    /** @brief 高精度向量元素求和，消除舍入误差 */
    ValType sumKahan(const UINT32& startIdx, const UINT32& length) const {
        FLOAT64 local_sum{0.0};
        FLOAT64 local_correction{0.0};
        FLOAT64 corrected_next_term{0.0}, new_sum{0.0};
        UINT32 maxIdx = startIdx + length;
        for (UINT32 i = startIdx; i < maxIdx; ++i) {
            corrected_next_term = getValue(i) + local_correction;
            new_sum = local_sum + local_correction;
            // 更新局部校正值，以进行下一次迭代
            local_correction = corrected_next_term - (new_sum - local_sum);
            local_sum = new_sum;
        }
        return local_sum + local_correction;
    }

    ValType sumKahan(UINT32 startIdx, UINT32 length, const std::function<ValType(ValType)>& func) const {
        FLOAT64 local_sum{0.0};
        FLOAT64 local_correction{0.0};
        FLOAT64 corrected_next_term{0.0}, new_sum{0.0};
        UINT32 maxIdx = startIdx + length;
        for (UINT32 i = startIdx; i < maxIdx; ++i) {
            corrected_next_term = func(getValue(i)) + local_correction;
            new_sum = local_sum + corrected_next_term;
            // 更新局部校正值，以进行下一次迭代
            local_correction = corrected_next_term - (new_sum - local_sum);
            local_sum = new_sum;
        }
        return local_sum + local_correction;
    }

#ifdef USE_OMP_VECTOR_FUNC
    /** @brief 高精度向量元素求和，消除舍入误差
     * @attention release模式下，多线程版本的Kahan求和精度比串行版本精度高一点（debug模式下则相同，因为取消了浮点优化）*/
    ValType sumKahanOMP(UINT32 startIdx, UINT32 length, const std::function<ValType(ValType)>& func) {
        UINT32 maxIdx = startIdx + length;
        FLOAT64 res = 0.0;
        FLOAT64 correction = 0.0;

#pragma omp parallel num_threads(THREAD_NUM) proc_bind(close)
        {
            FLOAT64 local_sum = 0.0;
            FLOAT64 local_correction = 0.0;
            FLOAT64 corrected_next_term, new_sum;

#pragma omp for simd schedule(static) nowait
            for (UINT32 i = startIdx; i < maxIdx; ++i) {
                FLOAT64 value = func(getValue(i)); // 内联处理 func 和 getValue
                corrected_next_term = value + local_correction;
                new_sum = local_sum + corrected_next_term;
                local_correction = corrected_next_term - (new_sum - local_sum);
                local_sum = new_sum;
            }

            // 合并所有线程的局部累加结果到全局结果
#pragma omp critical
            {
                corrected_next_term = local_sum + local_correction + correction;
                new_sum = res + corrected_next_term;
                correction = corrected_next_term - (new_sum - res);
                res = new_sum;
            }
        }

        return res + correction;
    }

    /** @brief 一个用于已经分配并行域的多线程版本Kahan求和
     * @attention 注意这里传的global参数必须在申请并行域时初始化并设为共享变量，并且必须在并行域外就初始化为0
     * @param [in] startIdx: 全局角度上的向量求和的数组下标
     * @param [in] length:  全局角度上的向量求和的数据长度
     * @param [in, out] globalSum:  求和之后的结果（结果由master线程管理），必须初始化为0
     * @param [in, out] globalCorrection:   求和过程中的校正值（由master线程管理），必须初始化为0
     * @param [in] func: 单个数组元素的变换函数（比如要对每个元素进行平方后再求和） */
    void sumKahanOuterOMP(UINT32 startIdx, UINT32 length, FLOAT64& globalSum, FLOAT64& globalCorrection,
                          const std::function<ValType(ValType)>& func) {
#ifndef NDEBUG
        // THROW_EXCEPTION(omp_get_num_threads() == 1, THROW_LOGIC_ERROR("The parallel region was not initialized!"))
        THROW_EXCEPTION(globalSum != 0 || globalCorrection != 0,
                        THROW_LOGIC_ERROR("The global parameter was not initialized!"))
#endif
        UINT32 maxIdx = startIdx + length;
        /*
        不能在已经开启并行域的情况下再让每个线程都给共享变量赋值，如果没有同步操作，就会形成竞态条件，导致后面的汇总结果不准确。
        一个显而易见的情况：
            有i个线程已经成功更新了global变量，第i+1线程走的慢，这时执行了global变量的赋值，就导致原来的一部分累加被清空，
            求和过程有误。
         */
        // globalSum = 0.0;
        // globalCorrection = 0.0;
        FLOAT64 local_sum = 0.0;
        FLOAT64 local_correction = 0.0;
        FLOAT64 corrected_next_term, new_sum;

#pragma omp for simd schedule(static)
        for (UINT32 i = startIdx; i < maxIdx; ++i) {
            FLOAT64 value = func(getValue(i)); // 内联处理 func 和 getValue
            corrected_next_term = value + local_correction;
            new_sum = local_sum + corrected_next_term;
            local_correction = corrected_next_term - (new_sum - local_sum);
            local_sum = new_sum;
        }

        // 合并所有线程的局部累加结果到全局结果
#pragma omp critical
        {
            corrected_next_term = local_sum + local_correction + globalCorrection;
            new_sum = globalSum + corrected_next_term;
            globalCorrection = corrected_next_term - (new_sum - globalSum);
            globalSum = new_sum;
        }
    }

#endif
};


#endif /* VECTOR_BASE_H */
