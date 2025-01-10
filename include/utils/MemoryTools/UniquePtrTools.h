/*
 * @author  邓轶丹
 * @date    2024/5/25
 * @details UniquePtr封装工具
 */

#ifndef PMSLS_NEW_UNIQUEPTRTOOLS_H
#define PMSLS_NEW_UNIQUEPTRTOOLS_H

#include "../../../config/config.h"
#include "../../../config/debug.h"
#include "../../../config/headers.h"
#include "../ErrorHandler.h"


template<typename ObjectName, typename = std::enable_if_t<std::is_class<ObjectName>::value>>
class UniqueObject {
private:
    std::unique_ptr<ObjectName> m_ptr;

public:
    UniqueObject() = default;

    ~UniqueObject() = default;

    // 拷贝操作对于unique_ptr是不合法的，底层指针只能独占式使用
    UniqueObject(const UniqueObject<ObjectName> &pre) = delete;

    UniqueObject &operator=(const UniqueObject<ObjectName> &pre) = delete;

    UniqueObject(UniqueObject<ObjectName> &&pre) {
        m_ptr = std::move(pre.m_ptr);
    }

    UniqueObject &operator=(UniqueObject<ObjectName> &&pre) {
        if (this == &pre) return *this;
        m_ptr = std::move(pre.m_ptr);
        return *this;
    }

    inline ObjectName &operator*() {
        return *m_ptr;
    }

    inline const ObjectName &operator*() const {
        return *m_ptr;
    }

    inline std::unique_ptr<ObjectName> &operator->() {
#ifndef NDEBUG
        THROW_EXCEPTION(m_ptr.get() == nullptr,
                        THROW_INVALID_ARGUMENT("You are trying to use object via null pointer which is illegal!"))
#endif
        return m_ptr;
    }

    inline const std::unique_ptr<ObjectName> &operator->() const {
#ifndef NDEBUG
        THROW_EXCEPTION(m_ptr.get() == nullptr,
                        THROW_INVALID_ARGUMENT("You are trying to use object via null pointer which is illegal!"))
#endif
        return m_ptr;
    }

    template<typename ...ArgType>
    UniqueObject(ArgType &&...args) {
        // 通过std::forward将参数转发给ObjectName类的构造函数。
        m_ptr = std::make_unique<ObjectName>(std::forward<ArgType>(args)...);
    }

    /** @brief 用来调用对应类的构造函数，初始化当前对象 */
    template<typename ...ArgType>
    inline void construct(ArgType &&...args) {
        // 通过std::forward将参数转发给ObjectName类的构造函数。
        m_ptr = std::make_unique<ObjectName>(std::forward<ArgType>(args)...);
    }


    inline const std::unique_ptr<ObjectName> &get() const {
        return m_ptr;
    }

    inline std::unique_ptr<ObjectName> &get() {
        return m_ptr;
    }

};

/** @brief 由智能指针unique_ptr组成的一维数组 */
template<typename ObjectName, typename = std::enable_if_t<std::is_class<ObjectName>::value>>
class UniquePtr1D {
private:
    std::unique_ptr<std::unique_ptr<ObjectName>[]> m_ptr;
    INT32 m_dim{0};

public:
    UniquePtr1D() = default;

    explicit UniquePtr1D(const INT32 &dim) {
        m_ptr = std::make_unique<std::unique_ptr<ObjectName>[]>(dim);
        m_dim = dim;
    }

    UniquePtr1D(const UniquePtr1D<ObjectName> &pre) = delete;

    UniquePtr1D(UniquePtr1D<ObjectName> &&pre) {
        m_ptr = std::move(pre.m_ptr);
        m_dim = pre.m_dim;
    }

    inline std::unique_ptr<ObjectName> &operator[](const INT32 &idx) {
#ifndef NDEBUG
        THROW_EXCEPTION(idx < 0 || idx >= m_dim, THROW_OUT_OF_RANGE("The idx for UniquePtr1D is out-of-range!"))
#endif
        return m_ptr[idx];
    }

    inline std::unique_ptr<ObjectName> &operator[](const INT32 &idx) const {
#ifndef NDEBUG
        THROW_EXCEPTION(idx < 0 || idx >= m_dim, THROW_OUT_OF_RANGE("The idx for UniquePtr1D is out-of-range!"))
#endif
        return m_ptr[idx];
    }

    UniquePtr1D &operator=(const UniquePtr1D<ObjectName> &pre) = delete;

    UniquePtr1D &operator=(UniquePtr1D<ObjectName> &&pre) {
        if (&pre == this) return *this;
        m_ptr = std::move(pre.m_ptr);
        m_dim = pre.m_dim;
        return *this;
    }


    inline void reset(const INT32 &newDim) {
        std::unique_ptr<std::unique_ptr<ObjectName>[]> newPtr = std::make_unique<std::unique_ptr<ObjectName>[]>(
                newDim);
        m_ptr = std::move(newPtr);
        m_dim = newDim;
    }

    inline void realloc(const INT32 &newDim) {
        if (newDim == m_dim) return;
        std::unique_ptr<std::unique_ptr<ObjectName>[]> newPtr = std::make_unique<std::unique_ptr<ObjectName>[]>(
                newDim);
        INT32 actualDim = std::min(m_dim, newDim);
        for (INT32 i = 0; i < actualDim; ++i) {
            newPtr[i] = std::move(m_ptr[i]);
        }
        m_ptr = std::move(newPtr);
        m_dim = newDim;
    }

    /** @brief 函数转发，主要用于根据参数列表匹配具体的构造函数，生成对应的类对象 */
    template<typename ...ArgType>
    inline void construct(const INT32 &objIdx, ArgType &&...args) {
#ifndef NDEBUG
        THROW_EXCEPTION(objIdx >= m_dim || objIdx < 0, THROW_OUT_OF_RANGE("The index of object is out-of-range!"))
#endif
        // 通过std::forward将参数转发给ObjectName类的构造函数。
        m_ptr[objIdx] = std::make_unique<ObjectName>(std::forward<ArgType>(args)...);
    }


    inline INT32 getDim() const {
        return m_dim;
    }

};

template<typename ObjectName, typename = std::enable_if_t<std::is_class<ObjectName>::value>>
class UniquePtr2D {
private:
    std::unique_ptr<std::unique_ptr<UniquePtr1D<ObjectName>>[]> m_ptr;
    INT32 m_dim{0};

public:
    UniquePtr2D() = default;

    explicit UniquePtr2D(const INT32 &dim) {
        m_ptr = std::make_unique<std::unique_ptr<UniquePtr1D<ObjectName>>[]>(dim);
        m_dim = dim;
        for (INT32 i = 0; i < dim; ++i) {
            m_ptr[i] = std::make_unique<UniquePtr1D<ObjectName>>();
        }
    }

    /** @brief 生成dim1 x dim2 维的ObjectName类智能指针*/
    UniquePtr2D(const INT32 &dim1, const INT32 &dim2) {
        m_ptr = std::make_unique<std::unique_ptr<UniquePtr1D<ObjectName>>[]>(dim1);
        m_dim = dim1;
        for (INT32 i = 0; i < dim1; ++i) {
            m_ptr[i] = std::make_unique<UniquePtr1D<ObjectName>>(dim2);
        }
    }

    UniquePtr2D(const UniquePtr2D<ObjectName> &pre) = delete;

    UniquePtr2D(UniquePtr2D<ObjectName> &&pre) {
        m_ptr = std::move(pre.m_ptr);
        m_dim = pre.m_dim;
    }

    inline UniquePtr1D<ObjectName> &operator[](const INT32 &idx) {
#ifndef NDEBUG
        THROW_EXCEPTION(idx < 0 || idx >= m_dim, THROW_OUT_OF_RANGE("The idx for UniquePtr2D is out-of-range!"))
#endif
        return *(m_ptr[idx]);
    }

    inline UniquePtr1D<ObjectName> &operator[](const INT32 &idx) const {
#ifndef NDEBUG
        THROW_EXCEPTION(idx < 0 || idx >= m_dim, THROW_OUT_OF_RANGE("The idx for UniquePtr2D is out-of-range!"))
#endif
        return *(m_ptr[idx]);
    }

    /** @brief 用于直接获取第idx1行第idx2列的对象 */
    inline ObjectName &operator()(const INT32 &idx1, const INT32 &idx2) {
#ifndef NDEBUG
        THROW_EXCEPTION(idx1 < 0 || idx1 >= m_dim, THROW_OUT_OF_RANGE("The 1st dim idx is out-of-range!"))
        THROW_EXCEPTION(idx2 < 0 || idx2 >= m_ptr[idx1]->getDim(),
                        THROW_OUT_OF_RANGE("The 2nd dim idx is out-of-range!"))
#endif
        return *(*m_ptr[idx1])[idx2];
    }

    /** @brief 用于直接获取第idx1行第idx2列的对象 */
    inline ObjectName &operator()(const INT32 &idx1, const INT32 &idx2) const {
#ifndef NDEBUG
        THROW_EXCEPTION(idx1 < 0 || idx1 >= m_dim, THROW_OUT_OF_RANGE("The 1st dim idx is out-of-range!"))
        THROW_EXCEPTION(idx2 < 0 || idx2 >= m_ptr[idx1]->getDim(),
                        THROW_OUT_OF_RANGE("The 2nd dim idx is out-of-range!"))
#endif
        return *(*m_ptr[idx1])[idx2];
    }

    UniquePtr2D &operator=(const UniquePtr2D<ObjectName> &pre) = delete;

    UniquePtr2D &operator=(UniquePtr2D<ObjectName> &&pre) {
        if (&pre == this) return *this;
        m_ptr = std::move(pre.m_ptr);
        m_dim = pre.m_dim;
        return *this;
    }


    inline void reset(const INT32 &newDim) {
        std::unique_ptr<std::unique_ptr<UniquePtr1D<ObjectName>>[]> newPtr = std::make_unique<std::unique_ptr<UniquePtr1D<ObjectName>>[]>(
                newDim);
        for (INT32 i = 0; i < newDim; ++i) {
            newPtr[i] = std::make_unique<UniquePtr1D<ObjectName>>();
        }
        m_ptr = std::move(newPtr);
        m_dim = newDim;
    }

    inline void realloc(const INT32 &newDim) {
        if (newDim == m_dim) return;
        std::unique_ptr<std::unique_ptr<UniquePtr1D<ObjectName>>[]> newPtr = std::make_unique<std::unique_ptr<UniquePtr1D<ObjectName>>[]>(
                newDim);
        INT32 actualDim = std::min(m_dim, newDim);
        for (INT32 i = 0; i < actualDim; ++i) {
            newPtr[i] = std::move(m_ptr[i]);
        }
        if (actualDim < newDim) {
            for (INT32 i = actualDim; i < newDim; ++i) {
                newPtr[i] = std::make_unique<UniquePtr1D<ObjectName>>();
            }
        }
        m_ptr = std::move(newPtr);
        m_dim = newDim;
    }

    inline INT32 getDim() const {
        return m_dim;
    }

};

#endif //PMSLS_NEW_UNIQUEPTRTOOLS_H
