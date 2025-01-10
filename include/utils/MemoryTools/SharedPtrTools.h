/*
 * @author  邓轶丹
 * @date    2024/5/28
 * @details SharedPtr封装工具
 */

#ifndef PMSLS_NEW_SHAREDPTRTOOLS_H
#define PMSLS_NEW_SHAREDPTRTOOLS_H

#include "../../../config/config.h"
#include "../../../config/debug.h"
#include "../../../config/headers.h"
#include "../../../include/utils/ErrorHandler.h"


template <typename ObjectName, typename = std::enable_if_t<std::is_class<ObjectName>::value>>
class SharedObject {
private:
    std::shared_ptr<ObjectName> m_ptr;

public:
    SharedObject() = default;

    ~SharedObject() = default;

    SharedObject(const SharedObject<ObjectName>& pre) {
        m_ptr = pre.m_ptr;
    }

    SharedObject(SharedObject<ObjectName>&& pre) {
        m_ptr = std::move(pre.m_ptr);
    }

    SharedObject& operator=(const SharedObject<ObjectName>& pre) {
        if (this == &pre) return *this;
        m_ptr = pre.m_ptr;
        return *this;
    }

    SharedObject& operator=(SharedObject<ObjectName>&& pre) {
        if (this == &pre) return *this;
        m_ptr = std::move(pre.m_ptr);
        return *this;
    }

    inline ObjectName& operator*() {
        return *m_ptr;
    }

    inline const ObjectName& operator*() const {
        return *m_ptr;
    }

    inline std::shared_ptr<ObjectName>& operator->() {
#ifndef NDEBUG
        THROW_EXCEPTION(m_ptr.get() == nullptr,
                        THROW_INVALID_ARGUMENT("You are trying to use object via null pointer which is illegal!"))
#endif
        return m_ptr;
    }

    inline const std::shared_ptr<ObjectName>& operator->() const {
#ifndef NDEBUG
        THROW_EXCEPTION(m_ptr.get() == nullptr,
                        THROW_INVALID_ARGUMENT("You are trying to use object via null pointer which is illegal!"))
#endif
        return m_ptr;
    }

    template <typename... ArgType>
    SharedObject(ArgType&&... args) {
        // 通过std::forward将参数转发给ObjectName类的构造函数。
        m_ptr = std::make_shared<ObjectName>(std::forward<ArgType>(args)...);
    }

    /** @brief 用来调用对应类的构造函数，初始化当前对象 */
    template <typename... ArgType>
    inline void construct(ArgType&&... args) {
        // 通过std::forward将参数转发给ObjectName类的构造函数。
        m_ptr = std::make_shared<ObjectName>(std::forward<ArgType>(args)...);
    }


    inline const std::shared_ptr<ObjectName>& get() const {
        return m_ptr;
    }

    inline std::shared_ptr<ObjectName>& get() {
        return m_ptr;
    }
};


/** @brief 由智能指针shared_ptr组成的一维数组 */
template <typename ObjectName, typename = std::enable_if_t<std::is_class<ObjectName>::value>>
class SharedPtr1D {
private:
    std::shared_ptr<std::shared_ptr<ObjectName>[]> m_ptr;
    INT32 m_dim{0};

public:
    SharedPtr1D() = default;

    explicit SharedPtr1D(const INT32& dim) {
        m_ptr = std::shared_ptr<std::shared_ptr<ObjectName>[]>(new std::shared_ptr<ObjectName>[dim],
                                                               std::default_delete<std::shared_ptr<ObjectName>[]>());
        m_dim = dim;
    }

    explicit SharedPtr1D(const SharedPtr1D<ObjectName>& pre) {
        m_ptr = pre.m_ptr;
        m_dim = pre.m_dim;
    }

    explicit SharedPtr1D(SharedPtr1D<ObjectName>&& pre) noexcept {
        m_ptr = std::move(pre.m_ptr);
        m_dim = pre.m_dim;
    }

    inline std::shared_ptr<ObjectName>& operator[](const INT32& idx) {
#ifndef NDEBUG
        THROW_EXCEPTION(idx < 0 || idx >= m_dim, THROW_OUT_OF_RANGE("The idx for SharedPtr1D is out-of-range!"))
#endif
        return m_ptr[idx];
    }

    inline const std::shared_ptr<ObjectName>& operator[](const INT32& idx) const {
#ifndef NDEBUG
        THROW_EXCEPTION(idx < 0 || idx >= m_dim, THROW_OUT_OF_RANGE("The idx for SharedPtr1D is out-of-range!"))
#endif
        return m_ptr[idx];
    }

    SharedPtr1D& operator=(const SharedPtr1D<ObjectName>& pre) {
        if (&pre == this) return *this;
        m_ptr = pre.m_ptr;
        m_dim = pre.m_dim;
        return *this;
    }

    SharedPtr1D& operator=(SharedPtr1D<ObjectName>&& pre) noexcept {
        if (&pre == this) return *this;
        m_ptr = std::move(pre.m_ptr);
        m_dim = pre.m_dim;
        return *this;
    }

    inline void reset(const INT32& newDim) {
        std::shared_ptr<std::shared_ptr<ObjectName>[]> newPtr(new std::shared_ptr<ObjectName>[newDim],
                                                              std::default_delete<std::shared_ptr<ObjectName>[]>());
        m_ptr = std::move(newPtr);
        m_dim = newDim;
    }

    inline void realloc(const INT32& newDim) {
        if (newDim == m_dim) return;
        std::shared_ptr<std::shared_ptr<ObjectName>[]> newPtr(new std::shared_ptr<ObjectName>[newDim],
                                                              std::default_delete<std::shared_ptr<ObjectName>[]>());
        INT32 actualDim = std::min(m_dim, newDim);
        for (INT32 i = 0; i < actualDim; ++i) {
            newPtr[i] = std::move(m_ptr[i]);
        }
        m_ptr = std::move(newPtr);
        m_dim = newDim;
    }

    inline INT32 getDim() const {
        return m_dim;
    }

    /** @brief 一个用于debug的函数，检查底层指向的对象地址是否相等，避免直接比较对象的值 */
    inline bool checkEqual(const SharedPtr1D<ObjectName>& another) {
        return m_ptr.get() == another.m_ptr.get();
    }

    /** @brief 函数转发，主要用于根据参数列表匹配具体的构造函数，生成对应的类对象 */
    template <typename... ArgType>
    inline void construct(const INT32& objIdx, ArgType&&... args) {
#ifndef NDEBUG
        THROW_EXCEPTION(objIdx >= m_dim || objIdx < 0, THROW_OUT_OF_RANGE("The index of object is out-of-range!"))
#endif
        // 通过std::forward将参数转发给ObjectName类的构造函数。
        m_ptr[objIdx] = std::make_shared<ObjectName>(std::forward<ArgType>(args)...);
    }
};


/** @brief 由智能指针shared_ptr组成的二维数组 */
template <typename ObjectName, typename = std::enable_if_t<std::is_class<ObjectName>::value>>
class SharedPtr2D {
private:
    std::shared_ptr<std::shared_ptr<SharedPtr1D<ObjectName>>[]> m_ptr;
    INT32 m_dim{0};

public:
    SharedPtr2D() = default;

    explicit SharedPtr2D(const INT32& dim) {
        // std::shared_ptr 默认的删除器用来删除单个对象而不是数组。如果不指定 std::default_delete<T[]>，当 std::shared_ptr 超出作用域时，它会调用错误的删除操作，从而导致未定义行为。
        m_ptr = std::shared_ptr<std::shared_ptr<SharedPtr1D<ObjectName>>[]>(
            new std::shared_ptr<SharedPtr1D<ObjectName>>[dim],
            std::default_delete<std::shared_ptr<SharedPtr1D<ObjectName>>[]>());
        m_dim = dim;
        for (INT32 i = 0; i < dim; ++i) {
            m_ptr[i] = std::make_shared<SharedPtr1D<ObjectName>>();
        }
    }

    /** @brief 生成dim1 x dim2 维的ObjectName类智能指针 */
    SharedPtr2D(const INT32& dim1, const INT32& dim2) {
        m_ptr = std::shared_ptr<std::shared_ptr<SharedPtr1D<ObjectName>>[]>(
            new std::shared_ptr<SharedPtr1D<ObjectName>>[dim1],
            std::default_delete<std::shared_ptr<SharedPtr1D<ObjectName>>[]>());
        m_dim = dim1;
        for (INT32 i = 0; i < dim1; ++i) {
            m_ptr[i] = std::make_shared<SharedPtr1D<ObjectName>>(dim2);
        }
    }

    SharedPtr2D(const SharedPtr2D<ObjectName>& pre) {
        m_ptr = pre.m_ptr;
        m_dim = pre.m_dim;
    }

    SharedPtr2D(SharedPtr2D<ObjectName>&& pre) noexcept {
        m_ptr = std::move(pre.m_ptr);
        m_dim = pre.m_dim;
    }

    inline SharedPtr1D<ObjectName>& operator[](const INT32& idx) {
#ifndef NDEBUG
        THROW_EXCEPTION(idx < 0 || idx >= m_dim, THROW_OUT_OF_RANGE("The idx for SharedPtr2D is out-of-range!"))
#endif
        return *m_ptr[idx];
    }

    inline SharedPtr1D<ObjectName>& operator[](const INT32& idx) const {
#ifndef NDEBUG
        THROW_EXCEPTION(idx < 0 || idx >= m_dim, THROW_OUT_OF_RANGE("The idx for SharedPtr2D is out-of-range!"))
#endif
        return *m_ptr[idx];
    }

    inline ObjectName& operator()(const INT32& idx1, const INT32& idx2) {
        return *(*m_ptr[idx1])[idx2];
    }

    inline ObjectName& operator()(const INT32& idx1, const INT32& idx2) const {
        return *(*m_ptr[idx1])[idx2];
    }

    SharedPtr2D& operator=(const SharedPtr2D<ObjectName>& pre) {
        if (&pre == this) return *this;
        m_ptr = pre.m_ptr;
        m_dim = pre.m_dim;
        return *this;
    }

    SharedPtr2D& operator=(SharedPtr2D<ObjectName>&& pre) noexcept {
        if (&pre == this) return *this;
        m_ptr = std::move(pre.m_ptr);
        m_dim = pre.m_dim;
        return *this;
    }

    inline void reset(const INT32& newDim) {
        std::shared_ptr<std::shared_ptr<SharedPtr1D<ObjectName>>[]> newPtr = std::shared_ptr<std::shared_ptr<SharedPtr1D
            <ObjectName>>[]>(
            new std::shared_ptr<SharedPtr1D<ObjectName>>[newDim],
            std::default_delete<std::shared_ptr<SharedPtr1D<ObjectName>>[]>());
        for (INT32 i = 0; i < newDim; ++i) {
            newPtr[i] = std::make_shared<SharedPtr1D<ObjectName>>();
        }
        m_ptr = std::move(newPtr);
        m_dim = newDim;
    }

    inline void realloc(const INT32& newDim) {
        if (newDim == m_dim) return;
        std::shared_ptr<std::shared_ptr<SharedPtr1D<ObjectName>>[]> newPtr = std::shared_ptr<std::shared_ptr<SharedPtr1D
            <ObjectName>>[]>(
            new std::shared_ptr<SharedPtr1D<ObjectName>>[newDim],
            std::default_delete<std::shared_ptr<SharedPtr1D<ObjectName>>[]>());
        INT32 actualDim = std::min(m_dim, newDim);
        for (INT32 i = 0; i < actualDim; ++i) {
            newPtr[i] = std::move(m_ptr[i]);
        }
        if (actualDim < newDim) {
            for (INT32 i = actualDim; i < newDim; ++i) {
                newPtr[i] = std::make_shared<SharedPtr1D<ObjectName>>();
            }
        }
        m_ptr = std::move(newPtr);
        m_dim = newDim;
    }

    inline INT32 getDim() const {
        return m_dim;
    }

    /** @brief 一个用于debug的函数，检查底层指向的对象地址是否相等，避免直接比较对象的值 */
    inline bool checkEqual(const SharedPtr2D<ObjectName>& another) {
        return m_ptr.get() == another.m_ptr.get();
    }
};


#endif //PMSLS_NEW_SHAREDPTRTOOLS_H
