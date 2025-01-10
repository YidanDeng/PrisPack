/*
 * @author  邓轶丹
 * @date    2024/3/22
 * @details 一些常用的工具函数
 */

#ifndef PMSLS_DEV_BASEUTILS_H
extern "C++" {

#include "../../config/headers.h"

template<typename T>
struct CompareStruct {
    T val;
    int ord;
};

/**
 * @brief CompareStruct结构体比大小（小于） 结构体a<结构体b
 */
template<class T>
struct CompareStructLess {
    bool operator()(T const &a, T const &b) const { return a.val < b.val; }
};

/**
 * @brief CompareStruct结构体比大小（小于） 结构体a>结构体b
 */
template<class T>
struct CompareStructGreater {
    bool operator()(T const &a, T const &b) const { return a.val > b.val; }
};


/** @brief 基于某个有序数组和搜索区间展开二分查找。
 * @param [in] vec: 待搜索的数组
 * @param [in] val: 需要搜索的值
 * @param [in] left: 搜索区间的上界（从0开始编号）
 * @param [in] right: 搜索区间的下界（默认为整个搜索区间的长度）
 * @param [in,out] mid: 搜索到的值的精确位置，如果元素未找到，则返回位于疑似精确位置右侧的最近位置。 */
template<typename T>
void findValByBinarySearch(T *&vec, const T &val, UINT32 left, UINT32 right, UINT32 &mid) {
    if (!vec) {// 如果被搜索数组指针为空，则搜索无意义，终止退出
#ifndef NDEBUG
        std::cerr << YELLOW << __func__ << ": " << L_PURPLE <<
                  "[ERROR] The array pointer of values is null!" << COLOR_NONE << std::endl;
#endif
        return;
    }

#ifndef NWARN
    if (left > right) {
        std::cout << YELLOW << __func__ << ": " << L_PURPLE
                  << "[WARNING] The value of upper-bound or lower-bound may be incorrect."
                  << COLOR_NONE << std::endl;
    }
#endif
    while (right >= left) {
        mid = left + (right - left) / 2;
        if (vec[left] == val) {
            mid = left;
            return;
        }
        if (vec[right] == val) {
            mid = right;
            return;
        }
        if (vec[mid] == val)
            return;
        else if (vec[mid] < val) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    // 假设元素未找到，则令返回值等于左值针的值
    if (mid != left)
        mid = left;
}

inline void transStrToLower(std::string &str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
}

}

#define PMSLS_DEV_BASEUTILS_H
#endif //PMSLS_DEV_BASEUTILS_H




