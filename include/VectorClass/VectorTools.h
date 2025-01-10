/*
 * @author  邓轶丹
 * @date    2024/5/6
 * @details 实现向量有关的工具函数
 */

#ifndef PMSLS_NEW_VECTORTOOLS_H
#define PMSLS_NEW_VECTORTOOLS_H

#include "AlignedVector.h"
#include "DenseVector.h"

#ifdef CUDA_ENABLED

#include "PageLockedVector.cuh"

#endif

namespace HOST {
    /** @brief 借助智能指针实例化vector对象，这样无需手动delete */
    template<typename ValType>
    void initializeVector(std::unique_ptr<HostVector<ValType> > &vecPtr, UINT32 length,
                          const memoryType_t &memoryType) {
        if (vecPtr) vecPtr.reset(nullptr);
        if (memoryType == memoryBase) {
            vecPtr = std::make_unique<DenseVector<ValType> >(length);
        } else if (memoryType == memoryAligned) {
            vecPtr = std::make_unique<AlignedVector<ValType> >(length);
        }
#ifdef CUDA_ENABLED     // 该宏在CmakeList中定义，由编译器自动检测当前主机是否支持CUDA架构
        else if (memoryType == memoryPageLocked) {
            vecPtr = std::make_unique<PageLockedVector<ValType>>(length);
        }
#endif
    }

    /** @brief 排序列表1，令列表1中的值由小到大排列，然后保持原列表1与列表2的相对顺序，重新排序列表2
     * @attention 底层逻辑使用简单的插入排序，所以被排序列表规模不能太大，否则会非常影响性能 */
    template<typename ValType1, typename ValType2>
    void sortVectorPair(ValType1 *vec1, ValType2 *vec2, UINT32 length) {
        ValType1 key;
        ValType2 val;
        for (INT32 i = 1; i < length; ++i) {
            key = vec1[i];
            val = vec2[i];
            INT32 j = i - 1;
            // 从当前元素的位置向前遍历
            while (j >= 0 && vec1[j] > key) {
                // 如果发现前面的元素比当前的元素大，则前面的元素都要向后移动，为当前元素插入列表腾出位置
                vec1[j + 1] = vec1[j];
                vec2[j + 1] = vec2[j];
                j--;
            }
            INT32 finalIdx = j + 1;
            vec1[finalIdx] = key;
            vec2[finalIdx] = val;
        }
    }


    /**@brief 归并升序排序，时间复杂度nlogn
     * @details 排序函数，实现对x中前numElememts个元素进行升序排序（小----->大）
     * @param x [in,out]待排序数组，直接改变原数组
     * @param numElements
     * @param vTmp: 辅助空间
     */
    template<typename ValType>
    void ArrayMergeSort(ValType *x, UINT32 numElements, ValType *vTmp) {
        if (numElements <= 10) {
            for (UINT32 ii = 1; ii < numElements; ii++) {
                INT32 tmpVal = x[ii];
                UINT32 jj;
                for (jj = ii; jj > 0 && x[jj - 1] > tmpVal; jj--)
                    x[jj] = x[jj - 1];
                x[jj] = tmpVal;
            }
        } else {
            UINT32 m = numElements / 2, ii = 0, jj = m, ll = 0;
            ArrayMergeSort(x, m, vTmp);
            ArrayMergeSort(x + m, numElements - m, vTmp);
            while (ii < m && jj < numElements) vTmp[ll++] = x[ii] < x[jj] ? x[ii++] : x[jj++];
            while (ii < m) vTmp[ll++] = x[ii++];
            for (UINT32 kk = 0; kk < ll; kk++) x[kk] = vTmp[kk];
        }
    }


    /** @brief 排序列表1，令列表1中的值由小到大排列，然后保持原列表1与列表2的相对顺序，重新排序列表2
     * @param [in,out] vec1 :   待排序列表1；
     * @param [in,out] vec2 :   待排序列表2；
     * @param [in] tmpVec1 :    辅助空间，长度大于或等于numElements，需要在外部自行分配内存；
     * @param [in] tmpVec2 :    辅助空间，长度大于或等于numElements，需要在外部自行分配内存；
     * @param [in] numElements :数组中的元素个数。
     * @attention 底层逻辑使用归并排序，对内存需要比较大，但速度较快。 */
    template<typename ValType1, typename ValType2>
    void mergeSortVectorPair(ValType1 *vec1, ValType2 *vec2, ValType1 *tmpVec1, ValType2 *tmpVec2,
                             UINT32 numElements) {
        if (numElements <= 10) {
            // 插入排序处理小规模数据
            for (UINT32 ii = 1; ii < numElements; ++ii) {
                ValType1 key1 = vec1[ii];
                ValType2 key2 = vec2[ii];
                UINT32 jj = ii;
                while (jj > 0 && vec1[jj - 1] > key1) {
                    vec1[jj] = vec1[jj - 1];
                    vec2[jj] = vec2[jj - 1];
                    --jj;
                }
                vec1[jj] = key1;
                vec2[jj] = key2;
            }
        } else {
            // 归并排序处理大规模数据
            UINT32 mid = numElements / 2;
            mergeSortVectorPair(vec1, vec2, tmpVec1, tmpVec2, mid); // 左半部分递归
            mergeSortVectorPair(vec1 + mid, vec2 + mid, tmpVec1, tmpVec2, numElements - mid); // 右半部分递归

            // 合并
            UINT32 i = 0, j = mid, k = 0;
            while (i < mid && j < numElements) {
                if (vec1[i] <= vec1[j]) {
                    tmpVec1[k] = vec1[i];
                    tmpVec2[k] = vec2[i];
                    ++i;
                } else {
                    tmpVec1[k] = vec1[j];
                    tmpVec2[k] = vec2[j];
                    ++j;
                }
                ++k;
            }
            while (i < mid) {
                tmpVec1[k] = vec1[i];
                tmpVec2[k] = vec2[i];
                ++i;
                ++k;
            }
            while (j < numElements) {
                tmpVec1[k] = vec1[j];
                tmpVec2[k] = vec2[j];
                ++j;
                ++k;
            }

            // 拷贝回原数组
            for (UINT32 ii = 0; ii < numElements; ++ii) {
                vec1[ii] = tmpVec1[ii];
                vec2[ii] = tmpVec2[ii];
            }
        }
    }

    /** @brief 快速排序拆分数组，实现选中数组前Ncut大个元素
     * @param a 待排序数组
     * @param ind 排列的下标
     * @param n 数组长度
     * @param Ncut 阈值
     */
    template<typename T>
    void qsplit(T *a, int *ind, int n, int Ncut) {
        double key, tmp;
        int j, itmp, first, mid, last, ncut;
        ncut = Ncut - 1;
        first = 0;
        last = n - 1;
        if (ncut < first || ncut > last) {
            return;
        }
        /* outer loop -- while mid != ncut */
        do {
            mid = first;
            key = fabs((double) a[mid]);
            for (j = first + 1; j <= last; j++) {
                if (fabs((double) a[j]) > key) {
                    mid = mid + 1;
                    tmp = (double) a[mid];
                    itmp = ind[mid];
                    a[mid] = a[j];
                    ind[mid] = ind[j];
                    a[j] = (T) tmp;
                    ind[j] = itmp;
                }
            }
            /*-------------------- interchange */
            tmp = (double) a[mid];
            a[mid] = a[first];
            a[first] = (T) tmp;
            itmp = ind[mid];
            ind[mid] = ind[first];
            ind[first] = itmp;
            /*-------------------- test for while loop */
            if (mid == ncut) {
                break;
            }
            if (mid > ncut) {
                last = mid - 1;
            } else {
                first = mid + 1;
            }
        } while (mid != ncut);
    }
} //HOST


#endif //PMSLS_NEW_VECTORTOOLS_H
