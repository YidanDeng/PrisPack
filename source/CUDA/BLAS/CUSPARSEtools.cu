/**
* @author  邓轶丹
 * @date    2024/6/2
 * @details 用于控制单GPU上使用多个流异步求解cusparse相关的计算，所有控制变量自动释放，待计算的值通过函数参数进行传递。 */


#include "../../../include/CUDA/BLAS/CUSPARSEtools.cuh"

namespace DEVICE {
    /* ================================== CusparseCSRDescriptor ================================== */
    template<typename ValType>
    CusparseCSRDescriptor<ValType>::CusparseCSRDescriptor(const std::shared_ptr<DeviceCSRMatrix<ValType> > &mat) {
        m_oriMat = mat;
        CHECK_CUDA(cudaSetDevice(mat->getDeviceID()))
        if (std::is_same<ValType, FLOAT32>::value)
            m_dataType = CUDA_R_32F;
        else if (std::is_same<ValType, FLOAT64>::value)
            m_dataType = CUDA_R_64F;
#ifndef NDEBUG
        else {
            std::cerr << L_RED << "[ERROR] Incompatible data type for " << __func__
                    << "func, the current type id is: "
                    << typeid(ValType).name() << ", the data type must be float or double!" << COLOR_NONE
                    << std::endl;
            return;
        }
#endif
        // 创建CSR结构的稀疏矩阵
        CHECK_CUSPARSE(cusparseCreateCsr(&m_mat, mat->getRowNum(), mat->getColNum(), mat->getNNZ(),
            mat->getRowOffsetPtr(), mat->getColIndicesPtr(), mat->getValuesPtr(),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, m_dataType))
    }

    template<typename ValType>
    CusparseCSRDescriptor<ValType>::~CusparseCSRDescriptor() {
        if (m_mat) {
            CHECK_CUDA(cudaSetDevice(m_oriMat->getDeviceID()))
            // 销毁矩阵/向量的descriptor
            CHECK_CUSPARSE(cusparseDestroySpMat(m_mat))
        }
    }

    /* ================================== CusparseCSRDescriptor end ================================== */


    /* ================================== CusparseDnVectorDescriptor ================================== */
    template<typename ValType>
    CusparseDnVectorDescriptor<ValType>::CusparseDnVectorDescriptor(
        const std::shared_ptr<SyncDeviceVector<ValType> > &vec) {
        m_oriVec = vec;
        CHECK_CUDA(cudaSetDevice(vec->getLocation()))
        if (std::is_same<ValType, FLOAT32>::value)
            m_dataType = CUDA_R_32F;
        else if (std::is_same<ValType, FLOAT64>::value)
            m_dataType = CUDA_R_64F;
#ifndef NDEBUG
        else {
            std::cerr << L_RED << "[ERROR] Incompatible data type for " << __func__
                    << "func, the current type id is: "
                    << typeid(ValType).name() << ", the data type must be float or double!" << COLOR_NONE
                    << std::endl;
            return;
        }
#endif
        CHECK_CUSPARSE(cusparseCreateDnVec(&m_vec, vec->getLength(), vec->getRawValPtr(), m_dataType))
    }

    template<typename ValType>
    CusparseDnVectorDescriptor<ValType>::~CusparseDnVectorDescriptor() {
        if (m_vec) {
            CHECK_CUDA(cudaSetDevice(m_oriVec->getLocation()))
            CHECK_CUSPARSE(cusparseDestroyDnVec(m_vec))
        }
    }

    /* ================================== CusparseDnVectorDescriptor end ================================== */


    template<typename ValType>
    CusparseCsrSpMV<ValType>::CusparseCsrSpMV(const std::shared_ptr<CusparseCSRDescriptor<ValType> > &matDesc,
                                              const std::shared_ptr<StreamController> &stream) {
#ifndef NDEBUG
        THROW_EXCEPTION(matDesc->getDeviceID() != stream->getDeviceID(),
                        THROW_LOGIC_ERROR("The stream controller and device CSR matrix are not on the same GPU!"))
#endif
        m_matA = matDesc;
        m_handler = std::make_shared<CusparseHandler>(stream);
    }

    template<typename ValType>
    CusparseCsrSpMV<ValType>::CusparseCsrSpMV(const std::shared_ptr<CusparseCSRDescriptor<ValType> > &matDesc,
                                              const std::shared_ptr<CusparseHandler> &handler) {
#ifndef NDEBUG
        THROW_EXCEPTION(matDesc->getDeviceID() != handler->getDeviceID(),
                        THROW_LOGIC_ERROR("The cusparse handler and device CSR matrix are not on the same GPU!"))
#endif
        m_matA = matDesc;
        m_handler = handler;
    }


    template<typename ValType>
    void CusparseCsrSpMV<ValType>::csrMultiplyVec(const cusparseOperation_t &matAop, const ValType &alpha,
                                                  const CusparseDnVectorDescriptor<ValType> &d_in,
                                                  const ValType &beta,
                                                  CusparseDnVectorDescriptor<ValType> &d_out) {
        INT32 currDevID = m_matA->getDeviceID();
        CHECK_CUDA(cudaSetDevice(currDevID))
        m_alpha = alpha;
        m_beta = beta;
#ifndef NDEBUG
        THROW_EXCEPTION(d_in.getLength() != m_matA->getColNum(),
                        THROW_LOGIC_ERROR("Matrix dim and vector dim are incompatible!"))
        THROW_EXCEPTION(d_in.getDeviceID() != currDevID || d_out.getDeviceID() != currDevID,
                        THROW_LOGIC_ERROR("The device vector and device CSR matrix are not on the same GPU!"))
        THROW_EXCEPTION(m_matA->getRowNum() != d_out.getLength(),
                        THROW_LOGIC_ERROR("The dim of two vectors are incompatible!"))
#endif
        // 如果需要，创建额外的缓冲区
        size_t newSize;
        UINT8 needChangeBuff{0};
        if (m_bufferSize == 0 || m_csrOpt != matAop) {
            CHECK_CUSPARSE(cusparseSpMV_bufferSize(**m_handler, matAop, &m_alpha, m_matA->getCSRDescriber(),
                d_in.getDenseVecDescriber(), &m_beta, d_out.getDenseVecDescriber(),
                m_matA->getCuDataType(), CUSPARSE_SPMV_CSR_ALG1, &newSize))
            needChangeBuff = 1;
            m_csrOpt = matAop;
        }
        if (needChangeBuff && newSize != m_bufferSize) {
            CHECK_CUDA(cudaFreeAsync(m_dBuffer, **m_handler->getStream()))
            CHECK_CUDA(cudaMallocAsync(&m_dBuffer, newSize, **m_handler->getStream()))
            m_bufferSize = newSize;
        }
        // 开始计算SpMV，即res = m_alpha * Ax + beta * res
        CHECK_CUSPARSE(
            cusparseSpMV(**m_handler, matAop, &m_alpha, m_matA->getCSRDescriber(), d_in.getDenseVecDescriber(),
                &m_beta, d_out.getDenseVecDescriber(), m_matA->getCuDataType(),
                CUSPARSE_SPMV_CSR_ALG1, m_dBuffer))
        // 还有改进的CUSPARSE_SPMV_CSR_ALG2
    }

    template<typename ValType>
    CusparseCsrSpMV<ValType>::~CusparseCsrSpMV() {
        if (m_dBuffer) {
            CHECK_CUDA(cudaSetDevice(m_matA->getDeviceID()))
            CHECK_CUDA(cudaFreeAsync(m_dBuffer, **m_handler->getStream()));
        }
    }

    /* ================================== CusparseCsrSpMV end ================================== */

    template<typename ValType>
    CusparseCSRTriSolve<ValType>::CusparseCSRTriSolve(const std::shared_ptr<CusparseCSRDescriptor<ValType> > &matDesc,
                                                      const cusparseFillMode_t &fillMode,
                                                      const cusparseDiagType_t &diagType,
                                                      const std::shared_ptr<CusparseHandler> &handler) {
#ifndef NDEBUG
        THROW_EXCEPTION(matDesc->getDeviceID() != handler->getDeviceID(),
                        THROW_LOGIC_ERROR("The stream controller and device CSR matrix are not on the same GPU!"))
#endif
        CHECK_CUDA(cudaSetDevice(matDesc->getDeviceID()))
        m_matA = matDesc;
        m_fillMode = fillMode;
        m_diagType = diagType;
        m_handler = handler;
        m_stream = handler->getStream();
        /* 设置稀疏三角矩阵相关参数 */
        CHECK_CUSPARSE(cusparseSpMatSetAttribute(m_matA->getCSRDescriber(), CUSPARSE_SPMAT_FILL_MODE,
            &m_fillMode, sizeof(m_fillMode)))
        CHECK_CUSPARSE(cusparseSpMatSetAttribute(m_matA->getCSRDescriber(), CUSPARSE_SPMAT_DIAG_TYPE,
            &m_diagType, sizeof(m_diagType)))
        /* 初始化稀疏三角方程组求解器 */
        CHECK_CUSPARSE(cusparseSpSV_createDescr(&m_spsvDescriptor))
    }

    template<typename ValType>
    void CusparseCSRTriSolve<ValType>::csrTriSolve(const cusparseOperation_t &matAop, ValType alpha,
                                                   const CusparseDnVectorDescriptor<ValType> &d_in,
                                                   CusparseDnVectorDescriptor<ValType> &d_out) {
        INT32 currDevID = m_matA->getDeviceID();
        CHECK_CUDA(cudaSetDevice(currDevID))
        m_alpha = alpha;
#ifndef NDEBUG
        THROW_EXCEPTION(d_in.getLength() != m_matA->getRowNum(),
                        THROW_LOGIC_ERROR("Matrix dim and vector dim are incompatible!"))
        THROW_EXCEPTION(d_in.getDeviceID() != currDevID || d_out.getDeviceID() != currDevID,
                        THROW_LOGIC_ERROR("The device vector and device CSR matrix are not on the same GPU!"))
        THROW_EXCEPTION(m_matA->getRowNum() != d_out.getLength(),
                        THROW_LOGIC_ERROR("The dim of two vectors are incompatible!"))
#endif
        // 如果需要，创建额外的缓冲区
        size_t newSize;
        UINT8 needChangeBuff{0};
        if (m_bufferSize == 0 || m_csrOpt != matAop) {
            CHECK_CUSPARSE(cusparseSpSV_bufferSize(**m_handler, matAop, &m_alpha, m_matA->getCSRDescriber(),
                d_in.getDenseVecDescriber(), d_out.getDenseVecDescriber(),
                m_matA->getCuDataType(), CUSPARSE_SPSV_ALG_DEFAULT, m_spsvDescriptor,
                &newSize))
            needChangeBuff = 1;
            m_csrOpt = matAop;
        }
        if (needChangeBuff) {
            if (newSize != m_bufferSize) {
                CHECK_CUDA(cudaFreeAsync(m_bufferPtr, **m_handler->getStream()))
                CHECK_CUDA(cudaMallocAsync(&m_bufferPtr, newSize, **m_handler->getStream()))
                m_bufferSize = newSize;
            }
            CHECK_CUSPARSE(cusparseSpSV_analysis(**m_handler, matAop, &m_alpha, m_matA->getCSRDescriber(),
                d_in.getDenseVecDescriber(), d_out.getDenseVecDescriber(),
                m_matA->getCuDataType(), CUSPARSE_SPSV_ALG_DEFAULT, m_spsvDescriptor,
                m_bufferPtr))
        }
        CHECK_CUSPARSE(cusparseSpSV_solve(**m_handler, matAop, &m_alpha, m_matA->getCSRDescriber(),
            d_in.getDenseVecDescriber(), d_out.getDenseVecDescriber(),
            m_matA->getCuDataType(), CUSPARSE_SPSV_ALG_DEFAULT, m_spsvDescriptor))
    }


    template<typename ValType>
    CusparseCSRTriSolve<ValType>::~CusparseCSRTriSolve() {
        if (m_spsvDescriptor) {
            CHECK_CUDA(cudaSetDevice(m_matA->getDeviceID()))
            if (m_bufferPtr) {
                CHECK_CUDA(cudaFreeAsync(m_bufferPtr, **m_handler->getStream()));
            }
            CHECK_CUSPARSE(cusparseSpSV_destroyDescr(m_spsvDescriptor))
        }
    }
} // namespace DEVICE
