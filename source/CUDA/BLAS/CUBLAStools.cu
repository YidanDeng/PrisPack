/**
 * @author  邓轶丹
 * @date    2024/4/3
 * @details 用于控制单GPU上使用多个流异步求解cublas相关的计算，所有控制变量自动释放，待计算的值通过函数参数进行传递。 */


#include "../../../include/CUDA/BLAS/CUBLAStools.cuh"
#include "../../../include/CUDA/cuKernel.cuh"


namespace DEVICE {
    template <typename ValType>
    CUBLAStools<ValType>::~CUBLAStools() {
        CHECK_CUDA(cudaSetDevice(m_deviceID))
        CHECK_CUBLAS(cublasDestroy(m_handler))
    }


    template <typename ValType>
    CUBLAStools<ValType>::CUBLAStools() {
        CHECK_CUDA(cudaSetDevice(DEFAULT_GPU))
        CHECK_CUBLAS(cublasCreate(&m_handler))
        CHECK_CUBLAS(cublasSetStream(m_handler, nullptr))
    }

    template <typename ValType>
    CUBLAStools<ValType>::CUBLAStools(const std::shared_ptr<StreamController>& stream)
        : m_deviceID(stream->getDeviceID()), m_stream(stream) {
#ifndef NINFO
        SHOW_INFO("Initialization for Async-CUBLAS begin...")
#endif
        // 切换到对应的GPU，并在该GPU上创建相关的流，将控制器与流关联起来
        CHECK_CUDA(cudaSetDevice(m_deviceID))
        CHECK_CUBLAS(cublasCreate(&m_handler))
        CHECK_CUBLAS(cublasSetStream(m_handler, **stream))
#ifndef NINFO
        SHOW_INFO("Initialization for Async-CUBLAS success!")
#endif
    }


    template <typename ValType>
    ValType CUBLAStools<ValType>::cublasVecNorm2(const DeviceVector<ValType>& dev_vec) {
#ifndef NDEBUG
        THROW_EXCEPTION(dev_vec.getLocation() != m_deviceID,
                        THROW_LOGIC_ERROR("The device vector and current cublas handler are not on the same GPU!"))
        THROW_EXCEPTION(m_deviceID != dev_vec.getLocation(),
                        THROW_LOGIC_ERROR("The cublas handler and the device vector are not on the same GPU!"))
#endif
        ValType result{0};
        if (dev_vec.getLength() == 0) return result;
        CHECK_CUDA(cudaSetDevice(m_deviceID))
        if (std::is_same<ValType, FLOAT64>::value) {
            CHECK_CUBLAS(cublasDnrm2(m_handler, dev_vec.getLength(), (double *) &dev_vec[0], 1, (double *) &result))
        } else if (std::is_same<ValType, FLOAT32>::value) {
            CHECK_CUBLAS(cublasSnrm2(m_handler, dev_vec.getLength(), (float *) &dev_vec[0], 1, (float *) &result))
        }
#ifndef NWARN
        else {
            // 一个兜底运算，尽量不要使用
            std::string outPut = "Incompatible value type of inner-production, type ID: " +
                std::string(typeid(ValType).name());
            SHOW_WARN(outPut)
            result = vecNorm2(PACK_RAW_PTR(&dev_vec[0]), 1, dev_vec.getLength());
        }
#endif

        return result;
    }

    template <typename ValType>
    ValType CUBLAStools<ValType>::cublasInnerProduct(const DeviceVector<ValType>& dev_x,
                                                     const DeviceVector<ValType>& dev_y) {
#ifndef NDEBUG
        THROW_EXCEPTION(dev_x.getLength() != dev_y.getLength(),
                        THROW_LOGIC_ERROR("The two device vectors have incompatible lengths!"))
        THROW_EXCEPTION(m_deviceID != dev_x.getLocation() || m_deviceID != dev_y.getLocation(),
                        THROW_LOGIC_ERROR("The cublas handler and two device vectors are not on the same GPU!"))
#endif
        if (dev_x.getLength() == 0) return 0;
        CHECK_CUDA(cudaSetDevice(m_deviceID))
        ValType result{0};
        const ValType* devXptr = &dev_x[0];
        const ValType* devYptr = &dev_y[0];

        if (std::is_same<ValType, FLOAT64>::value) {
            CHECK_CUBLAS(cublasDdot(m_handler, dev_x.getLength(), (double *) devXptr, 1,
                (double *) devYptr, 1, (double *) &result))
        } else if (std::is_same<ValType, FLOAT32>::value) {
            CHECK_CUBLAS(cublasSdot(m_handler, dev_x.getLength(), (float *) devXptr, 1, (float *) devYptr, 1,
                (float *) &result))
        } else {
            // 一个兜底运算，尽量不要使用
            std::cout << "[WARNING] Incompatible value type of inner-production, type ID: " << typeid(ValType).name()
                << std::endl;
            ValType init{0};
            result = thrust::inner_product(thrust::device, devXptr, devXptr + dev_x.getLength(), devYptr, init);
            return result;
        }

        return result;
    }

    template <typename ValType>
    void CUBLAStools<ValType>::cublasVecAdd(const ValType& alpha, const DeviceVector<ValType>& dev_x,
                                            DeviceVector<ValType>& dev_y) {
#ifndef NDEBUG
        THROW_EXCEPTION(dev_x.getLength() != dev_y.getLength(),
                        THROW_LOGIC_ERROR("The two device vectors have incompatible lengths!"))
        THROW_EXCEPTION(m_deviceID != dev_x.getLocation() || m_deviceID != dev_y.getLocation(),
                        THROW_LOGIC_ERROR("The cublas handler and two device vectors are not on the same GPU!"))
#endif
        CHECK_CUDA(cudaSetDevice(m_deviceID))
        if (std::is_same_v<ValType, FLOAT64>) {
            const auto local_alpha = static_cast<FLOAT64>(alpha);
            CHECK_CUBLAS(cublasDaxpy(m_handler, dev_x.getLength(), &local_alpha, (double *) &dev_x[0], 1,
                (double *) &dev_y[0], 1))
        } else if (std::is_same_v<ValType, FLOAT32>) {
            const auto local_alpha = static_cast<FLOAT32>(alpha);
            CHECK_CUBLAS(cublasSaxpy(m_handler, dev_x.getLength(), &local_alpha, (float *) &dev_x[0], 1,
                (float *) &dev_y[0], 1))
        } else {
            // 一个兜底运算，尽量不要使用
            std::cout << "[WARNING] Incompatible value type of vec-add, type ID: " << typeid(ValType).name()
                << std::endl;
            INT32 grid_size = (dev_x.getLength() + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
            vecAddDevice<ValType, ValType, ValType><<<grid_size, MAX_BLOCK_SIZE, 0, **m_stream>>>(
                alpha, &dev_x[0], (ValType)1.0, &dev_y[0], &dev_y[0], dev_x.getLength());
        }
    }

    template <typename ValType>
    void CUBLAStools<ValType>::cublasMatVecMul(ValType alpha, cublasOperation_t matOption,
                                               const DeviceVector<ValType>& dev_matrix,
                                               const DeviceVector<ValType>& dev_vector,
                                               ValType beta, DeviceVector<ValType>& dev_result, INT32 rows,
                                               INT32 cols) {
#ifndef NDEBUG
        THROW_EXCEPTION(m_deviceID != dev_matrix.getLocation() || m_deviceID != dev_vector.getLocation() ||
                        m_deviceID != dev_result.getLocation(),
                        THROW_LOGIC_ERROR("The cublas handler and vectors are not on the same GPU!"))
        THROW_EXCEPTION(dev_matrix.getLength() != rows * cols,
                            THROW_LOGIC_ERROR("Matrix size does not match the provided dimensions!"))
        if (matOption == CUBLAS_OP_N) {
            THROW_EXCEPTION(dev_vector.getLength() != cols,
                            THROW_LOGIC_ERROR("Vector size does not match the matrix columns!"))
            THROW_EXCEPTION(dev_result.getLength() != rows,
                            THROW_LOGIC_ERROR("Result vector size does not match the matrix rows!"))
        } else if (matOption == CUBLAS_OP_T) {
            THROW_EXCEPTION(dev_vector.getLength() != rows,
                            THROW_LOGIC_ERROR("Vector size does not match the matrix columns!"))
            THROW_EXCEPTION(dev_result.getLength() != cols,
                            THROW_LOGIC_ERROR("Result vector size does not match the matrix rows!"))
        }
#endif
        CHECK_CUDA(cudaSetDevice(m_deviceID));

        if (std::is_same_v<ValType, FLOAT64>) {
            const auto local_alpha = static_cast<FLOAT64>(alpha);
            const auto local_beta = static_cast<FLOAT64>(beta);
            CHECK_CUBLAS(cublasDgemv(m_handler, matOption, rows, cols, &local_alpha, (double*)&dev_matrix[0], rows,
                (double*)&dev_vector[0], 1, &local_beta, (double*)&dev_result[0], 1));
        } else if (std::is_same_v<ValType, FLOAT32>) {
            const auto local_alpha = static_cast<FLOAT32>(alpha);
            const auto local_beta = static_cast<FLOAT32>(beta);
            CHECK_CUBLAS(cublasSgemv(m_handler, matOption, rows, cols, &local_alpha, (float*)&dev_matrix[0], rows,
                (float*)&dev_vector[0], 1, &local_beta, (float*)&dev_result[0], 1));
        } else {
            std::cout << "[WARNING] Incompatible value type for matrix-vector multiplication, type ID: " <<
                typeid(ValType).name() << std::endl;
        }
    }

    template <typename ValType>
    void CUBLAStools<ValType>::cublasMatMatMul(ValType alpha, cublasOperation_t transa,
                                               const DeviceVector<ValType>& dev_matrixA,
                                               cublasOperation_t transb, const DeviceVector<ValType>& dev_matrixB,
                                               ValType beta, DeviceVector<ValType>& dev_result, INT32 rowsA,
                                               INT32 colsA, INT32 colsB) {
#ifndef NDEBUG
        THROW_EXCEPTION(dev_matrixA.getLength() < rowsA * colsA,
                        THROW_LOGIC_ERROR("Matrix A size does not match the provided dimensions!"))
        THROW_EXCEPTION(dev_matrixB.getLength() < colsA * colsB,
                        THROW_LOGIC_ERROR("Matrix B size does not match the provided dimensions!"))
        THROW_EXCEPTION(dev_result.getLength() < rowsA * colsB,
                        THROW_LOGIC_ERROR("Result matrix size does not match the expected size!"))
        THROW_EXCEPTION(
            m_deviceID != dev_matrixA.getLocation() || m_deviceID != dev_matrixB.getLocation() || m_deviceID !=
            dev_result.getLocation(),
            THROW_LOGIC_ERROR("The cublas handler and matrices are not on the same GPU!"))
#endif
        CHECK_CUDA(cudaSetDevice(m_deviceID));
        /*  A: (m, k)
         *  B: (k, n)
         *  C: (m, n)
         */
        if (std::is_same_v<ValType, FLOAT64>) {
            const auto local_alpha = static_cast<FLOAT64>(alpha);
            const auto local_beta = static_cast<FLOAT64>(beta);
            CHECK_CUBLAS(cublasDgemm(m_handler, transa, transb, rowsA, colsB, colsA, &local_alpha,
                (double*)&dev_matrixA[0], rowsA, (double*)&dev_matrixB[0], colsA, &local_beta, (double*)&dev_result[0],
                rowsA
            ));
        } else if (std::is_same_v<ValType, FLOAT32>) {
            const auto local_alpha = static_cast<FLOAT32>(alpha);
            const auto local_beta = static_cast<FLOAT32>(beta);
            CHECK_CUBLAS(cublasSgemm(m_handler, transa, transb, rowsA, colsB, colsA, &local_alpha,
                (float*)&dev_matrixA[0], rowsA, (float*)&dev_matrixB[0], colsA, &local_beta, (float*)&dev_result[0],
                rowsA));
        } else {
            std::cout << "[WARNING] Incompatible value type for matrix-matrix multiplication, type ID: " <<
                typeid(ValType).name() << std::endl;
        }
    }
}
