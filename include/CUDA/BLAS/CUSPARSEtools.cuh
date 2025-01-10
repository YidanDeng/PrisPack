/**
 * @author  邓轶丹
 * @date    2024/6/2
 * @details 用于控制单GPU上使用多个流异步求解cusparse相关的计算，所有控制变量自动释放，待计算的值通过函数参数进行传递。 */

#ifndef PMSLS_NEW_CUSPARSETOOLS_CUH
#define PMSLS_NEW_CUSPARSETOOLS_CUH


#include "../../../config/config.h"
#include "../../../config/CUDAheaders.cuh"
#include "../../../include/MatrixClass/DeviceCSRMatrix.cuh"

namespace DEVICE {
    class CusparseHandler {
    private:
        cusparseHandle_t m_handler{nullptr};
        INT32 m_deviceID{DEFAULT_GPU};
        std::shared_ptr<StreamController> m_stream; ///< 转存变量

    public:
        CusparseHandler() = default;

        ~CusparseHandler() {
            if (m_handler) {
                CHECK_CUDA(cudaSetDevice(m_deviceID))
                CHECK_CUSPARSE(cusparseDestroy(m_handler))
            }
        }

        CusparseHandler(const CusparseHandler &other) = delete;

        CusparseHandler(CusparseHandler &&other) noexcept
            : m_handler(other.m_handler),
              m_deviceID(other.m_deviceID),
              m_stream(std::move(other.m_stream)) {
            other.m_handler = nullptr;
        }

        CusparseHandler &operator=(const CusparseHandler &other) = delete;

        CusparseHandler &operator=(CusparseHandler &&other) noexcept {
            if (this == &other)
                return *this;
            m_handler = other.m_handler;
            m_deviceID = other.m_deviceID;
            m_stream = std::move(other.m_stream);
            other.m_handler = nullptr;
            return *this;
        }

        explicit CusparseHandler(const INT32 &devID) {
            m_deviceID = devID;
            CHECK_CUDA(cudaSetDevice(m_deviceID))
            CHECK_CUSPARSE(cusparseCreate(&m_handler))
            m_stream = std::make_shared<StreamController>();
            CHECK_CUSPARSE(cusparseSetStream(m_handler, **m_stream))
        }

        explicit CusparseHandler(const std::shared_ptr<StreamController> &stream) {
#ifndef NDEBUG
            // 如果stream的指针没有实例化具体对象，则这个参数是无效的
            THROW_EXCEPTION(stream.get() == nullptr, THROW_INVALID_ARGUMENT("The stream pointer was not initialized!"))
#endif
            m_deviceID = stream->getDeviceID();
            m_stream = stream;
            CHECK_CUDA(cudaSetDevice(m_stream->getDeviceID()))
            CHECK_CUSPARSE(cusparseCreate(&m_handler))
            CHECK_CUSPARSE(cusparseSetStream(m_handler, **stream))
        }

        inline cusparseHandle_t &operator*() {
            return m_handler;
        }

        inline const cusparseHandle_t &operator*() const {
            return m_handler;
        }

        inline INT32 getDeviceID() const {
            return m_deviceID;
        }

        inline const std::shared_ptr<StreamController> &getStream() const {
            return m_stream;
        }
    };


    /* ================================== CSRDescriptor ================================== */
    template<typename ValType>
    class CusparseCSRDescriptor {
    private:
        cusparseSpMatDescr_t m_mat{nullptr}; ///< 用于存放稀疏矩阵的描述符
        cudaDataType_t m_dataType{CUDA_R_32F};
        std::shared_ptr<DeviceCSRMatrix<ValType> > m_oriMat;

    public:
        explicit CusparseCSRDescriptor(const std::shared_ptr<DeviceCSRMatrix<ValType> > &mat);

        ~CusparseCSRDescriptor();

        CusparseCSRDescriptor(const CusparseCSRDescriptor &other) = delete;

        CusparseCSRDescriptor(CusparseCSRDescriptor &&other) noexcept
            : m_mat(other.m_mat),
              m_dataType(other.m_dataType),
              m_oriMat(std::move(other.m_oriMat)) {
            other.m_mat = nullptr;
        }

        CusparseCSRDescriptor &operator=(const CusparseCSRDescriptor &other) = delete;

        CusparseCSRDescriptor &operator=(CusparseCSRDescriptor &&other) noexcept {
            if (this == &other)
                return *this;
            m_mat = other.m_mat;
            m_dataType = other.m_dataType;
            m_oriMat = std::move(other.m_oriMat);
            other.m_mat = nullptr;
            return *this;
        }


        inline DeviceCSRMatrix<ValType> &operator*() {
            return *m_oriMat;
        }

        inline const DeviceCSRMatrix<ValType> &operator*() const {
            return *m_oriMat;
        }

        inline cusparseSpMatDescr_t getCSRDescriber() const {
            return m_mat;
        }

        inline INT32 getDeviceID() const {
            return m_oriMat->getDeviceID();
        }

        inline UINT32 *getRowOffsetPtr() const {
            return m_oriMat->getRowOffsetPtr();
        }


        inline UINT32 *getColIndicesPtr() const {
            return m_oriMat->getColIndicesPtr();
        }


        inline ValType *getValuesPtr() const {
            return m_oriMat->getValuesPtr();
        }


        inline UINT32 getNNZ() const {
            return m_oriMat->getNNZ();
        }

        inline UINT32 getRowNum() const {
            return m_oriMat->getRowNum();
        }

        inline UINT32 getColNum() const {
            return m_oriMat->getColNum();
        }

        inline cudaDataType_t getCuDataType() const {
            return m_dataType;
        }
    };


    template class CusparseCSRDescriptor<FLOAT32>;
    template class CusparseCSRDescriptor<FLOAT64>;
    /* ================================== CSRDescriptor end ================================== */


    /* ================================== DnVectorDescriptor ================================== */
    template<typename ValType>
    class CusparseDnVectorDescriptor {
    private:
        cusparseDnVecDescr_t m_vec{nullptr};
        cudaDataType_t m_dataType{CUDA_R_32F};
        std::shared_ptr<SyncDeviceVector<ValType> > m_oriVec;

    public:
        explicit CusparseDnVectorDescriptor(const std::shared_ptr<SyncDeviceVector<ValType> > &vec);

        ~CusparseDnVectorDescriptor();

        CusparseDnVectorDescriptor(const CusparseDnVectorDescriptor &other) = delete;

        CusparseDnVectorDescriptor(CusparseDnVectorDescriptor &&other) noexcept
            : m_vec(other.m_vec),
              m_dataType(other.m_dataType),
              m_oriVec(std::move(other.m_oriVec)) {
            other.m_vec = nullptr;
        }

        CusparseDnVectorDescriptor &operator=(const CusparseDnVectorDescriptor &other) = delete;

        CusparseDnVectorDescriptor &operator=(CusparseDnVectorDescriptor &&other) noexcept {
            if (this == &other)
                return *this;
            m_vec = other.m_vec;
            m_dataType = other.m_dataType;
            m_oriVec = std::move(other.m_oriVec);
            other.m_vec = nullptr;
            return *this;
        }

        inline SyncDeviceVector<ValType> &operator*() {
            return *m_oriVec;
        }

        inline const SyncDeviceVector<ValType> &operator*() const {
            return *m_oriVec;
        }

        inline cusparseDnVecDescr_t getDenseVecDescriber() const {
            return m_vec;
        }

        inline INT32 getDeviceID() const {
            return m_oriVec->getLocation();
        }

        inline ValType *getRawValPtr() const {
            return m_oriVec->getRawValPtr();
        }

        inline UINT32 getLength() const {
            return m_oriVec->getLength();
        }

        inline size_t getByteSize() const {
            return m_oriVec->getByteSize();
        }
    };


    template
    class CusparseDnVectorDescriptor<INT32>;

    template
    class CusparseDnVectorDescriptor<UINT32>;

    template
    class CusparseDnVectorDescriptor<FLOAT32>;

    template
    class CusparseDnVectorDescriptor<FLOAT64>;
    /* ================================== DnVectorDescriptor end ================================== */

    /* ================================== CSRSpMV ================================== */
    /* 基于cusparse库实现矩阵向量乘法（该函数适合固定矩阵乘多个向量的情况）*/
    template<typename ValType>
    class CusparseCsrSpMV {
    private:
        std::shared_ptr<CusparseHandler> m_handler; ///< cusparse控制器，可内部初始化，也可以转存外部变量
        std::shared_ptr<CusparseCSRDescriptor<ValType> > m_matA; ///< 用于存放稀疏矩阵的描述符，转存变量

        ValType m_alpha{1};
        ValType m_beta{0};
        void *m_dBuffer{nullptr}; ///< 用于存放乘法过程中用到的缓冲区，对于同一个CSR矩阵仅初始化一次
        size_t m_bufferSize{0}; ///< 记录缓冲区大小
        cusparseOperation_t m_csrOpt{CUSPARSE_OPERATION_NON_TRANSPOSE};

    public:
        /** @brief 初始化GPU上的稀疏矩阵向量乘法控制器
         * @details 根据指定流控制器，在类内部初始化cusparse的控制器，并绑定到对应流上 */
        CusparseCsrSpMV(const std::shared_ptr<CusparseCSRDescriptor<ValType> > &matDesc,
                        const std::shared_ptr<StreamController> &stream);


        /** @brief 初始化GPU上的稀疏矩阵向量乘法控制器 */
        CusparseCsrSpMV(const std::shared_ptr<CusparseCSRDescriptor<ValType> > &matDesc,
                        const std::shared_ptr<CusparseHandler> &handler);

        ~CusparseCsrSpMV();

        CusparseCsrSpMV(const CusparseCsrSpMV &other) = delete;

        CusparseCsrSpMV(CusparseCsrSpMV &&other) noexcept
            : m_handler(std::move(other.m_handler)),
              m_matA(std::move(other.m_matA)),
              m_alpha(std::move(other.m_alpha)),
              m_beta(std::move(other.m_beta)),
              m_dBuffer(other.m_dBuffer),
              m_bufferSize(other.m_bufferSize) {
            other.m_dBuffer = nullptr;
        }

        CusparseCsrSpMV &operator=(const CusparseCsrSpMV &other) = delete;

        CusparseCsrSpMV &operator=(CusparseCsrSpMV &&other) noexcept {
            if (this == &other)
                return *this;
            m_handler = std::move(other.m_handler);
            m_matA = std::move(other.m_matA);
            m_alpha = std::move(other.m_alpha);
            m_beta = std::move(other.m_beta);
            m_dBuffer = other.m_dBuffer;
            m_bufferSize = other.m_bufferSize;
            other.m_dBuffer = nullptr;
            return *this;
        }

        /** @brief 计算Y = alpha * A * X + beta * Y*/
        void csrMultiplyVec(const cusparseOperation_t &matAop, const ValType &alpha,
                            const CusparseDnVectorDescriptor<ValType> &d_in, const ValType &beta,
                            CusparseDnVectorDescriptor<ValType> &d_out);

        inline void csrMultiplyVec(const cusparseOperation_t &matAop, const ValType &alpha,
                                   const std::shared_ptr<CusparseDnVectorDescriptor<ValType> > &d_in,
                                   const ValType &beta, std::shared_ptr<CusparseDnVectorDescriptor<ValType> > &d_out) {
#ifndef NDEBUG
            THROW_EXCEPTION(d_in->getDeviceID() != m_matA->getDeviceID() ||
                            d_out->getDeviceID() != m_matA->getDeviceID(),
                            THROW_LOGIC_ERROR("The device vector and CSR matrix are not on the same GPU!"))
#endif
            csrMultiplyVec(matAop, alpha, *d_in, beta, *d_out);
        }

        inline void csrMultiplyVec(const CusparseDnVectorDescriptor<ValType> &d_in,
                                   CusparseDnVectorDescriptor<ValType> &d_out) {
            csrMultiplyVec(CUSPARSE_OPERATION_NON_TRANSPOSE, 1, d_in, 0, d_out);
        }

        inline void csrMultiplyVec(const std::shared_ptr<CusparseDnVectorDescriptor<ValType> > &d_in,
                                   std::shared_ptr<CusparseDnVectorDescriptor<ValType> > &d_out) {
            csrMultiplyVec(CUSPARSE_OPERATION_NON_TRANSPOSE, 1, *d_in, 0, *d_out);
        }


        inline INT32 getDeviceID() {
            return m_matA->getDeviceID();
        }

        inline cudaDataType_t getCuDataType() {
            return m_matA->getCuDataType();
        }
    };

    template
    class CusparseCsrSpMV<FLOAT32>;

    template
    class CusparseCsrSpMV<FLOAT64>;
    /* ================================== CSRSpMV end ================================== */


    /* ================================== CSRSpSV ================================== */
    template<typename ValType>
    class CusparseCSRTriSolve {
    private:
        std::shared_ptr<CusparseHandler> m_handler; ///< cusparse控制器
        std::shared_ptr<CusparseCSRDescriptor<ValType> > m_matA; ///< 用于存放原矩阵的描述符，转存变量
        std::shared_ptr<StreamController> m_stream; ///< cuda流，转存变量

        ValType m_alpha{1};
        size_t m_bufferSize{0}; ///< 计算所需的缓冲区大小（单位：字节）
        void *m_bufferPtr{nullptr}; ///< 缓冲区指针
        cusparseSpSVDescr_t m_spsvDescriptor{nullptr}; ///< 稀疏三角求解描述符
        cusparseFillMode_t m_fillMode{CUSPARSE_FILL_MODE_LOWER}; ///< 填充模式（描述稀疏三角块是上三角还是下三角）
        cusparseDiagType_t m_diagType{CUSPARSE_DIAG_TYPE_UNIT}; ///< 对角块模式（描述稀疏三角块包不包括对角元）
        cusparseOperation_t m_csrOpt{CUSPARSE_OPERATION_NON_TRANSPOSE}; ///< 标记计算时A是否转置

    public:
        CusparseCSRTriSolve() = default;

        /** @brief 初始化GPU上的稀疏三角方程组求解控制器 */
        CusparseCSRTriSolve(const std::shared_ptr<CusparseCSRDescriptor<ValType> > &matDesc,
                            const cusparseFillMode_t &fillMode,
                            const cusparseDiagType_t &diagType, const std::shared_ptr<CusparseHandler> &handler);

        ~CusparseCSRTriSolve();

        CusparseCSRTriSolve(const CusparseCSRTriSolve &other) = delete;

        CusparseCSRTriSolve(CusparseCSRTriSolve &&other) noexcept
            : m_handler(std::move(other.m_handler)),
              m_matA(std::move(other.m_matA)),
              m_stream(std::move(other.m_stream)),
              m_alpha(other.m_alpha),
              m_bufferSize(other.m_bufferSize),
              m_bufferPtr(other.m_bufferPtr),
              m_spsvDescriptor(other.m_spsvDescriptor),
              m_fillMode(other.m_fillMode),
              m_diagType(other.m_diagType),
              m_csrOpt(other.m_csrOpt) {
            other.m_bufferPtr = nullptr;
            other.m_bufferSize = 0;
            other.m_spsvDescriptor = nullptr;
        }

        CusparseCSRTriSolve &operator=(const CusparseCSRTriSolve &other) = delete;

        CusparseCSRTriSolve &operator=(CusparseCSRTriSolve &&other) noexcept {
            if (this == &other)
                return *this;
            m_handler = std::move(other.m_handler);
            m_matA = std::move(other.m_matA);
            m_stream = std::move(other.m_stream);
            m_alpha = other.m_alpha;
            m_bufferSize = other.m_bufferSize;
            m_bufferPtr = other.m_bufferPtr;
            m_spsvDescriptor = other.m_spsvDescriptor;
            m_fillMode = other.m_fillMode;
            m_diagType = other.m_diagType;
            m_csrOpt = other.m_csrOpt;
            other.m_bufferPtr = nullptr;
            other.m_bufferSize = 0;
            other.m_spsvDescriptor = nullptr;
            return *this;
        }

        /** @brief 求解 tri(A) * Y = alpha * X */
        void csrTriSolve(const cusparseOperation_t &matAop, ValType alpha,
                         const CusparseDnVectorDescriptor<ValType> &d_in, CusparseDnVectorDescriptor<ValType> &d_out);

        inline void csrTriSolve(const cusparseOperation_t &matAop,
                                const std::shared_ptr<CusparseDnVectorDescriptor<ValType> > &d_in,
                                std::shared_ptr<CusparseDnVectorDescriptor<ValType> > &d_out) {
            csrTriSolve(matAop, 1, *d_in, *d_out);
        }

        inline void csrTriSolve(const std::shared_ptr<CusparseDnVectorDescriptor<ValType> > &d_in,
                                std::shared_ptr<CusparseDnVectorDescriptor<ValType> > &d_out) {
            csrTriSolve(CUSPARSE_OPERATION_NON_TRANSPOSE, 1, *d_in, *d_out);
        }
    };


    template
    class CusparseCSRTriSolve<FLOAT32>;

    template
    class CusparseCSRTriSolve<FLOAT64>;
}


#endif //PMSLS_NEW_CUSPARSETOOLS_CUH
