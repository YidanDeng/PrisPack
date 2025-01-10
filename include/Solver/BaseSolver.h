#ifndef PMSLS_NEW_BASESOLVER_H
#define PMSLS_NEW_BASESOLVER_H

#include "../MatrixClass/CSRMatrix.h"
#include "../MatrixClass/DenseMatrix.h"
#include "../utils/MemoryTools/SharedPtrTools.h"


template<typename ValType>
class BaseSolver {
public:
    BaseSolver() = default;

    virtual ~BaseSolver() = default;

    virtual void solve(const HOST::CSRMatrix<ValType> &matA, const HostVector<ValType> &b, HostVector<ValType> &x) = 0;
};

#endif //PMSLS_NEW_BASESOLVER_H
