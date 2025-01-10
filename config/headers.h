#ifndef CUDATOOLS_HEADERS_H
#define CUDATOOLS_HEADERS_H

extern "C++" {
#include <sys/resource.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <climits>
#include <string>
#include <ctime>
#include <utility>
#include <chrono>
#include <cstring>
#include <random>
#include <cfloat>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <memory>
#include <mutex>
#include <complex>


#ifdef _WIN32
#include <nmmintrin.h>
#include <immintrin.h>
#endif

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

#ifdef HAVE_FOG_VECTOR_CLASS
#define MAX_VECTOR_SIZE 512
#define VCL_NAMESPACE vcl
#include "vectorclass.h"    // 避免命名空间污染，尽量不用using namespace
#endif //HAVE_FOG_VECTOR_CLASS
}

#endif //CUDATOOLS_HEADERS_H
