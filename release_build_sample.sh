rm -rf ./cmake-build-release/*;
cmake -S . -G "Unix Makefiles" -D CMAKE_C_COMPILER="/usr/bin/gcc" -D CMAKE_CXX_COMPILER="/usr/bin/g++" -DCMAKE_BUILD_TYPE=Release -B cmake-build-release && cmake --build cmake-build-release -- -j 12;
# 如果需要编译CUDA代码，则加上： -D CMAKE_CUDA_COMPILER="/usr/local/cuda-12.3/bin/nvcc"