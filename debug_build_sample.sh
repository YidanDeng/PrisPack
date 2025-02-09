rm -rf ./cmake-build-debug/*;
cmake -S . -G "Unix Makefiles" \
  -D CMAKE_C_COMPILER="/usr/bin/gcc" \
  -D CMAKE_CXX_COMPILER="/usr/bin/g++" \
  -D CMAKE_BUILD_TYPE=Debug \
  -B cmake-build-debug && cmake --build cmake-build-debug -- -j 12;
# 如果需要编译CUDA代码，则加上： -D CMAKE_CUDA_COMPILER="/usr/local/cuda-12.3/bin/nvcc"





