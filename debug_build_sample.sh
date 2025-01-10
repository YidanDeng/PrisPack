rm -rf ./cmake-build-debug/*;
cmake -S . -G "Unix Makefiles" -D CMAKE_C_COMPILER="/usr/bin/gcc" -D CMAKE_CXX_COMPILER="/usr/bin/g++" -DCMAKE_BUILD_TYPE=Debug -B cmake-build-debug && cmake --build cmake-build-debug -- -j 12;
