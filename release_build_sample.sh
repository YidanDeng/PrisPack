rm -rf ./cmake-build-release/*;
cmake -S . -G "Unix Makefiles" -D CMAKE_C_COMPILER="/usr/bin/gcc" -D CMAKE_CXX_COMPILER="/usr/bin/g++" -DCMAKE_BUILD_TYPE=Release -B cmake-build-release && cmake --build cmake-build-release -- -j 12;
