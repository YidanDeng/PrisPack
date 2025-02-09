#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7};

    // 随机数生成器
    std::random_device rd;
    std::mt19937 g(rd());

    // 洗牌
    std::shuffle(numbers.begin(), numbers.end(), g);

    // 取前3个数
    for (int i = 0; i < 3; ++i) {
        std::cout << numbers[i] << " ";
    }

    return 0;
}