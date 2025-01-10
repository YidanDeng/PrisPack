#include <iostream>
#include <fstream>
#include <vector>

inline void getBuildFolderAbsolutePath(std::stringstream &ss) {
    char pathBuffer[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", pathBuffer, sizeof(pathBuffer) - 1);
    if (len != -1) {
        pathBuffer[len] = '\0'; // 确保以 null 结尾
    } else {
        TRY_CATCH(THROW_LOGIC_ERROR("Could not get absolute path of current exe file!"))
    }
    INT32 status = chdir(dirname(pathBuffer));  // 去掉文件后缀，只留文件夹的路径
    THROW_EXCEPTION(status == -1, THROW_INVALID_ARGUMENT("Changing work dir failed!"))
    ss << pathBuffer;
}


template <typename ValType>
inline void writeVectorToFile(ValType* valPtr, UINT32 length, const char* folderPath, const char* relativePath) {
    std::stringstream ss;
    ss << folderPath << "/" << relativePath;
    // 打开文件，使用 ios::out 模式（如果文件存在会覆盖）
    std::ofstream outFile(ss.str().c_str());
    // 检查文件是否成功打开
    if (!outFile) {
        std::cerr << "Can not open file: " << relativePath << std::endl;
        return;
    }
    // 遍历向量并写入文件
    for (UINT32 i = 0; i < length; i++) {
        outFile << valPtr[i] << std::endl; // 每个值一行
    }
    // 关闭文件
    outFile.close();
}
