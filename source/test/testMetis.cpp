/*
 * @author  邓轶丹
 * @date    2024/6/28
 * @details 测试Metis
 */
#include <metis.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

vector<idx_t> func(vector<idx_t> &xadj, vector<idx_t> &adjncy, vector<idx_t> &adjwgt, decltype(METIS_PartGraphKway) *METIS_PartGraphFunc) {
    idx_t nVertices = xadj.size() - 1; // 节点数
    idx_t nEdges = adjncy.size() / 2;  // 边数
    idx_t nWeights = 1;                // 节点权重维数
    idx_t nParts = 3;                  // 子图个数≥2
    idx_t objval;                      // 目标函数值
    vector<idx_t> part(nVertices, 0);  // 划分结果

    int ret = METIS_PartGraphFunc(&nVertices, &nWeights, xadj.data(), adjncy.data(),
                                  NULL, NULL, adjwgt.data(), &nParts, NULL,
                                  NULL, NULL, &objval, part.data());

    if (ret != rstatus_et::METIS_OK) { cout << "METIS_ERROR" << endl; }
    cout << "METIS_OK" << endl;
    cout << "objval: " << objval << endl;
    for (unsigned part_i = 0; part_i < part.size(); part_i++) {
        cout << part_i + 1 << " " << part[part_i] << endl;
    }

    return part;
}

int main() {
    ifstream ingraph("../datasets/graph.txt");
    int vexnum, edgenum;
    string line;
    getline(ingraph, line);
    istringstream tmp(line);
    tmp >> vexnum >> edgenum;
    vector<idx_t> xadj(0);
    vector<idx_t> adjncy(0); // 压缩图表示
    vector<idx_t> adjwgt(0); // 边权重

    idx_t a, w;
    for (int i = 0; i < vexnum; i++) {
        xadj.push_back(adjncy.size());
        getline(ingraph, line);
        istringstream tmp(line);
        while (tmp >> a >> w) {
            adjncy.push_back(a - 1); // 节点id从0开始
            adjwgt.push_back(w);
        }
    }
    xadj.push_back(adjncy.size());
    ingraph.close();

    vector<idx_t> part = func(xadj, adjncy, adjwgt, METIS_PartGraphRecursive);
//    vector<idx_t> part = func(xadj, adjncy, adjwgt, METIS_PartGraphKway);

    ofstream outpartition("partition.txt");
    for (int i = 0; i < part.size(); i++) { outpartition << i + 1 << " " << part[i] << endl; }
    outpartition.close();

    return 0;
}
