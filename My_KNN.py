from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
from utils import wrappers
from sklearn.metrics import pairwise

class MyKnn:
    def __init__(self, test_data: np.ndarray, scaler):
        self.test_data = test_data
        scaler.transform(self.test_data)

    @wrappers.time_counter
    def classifyByKNN(self, train_data: str):

        classifer = knn(algorithm="brute")
        classifer.fit(train_data[:, :-1], train_data[:, -1:])

        return classifer.predict(self.test_data)

    @wrappers.time_counter
    def classifyByImprovedKNN(self, indices):

        # indices是直接读取的json文件，先进行转换
        if isinstance(indices, list):
            from utils.transformers import jsonToIndices

            indices = jsonToIndices(indices)

        core_samples, max_dists, clusters, labels = indices

        # N个聚簇
        N = len(core_samples)
        # 分别对每个聚簇建立分类器
        classifiers = [knn(n_neighbors=1)] * N

        # 从这里开始计时
        for i in range(N):
            classifiers[i].fit(clusters[i], labels[i])

        
        # 测试样本i到第j个聚簇核心点的距离
        mat=pairwise.euclidean_distances(test_data,core_samples)
        # 样本*候选聚簇矩阵，表示聚簇j是不是第i个样本的候选聚簇
        mat=mat<=max_dists

        res = []
        # 对每条测试样本分别筛选候选聚簇，并得出预测结果
        for i in range(len(self.test_data)):
            tmp=[]
            for j in range(N):
                if(mat[i][j]):
                    # todo: 找到每个簇里最近的样本
                    ...
            # 没有候选聚簇直接判定为异常样本
            if(len(tmp)==0):
                res.append(-1)
            else:
                # todo：统计得出预测结果
                ...

        # 返回预测类别
        return np.array(res)


if __name__ == "__main__":
    from sklearn.preprocessing import MinMaxScaler
    import json

    # 未归一化的测试集
    test_data = np.array(
        [
            [0.77306, 0.8216767, 0.83963],
            [-1.199098, 3.365850, 0.189617],
            [0.130689, -4.946325, -0.936443],
        ]
    )

    # 归一化统一用原始数据集
    train_data = np.loadtxt("./mocked_data5.txt", delimiter=" ")
    scaler = MinMaxScaler()
    scaler.fit(train_data[:, :-1])

    # -----KNN-----
    # train_data[:,:-1]=scaler.transform(train_data[:,:-1])

    # mk=MyKnn(test_data,scaler)
    # print(mk.classifyByKNN(train_data))

    # -----improved_KNN-----
    # with open("./test_indices.json","r") as f:
    #     js=json.load(f)
    #     mk=MyKnn(test_data,scaler)
    #     mk.classifyByImprovedKNN(js)

    

    core_samples = np.array([
            [0.77306938, 0.82167679, 0.8396377], 
            [0.50482839, 0.52072752, 0.59595394]
            ])
    max_dists=np.array([0.30798244,4])
    # 测试样本i到第j个聚簇核心点的距离
    mat=pairwise.euclidean_distances(
        test_data,core_samples
    )
    mat=mat<=max_dists
    print(mat)
    # print(np.linalg.norm(np.array([0.77306, 0.8216767, 0.83963])-np.array([0.50482839, 0.52072752, 0.59595394])))
