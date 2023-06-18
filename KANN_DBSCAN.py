import copy
import numpy as np
from sklearn.cluster import DBSCAN
import sklearn.metrics.pairwise as pairwise
from utils.wrappers import time_counter

class KANN_DBSCAN:

    def __init__(self,data:np.ndarray):
        self.X=data
        self.DistMatrix=0 #距离矩阵
        self.EpsCandidates=None #候选Eps集合
        self.MinptsCandidates=None #候选MinPts集合
        self.all_param_list = None #所有聚类情况的参数

    def _returnEpsCandidates(self):
        """
        param dataSet: 数据集
        return: eps候选集合
        """
        self.DistMatrix=pairwise.euclidean_distances(self.X)
        dis_mat=copy.deepcopy(self.DistMatrix)
        dis_mat.sort(1)
        dis_mat=dis_mat.mean(0)
        
        # 返回EpsCandidates
        return dis_mat[1:]

    def _returnMinptsCandidates(self, DistMatrix, EpsCandidates):
        """
        param DistMatrix: 距离矩阵
        param EpsCandidates: Eps候选列表
        return: Minpts候选列表
        """
        MinptsCandidates = [int(np.sum(DistMatrix <= eps,axis=1).mean())
                             for eps in EpsCandidates]
        
        return MinptsCandidates

    def fit(self):
        self.EpsCandidates = self._returnEpsCandidates()
        self.MinptsCandidates = self._returnMinptsCandidates(
            self.DistMatrix, self.EpsCandidates
        )
        self._do_multi_dbscan()
        

    def _do_multi_dbscan(self):
        # 根据候选Eps和MinPts多次进行聚类
        # self.all_predict_list = []
        self.all_param_list = []

        for i in range(len(self.EpsCandidates)):
            eps = self.EpsCandidates[i]
            minpts = self.MinptsCandidates[i]
            
            db = DBSCAN(eps=eps, min_samples=minpts).fit(self.X)
            num_clusters = max(db.labels_) + 1

            # self.all_predict_list.append(db.labels_)
            self.all_param_list.append((num_clusters,eps,minpts))

    # 获取当前Multi-DBSCAN的聚类参数信息,格式为{聚类个数:[{"eps":x1,"minpts":y1},{"eps":x2,"minpts":y2},...]}
    def get_info_dict(self):
        if self.all_param_list is None:
            raise RuntimeError("get_info_dict before fit")
        return self.all_param_list

    def _get_best_cluster(self):
        """
        找到最佳聚类方案
        return: 最佳Eps和MinPts
        """
        size=len(self.all_param_list)
        if(size<3):
            return self.all_param_list[0][1:]

        for i in range(2,size):
            if(self.all_param_list[i][0]==self.all_param_list[i-1][0]==self.all_param_list[i-2][0]):
                j=i+1
                if(j>=size or self.all_param_list[j][0]!=self.all_param_list[i][0]):
                    return self.all_param_list[j-1][1:]
        
        # 退化为普通KNN
        return self.all_param_list[-1][1:]
    
    def save_index(self,file_name:str):
        """
        param file_name:保存的聚簇索引json文件名
        保存聚簇索引
        """

        import json
        eps,minpts=self._get_best_cluster()
        db = DBSCAN(eps=eps, min_samples=minpts).fit(self.X)
        N=max(db.labels_)+1
        clusters=[self.X[np.where(db.labels_==i)] for i in range(N)]
        cores=[np.mean(cluster,axis=0) for cluster in clusters]
        
        indices=[
            {
                "core_sample":cores[i].tolist(),
                "max_dist":np.max(np.linalg.norm(clusters[0]-cores[0],axis=1)).item(),
                "full_cluster":clusters[i].tolist()
            }
            for i in range(N)
        ]
        # print(indices)
        with open(f"./{file_name}.json","w") as f:
            json.dump(indices,f)




if __name__ =='__main__':
    @time_counter
    def run_KD(data):
        kd=KANN_DBSCAN(data)
        kd.fit()
        # print(kd.get_info_dict())
        kd.save_index("test_indices")
        return kd._get_best_cluster()
    
    from sklearn.preprocessing import MinMaxScaler
    import seaborn as sns
    from matplotlib import pyplot as plt   
    import pandas as pd 

    data=np.loadtxt("./mocked_data.txt",delimiter=" ",dtype=np.float32)
    
    
    
    scaler=MinMaxScaler()
    scaler.fit(data[:,:2])
    data=scaler.transform(data[:,:2])
    res=run_KD(data)
    print(f"best param:{res}")

    ds=DBSCAN(eps=res[0],min_samples=res[1])
    labels=ds.fit(data).labels_
    p_data=pd.DataFrame(data={"A":data[:,0].flatten(),"B":data[:,1].flatten(),"L":labels})
    sns.scatterplot(data=p_data,x="A",y="B",hue="L")
    plt.show()

    