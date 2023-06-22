from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
from utils import wrappers


class MyKnn:

    def __init__(self,test_data:np.ndarray,scaler):
        self.test_data=test_data
        scaler.transform(self.test_data)
        
    @wrappers.time_counter
    def classifyByKNN(self,train_data:str):
        
        classifer=knn(algorithm='brute')
        classifer.fit(train_data[:,:-1],train_data[:,-1:])
        
        return classifer.predict(self.test_data)

    @wrappers.time_counter
    def classifyByImprovedKNN(self,indices):
        
        # indices是直接读取的json文件，先进行转换
        if(isinstance(indices,list)):
            from utils.transformers import jsonToIndices
            indices=jsonToIndices(indices)
        
        keys=list(indices.keys())
        N=len(keys)
        classifers=[knn(n_neighbors=1)]*N
        
        # 从这里开始计时
        for i in range(N):
            crt=np.array(indices[keys[i]])
            classifers[i].fit(crt[:,:-1],crt[:,-1:])


        mat=[c.kneighbors(self.test_data) for c in classifers]
        print(mat)
        res=[]
        # 返回预测类别
        return np.array(res)


if __name__=='__main__':
    from sklearn.preprocessing import MinMaxScaler
    import json
    test_data=np.array([[0.77306, 0.8216767, 0.83963],
                        [-1.199098,3.365850,0.189617],
                        [0.130689,-4.946325,-0.936443]])
    
    # 归一化统一用原始数据集
    train_data=np.loadtxt("./mocked_data5.txt",delimiter=" ")
    scaler=MinMaxScaler()
    scaler.fit(train_data[:,:-1])
    
    # -----KNN-----
    # train_data[:,:-1]=scaler.transform(train_data[:,:-1])
    
    # mk=MyKnn(test_data,scaler)
    # print(mk.classifyByKNN(train_data))
    
    # -----improved_KNN-----
    with open("./test_indices.json","r") as f:
        js=json.load(f)
        mk=MyKnn(test_data,scaler)
        mk.classifyByImprovedKNN(js)
    
    