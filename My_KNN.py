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

    
    def classifyByImprovedKNN(self,train_data_path:str):
        ...



if __name__=='__main__':
    from sklearn.preprocessing import MinMaxScaler

    test_data=np.array([[4.234278,3.462615,4.734509],
                        [-1.199098,3.365850,0.189617],
                        [0.130689,-4.946325,-0.936443]])
    
    train_data=np.loadtxt("./mocked_data5.txt",delimiter=" ")
    scaler=MinMaxScaler()
    scaler.fit(train_data[:,:-1])
    train_data[:,:-1]=scaler.transform(train_data[:,:-1])
    
    mk=MyKnn(test_data,scaler)
    

    print(mk.classifyByKNN(train_data))
    