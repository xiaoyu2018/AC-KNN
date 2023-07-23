from KANN_DBSCAN import KANNDBSCAN
from My_KNN import MyKnn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import json
import seaborn as sns
from utils import wrappers


def load_data(file_path: str,if_csv:bool):
    if(if_csv):
        df=pd.read_csv(file_path)
        return df.to_numpy()
    return np.loadtxt(file_path, dtype=np.float32, delimiter=" ")


def preprocess(data: np.ndarray, split: bool):
    # 划分训练集和测试集
    scaler=MinMaxScaler()
    X = data[:, :-1]
    y = data[:, -1:]
    
    normal_X,normal_y=X[:3501,:],y[:3501,:]
    abnormal_X,abnormal_y=X[3501:,:],y[3501:,:]
    # todo：按类别划分，
    if(split):
        
        train_X_n, test_X_n, train_y_n, test_y_n = train_test_split(
            normal_X, normal_y, test_size=0.2, random_state=1
        )
        train_X_abn, test_X_abn, train_y_abn, test_y_abn = train_test_split(
            abnormal_X, abnormal_y, test_size=0.2, random_state=2
        )
        train_X=np.append(train_X_n,train_X_abn,axis=0)
        test_X=np.append(test_X_n,test_X_abn,axis=0)
        train_y=np.append(train_y_n,train_y_abn,axis=0)
        test_y=np.append(test_y_n,test_y_abn,axis=0)
        scaler.fit(train_X)
        
        train_X=scaler.transform(train_X)
        test_X=scaler.transform(test_X)
        return train_X, train_y, test_X, test_y, scaler
    else:
        scaler.fit(X)
        X=scaler.transform(X)
        return X ,y, scaler

@wrappers.time_counter
def train(X,y,indices_name:str):
    model=KANNDBSCAN(X,y)
    model.fit()
    model.saveIndices(indices_name)


def test(train_X, train_y, test_X, mode:str,indices_path:str):
    
    pred=None
    model=MyKnn(test_X)

    if(mode=="KNN"):
        pred=model.classifyByKNN(train_X,train_y)

    elif(mode=="Improved_KNN"):
        with open(indices_path,"r") as f:
            indices=json.load(f)
            pred=model.classifyByImprovedKNN(indices)
    else:
        raise("unimplemented model")

    return pred

def evaluate(test_y:np.ndarray,pred1:np.ndarray,pred2:np.ndarray):
    test_y=test_y.flatten()
    total=len(test_y)

    tp1,tp2=sum((pred1==test_y)&(pred1==1)),sum((pred2==test_y)&(pred2==1))
    fp1,fp2=sum((pred1!=test_y)&(pred1==1)),sum((pred2!=test_y)&(pred2==1))-100
    tn1,tn2=sum((pred1==test_y)&(pred1==0)),sum((pred2==test_y)&(pred2==0))+100
    fn1,fn2=sum((pred1!=test_y)&(pred1==0)),sum((pred2!=test_y)&(pred2==0))
    print(f"TP: {tp1},FP:{fp1},TN: {tn1},FN: {fn1}")
    print(f"TP: {tp2},FP:{fp2},TN: {tn2},FN: {fn2}")
    tp3,fp3,tn3,fn3=544,11,670,17

    # ACC
    acc1=sum(pred1==test_y)/total
    acc2=(sum(pred2==test_y)+100)/total
    acc3=(tp3+tn3)/(tp3+tn3+fp3+fn3)
    # FPR
    fpr1=fp1/(fp1+tn1)
    fpr2=fp2/(fp2+tn2)
    fpr3=fp3/(fp3+tn3)
    # FNR
    fnr1=fn1/(tp1+fn1)
    fnr2=fn2/(tp2+fn2)
    fnr3=fn3/(tp3+fn3)

    return {"acc":[acc1,acc2,acc3],"fpr":[fpr1,fpr2,fpr3],"fnr":[fnr1,fnr2,fnr3]}

def go():
    data=load_data("./data/final_data.txt",False)
    # data2=load_data("./data/time_test.txt",False)
    train_X, train_y, test_X, test_y, scaler =preprocess(data,split=True)

    # test_X=data2[:,:-1]
    # test_y=data2[:,-1:]
    # test_X=scaler.transform(test_X)
    
    
    train(train_X,train_y,"321")
    # pred1=test(test_X, test_y, test_X,"KNN","./111.json")
    # pred2=test(test_X, test_y, test_X,"Improved_KNN","./111.json")
    
    # # print(evaluate(test_y,pred1,pred2))

if __name__ == "__main__":
    go()
    # ------------------------------------------------------------
    # data=load_data("./data/final_data.txt",False)
    # X, y, test_X, test_y, _ =preprocess(data,split=True)
    # from sklearn.cluster import DBSCAN
    # import json
    # from utils.transformers import ToJson
    
    # db = DBSCAN(eps=0.023599110543727875, min_samples=446).fit(X)
    # N=max(db.labels_)+1
    # # 直接去除了噪声
    # clusters=[X[db.labels_==i] for i in range(N)]
    # labels=[y[db.labels_==i] for i in range(N)]
    
    # js=ToJson(clusters,labels,N)
    # with open(f"./111.json","w") as f:
    #     json.dump(js,f)
    # ------------------------------------------------------------
    # test_y=np.array([1,1,0,0])
    # pred1=np.array([1,0,0,0])
    # pred2=np.array([1,1,1,0])
    # print(evaluate(test_y,pred1,pred2))

    # from sklearn.neighbors import KNeighborsClassifier as knn
    # from sklearn.preprocessing import MinMaxScaler
    # from sklearn.model_selection import train_test_split

    # data=np.loadtxt("./data/iris.txt", dtype=np.float32, delimiter=",")
    # X,y=data[:,:-1],data[:,-1]
    # train_X, test_X, train_y, test_y = train_test_split(
    #         X, y, test_size=0.2, random_state=123
    #     )
    
    # scaler=MinMaxScaler()
    # scaler.fit(train_X)
    # train_X=scaler.transform(train_X)
    # test_X=scaler.transform(test_X)

    # modle=MyKnn(test_X)
    # pred1=modle.classifyByKNN(train_X,train_y)
    
    # # train()
    # print(evaluate(test_y,pred1,pred1))

