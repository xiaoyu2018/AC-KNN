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
    
    # todo：按类别划分
    if(split):
        
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=1
        )
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


def test(train_X, train_y, test_X, scaler:MinMaxScaler,mode:str,indices_path:str):
    train_X=scaler.transform(train_X)
    pred=None
    model=MyKnn(test_X,scaler)

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
    fp1,fp2=sum((pred1!=test_y)&(pred1==1)),sum((pred2!=test_y)&(pred2==1))
    tn1,tn2=sum((pred1==test_y)&(pred1==0)),sum((pred2==test_y)&(pred2==0))
    fn1,fn2=sum((pred1!=test_y)&(pred1==0)),sum((pred2!=test_y)&(pred2==0))
    # ACC
    acc1=sum(pred1==test_y)/total
    acc2=sum(pred2==test_y)/total
    
    # FPR
    fpr1=fp1/(fp1+tn1)
    fpr2=fp2/(fp2+tn2)
    # FNR
    fnr1=fn1/(tp1+fn1)
    fnr2=fn2/(tp2+fn2)

    return {"acc":[acc1,acc2],"fpr":[fpr1,fpr2],"fnr":[fnr1,fnr2]}

def go():
    data=load_data("./data/final_data.txt",False)
    train_X, train_y, test_X, test_y, scaler=preprocess(data,split=True)
    # print(train_y)
    # train(train_X,train_y,"1")
    pred1=test(train_X, train_y, test_X, scaler,"KNN","./1.json")
    pred2=test(train_X, train_y, test_X, scaler,"Improved_KNN","./1.json")
    
    print(evaluate(test_y,pred1,pred2))

if __name__ == "__main__":
    go()
    # test_y=np.array([1,1,0,0])
    # pred1=np.array([1,0,0,0])
    # pred2=np.array([1,1,1,0])
    # print(evaluate(test_y,pred1,pred2))