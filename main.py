from KANN_DBSCAN import KANNDBSCAN
from My_KNN import MyKnn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import json
import seaborn as sns
from utils import wrappers

sns.set_style("darkgrid")
sns.set_context("paper")

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

@wrappers.time_counter
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

def evaluate():
    ...




if __name__ == "__main__":
    data=load_data("./data/final_data.txt",False)
    train_X, train_y, test_X, test_y, scaler=preprocess(data,split=True)
    # print(train_y)
    train(train_X,train_y,"1")
    # test()