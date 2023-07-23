import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle as pkl
sns.set_style("darkgrid")
sns.set_context("paper")
import numpy as np

def get_data_dist():

    df=pd.DataFrame(data={
        "Dataset":["train_normal","train_anomaly","test_normal","test_anomaly"],
        "Sample amount":[2800,2242,700,560]
    })
    sns.barplot(data=df,x="Dataset",y="Sample amount")

    plt.savefig("1.jpg",dpi=300)

def get_K_cLuster_curve():
    
    # with open("./utils/cluster_info.pkl","rb") as f1:
    #     data=pkl.load(f1)
        
    #     with open("./cluster_info.txt","w") as f2:
    #         for line in data:
    #             for i,v in enumerate(line):
    #                 f2.write(f"{v}")
    #                 if(i!=2):
    #                     f2.write(",")
    #             f2.write("\n")
    
    data=np.loadtxt("./cluster_info.txt",delimiter=",")

    # print(len(data))
    df=pd.DataFrame({
        "K-value":[i for i in range(2,102)],
        "Cluster quantity":[v[0] for v in data[:100]]
    })

    sns.lineplot(data=df,x="K-value",y="Cluster quantity",markers="o")

    plt.savefig("2.jpg",dpi=300)


if __name__=='__main__':
    get_K_cLuster_curve()