import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def get_data_dist():

    df=pd.DataFrame(data={
        "Dataset":["train_normal","train_anomaly","test_normal","test_anomaly"],
        "Sample amount":[3600,1841,900,461]
    })
    sns.barplot(data=df,x="Dataset",y="Sample amount")

    plt.savefig("1.jpg",dpi=300)

def get_K_cLuster_curve():
    ...



if __name__=='__main__':
    ...