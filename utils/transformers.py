import numpy as np
# from .wrappers import time_counter #此处使用相对导入,否则顶层文件绝对导入本文件时报错

def ToJson(clusters: list, labels: list, N: int):
    cores = [np.mean(cluster, axis=0) for cluster in clusters]
    return [
        {
            "core_sample": cores[i].tolist(),
            "max_dist": np.max(np.linalg.norm(clusters[i] - cores[i], axis=1)).item(),
            "full_cluster": [
                {"sample": clusters[i][j].tolist(), "label": labels[i][j].item()}
                for j in range(clusters[i].shape[0])
            ],
        }
        for i in range(N)
    ]

# @time_counter
def jsonToIndices(data: list):
    '''
    param data: 直接json.load得到的列表
    return: 返回三个列表[core_smple] [max_dist] [full_cluster] [labels]
    '''
    
    res1=np.array([v["core_sample"] for v in data])
    res2=np.array([v['max_dist'] for v in data])
    # 聚簇内样本点数量不同，所有就不转换成张量了
    res3=[[sample["sample"] for sample in v["full_cluster"]] for v in data]
    res4=[[sample["label"] for sample in v["full_cluster"]] for v in data]
    
    return res1,res2,res3,res4


if __name__=='__main__':
    import json

    with open("./test_indices.json","r") as f:
        js=json.load(f)
        for v in jsonToIndices(js):
            print(v)
            print()

    
