import numpy as np
# from .wrappers import time_counter #此处使用相对导入,否则顶层文件绝对导入本文件时报错

def ToJson(clusters: list, labels: list, N: int):
    cores = [np.mean(cluster, axis=0) for cluster in clusters]
    return [
        {
            "core_sample": cores[i].tolist(),
            "max_dist": np.max(np.linalg.norm(clusters[0] - cores[0], axis=1)).item(),
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
    return: {(core_smple,max_dist):full_cluster}这样的字典(最后一列是标签)
    '''
    
    res={
        (tuple(val["core_sample"]),val['max_dist']):[i['sample']+[i["label"]] for i in val['full_cluster']]
        for val in data
    }
    return res


if __name__=='__main__':
    import json

    with open("./test_indices.json","r") as f:
        js=json.load(f)
        print(jsonToIndices(js))
