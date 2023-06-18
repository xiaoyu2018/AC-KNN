import numpy as np
import sklearn.metrics.pairwise as pairwise

def one():
    from time import time
    a=[[1]*23 for _ in range(8500)]

    b=[2]*23

    start=time()
    for s in a:
        tmp=0
        for i in range(len(b)):
            tmp+=(b[i]-s[i])**2   

    end=time()

    print(end-start)

def two():
    data=np.array([
        [1,2,3],
        [1,1,1],
        [2,3,4]
    ])
    dis_mat=pairwise.euclidean_distances(data)
    print(dis_mat)
    dis_mat.sort(1)
    dis_mat=dis_mat.mean(0)
    print(dis_mat[1:])

    import copy
    DistMatrix = pairwise.euclidean_distances(data)
    tmp_matrix = copy.deepcopy(DistMatrix)
    for i in range(len(tmp_matrix)):
        tmp_matrix[i].sort()
    EpsCandidates = []
    print()
    for k in range(1, len(data)):
        Dk = tmp_matrix[:, k]
        
        # 快160+倍
        DkAverage = np.mean(Dk)
        EpsCandidates.append(DkAverage)
    print(EpsCandidates)


def three():
    EpsCandidates= [1.90005653,3.23979425]
    DistMatrix=np.array([
        [0.         ,2.23606798 ,1.73205081],
        [2.23606798 ,0.         ,3.74165739],
        [1.73205081 ,3.74165739 ,0.        ],
    ])
    print(DistMatrix<=2)
    print(np.sum(DistMatrix <= 2,axis=1))

def main():
    three()


if __name__=='__main__':
    main()