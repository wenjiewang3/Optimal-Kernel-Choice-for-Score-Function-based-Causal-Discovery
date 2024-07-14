import cdt
import numpy as np
import matplotlib.pyplot as plt

def F1_score_nparray(gt, est):
    """
    Compute and store the adjacency confusion between two graphs.
    gt : Truth graph (numpy.array)
    est: Estimated graph(numpy.array)
    """
    adjFn = 0; adjTp = 0; adjFp = 0; adjTn = 0
    assert gt.shape[0] == gt.shape[1]
    lens_of_nodes = gt.shape[0]

    for i in np.arange(0, lens_of_nodes):
        for j in np.arange(i+1, lens_of_nodes):
            estAdj = (est[j, i] != 0)
            truthAdj  = (gt[j, i] != 0)

            if truthAdj and not estAdj:
                adjFn = adjFn + 1
            elif estAdj and not truthAdj:
                adjFp = adjFp + 1
            elif estAdj and truthAdj:
                adjTp = adjTp + 1
            elif not estAdj and not truthAdj:
                adjTn = adjTn + 1

    precesion = adjTp / (adjTp + adjFp) if (adjTp + adjFp) != 0 else 0
    recall = adjTp / (adjTp + adjFn) if (adjTp + adjFn) != 0 else 0
    # print("pre: ", precesion, "recall: ", recall)
    F1_score = 2*(recall*precesion) / (recall + precesion) if (recall + precesion) != 0 else 0
    return F1_score

def SHD_nparray(gt, est, double_for_anticausal=True):
    n = gt.shape[-1]
    SHD = cdt.metrics.SHD(gt, est, double_for_anticausal) / (n * (n-1))
    return SHD

def G_transfer(graph):
    graph[np.abs(graph) < 0.1] = 0
    graph[graph > 0.1] = -1
    graph[graph < -0.1] = 1
    gt = -graph.T
    return graph + gt

def G_onlyone(graph):
    graph[graph == 1] = 0
    graph[graph == -1] = 1
    return graph
