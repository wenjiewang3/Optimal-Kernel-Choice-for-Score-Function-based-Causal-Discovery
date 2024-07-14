from typing import Optional

import torch
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.GESUtils import *
from causallearn.utils.PDAG2DAG import pdag2dag

from Score.GPScore import local_score_gp_lbfgsb
from Score.MargScore import local_score_marg_lbfgsb
from Score.MIScore import local_score_mi_lbfgsb


def ges(Data_dir, score_func: str = 'local_score_BIC', parameters = None):
    """
    Perform greedy equivalence search (Search) algorithm

    Parameters
    ----------
    X : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    Data_dir : data dir    X = data_dir['data_mat'] (numpy ndarray), shape (n_samples, n_features)
                           The input data, where n_samples is the number of samples and n_features is the number of features.
    score_func : the string name of score function. (str(one of 'local_score_CV_general', 'local_score_marginal_general',
                    'local_score_cat')).
    maxP : allowed maximum number of parents when searching the graph
    parameters : when using CV likelihood,
                  parameters['kfold']: k-fold cross validation
                  parameters['lambda']: regularization parameter
                  parameters['dlabel']: for variables with multi-dimensions,
                               indicate which dimensions belong to the i-th variable.
    -------
    Record['G']: learned causal graph, where Record['G'].graph[j,i]=1 and Record['G'].graph[i,j]=-1 indicates  i --> j ,
                    Record['G'].graph[i,j] = Record['G'].graph[j,i] = -1 indicates i --- j.
    Record['update1']: each update (Insert operator) in the forward step
    Record['update2']: each update (Delete operator) in the backward step
    Record['G_step1']: learned graph at each step in the forward step
    Record['G_step2']: learned graph at each step in the backward step
    Record['score']: the score of the learned graph
    """
    X = Data_dir['data_mat']
    idx_list = Data_dir['var_idx']
    if score_func != 'local_score_CV_multi':
        parameters['var_idx'] = idx_list
    N = len(Data_dir['var_idx'])  # number of variables
    maxP = int(N/2) # maximum number of parents
    threshold = 0
    if X.shape[0] < X.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    X = np.mat(X)
    if score_func == 'local_score_Marg':  # negative marginal likelihood based on regression in RKHS
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_marg_lbfgsb, parameters=parameters)

    elif score_func == 'local_score_MI': # Mutual information-based score function
        parameters['width_init'] = Data_dir.get('width_init')
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_mi_lbfgsb, parameters=parameters)

    elif score_func == 'local_score_GP': # vanilla GP score function
        parameters['width_init'] = Data_dir.get('width_init')
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_gp_lbfgsb, parameters=parameters)
    else:
        raise Exception('Unknown score function')

    score_func = localScoreClass
    node_names = [("x%d" % i) for i in range(N)]
    nodes = []

    for name in node_names:
        node = GraphNode(name)
        nodes.append(node)

    G = GeneralGraph(nodes)
    # G = np.matlib.zeros((N, N)) # initialize the graph structure
    score = score_g(X, G, score_func, parameters)  # initialize the score

    # print(G)

    G = pdag2dag(G)
    G = dag2cpdag(G)

    # print(G)

    ## --------------------------------------------------------------------
    ## forward greedy search
    record_local_score = [[] for i in range(N)]
    # record the local score calculated each time. Thus when we transition to the second phase,
    # many of the operators can be scored without an explicit call the the scoring function
    # record_local_score{trial}{j} record the local scores when Xj as a parent

    score_new = score
    count1 = 0
    update1 = []
    G_step1 = []
    score_record1 = []
    graph_record1 = []
    while True:
        count1 = count1 + 1
        score = score_new
        score_record1.append(score)
        graph_record1.append(G)
        min_chscore = 1e7
        min_desc = []
        for i in range(N):
            for j in range(N):
                if (G.graph[i, j] == Endpoint.NULL.value and G.graph[j, i] == Endpoint.NULL.value
                        and i != j and len(np.where(G.graph[j, :] == Endpoint.ARROW.value)[0]) <= maxP):
                    # find a pair (Xi, Xj) that is not adjacent in the current graph , and restrict the number of parents
                    Tj = np.intersect1d(np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                                        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0])  # neighbors of Xj  Xj中的连接但未确定方向的所有元素

                    Ti = np.union1d(np.where(G.graph[:, i] != Endpoint.NULL.value)[0],
                                    np.where(G.graph[i, 0] != Endpoint.NULL.value)[0])  # adjacent to Xi 所有与Xi相连的所有元素

                    NTi = np.setdiff1d(np.arange(N), Ti)   # 与Xi不相连的元素
                    T0 = np.intersect1d(Tj, NTi)  # 在Tj中找到所有与Ti不相连的元素   find the neighbours of Xj that are not adjacent to Xi
                    # for any subset of T0
                    sub = Combinatorial(T0.tolist())  # find all the subsets for T0
                    S = np.zeros(len(sub))
                    # S indicate whether we need to check sub{k}.
                    # 0: check both conditions.
                    # 1: only check the first condition
                    # 2: check nothing and is not valid.
                    for k in range(len(sub)):
                        if (S[k] < 2):  # S indicate whether we need to check subset(k)
                            V1 = insert_validity_test1(G, i, j, sub[k])  # Insert operator validation test:condition 1
                            if (V1):
                                if (not S[k]):
                                    V2 = insert_validity_test2(G, i, j,
                                                               sub[k])  # Insert operator validation test:condition 2
                                else:
                                    V2 = 1
                                if (V2):
                                    Idx = find_subset_include(sub[k], sub)  # find those subsets that include sub(k)
                                    S[np.where(Idx == 1)] = 1
                                    chscore, desc, record_local_score = insert_changed_score(X, G, i, j, sub[k],
                                                                                             record_local_score,
                                                                                             score_func,
                                                                                             parameters)
                                    if isinstance(chscore, torch.Tensor):
                                        chscore = chscore.item()
                                    # print("add--> i: ", i, ", j: ", j, ", min_chscore: ",min_chscore,
                                    #      ", chscore: ", chscore, ", sub[k]: ", sub[k])
                                    # calculate the changed score after Insert operator
                                    # desc{count} saves the corresponding (i,j,sub{k})
                                    # sub{k}:
                                    if (chscore < min_chscore):
                                        min_chscore = chscore
                                        min_desc = desc
                            else:
                                Idx = find_subset_include(sub[k], sub)  # find those subsets that include sub(k)
                                S[np.where(Idx == 1)] = 2

        if (len(min_desc) != 0):
            score_new = score + min_chscore
            if (score - score_new <= threshold):
                break
            # print("score_new: ", score_new, score)
            G = insert(G, min_desc[0], min_desc[1], min_desc[2])
            update1.append([min_desc[0], min_desc[1], min_desc[2]])
            # print(G.graph)
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step1.append(G)
        else:
            score_new = score
            break

    ## --------------------------------------------------------------------
    # backward greedy search
    # print('backward')
    count2 = 0
    score_new = score
    update2 = []
    G_step2 = []
    score_record2 = []
    graph_record2 = []
    while True:
        count2 = count2 + 1
        score = score_new
        score_record2.append(score)
        graph_record2.append(G)
        min_chscore = 1e7
        min_desc = []
        for i in range(N):
            for j in range(N):
                if ((G.graph[j, i] == Endpoint.TAIL.value and G.graph[i, j] == Endpoint.TAIL.value)
                        or G.graph[j, i] == Endpoint.ARROW.value):  # if Xi - Xj or Xi -> Xj
                    Hj = np.intersect1d(np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                                        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0])  # neighbors of Xj
                    Hi = np.union1d(np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
                                    np.where(G.graph[:, i] != Endpoint.NULL.value)[0])  # adjacent to Xi
                    H0 = np.intersect1d(Hj, Hi)  # find the neighbours of Xj that are adjacent to Xi
                    # for any subset of H0
                    sub = Combinatorial(H0.tolist())  # find all the subsets for H0
                    S = np.ones(len(sub))  # S indicate whether we need to check sub{k}.
                    # 1: check the condition,
                    # 2: check nothing and is valid;
                    for k in range(len(sub)):
                        if (S[k] == 1):
                            V = delete_validity_test(G, i, j, sub[k])  # Delete operator validation test
                            if (V):
                                # find those subsets that include sub(k)
                                Idx = find_subset_include(sub[k], sub)
                                S[np.where(Idx == 1)] = 2  # and set their S to 2
                        else:
                            V = 1

                        if (V):
                            chscore, desc, record_local_score = delete_changed_score(X, G, i, j, sub[k],
                                                                                     record_local_score, score_func,
                                                                                     parameters)
                            if isinstance(chscore, torch.Tensor):
                                chscore = chscore.item()
                            # print("delete--> i: ", i, ", j: ", j, ", min_chscore: ", min_chscore,
                            #      ", chscore: ", chscore, ", sub[k]: ", sub[k])
                            # calculate the changed score after Insert operator
                            # desc{count} saves the corresponding (i,j,sub{k})
                            if (chscore < min_chscore):
                                min_chscore = chscore
                                min_desc = desc

        if len(min_desc) != 0:
            score_new = score + min_chscore
            if score - score_new <= threshold:
                break
            # print("score_new: ", score_new, score)
            G = delete(G, min_desc[0], min_desc[1], min_desc[2])
            update2.append([min_desc[0], min_desc[1], min_desc[2]])
            # print(G.graph)
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step2.append(G)
        else:
            score_new = score
            break

    Record = {'update1': update1, 'update2': update2, 'G_step1': G_step1, 'G_step2': G_step2, 'G': G, 'score': score}
    return Record
