import numpy as np

def graph_generate(data_nums, variable_nums, graph_density, max_dim = 4, seeds = np.random.randint(0, 1000)):
    np.random.seed(seeds)
    Graph = np.zeros([variable_nums, variable_nums])
    edge_nums = int(0.5 * (variable_nums ** 2) - variable_nums)
    edge_list = []
    max_dim = min(max_dim, int(variable_nums/2))

    for i in np.arange(0, variable_nums):
        for j in np.arange(i+1, variable_nums):
            edge_list.append([i, j])
    edge_list = np.array(edge_list)
    n = edge_list.shape[0]
    selected_idx = np.random.choice(np.arange(n), size=int(graph_density*edge_nums), replace=False)
    edges_selected = edge_list[selected_idx]
    for [i, j] in edges_selected:
        Graph[i, j] = -1; Graph[j, i] = 1
    return generate_data_multi(Graph, data_nums, max_dim, seeds = seeds)

def generate_data_multi(G, data_nums, max_dim, seeds):
    np.random.seed(seeds)
    assert G.shape[0] == G.shape[1]
    variable_nums = G.shape[-1]
    func_dic = {1: 'np.sin', 2: 'np.cos', 3: 'np.tanh', 4: 'np.tanh'}
    Data = {}
    for i in range(0, variable_nums):
        Data[i] = np.array([])
    leaf_idx = []
    for i in range(0, variable_nums):
        if np.max(G[i]) < 1:
            dim_j = np.random.randint(1, max_dim+1)
            if(np.random.randint(0, 2) == 0):
                Data[i] = np.random.randn(data_nums, dim_j)
            else:
                Data[i] = np.random.random((data_nums, dim_j)) - 0.5
            leaf_idx.append(i)
    no_leaf_node_idx = np.setdiff1d(np.arange(0, variable_nums), np.array(leaf_idx))
    # print('leaf_nodes: ', leaf_idx)
    # print("no_leaf_node_idx: ", no_leaf_node_idx)

    for j in no_leaf_node_idx:
        PA = np.where(G[:, j] == -1)[0]
        nums_PA = PA.shape[0]
        dim_j = np.random.randint(1, max_dim+1)
        func_id_f = np.random.randint(0, 3)
        func_id_g = np.random.randint(0, 4)
        noise_id = np.random.randint(0, 2)

        for i in range(nums_PA):
            if(i == 0):
                PA_Data = Data[PA[i]]
            else:
                PA_Data = np.concatenate((PA_Data, Data[PA[i]]), axis=1)
        Data_changed = np.matmul(PA_Data,np.ones((PA_Data.shape[-1], 1)))
        if(func_id_f == 0):
            Data[j] = 1.7*Data_changed/(PA_Data.shape[-1]+1)
        elif(func_id_f == 1):
            power = np.random.randint(1, 3)
            Data[j] = np.power(Data_changed, power)
        else:
            idx = np.random.randint(1, 4)
            Data[j] = eval(func_dic[idx])(Data_changed)

        if(noise_id == 0):
            Data[j] += 0.25*np.random.randn(data_nums, 1)
        else:
            Data[j] += 0.25*(np.random.random((data_nums, 1)) - 0.5)

        if(func_id_g == 0):
            Data[j] = 3*Data_changed/(PA_Data.shape[-1]+1)
        elif(func_id_g == 1):
            Data[j] += np.exp(Data[j])
        elif(func_id_g == 2):
            power = np.random.randint(1, 3)
            Data[j] = np.power(Data[j], power)
        else:
            idx = np.random.randint(1, 4)
            Data[j] = eval(func_dic[idx])(Data_changed)

    for i in range(0, variable_nums):
        if i == 0:
            data_mat = Data[0]
        else:
            data_mat = np.concatenate((data_mat, Data[i]), axis=1)

    var_idx = []
    idx = 0
    for i in range(0, variable_nums):
        var_dim = Data[i].shape[-1]
        var_idx.append(list(range(idx, idx+var_dim)))
        idx += var_dim
        if i == 0:
            data_mat = Data[0]
        else:
            data_mat = np.concatenate((data_mat, Data[i]), axis=1)
    Data['G'] = G
    Data['data_mat'] = data_mat
    Data['var_idx'] = var_idx
    Data['threshold'] = 0.02
    return Data

