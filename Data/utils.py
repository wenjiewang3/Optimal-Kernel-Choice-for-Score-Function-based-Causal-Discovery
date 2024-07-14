import numpy as np

def Continuous2Discrete(Data_dir=None, bins_nums=20):
    Data_dir['threshold'] = 0.01
    Data_dir['width_init'] = 0.01
    data = Data_dir['data_mat']
    for i in range(data.shape[-1]):
        data_c = data[:, i]
        max = np.max(data_c)
        min = np.min(data_c)
        bin_nums = bins_nums
        bins = np.linspace(min, max, num=bin_nums)
        data_d = np.digitize(data_c, bins=bins)
        data[:, i] = data_d
    return data

def Part2Discrete(Data_dir=None, bins_nums=20, ratio = 0.5):
    Data_dir['threshold'] = 0.01
    Data_dir['width_init'] = 0.01
    data = Data_dir['data_mat']
    n = data.shape[-1]
    DisIdx = np.random.choice(np.arange(n), size=int(ratio*n), replace=False)
    for i in DisIdx:
        data_c = data[:, i]
        max = np.max(data_c)
        min = np.min(data_c)
        bin_nums = bins_nums
        bins = np.linspace(min, max, num=bin_nums)
        data_d = np.digitize(data_c, bins=bins)
        data[:, i] = data_d
    return data
