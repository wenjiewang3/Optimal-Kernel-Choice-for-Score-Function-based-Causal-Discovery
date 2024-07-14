import argparse
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from Data.SyntheticData import graph_generate
from Data.utils import Continuous2Discrete, Part2Discrete
from GES.GES import ges
from Utils.Evaluate import F1_score_nparray, SHD_nparray


def config():
    parser = argparse.ArgumentParser(description="parameter setting")
    # dataset config
    parser.add_argument("--data_type", type=str, default="multi",
                        help="dataset name, 'con' stands for continues, 'dis' for discrete, 'multi' for multi-dimensional data"
                             "'mix' for continues and discrete mixed data, 'multi' for multi-dimensional data")
    parser.add_argument("--n", type=int, default=30, help="sample size")
    parser.add_argument("--gd", type=float, default=0.4, help="graph density")

    # model config
    parser.add_argument("--score", type=str, default='GP',
                        help="score function option, 'GP', 'Marg' and 'MI' ")
    parser.add_argument("--epochs", type=int, default=200, help="training epochs if using adam optimizer")
    parser.add_argument("--optim", type=str, default="adam", help="optimizer: 'lg' for LBFGSB and 'adam'")
    parser.add_argument("--threshold", type=float, default=0.001, help= "to prevent numerical errors under the square root in the Jacobian determinant" )

    # training setting
    parser.add_argument("--device", type=str, default="cpu", help='training on "cpu" or "cuda" ')
    parser.add_argument("--seed", type=int, default=np.random.randint(1, 1e5), help="random seed")


    opt = parser.parse_args()
    return opt

def main(opt):

    assert opt.data_type in ['con', 'dis', 'mix', 'multi'], print("data type not implemented")
    Data_dir = graph_generate(data_nums=opt.n,  variable_nums=6, graph_density=opt.gd,
                              seeds=opt.seed, max_dim=1 if opt.data_type == "multi" else 1)
    if opt.data_type == 'dis':
        Data_dir['data_mat'] = Continuous2Discrete(Data_dir)
    if opt.data_type == 'mix':
        Data_dir['data_mat'] = Part2Discrete(Data_dir)
    print("dataset setting -> data type: ", opt.data_type, ", sample size: ", opt.n, ", graph density : ", opt.gd)
    print("model   setting -> score: ", opt.score, " , device: ", opt.device, ", optim: ", opt.optim, ", seed: ", opt.seed)

    parameters = {'epochs': 200, 'device': opt.device, 'optim': opt.optim, 'threshold': opt.threshold}

    Gt = Data_dir['G']
    print("truth graph")
    print(Gt)
    assert opt.score in ['CV', 'Marg', 'MI', 'GP']
    if opt.score == 'Marg':
        Record = ges(Data_dir, 'local_score_Marg',  parameters=parameters)
    elif opt.score == "GP":
        Record = ges(Data_dir, 'local_score_GP',  parameters=parameters)
    elif opt.score == 'MI':
        Record = ges(Data_dir, 'local_score_MI',  parameters=parameters)
    else:
        raise NotImplementedError("score not implemented")

    # Visualization
    # pyd = GraphUtils.to_pydot(Record['G'])
    # tmp_png = pyd.create_png(f="png")
    # fp = io.BytesIO(tmp_png)
    # img = mpimg.imread(fp, format='png')
    # plt.axis('off')
    # plt.imshow(img)
    # plt.show()

    print("cat est graph")
    print(Record['G'].graph)
    F1_score = F1_score_nparray(Gt, Record['G'].graph)
    SHD = SHD_nparray(Gt, Record['G'].graph)
    print("F1_score: ", F1_score, "SHD: ", SHD)
    return F1_score, SHD, Record['G'].graph

if __name__ == '__main__':
    opt = config()
    main(opt)