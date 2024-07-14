

# Optimal Kernel choice for score function-based causal discovery

This repository contains an implementation of the federated DAG structure learning methods described in ["Optimal Kernel choice for score function-based causal discovery (ICML 2024)"](https://icml.cc/virtual/2024/poster/34621).

Dependencies: `python3.9, pytorch 1.12, causal_learn 0.1.2.7, cdt 0.6.0`


## Run
First, run the requirements with `pip install -r requirements.txt`.

You can run `bash run.sh` to test the codes. Our code can run on a GPU if a GPU device is available with the option "--device cuda."

## Acknowledgments
- Our implementation is highly based on causal discovery python package named causal-learn [pip link](https://github.com/py-why/causal-learn) and [ducoment link](https://causal-learn.readthedocs.io/en/latest/)， which is primarily used to implement the GES algorithm in our code.

If you find it useful, please consider citing: 
```bibtex
@inproceedings{
wang2024optimal,
title={Optimal Kernel choice for score function-based causal discovery},
author={Wenjie Wang, Biwei Huang, Feng Liu, Xinge You, Tongliang Liu, Kun Zhang, Mingming Gong},
booktitle={International conference on machine learning},
year={2024},
organization={PMLR}
}
```

```bibtex
@inproceedings{huang2018generalized,
  title={Generalized score functions for causal discovery},
  author={Huang, Biwei and Zhang, Kun and Lin, Yizhu and Sch{\"o}lkopf, Bernhard and Glymour, Clark},
  booktitle={Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery \& data mining},
  pages={1551--1560},
  year={2018}
}
```