# run GES with Conditional Marginal likelihood as score function
python main.py --data_type con --score Marg --n 300 --gd 0.5 --device cpu

# run GES with Mutual Information-based score function
#python main.py --data_type con --score MI --n 300 --gd 0.5 --device cpu