from scipy.stats import spearmanr
import numpy as np
import pandas as pd

#load the csvs
human_eval = pd.read_csv('human_eval_bin.csv', index_col=0)
auto_eval_full = pd.read_csv('auto_eval_toy.csv')
auto_eval = auto_eval_full['fkre.GEN']

#transpose the csv
human_eval_transpose = human_eval.transpose()
human_eval_transpose.index = human_eval_transpose.index.astype(int)
auto_eval = auto_eval.loc[human_eval_transpose.index]

#get rid of the zero columns (representing empty data)
nonzero_idxs = (human_eval_transpose != 0).any(axis=1)
nonzero_human_eval = human_eval_transpose[nonzero_idxs]
nonzero_auto_eval = auto_eval[nonzero_idxs]

#take the average across all human readability metrics
averaged_human_eval = nonzero_human_eval.mean(axis=1)

#get the spearman correlation
correlation, pval = spearmanr(averaged_human_eval, nonzero_auto_eval)
print(f"Correlation: {correlation:.5f}, p-value: {pval:.5f}")