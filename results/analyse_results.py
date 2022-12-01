import pandas as pd
import numpy as np
import scipy.stats


def mean_confidence_interval(data, round=3, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return np.round(m, round), np.round(m-h, round), np.round(m+h, round)



round = 3

data = pd.read_csv(f'results.csv')#, encoding='latin')


models = list(set(data.Model))
datasets = list(set(data.Dataset))
# force order
methods = ['Raw/Q', 'Prep/Q', 'Raw/BOW', 'Prep/BOW']

for model in models:
    print(model)
    for dataset in datasets:
        print(f'  {dataset}')
        for method in methods:
            current_data = data[(data.Model==model) & (data.Dataset==dataset) & (data.Method==method)]
            f1, f1_l, f1_h = mean_confidence_interval(list(current_data.F1), round=round)
            acc, acc_l, acc_h = mean_confidence_interval(list(current_data.Accuracy), round=round)
            # todo: round
            print(f'    {method}: f1: {f1}, [{f1_l},{f1_h}] | acc: {acc}, [{acc_l},{acc_h}]')
    print('\n')


