#Import libraries
import sys
import os
import pandas as pd
import mmh3
import numpy as np

import math
from sklearn.feature_extraction.text import CountVectorizer
from mlxtend.frequent_patterns import apriori
from sklearn.model_selection import RepeatedKFold, KFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score
import time
from mlxtend.preprocessing import TransactionEncoder

from ast import literal_eval
from tqdm.notebook import tqdm
import mmh3
from itertools import product
import itertools



# create shingle function
def shingles(string: str, q: int):
    output = set()
    for i in range(len(string) + 1):
        if i < q:
            pass
        else:
            output.add(' '.join(string[i - q:i]))
    return output


# create jaccard sim function
def jaccard(doc1, doc2):
    doc1 = set(doc1)
    doc2 = set(doc2)
    intersect = doc1.intersection(doc2)
    union = doc1.union(doc2)
    if len(union) != 0:
        return len(intersect) / len(union)
    else:
        return 0


# create document sim function
def similarity(docs: dict):
    output = np.zeros((len(docs.keys()), len(docs.keys())))
    for key1, value1 in docs.items():
        for key2, value2 in docs.items():
            if key1 <= key2:
                pass
            else:
                jac_value = jaccard(value1, value2)
                output[key1, key2] = jac_value

    return np.tril(output) + np.triu(output.T, 1)


def weighted_knn(x, y_train,test_idx,low,high,k_neighbours=5):
    y_test = []
    mask = np.ones(len(x),bool)
    mask[y_test] = False
    for i in test_idx:
        ind = []
        temp = np.argpartition(x[i], -k_neighbours)[-k_neighbours:]
        temp = np.flip(temp)
        for idx in temp:
            if idx>=high or idx < low:
                ind.append(idx)
        topk = x[i][ind]
        labels = {j: y_train[j] for j in ind}
        ham = 0
        spam = 0
        for key, value in labels.items():
            if value == 'ham':
                ham += x[i][key]
            if value == 'spam':
                spam += x[i][key]
        if ham>spam:
            y_test.append('ham')
        if ham == spam:
            y_test.append('ham')
        if ham<spam:
            y_test.append('spam')
    return y_test


def get_freq_itemset(data, labels, look_up, num_itemsets, max_len):
    freq_itemsets_dict = {}
    for idx, label in enumerate(labels):
        support=0.1
        itemsets=[]
        while len(itemsets) <= num_itemsets[idx]:
            filtered_data = data[data.label == label]
            look_up_data = list(filtered_data[look_up].values)
            te = TransactionEncoder()
            te_ary = te.fit(look_up_data).transform(look_up_data)
            df = pd.DataFrame(te_ary, columns=te.columns_)
            itemsets = apriori(df, min_support=support, use_colnames=True, max_len=max_len)
            support -= support/20
        # finding top supported
        formatting = itemsets.sort_values(by='support', ascending=False).iloc[:num_itemsets[idx]].reset_index(drop=True)
        # todo: normalizing
        #formatting.support = formatting.support/formatting.support.sum()
        #formatting.support = (formatting.support - formatting.support.min()) / (formatting.support.max() - formatting.support.min())
        freq_itemsets_dict[label] = formatting

    return freq_itemsets_dict


def filter_apriori(data, labels, look_up, num_itemsets, max_len):

    freq_itemsets = get_freq_itemset(data, labels, look_up, num_itemsets, max_len=max_len)

    # finde hvor mange unikke
    itemsets = set(list(freq_itemsets[labels[0]].itemsets) + list(freq_itemsets[labels[1]].itemsets))

    total_tokens = []
    for itemset in list(itemsets):
        token = ' '.join(list(itemset))
        total_tokens.append(token)

    for idx, row in data.iterrows():
        row_tokens = []
        for token in row.tokens:
            if token in total_tokens:
                row_tokens.append(token)
        data.iloc[idx].tokens = row_tokens

    return data


from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from ast import literal_eval

emails = pd.read_csv(f'../data/clean_spam.csv', encoding='latin')
emails.tokens = emails.tokens.apply(literal_eval)










labels = ['ham', 'spam']
look_up = 'tokens'
num_itemsets = [150, 150]
max_len = 1
emails = filter_apriori(emails, labels, look_up, num_itemsets, max_len)


start_time = time.time()

#get bag of words as sets for each email
email_bow = {emails.index[i]: set(emails['tokens'][i]) for i in range(len(emails))}

#get similarity between documents
email_bow_sim = similarity(email_bow)

from sklearn.metrics import f1_score
kfold=5
f1_scores = []
test_set_percent = ((len(emails)/kfold)/len(emails))
test_size = round(test_set_percent*len(emails))
former_test_idx = 0
y_tests = []
y_pred2 = []
for i in tqdm(range(kfold)):
    y_test = emails['label'][former_test_idx:(i+1)*test_size]
    y_tests = y_tests + list(y_test)
    y_idx = y_test.index
    mask = np.ones(len(emails), bool)
    mask[y_idx] = False
    y_train = emails['label'][mask]
    y_pred = weighted_knn(email_bow_sim,y_train,y_idx,former_test_idx,(i+1)*test_size,5)
    y_pred2 = y_pred2 + list(y_pred)
    f1_scores.append(f1_score(y_test,y_pred, pos_label='spam'))
    former_test_idx += test_size
print(f1_scores)


print("--- %s seconds ---" % (time.time() - start_time))




