

import pandas as pd
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

seed = 42





# create shingle function
def shingles(string :str ,q :int):
    output = set()
    for i in range(len(string ) +1):
        if i < q:
            pass
        else:
            output.add(''.join(string[ i -q:i]))
    return output


# create jaccard sim function
def jaccard(doc1, doc2):
    doc1 =set(doc1)
    doc2 =set(doc2)
    intersect = doc1.intersection(doc2)
    union = doc1.union(doc2)
    if len(union) != 0:
        return len(intersect) / len(union)
    else:
        return 0


# create document sim function
def similarity(docs :dict):
    output = np.zeros((len(docs.keys()), len(docs.keys())))
    # iterating each document
    for key1, value1 in docs.items():
        for key2, value2 in docs.items():
            # don't need any lower
            if key1 <= key2:
                pass
            else:
                jac_value = jaccard(value1, value2)
                output[key1, key2] = jac_value

    # creating symmetric matrix now
    return np.tril(output) + np.triu(output.T, 1)


def weighted_knn(x, y_train, test_idx, low, high, k_neighbours=5):
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

        # train.label
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



def knn(train, test, method, k_neighbours):

    if method == 'terms':
        look_up = 'tokens'
    elif method == 'shingles':
        look_up = 'shingles'
    else:
        raise NotImplementedError

    email_similarity = similarity(train[look_up])

    y_test = []
    y_pred = []

    for idx, x in test.iterrows():
        # by x.tokens we remove the 'label' data
        # classifying
        #pred = NB_classify(x[look_up], freq_itemsets_dict, labels, priors)
        pred = model.predict(x[look_up])
        y_pred.append(pred)
        y_test.append(x.label)

        y_pred = weighted_knn(email_similarity, train.labels, y_idx, former_test_idx, (i + 1) * test_size, k_neighbours=k_neighbours)


# loading dataset
data = pd.read_csv(f'../data/SMS.csv')
data.tokens = data.tokens.apply(literal_eval)

# splitting dataset
train, test = train_test_split(data, test_size=0.33, random_state=0)
train.reset_index(inplace=True)
test.reset_index(inplace=True)

q=5
method = 'terms'

df_shingles = {i: [shingles(train['tokens'][i], q=q)] for i in list(train.index)}
train['shingles'] = pd.DataFrame.from_dict(df_shingles, orient='index')
df_shingles = {i: [shingles(test['tokens'][i], q=q)] for i in list(test.index)}
test['shingles'] = pd.DataFrame.from_dict(df_shingles, orient='index')
#train.shingles = train.shingles.apply(literal_eval)

# testing low min support, more itemsets

start_time = time.time()
out = knn(train=train, test=test, method=method, k_neighbours=5)
print("--- %s seconds ---" % (time.time() - start_time))







