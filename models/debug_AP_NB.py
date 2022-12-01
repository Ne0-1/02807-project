
# importing modules

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
def shingles(term_list:list,q:int):
    string = ' '.join(term_list)
    output = set()
    for i in range(len(string)+1):
        if i < q:
            pass
        else:
            output.add(''.join(string[i-q:i]))
    return list(output)


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

def count_occurences(word_set, x):
    if len(word_set) == 1:
        return x.count(list(word_set)[0])
    else:
        diff_counts = []
        for string in list(word_set):
            diff_counts.append(x.count(string))
        # returning minimum value
        return min(diff_counts)


# Complete algorithm

class naiveBayes():

    def __init__(self, freq_itemsets_dict, priors, N_data):
        self.priors = np.log(list(priors.values()))
        self.labels = list(freq_itemsets_dict.keys())
        self.loglikelihood = {}
        combined = list(freq_itemsets_dict[self.labels[0]].itemsets) + list(freq_itemsets_dict[self.labels[1]].itemsets)
        for label_idx,label in enumerate(self.labels):
            #denominator = freq_itemsets_dict[label].support.sum()
            N_itemsets = len(freq_itemsets_dict[label])
            N_labels = N_data[label_idx]
            for itemset in combined:
                if itemset in list(freq_itemsets_dict[label].itemsets):
                    supp = float(freq_itemsets_dict[label].loc[freq_itemsets_dict[label].itemsets == itemset].support)
                    self.loglikelihood[(' '.join(list(itemset)), label)] =np.log((supp * N_labels + 1) / (N_labels + N_itemsets))  # todo laplace smoothing
                else:
                    self.loglikelihood[(' '.join(list(itemset)), label)] = np.log(1/N_itemsets)

            #for idx, row in freq_itemsets_dict[label].iterrows():
            #    supp = (row.support*N_labels+1)/(N_labels+N_itemsets) # todo laplace smoothing
            #    #self.loglikelihood[(' '.join(list(row.itemsets)), label)] = np.log(row.support) #np.log(row.support/denominator)
            #    self.loglikelihood[(' '.join(list(row.itemsets)), label)] = supp

    def predict(self, x):
        scores = self.priors
        for idx, label in enumerate(self.labels):
            used_tokens = []
            for token in x:
                try:
                    scores[idx] += self.loglikelihood[(token, label)]
                    used_tokens.append(token)
                except Exception as e:
                    pass
            # create two fold combinations
            # loop

        return self.labels[np.argmax(scores)] # return label name


def apriori_NB(train, test, method, hparams, max_len):

    if method == 'terms':
        look_up = 'tokens'
    elif method == 'shingles':
        look_up = 'shingles'
    else:
        raise NotImplementedError


    num_itemsets = list(hparams)

    # hyperparameters
    labels = list(np.unique(train.label))  # currently repeated

    # getting frequent itemsets
    freq_itemsets_dict = get_freq_itemset(train, labels, look_up, num_itemsets, max_len)
    # calculating prior for NB
    priors = {}
    N = []
    for label in labels:
        # prior is based on the balance of labels in dataset
        N.append(sum(train.label == label))
        prior = sum(train.label == label) / len(train)
        priors[label] = prior

    # word_freq_dict =  apriori_for_binary(train, labels, minimum_support, look_up)
    model = naiveBayes(freq_itemsets_dict, priors, N)

    # classifying using the naive bayes
    y_test = []
    y_pred = []

    for idx, x in test.iterrows():
        # by x.tokens we remove the 'label' data
        # classifying
        #pred = NB_classify(x[look_up], freq_itemsets_dict, labels, priors)
        pred = model.predict(x[look_up])
        y_pred.append(pred)
        y_test.append(x.label)

    f1 = f1_score(y_test, y_pred, pos_label='spam')
    acc = accuracy_score(y_test, y_pred)

    # TODO: f1_score
    print(f'Performance: f1={np.round(f1, 5)}  |  acc={np.round(acc, 5)}')
    # check size of sets based om support values

    print(f'   sets={num_itemsets}: [{len(freq_itemsets_dict[labels[0]])},{len(freq_itemsets_dict[labels[1]])}]\n')

    return (f1, acc)




# loading dataset
data = pd.read_csv(f'../data/clean_spam.csv')
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
out = apriori_NB(train=train, test=test, method=method, hparams=(500, 500), max_len=1)
print("--- %s seconds ---" % (time.time() - start_time))



itemsets_ham = list(np.linspace(100, 1000, 10).astype(int))

itemsets_spam = list(np.linspace(100, 1000, 10).astype(int))

#q_shingles = np.arange(2,9)

# list of all hparams
hparams = [itemsets_ham, itemsets_spam]#, list(q_shingles)]
# running combination of hparams
combs = list(product(*hparams))

#combs = [(100,100), (300,300), (600,600), (900,900)]



df_results = pd.DataFrame(columns=["hparam", "f1", "acc"])  # from low to high conf

for hparam in tqdm(combs):
    out = apriori_NB(train=train, test=test, method=method, hparams=hparam, max_len=1)
    f1, acc = out
    df_single_results = pd.DataFrame({'hparam': [hparam], 'f1': f1, 'acc': acc})
    df_results = df_results.append(df_single_results)

df_results.to_csv('results.csv')

