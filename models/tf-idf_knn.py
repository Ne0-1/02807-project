
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
from tqdm import tqdm
import mmh3
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats

seed = 42

#TDIDF
def identity_tokenizer(text):
    return text



# data
# clean_completeSpamAssassin
# clean_spam
data = pd.read_csv(f'../data/clean_completeSpamAssassin.csv', encoding='latin')
data.tokens = data.tokens.apply(literal_eval)


df_text_tokenized = {i: [data['text'][i].split(' ')] for i in list(data.index)}
data['text_tokenized'] = pd.DataFrame.from_dict(df_text_tokenized, orient='index')


# raw/BOW, raw/Q, prep/BOW, prep/Q:
look_ups = ['text_tokenized', 'text', 'tokens', 'str_tokens']

for look_up in look_ups:
    labels = ['ham', 'spam']
    q = 5
    num_neighs = 5 # 2
    num_folds = 5

    kf = RepeatedKFold(n_splits=num_folds, n_repeats=1, random_state=seed)
    f1_scores = []
    accuracies = []

    for train_idx, test_idx in tqdm(kf.split(data)):
        # sorting data
        df_train = data.loc[train_idx]
        df_test = data.loc[test_idx]
        # generating vectors
        if look_up in ['text', 'str_tokens']:
            # shingles
            corpus = list(data[look_up])
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(q, q), min_df=0.001).fit(corpus)
            #vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(q, q), max_df=0.7).fit(corpus)

            X = vectorizer.fit_transform(df_train[look_up])

        else:
            vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, min_df=0.001)
            #vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, max_df=0.7)
            vecs = vectorizer.fit_transform(df_train[look_up])
            feature_names = vectorizer.get_feature_names_out()
            dense = vecs.todense()
            lst1 = dense.tolist()
            TDM = pd.DataFrame(lst1, columns=feature_names).dropna()
            X = vecs
        # training classifier
        neigh = KNeighborsClassifier(n_neighbors=num_neighs, metric='cosine')
        neigh.fit(X, df_train.label)
        # predicting
        test_vecs = vectorizer.transform(df_test[look_up])
        pred_prob = neigh.predict_proba(test_vecs)
        y_pred = [labels[pred_idx] for pred_idx in pred_prob.argmax(axis=1)]
        y_test = df_test.label
        # generating predictions
        f1 = f1_score(y_test, y_pred, pos_label='spam')
        acc = accuracy_score(y_test, y_pred)
        f1_scores.append(f1_score(y_test, y_pred, pos_label='spam'))
        accuracies.append(accuracy_score(y_test, y_pred))
    print(look_up)
    print(f'f1_mean: {np.mean(f1_scores)}  |  f1: {f1_scores}')
    print(f'acc_mean: {np.mean(accuracies)}  |  acc: {accuracies}')
    print('\n')














