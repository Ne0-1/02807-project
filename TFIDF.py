
# With inspiration from: 
# https://github.com/ochaton/mrjob-tfidf/blob/master/run.sh

from mrjob.job import MRJob
import re
import numpy as np
import pandas as pd

WORD_RE = re.compile(r"[\w']+")

#!/usr/bin/env python3
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.compat import jobconf_from_env
from mrjob.protocol import RawValueProtocol

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from math import log

import sys
import os

import re

# Splits at words
WORD_RE = re.compile(r"[\w']+")

data = pd.read_csv('data/tokens.csv', usecols=[1], names=['i', 'tokens'])
unique_tokens = np.unique(np.concatenate([str(line).split() for line in data['tokens']]))
# Word 2 index dictionary
word2idxDict = dict(zip(unique_tokens, np.arange(len(unique_tokens))))

NUMBER_OF_DOCUMENTS = 5572
NUMBER_OF_UNIQUE_TOKENS = len(unique_tokens)
TFIDF = np.zeros((NUMBER_OF_DOCUMENTS, NUMBER_OF_UNIQUE_TOKENS))

class MRTFIDF(MRJob):
    
    # Mapper1: 
    # Assigns value 1 to word-document pairs. 
    def get_words_from_line(self, _, line):
        """
        Maps to: 
        
        Key: tuple: (word, document)
        Value: 1
        """
        # splitting docname from text
        line = line.split(',')
        docname, line = line[0], ' '.join(line[1:])
        
        # Loop through words in line
        for term in WORD_RE.findall(line):
            # Key-Value pair
            yield (term, docname), 1

    # Reducer1: 
    # Reduces values of identical keys (word-doc pais) by summing
    def term_frequency_per_doc(self, term_doc, occurences):
        """
        Reduces: values to sum of values
        """
        term, docname = term_doc[0], term_doc[1]
        # summing occurences of terms in each term-doc pair
        yield (term, docname), sum(occurences)

    # Mapper2: 
    # Maps all keys of documents to a list of terms and their frequencies
    def get_docs_tf(self, term_doc, freq):
        """
        Maps to: 
        
        Key: document
        Value: tuple: (term, frequency)
        """
        term, doc = term_doc[0], term_doc[1]
        yield doc, (term, freq)

    # Reducer2: 
    # Word-document pairs as keys, 
    # assigns values (frequency of word in doc, total word_count in doc)
    def number_of_terms_per_each_doc(self, doc, term_freqs):
        """
        Key: tuple: (term, doc)
        Value: tuple: (frequncy of term, total document wordcount)
        """        
        terms = []
        freqs = []
        terms_in_doc = 0
        for term_freq in term_freqs:
            term, freq = term_freq[0], term_freq[1]
            terms.append(term)
            freqs.append(freq)
            terms_in_doc += freq

        for i in range(len(terms)):
            yield (terms[i], doc), (freqs[i], terms_in_doc)

    # Mapper3: 
    # Maps all keys of words to values of tuple(doc, frequency, total number of words)
    def get_terms_per_corpus(self, term_doc, freq_docWords):
        """
        Maps to: 
        
        Key: term
        Value: tuple: (doc, frequency, total number of words in doc)
        """        
        term, doc = term_doc[0], term_doc[1]
        freq, terms_in_doc = freq_docWords[0], freq_docWords[1]
        yield term, (doc, freq, terms_in_doc)

    # Reducer3: 
    # Get all term-doc keys, use values (word_frequency, total_words in doc, number of docs containing term)
    def term_appearence_in_corpus(self, term, doc_freq_nwords):
        """
        Key: tuple: (term, doc)
        Value: tuple: (frequncy of term, total number of words in doc, total number of docs containing term)
        """
        docs_containing_term = 0
        docs = []
        freqs = []
        terms_in_docs = []
        
        # Creating lists term-doc pair
        for dfn in doc_freq_nwords:
            docs_containing_term += 1
            docs.append(dfn[0])
            freqs.append(dfn[1])
            terms_in_docs.append(dfn[2])

        for i in range(len(docs)):
            yield (term, docs[i]), (freqs[i], terms_in_docs[i], docs_containing_term)

    # Mapper4
    # Maps the calculated tfidf score based on 
    # frequncy of term, total number of words in doc, total number of docs containing term
    def calculate_tf_idf(self, term_doc, tf_n_df):
        """
        Key: tuple: (term, doc)
        Value: TFIDF-value
        """
        term, doc = term_doc[0], term_doc[1]
        freqs, terms_in_doc, docs_containing_term = tf_n_df[0], tf_n_df[1], tf_n_df[2]
        
        # Calculating TF and IDF
        TF = (freqs / terms_in_doc)
        IDF = log(NUMBER_OF_DOCUMENTS / docs_containing_term)
        
        # Calculating actual TFIDF. 
        tfidf = TF * IDF
        
        # Accessing word index
        wordIdx = word2idxDict[term]
        
        # Inputting TFIDF in matrix
        TFIDF[int(doc), wordIdx] = tfidf
        
        # yield (term, doc), tfidf
        

    def steps(self):
        return [
            MRStep(
                mapper=self.get_words_from_line,
                reducer=self.term_frequency_per_doc,
            ),
            MRStep(
                mapper=self.get_docs_tf,
                reducer=self.number_of_terms_per_each_doc,
            ),
            MRStep(
                mapper=self.get_terms_per_corpus,
                reducer=self.term_appearence_in_corpus,
            ),
            MRStep(
                mapper=self.calculate_tf_idf,
            ),
        ]

if __name__ == '__main__':
    MRTFIDF.run()
    TFIDF.tofile('data/TFIDF.dat')
