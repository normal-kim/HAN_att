import pandas as pd
import numpy as np
from itertools import chain

import nltk
import kss
from nltk.tokenize import sent_tokenize
from konlpy.tag import Okt
from gensim.models import Word2Vec


okt = Okt()

def flatten(data):
    return list(chain(*data))


def process_NVA(corpus_sent_tok):
    corp = []
    for ind in range(len(corpus_sent_tok)):
        doc = []
        for s_ind in range(len(corpus_sent_tok[ind])):
            tmp_lst = []
            for tok in okt.pos( corpus_sent_tok[ind][s_ind]):
                if tok[1] in ['Noun', 'Verb', 'Adjective']:
                    tmp_lst.append(tok[0])
            if len(tmp_lst) > 1:
                doc.append(tmp_lst)
        corp.append(doc)        
    return corp

def do_preprocessing (res, sentiment_data, w2v_model):
    def do_w2v(data, model):
        tmp = []
        for elem in data:
            try: 
                tmp.append(model[elem])
            except Exception as ex:
                tmp.append(np.zeros(100))
        return np.array(tmp)

    corpus = sentiment_data['review'].apply(lambda x: kss.split_sentences(x)).tolist()
    
    df_in = pd.DataFrame( [res] ).T
    df_in.columns = {'tokenized'}
    df_in['tokenized_flat'] = df_in['tokenized'].map(lambda x: flatten(x))
    df_in['sent_org'] = corpus
    df_in['num_sent'] = df_in['tokenized'].apply(len)
    df_in['num_words'] = df_in['tokenized'].map(lambda x: [len(i) for i in x])
    df_in['w2v'] = df_in['tokenized'].map(lambda x : [do_w2v(i, w2v_model) for i in x])
    df_in['seq_len'] = df_in['tokenized_flat'].map(lambda x: len(x))
    df_in['aspect_cat'] = sentiment_data['aspect category']
    df_in['polarity'] = sentiment_data['polarity']
    return df_in


def train_test_split( df_in, kw = 'polarity', random_st = 21):    
    train_min_cls_size = int(df_in[kw].value_counts().min() * 0.9)
    
    df_train = df_in.groupby(kw).apply(lambda x: x.sample(train_min_cls_size, random_state = random_st))
    sampled_indices = [ ind[1] for ind in list(df_train.index) ]
    df_train = df_train.sample(frac = 1).reset_index(drop = True)
    print('train_size:\n', df_train[kw].value_counts())
    
    msk = df_in.index.isin(sampled_indices)
    df_test = df_in[~msk]
    test_cls_size = df_test[kw].value_counts().min()
    df_test = df_test.groupby(kw).apply(lambda x: x.sample(test_cls_size, random_state = random_st))
    df_test = df_test.sample(frac = 1).reset_index(drop = True)
    print('test_size:\n', df_test[kw].value_counts())
    return df_train, df_test
