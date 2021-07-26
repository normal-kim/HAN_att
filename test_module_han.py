import matplotlib
from IPython.display import HTML
from model_han import *
from data_loader_han import *
import torch
from konlpy.tag import Okt
import numpy as np 
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

okt = Okt()
def map_sentence_to_color(words, scores, sent_score):
    sentencemap = matplotlib.cm.get_cmap('RdBu')
    wordmap = matplotlib.cm.get_cmap('RdBu')
#     result = '<p><span style="margin:30px; padding:5px; background-color: {}">'\
#         .format(matplotlib.colors.rgb2hex(sentencemap(sent_score))[:3])
    result = '<p><span style="margin:1px; padding:2px; background-color: {}">'\
       .format(matplotlib.colors.rgb2hex(sentencemap(sent_score)[:3]))
    template = '<span class = "barcode"; style ="color: black; background-color: {}">{}</span>'
    for word, score in zip(words, scores):
        color = matplotlib.colors.rgb2hex(wordmap(score)[:3])
        result += template.format(color, '&nbsp' + word + '&nbsp')
    result += '</span><p>'
    return result

def get_test_sample(df_test, idx):
    orig_doc = df_test.loc[idx, 'tokenized']
    doc, num_sents, num_words = df_test.loc[idx, 'w2v'], df_test.loc[idx, 'num_sent'], df_test.loc[idx, 'num_words']
    ground_truth = df_test.loc[idx, 'polarity']
    samples = [{'w2v': doc, 
             'label': ground_truth, 
             'num_sent' : num_sents, 
             'num_words': num_words}]
    docs_tensor, labels, doc_lengths, sent_lengths = han_collate_fn(samples)
    
    x_s = (docs_tensor, doc_lengths, sent_lengths )
    return orig_doc, x_s, ground_truth

def attention_histogram(a_i, a_it):
    grp_no = a_i[0].shape[0]
    wd_att_np = a_it[0].data.numpy()
    np_shape = wd_att_np.shape[1]
    sent_indc = (np.array(list(map(int, ''.join([ str(i) * np_shape for i in range(grp_no) ])))) + 1).tolist()
    wd_indc = grp_no * list(range(1, np_shape + 1))
    df_att = pd.DataFrame(wd_att_np.flatten())
    df_att.columns = {'attention'}
    df_att['sent_indx'] = sent_indc
    df_att['wd_indx'] = wd_indc
    df_att = df_att [ abs(df_att['attention']) > 1e-20]
    #plot = sns.displot( data = df_att, x = "attention",  hue = 'sent_indx', rug = True)
    #plot.fig.suptitle('Word Attention Histogram for each Sentence')

    return df_att


def attention_word_positional(df_att):
    g = sns.barplot( data = df_att, x = "wd_indx", y = "attention",  hue = "sent_indx", 
           palette = "Accent", )
    #g.set_title(f'Attention Score of Words Positioned in a Sentence\n with Entropy Value')
    g.set_title(f'Attention Score of Words Positioned in a Sentence')

    entropy_dist = df_att.groupby('sent_indx').apply(lambda x: entropy(x['attention']) )

    #new_labels_ = [ str(1 +indx)+', entropy:'+str(round(i, 4)) for indx, i in enumerate(entropy_dist)]
    new_labels_ = [ 'sent' + str(1 +indx) for indx, i in enumerate(entropy_dist)]
    for t, l in zip(g.legend_.texts, new_labels_): t.set_text(l)
    return g


def attention_sentence(a_i):
    dist_a = a_i.data.numpy().flatten()
    df_sent = pd.DataFrame( dist_a )
    df_sent.columns = {'sentence attention'}
    df_sent['sent_indx'] = list(range(1, len(df_sent)+1))

    pl = sns.barplot( data = df_sent, x = "sent_indx", y = "sentence attention",  hue = "sent_indx", 
           palette = "OrRd", )
    #pl.set_title(f'Attention Score of Sentences\nEntropy:{entropy(dist_a):.4f}')
    pl.set_title(f'Attention Score of Sentences')
    return pl

def visualize_att(classifier, df_test, ind, random_st = 21 ):
    #han_encoder = classifier.encoder
    orig_doc, x_s, gt = get_test_sample(df_test, ind)
    sig_out, a_it, a_i = classifier(x_s)
    #docs, doc_lengths, sent_lengths = x_s
    print(f'Ground Truth: {gt}, \nPredicted: {sig_out.data.numpy().flatten()[0]:.3f}')
    # get attention values with the trained model
    #v, a_it, a_i = han_encoder(docs, doc_lengths, sent_lengths)
    
    # get raw data
    raw_sent = df_test.iloc[ind, :]['sent_org']
    raw_sent_tok = [ okt.pos(elem) for elem in raw_sent ]
    processed_raw = [ list(zip(*raw_sent_tok[i]))[0] for i in range(len(raw_sent_tok))]
    processed_raw = [ list(i) for i in processed_raw] 
    x = len(processed_raw)
    y = max( [len(i) for i in processed_raw] )
    update_att = np.zeros((x, y)).tolist()
    
    # fill in word attentions
    wd_atts = a_it.data.tolist()[0]
    diff = y - len(wd_atts[0])
    wd_atts = [ i + np.zeros(diff).tolist() for i in wd_atts ]
    
    # update attention
    for s, z in enumerate(zip(orig_doc, processed_raw)):
        cnt = 0
        for wd, val in enumerate(z[1]): 
            if val in z[0]:
                update_att[s][wd] = wd_atts[s][cnt]  
                cnt += 1
    
    # Vosia;oze
    words = processed_raw
    sent_score = a_i.tolist()[0]
    word_score = update_att# update_att.tolist()[0]
    result = "<h2>Attention Visualization</h2>"
    for sent, word_att, sent_att in zip(words, word_score, sent_score):
        result += map_sentence_to_color( sent, word_att, sent_att)


    df_att = attention_histogram(a_i, a_it)
    # plt.figure()
    att_word_pos = attention_word_positional(df_att)
    plt.figure()
    att_sent = attention_sentence(a_i)
    plt.figure()

    display(HTML(result))
    # display(att_hist.fig)
    # display(att_word_pos.figure)
    # display(att_sent.figure)

    with open(f'test_ind{ind}.html', 'w') as f:
        f.write(result)
    return result, a_it, a_i, gt


