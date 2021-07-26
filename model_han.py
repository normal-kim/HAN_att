import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


# context vector bias = True 로 해보기

class ReviewHAN(nn.Module):
    def __init__ (self, params_dict):
        super().__init__()
        self.params_dict = params_dict
        self.embed_dim = params_dict['embed_dim']
        self.word_gru_h_dim = params_dict['word_gru_h_dim']
        self.sent_gru_h_dim = params_dict['sent_gru_h_dim']
        self.word_gru_n_layers = params_dict['word_gru_n_layers']
        self.sent_gru_n_layers = params_dict['sent_gru_n_layers']
        self.word_att_dim = params_dict['word_att_dim']
        self.sent_att_dim = params_dict['sent_att_dim']
        self.dropgru_s = params_dict['dropgru_s']
        self.dropgru_w = params_dict['dropgru_w']
        self.Dropout = nn.Dropout(params_dict['dropval'])

        # sentence
        self.sent_gru = nn.GRU( 2 * self.word_gru_h_dim, self.sent_gru_h_dim, 
                                num_layers = self.sent_gru_n_layers, batch_first = True,
                                bidirectional = True, dropout = self.dropgru_s)
        self.sent_layer_norm = nn.LayerNorm( 2 * sent_gru_h_dim, elementwise_affine= True)
        self.sent_attention = nn.Linear(2 * self.sent_gru_h_dim, self.sent_att_dim)
        self.sentence_context_vector = nn.Linear(self.sent_att_dim, 1, bias = False)

        # word
        self.word_gru = nn.GRU(self.embed_dim, self.word_gru_h_dim, 
                                num_layers = self.word_gru_n_layers, 
                              batch_first = True, bidirectional = True, dropout = self.dropgru_w)
        self.word_layer_norm = nn.LayerNorm( 2* self.word_gru_h_dim, elementwise_affine=True)
        self.word_attention = nn.Linear( 2 * self.word_gru_h_dim, self.word_att_dim)
        self.word_context_vector = nn.Linear(self.word_att_dim, 1, bias = False)

        # attention methods
        self.attention_dict = {'softmax' : self.softmax_attention, 
                               'tanh' : self.tanh_attention, 
                               'de_attention' : self.de_attention}

    def forward (self, docs, doc_lengths, sent_lengths):
        # 1. Packing
        ## 1-1 reorder
        doc_lengths, doc_perm_idx = doc_lengths.sort(dim = 0, descending = True)
        docs = docs[doc_perm_idx]
        sent_lengths = sent_lengths[doc_perm_idx]
        ## 1-2 packing
        packed_sents = pack_padded_sequence(docs, lengths=doc_lengths.tolist(), batch_first = True)
        packed_sent_lengths = pack_padded_sequence( sent_lengths, lengths= doc_lengths.tolist(), 
                                                  batch_first=True)
        valid_bsz_sent = packed_sents.batch_sizes
        
        # 2. Word Attention
        ## 2-1. packing input data
        sents, sent_lengths = packed_sents.data, packed_sent_lengths.data
        # reorder
        sent_lengths, sent_perm_idx = sent_lengths.sort(dim = 0, descending = True)
        sents = sents[sent_perm_idx]
        
        # embedding done already, do dropout
        #sents = self.Dropout(sents)
        packed_words = pack_padded_sequence( sents, lengths = sent_lengths.tolist(), batch_first=True)
        valid_bsz_word = packed_words.batch_sizes
        
        ##2-2 NN: 
        # hidden layer
        h_it, _ = self.word_gru( packed_words )
        h_it_normed = self.word_layer_norm(h_it.data)
        h_it_pad, _ = pad_packed_sequence ( h_it, batch_first = True )
        
        # context mapping
        u_it = torch.tanh( self.word_attention( h_it_normed.data ))

        # calculate similarity
        u_it_cv = self.word_context_vector( u_it).squeeze(1)
        
        # attention weights
        a_it = self.attention_dict[self.params_dict['attention'] ](u_it_cv, 
                                                  valid_bsz_word, u_it, sent_lengths, True)
        # output
        s_i = (h_it_pad * a_it.unsqueeze(2)).sum(dim = 1)
        
        ## 2-3 reorder
        _, sent_unperm_idx = sent_perm_idx.sort(dim = 0, descending = False)
        s_i = s_i[sent_unperm_idx] 
        a_it = a_it[sent_unperm_idx] 
        

        # 3. Sentence Attention
        #sents = self.Dropout(sents)
        # 3-1 NN
        # hidden layer
        h_i, _ = self.sent_gru(PackedSequence(s_i, valid_bsz_sent))
        h_i_normed = self.sent_layer_norm( h_i.data )
        h_i_pad, _ = pad_packed_sequence( h_i, batch_first = True )
        
        # context mapping
        u_i = torch.tanh( self.sent_attention( h_i_normed.data ))
        
        # calculate similarity
        u_i_cv = self.sentence_context_vector(u_i).squeeze(1)

        # attention weights
        a_i = self.attention_dict[ self.params_dict['attention'] ](u_i_cv, 
                                                    valid_bsz_sent, u_i, doc_lengths, False)
        
        # document vector
        v = ( h_i_pad * a_i.unsqueeze(2)).sum(dim = 1)
        
        # 3-2 reorder
        a_it, _ = pad_packed_sequence( PackedSequence( a_it, valid_bsz_sent), 
                                                 batch_first = True)
        _, doc_unperm_idx = doc_perm_idx.sort(dim = 0, descending = False)
        
        # 4. Final Output
        v = v[doc_unperm_idx] 
        a_it = a_it[ doc_unperm_idx ] 
        a_i = a_i [ doc_unperm_idx ]
        
        return v, a_it, a_i 

    def softmax_attention (self, u_cv, valid_bs, *args):
        # SOFTMAX TYPE ATTENTION      
        a_exp = torch.exp( u_cv - u_cv.max() )
        a_exp_pad, _ = pad_packed_sequence( PackedSequence(a_exp, valid_bs), batch_first = True)
        return a_exp_pad / torch.sum ( a_exp_pad, dim = 1, keepdim = True)

    def tanh_attention (self, u_cv, valid_bs, *args):
        alpha_ = self.params_dict['tan_a']
        a_tanh = torch.tanh( alpha_ * u_cv )
        a_tanh_pad, _ = pad_packed_sequence( PackedSequence( a_tanh, valid_bs), 
                                      batch_first= True)
        return a_tanh_pad

    def de_attention (self, u_cv, valid_bs, u_it, lengths, word_flag = True):
        if word_flag == True:
          query = self.word_context_vector.weight
        else: 
          query = self.sentence_context_vector.weight

        key = u_it
        alpha = self.params_dict['alpha_de']
        beta = self.params_dict['beta_de']

        # N Vector
        N_vec = -beta * abs(query - key)
        N_vec = N_vec.sum(dim = 1, keepdim = True)
        N_vec_pad, _ = pad_packed_sequence( PackedSequence(N_vec, valid_bs), batch_first = True)
        value_mask = (1 * (N_vec_pad != 0))
        N_mean = torch.sum(N_vec_pad, 1) / lengths.unsqueeze(1)
        N_meaned = N_vec_pad - value_mask*N_mean.unsqueeze(2)

        # E Vector
        E_vec = alpha * torch.matmul( key, query.T )
        E_vec_pad, _ = pad_packed_sequence( PackedSequence(E_vec, valid_bs), batch_first = True)
        E_mean = torch.sum(E_vec_pad, 1) / lengths.unsqueeze(1)
        E_meaned = E_vec_pad - value_mask*E_mean.unsqueeze(2)

        A = torch.tanh( E_meaned ) * torch.sigmoid( N_meaned)

        return A.squeeze(2)