from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
import torch
import numpy as np

class HAN_dataset(Dataset):
    def __init__(self, review_df):
        self.review = review_df.reset_index(drop = True)
        
    def __len__(self):
        return len(self.review)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        label = self.review.loc[idx, 'polarity']
        w2v = self.review.loc[idx, 'w2v']
        num_words = self.review.loc[idx, 'num_words']
        num_sent = self.review.loc[idx, 'num_sent']
        
        sample = {'w2v': w2v, 
                 'label': label, 
                 'num_sent' : num_sent, 
                 'num_words': num_words}
        return sample


def han_collate_fn(samples):
    labels = [sample['label'] for sample in samples ]
    w2v = [sample['w2v'] for sample in samples]
    doc_lengths = [sample['num_sent'] for sample in samples]
    sent_lengths = [sample['num_words'] for sample in samples]
    
    bsz = len(labels)
    batch_max_doc_length = max(doc_lengths)
    batch_max_sent_length = max( [max(sl) if sl else 0 for sl in sent_lengths])
    docs_tensor = torch.zeros((bsz, batch_max_doc_length, batch_max_sent_length, 100), dtype = torch.float)
    sent_lengths_tensor = torch.zeros((bsz, batch_max_doc_length)).long()
    
    for doc_idx, doc in enumerate(w2v):
        doc_length = doc_lengths[doc_idx]
        sent_lengths_tensor[doc_idx, :doc_length] = torch.Tensor(sent_lengths[doc_idx])
        for sent_idx, sent in enumerate(doc):
            sent_length = sent_lengths[doc_idx][sent_idx]
            docs_tensor[doc_idx, sent_idx, :sent_length, :] = torch.FloatTensor(sent)
            
    return ( docs_tensor, torch.Tensor(labels), torch.Tensor(doc_lengths), sent_lengths_tensor)


class HanDataLoader(DataLoader):
    def __init__(self, dataset, params_dict, shuffle = True):
        self.n_samples = len(dataset)
        self.init_kwargs = {
            'dataset': dataset, 
            'batch_size' : params_dict['batch_size'],
            'collate_fn' : han_collate_fn, 
            'shuffle': shuffle
        }
        super().__init__(**self.init_kwargs)


def get_x_s(sample):
    return sample[0].to('cuda:0'), sample[2].to('cuda:0'), sample[3].to('cuda:0')


def get_trainloader(df_train, params_dict):
    msk = np.random.rand(len(df_train)) < 0.85
    train = df_train[msk]
    valid = df_train[~msk]
    return (HanDataLoader(HAN_dataset(train), params_dict), 
            HanDataLoader(HAN_dataset(valid), params_dict) )