from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
import math
import numpy as np

class POSDataset(Dataset):
    def __init__(self, instanceDict, vocab, tag2id, id2tag, max_sent_len=60, reverse=False):
        self.root = instanceDict['tagged_sents']
        self.vocab = vocab
        self.tag2id = tag2id
        self.id2tag = id2tag
        
        self.sents = [[s[0] for s in sentences] for sentences in self.root]
        self.input_sents = []
        self.output_sents = []
        self.tags = []
        for sample in self.sents:
            
            if max_sent_len == None:
                mlength = len(sample)
            else:
                mlength = max_sent_len
            
            if reverse:
                sample = sample[::-1]
                
            newsample = [Vocabulary.BOS] + sample[:mlength] + [Vocabulary.EOS]
            input_toks = self.vocab.encode_token_seq(newsample[:-1])
            output_toks = [self.vocab.encode_token_seq_tag(newsample[1:], self.id2tag[tagid]) for tagid in self.id2tag]
            self.input_sents.append(input_toks)
            self.output_sents.append(output_toks)
            
        for sentences in self.root:
            
            if max_sent_len == None:
                mlength = len(sentences)
            else:
                mlength = max_sent_len
            
            if reverse:
                sentences = sentences[::-1]
            
            outputsample = sentences[:mlength] + [(Vocabulary.EOS, 'UNKNOWN')]
            outputsample = [self.tag2id[tup[1]] if tup[1] in self.tag2id else self.tag2id['UNKNOWN'] for tup in outputsample]
            self.tags.append(outputsample)
    
    def __len__(self):
        return len(self.root)
    
    def __getitem__(self,idx):
        target_tensor = torch.as_tensor(self.tags[idx], dtype=torch.long)
        input_tensor = torch.as_tensor(self.input_sents[idx], dtype=torch.long)
        output_tensor = torch.as_tensor(self.output_sents[idx], dtype=torch.long)
        return (input_tensor, output_tensor, target_tensor)