import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import math


class LM_latent(nn.Module):
    def __init__(self, vocab_size, tag_wise_vocab_size, hidden_size, token_embedding_size, tag_embedding_size, lstm_layers=1):

        super(LM_latent, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.token_embedding_size = token_embedding_size
        self.tag_embedding_size = tag_embedding_size
        self.num_tags = len(tag_wise_vocab_size)
        self.tag_wise_vocabsize = tag_wise_vocab_size
        
        self.token_embedding = nn.Embedding(self.vocab_size, self.token_embedding_size)
        self.lstm = nn.LSTM(self.token_embedding_size, self.hidden_size, num_layers = lstm_layers, batch_first = True)
        self.tag_linear = nn.Linear(self.tag_embedding_size, self.num_tags)
        self.lower_hidden = nn.Linear(self.hidden_size, self.tag_embedding_size)
        self.tag_projections = nn.ModuleList([nn.Linear(self.hidden_size, self.tag_wise_vocabsize[i]) for i in range(self.num_tags)])
        
    def forward(self,input_seq):
        
        batch_size,sent_len = input_seq.shape[0],input_seq.shape[1]
        
        embeddings = self.token_embedding(input_seq) #batch_size, sent_len, embed_size
        h, _ = self.lstm(embeddings) #batch_size, sent_len, hidden_size
        
        h_lower = self.lower_hidden(h) #batch_size , sent_len, 100
        tag_logits = self.tag_linear(h_lower) #batch_size, sent_len, num_tags
        
        word_distribution_logits = [self.tag_projections[i](h) for i in range(self.num_tags)] #for ith tag -> batch_size, sent_len, i_vocab_size       
        return tag_logits, word_distribution_logits