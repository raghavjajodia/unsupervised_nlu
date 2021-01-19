import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import math
from transformers import RobertaModel

class LM_latent(nn.Module):
    def __init__(self, vocab_size, tag_wise_vocab_size, hidden_size, token_embedding_size, tag_embedding_size, lstm_layers=1, dropout_p=0.1):

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
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self,input_seq):
        
        batch_size,sent_len = input_seq.shape[0],input_seq.shape[1]
        
        embeddings = self.token_embedding(input_seq) #batch_size, sent_len, embed_size
        embeddings = self.dropout(embeddings)
        h, _ = self.lstm(embeddings) #batch_size, sent_len, hidden_size
        
        h_lower = self.lower_hidden(h) #batch_size , sent_len, 100
        tag_logits = self.tag_linear(h_lower) #batch_size, sent_len, num_tags
        
        word_distribution_logits = [self.tag_projections[i](h) for i in range(self.num_tags)] #for ith tag -> batch_size, sent_len, i_vocab_size       
        return tag_logits, word_distribution_logits
    
class LM_latent_type_rep(nn.Module):
    def __init__(self, vocab_size, tag_wise_vocab_size, hidden_size, token_embedding_size, tag_embedding_size, device, lstm_layers=1, dropout_p=0.1):

        super(LM_latent_type_rep, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.token_embedding_size = token_embedding_size
        self.tag_embedding_size = tag_embedding_size
        self.num_tags = len(tag_wise_vocab_size)
        self.tag_wise_vocabsize = tag_wise_vocab_size
        self.device = device
        self.lstm_layers = lstm_layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.token_embedding_size)
        self.lstm = nn.LSTM(self.token_embedding_size + self.tag_embedding_size, self.hidden_size, num_layers = lstm_layers, batch_first = True, bias = False)
        self.tag_linear = nn.Linear(self.tag_embedding_size, self.num_tags)
        self.lower_hidden = nn.Linear(self.hidden_size, self.tag_embedding_size)
        self.tag_projections = nn.ModuleList([nn.Linear(self.hidden_size, self.tag_wise_vocabsize[i]) for i in range(self.num_tags)])
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self,input_seq):
        
        batch_size,sent_len = input_seq.shape[0],input_seq.shape[1]
        h = torch.zeros((self.lstm_layers, batch_size,self.hidden_size),device=self.device)
        c = torch.zeros((self.lstm_layers, batch_size,self.hidden_size),device=self.device)
        embeddings = self.token_embedding(input_seq) #batch_size, sent_len, embed_size
        tag_logits_sent_norm = torch.ones((batch_size,self.num_tags),device=self.device)*1/self.num_tags
        tag_logits = torch.zeros((batch_size,sent_len,self.num_tags),device=self.device)
        word_distribution_logits = [torch.zeros((batch_size,sent_len,self.tag_wise_vocabsize[i]),device=self.device) for i in range(self.num_tags)]
        for idx in range(sent_len):
            embedding_input = torch.cat((embeddings[:,idx,:], torch.mm(tag_logits_sent_norm, self.tag_linear.weight)), 1).unsqueeze(1)
            embedding_input = self.dropout(embedding_input)
            _,(h,c) = self.lstm(embedding_input,(h,c))
            h_lower = self.lower_hidden(h[-1]) #batch_size,100
            tag_logits_sent = self.tag_linear(h_lower) #batch_size,num_tags
            tag_logits_sent_norm = nn.functional.softmax(tag_logits_sent, dim = 1)
            word_distribution_logits_sent = [((self.tag_projections[i](h[-1]))) for i in range(self.num_tags)] #batch_size,i_vocab_size
            tag_logits[:,idx,:] = tag_logits_sent
            for i in range(self.num_tags):
                word_distribution_logits[i][:,idx,:] = word_distribution_logits_sent[i]  
        return tag_logits, word_distribution_logits
    
    
class Autoregressive_Roberta_Autoencoder(nn.Module):
    def __init__(self, vocab_size, tag_wise_vocab_size, hidden_size, token_embedding_size, tag_embedding_size, lstm_layers=1, dropout_p=0.1):

        super(Autoregressive_Roberta_Autoencoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.token_embedding_size = token_embedding_size
        self.tag_embedding_size = tag_embedding_size
        self.num_tags = len(tag_wise_vocab_size)
        self.tag_wise_vocabsize = tag_wise_vocab_size
        self.num_layers = lstm_layers
        
        self.robertaEncoder = RobertaModel.from_pretrained('roberta-base')
        self.robertaLowering = nn.Linear(768, self.hidden_size)
        self.token_embedding = nn.Embedding(self.vocab_size, self.token_embedding_size)
        self.lstm = nn.LSTM(self.token_embedding_size, self.hidden_size, num_layers = lstm_layers, batch_first = True)
        self.tag_linear = nn.Linear(self.tag_embedding_size, self.num_tags)
        self.lower_hidden = nn.Linear(self.hidden_size, self.tag_embedding_size)
        self.tag_projections = nn.ModuleList([nn.Linear(self.hidden_size, self.tag_wise_vocabsize[i]) for i in range(self.num_tags)])
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self,input_seq, roberta_input, attention_mask):
        
        batch_size,sent_len = input_seq.shape[0],input_seq.shape[1]
        
        roberta_encoded = self.robertaEncoder(roberta_input, attention_mask)
        roberta_encoded = torch.mean(roberta_encoded[0], dim=1)
        roberta_encoded = self.robertaLowering(roberta_encoded) #batch_size , hidden_size
        
        h_0 = roberta_encoded.unsqueeze(0).expand((self.num_layers, batch_size, self.hidden_size)).contiguous()
        c_0 = roberta_encoded.unsqueeze(0).expand((self.num_layers, batch_size, self.hidden_size)).contiguous()
        
        embeddings = self.token_embedding(input_seq) #batch_size, sent_len, embed_size
        embeddings = self.dropout(embeddings)
        h, _ = self.lstm(embeddings, (h_0, c_0)) #batch_size, sent_len, hidden_size
        
        h_lower = self.lower_hidden(h) #batch_size , sent_len, 100
        tag_logits = self.tag_linear(h_lower) #batch_size, sent_len, num_tags
        
        word_distribution_logits = [self.tag_projections[i](h) for i in range(self.num_tags)] #for ith tag -> batch_size, sent_len, i_vocab_size       
        return tag_logits, word_distribution_logits