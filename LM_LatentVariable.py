#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
from collections import defaultdict,deque,Counter,OrderedDict
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import os
import time
import copy

from models import LM_latent
from vocab import Vocabulary


# In[2]:


general_vocab_size = 10000
batch_size = 64 #takes about 6gb memory
hidden_size = 512
token_embedding_size = 128
tag_embedding_size = 128
lstm_layers = 2
max_sent_len = 60

num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    device = 'cuda'
else:
    device = 'cpu'


# In[3]:


train_pickle_file = '/data/rj1408/ptb_wsj_pos/train.p'
val_pickle_file = '/data/rj1408/ptb_wsj_pos/val.p'
test_pickle_file = '/data/rj1408/ptb_wsj_pos/test.p'

with open(train_pickle_file,"rb") as f:
    traindict = pkl.load(f)
with open(val_pickle_file,"rb") as f:
    valdict = pkl.load(f)
with open(test_pickle_file,"rb") as f:
    testdict = pkl.load(f)


# In[4]:


with open('tagset.txt') as f:
    alltags = f.read()

alltags = list(map(lambda strline: strline.split('\t')[1], alltags.split('\n')))
alltags = set(alltags)

tag2id = defaultdict(int)
id2tag = defaultdict(str)
for i, tag in enumerate(alltags):
    tag2id[tag] = i
    id2tag[i] = tag
UNKNOWN_TAG = -50
PAD_TAG_ID = -51

# In[5]:


class POSDataset(object):
    def __init__(self, instanceDict, vocab, tag2id, id2tag, max_sent_len=60):
        self.root = instanceDict['tagged_sents']
        self.vocab = vocab
        self.tag2id = tag2id
        self.id2tag = id2tag
        
        self.sents = [[s[0] for s in sentences] for sentences in self.root]
        self.input_sents = []
        self.output_sents = []
        for sample in self.sents:
            newsample = [Vocabulary.BOS] + sample[:max_sent_len] + [Vocabulary.EOS]
            input_toks = self.vocab.encode_token_seq(newsample[:-1])
            output_toks = [self.vocab.encode_token_seq_tag(newsample[1:], self.id2tag[tagid]) for tagid in self.id2tag]
            self.input_sents.append(input_toks)
            self.output_sents.append(output_toks)
        
        self.tags = [([self.tag2id[s[1]] if s[1] in self.tag2id else UNKNOWN_TAG for s in sentences][:max_sent_len]) + [UNKNOWN_TAG] for sentences in self.root]
    
    def __len__(self):
        return len(self.root)
    
    def __getitem__(self,idx):
        target_tensor = torch.as_tensor(self.tags[idx], dtype=torch.long)
        input_tensor = torch.as_tensor(self.input_sents[idx], dtype=torch.long)
        output_tensor = torch.as_tensor(self.output_sents[idx], dtype=torch.long)
        return (input_tensor, output_tensor, target_tensor)


# In[6]:


def pad_list_of_tensors(list_of_tensors, pad_token):
    max_length = max([t.size(-1) for t in list_of_tensors])
    padded_list = []
    for t in list_of_tensors:
        padding = torch.zeros(list(t.shape)[:-1] + [max_length - t.size(-1)], dtype=torch.long) + pad_token
        padded_tensor = torch.cat([t, padding], dim = -1)
        padded_list.append(padded_tensor)
    padded_tensor = torch.stack(padded_list)
    return padded_tensor

def pad_collate_fn_pos(batch):
    # batch is a list of sample tuples
    input_list = [s[0] for s in batch]
    target_list = [s[1] for s in batch]
    target_labels = [s[2] for s in batch]
    pad_token_input = 2
    pad_token_output = Vocabulary.PADTOKEN_FOR_TAGVOCAB
    pad_token_tags = PAD_TAG_ID
    input_tensor = pad_list_of_tensors(input_list, pad_token_input)
    target_tensor = pad_list_of_tensors(target_list, pad_token_output)
    target_labels = pad_list_of_tensors(target_labels, pad_token_tags)
    return input_tensor, target_tensor, target_labels


# In[7]:


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

def latent_loss(outputs, target, device):
    #target dim # btchsize x numtags x sentLen
    tag_logits = outputs[0] #btchsize x sentlen x numtags
    word_dist_logits = outputs[1] #list #for jth tag -> batch_size, sent_len, j_vocab_size
    
    numtags = len(word_dist_logits)
    btchSize = tag_logits.shape[0]
    sentLen = tag_logits.shape[1]
    
    crossEntropy_tag = nn.CrossEntropyLoss(reduction='none')
    taglogitloss = [-crossEntropy_tag(tag_logits.transpose(1,2), torch.zeros((btchSize, sentLen), dtype=torch.long, device=device) + j) for j in range(numtags)]
    
    ignore_mask = (target == Vocabulary.TOKEN_NOT_IN_TAGVOCAB | target == Vocabulary.PADTOKEN_FOR_TAGVOCAB)
    target_with_ignore = target.clone()
    target_with_ignore[ignore_mask] == -100
    crossEntropy_word = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    wordlogitloss = [-crossEntropy_word(word_logit.transpose(1,2), target_with_ignore[:, j, :])  for j, word_logit in enumerate(word_dist_logits)]
    
    taglogitloss = torch.stack(taglogitloss)
    wordlogitloss = torch.stack(wordlogitloss)
    totalloss = taglogitloss + wordlogitloss
    totalloss = totalloss.view(btchSize, sentLen, numtags)
    
    #mask the tags if output word is not present in tag's vocab
    outofvocab_mask = (target.view(btchSize, sentLen, numtags) == Vocabulary.TOKEN_NOT_IN_TAGVOCAB)
    temp = outofvocab_mask*1.0
    temp[outofvocab_mask] = float('-inf')
    totalloss = totalloss + temp
    
    finalLoss = -log_sum_exp(totalloss, dim=-1)
    
    #mask the loss from tokens, if the output token is not present in even single tag category
    presentInZeroTagMask = torch.all((target.view(btchSize, sentLen, numtags) == Vocabulary.TOKEN_NOT_IN_TAGVOCAB), dim=-1)
    
    #mask the loss of padding tokens
    paddingMask = (target[:, 0, :] == Vocabulary.PADTOKEN_FOR_TAGVOCAB)
    
    tokenContributingToZeroLoss = (presentInZeroTagMask | paddingMask)
    useful_tokens = (~tokenContributingToZeroLoss).sum().item()
    
    return torch.sum(finalLoss[~tokenContributingToZeroLoss]), useful_tokens


# In[8]:


def train_model(model, criterion, optimizer, scheduler, device, checkpoint_path, f, verbIter, hyperparams, padtoken, num_epochs=25):
    metrics_dict = {}
    metrics_dict["train"] = {}
    metrics_dict["valid"] = {}
    metrics_dict["train"]["loss"] = {}
    metrics_dict["train"]["loss"]["epochwise"] = []
    metrics_dict["train"]["loss"]["stepwise"] = []
    metrics_dict["valid"]["loss"] = []
    metrics_dict["valid"]["tokenacc"] = []
    metrics_dict["valid"]["sentacc"] = []
        
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 9999999999999999

    for epoch in range(num_epochs):
        f.write('Epoch {}/{} \n'.format(epoch, num_epochs - 1))
        f.write('-' * 10)
        f.write('\n')
        f.flush()
        
        #train phase
        scheduler.step()
        model.train()
        
        running_loss = 0.0
        n_samples = 0
        non_pad_tokens_cache = 0

        end = time.time()
        
        for batch_num, (inputs, target, labels) in enumerate(dataloaders["train"]):
            
            data_time = time.time() - end
            inputs = inputs.to(device)
            target = target.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            batchSize = inputs.size(0)
            n_samples += batchSize

            # forward
            # track history if only in train
            forward_start_time  = time.time()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss, useful_tokens = criterion(outputs, target, device)

                # statistics
                running_loss += loss.item()

                non_pad_tokens_cache += useful_tokens

                loss /= useful_tokens

                loss.backward()
                optimizer.step()
            
            forward_time = time.time() - forward_start_time

            if batch_num % verbIter == 0:
                # Metrics
                epoch_loss = running_loss / non_pad_tokens_cache

                f.write('Train Loss: {:.4f} \n'.format(epoch_loss))
                f.write('Full Batch time: {} , Data load time: {} , Forward time: {}\n'.format(time.time() - end, data_time, forward_time))
                f.flush()

                metrics_dict["train"]["loss"]["stepwise"].append(epoch_loss)

            end = time.time()
        
        # Metrics
        epoch_loss = running_loss / non_pad_tokens_cache
        f.write('Train Loss: {:.4f} \n'.format(epoch_loss))
        f.flush()
        metrics_dict["train"]["loss"]["epochwise"].append(epoch_loss)

        
        #val phase
        epoch_loss, tokenaccuracy, sentaccuracy = evaluate(model, criterion, device, dataloaders["valid"], padtoken)
        f.write('Validation Loss: {:.4f}, Perplexity: {},  TokenAccuracy: {}, SentAccuracy: {} \n'.format(epoch_loss, perplexity(epoch_loss), tokenaccuracy, sentaccuracy))
        f.flush()
        metrics_dict["valid"]["loss"].append(epoch_loss)
        metrics_dict["valid"]["tokenacc"].append(tokenaccuracy)
        metrics_dict["valid"]["sentacc"].append(sentaccuracy)
        
            
        # deep copy the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
                
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'full_metrics': metrics_dict,
        'hyperparams': hyperparams
        }, '%s/net_epoch_%d.pth' % (checkpoint_path, epoch))

    time_elapsed = time.time() - since
    f.write('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    f.write('Best val loss: {:4f} \n'.format(best_loss))
    f.flush()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[9]:


def perplexity(avg_epoch_loss):
    return 2**(avg_epoch_loss/np.log(2))

def getTokenAccuracy(tag_logits, labels):
    #tag_logits ->  btchsize x sentlen x numtags
    #labels -> btchsize x sentlen
    num_tags = tag_logits.shape[2]
    predictions = torch.max(tag_logits, dim=-1).indices #btchsize x sentlen
    mask = ((labels != UNKNOWN_TAG) & (labels != PAD_TAG_ID))
    num = (predictions[mask] == labels[mask]).sum().item()
    den = labels[mask].shape[0]
    return num, den

def getSentenceAccuracy(tag_logits, labels):
    #tag_logits ->  btchsize x sentlen x numtags
    #labels -> btchsize x sentlen
    num_tags = tag_logits.shape[2]
    predictions = torch.max(tag_logits, dim=-1).indices #btchsize x sentlen
    mask = ((labels != UNKNOWN_TAG) & (labels != PAD_TAG_ID))
    
    sentCount = 0
    for i in range(tag_logits.shape[0]):
        maski = mask[i]
        labeli = labels[i,:]
        predictioni = predictions[i,:]
        result = torch.equal(labeli[maski], predictioni[maski])
        sentCount += result*1
    
    return sentCount, tag_logits.shape[0]


def evaluate(model, criterion, device, validation_loader, padtoken):  
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    running_word = 0
    running_sent = 0
    total_words = 0
    total_sents = 0
    n_samples = 0
    non_pad_tokens_cache = 0

    # Iterate over data.
    for batch_num, (inputs, targets, labels) in enumerate(validation_loader):

        inputs = inputs.to(device)
        targets = targets.to(device)
        labels = labels.to(device)

        batchSize = inputs.size(0)
        n_samples += batchSize

        outputs = model(inputs)
        loss, useful_tokens = criterion(outputs, targets, device)
                    
        # statistics
        running_loss += loss.item()
        num, den = getTokenAccuracy(outputs[0], labels)
        running_word += num
        total_words += den
        num, den = getSentenceAccuracy(outputs[0], labels)
        running_sent += num
        total_sents += den

        non_pad_tokens_cache += useful_tokens

    # Metrics
    epoch_loss = running_loss / non_pad_tokens_cache
    tokenaccuracy = running_word/total_words
    sentaccuracy = running_sent/total_sents
    return epoch_loss, tokenaccuracy, sentaccuracy


# In[10]:


vocab = Vocabulary(traindict, general_vocab_size, alltags)
tag_wise_vocabsize = dict([(tag2id[tup[0]], tup[1][2]) for tup in vocab.tag_specific_vocab.items()])

datasets = {}
dataloaders = {}

datasets["train"] = POSDataset(traindict, vocab, tag2id, id2tag)
datasets["valid"] = POSDataset(valdict, vocab, tag2id, id2tag)
datasets["test"] = POSDataset(testdict, vocab, tag2id, id2tag)

dataloaders["train"] = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn_pos, pin_memory=True)
dataloaders["valid"] = DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn_pos, pin_memory=True)
dataloaders["test"] = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn_pos, pin_memory=True)


# In[ ]:


options = {"vocab":vocab, "hidden_size": hidden_size, "token_embedding":token_embedding_size, 
           "tag_emb_size":tag_embedding_size, "lstmLayers": lstm_layers, "tagtoid":tag2id}

lr = 0.01
stepsize = 5
epochs = 30
outfolder = '/data/rj1408/ptb_wsj_pos/models/basic_imp2/a/'

model = LM_latent(vocab.vocab_size, tag_wise_vocabsize, hidden_size, token_embedding_size, tag_embedding_size, lstm_layers).to(device)
criterion = latent_loss
model_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(model_parameters, lr=lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=0.1)
f = open(os.path.join(outfolder, 'training_logs.txt'), 'w+')
bst_model = train_model(model, criterion, optimizer, exp_lr_scheduler, device, outfolder, f, 50, options, vocab.get_id(vocab.PADDING), epochs)


# In[ ]:




