#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import pickle as pkl
from collections import defaultdict,deque,Counter,OrderedDict
from torch.utils.data import DataLoader,Dataset
from torch.optim import lr_scheduler
import os
import time
import copy
import argparse
import random

from models import Autoregressive_Roberta_Autoencoder
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from vocab import Vocabulary
from datasets import POS_RobertaDataset
from utils import pad_list_of_tensors, pad_collate_fn_pos, log_sum_exp, pad_collate_roberta


############################################################ Parsing

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate, default=0.00001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netCont', default='', help="path to net (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, default=9090, help='manual seed')
parser.add_argument('--verbIter', type=int, default=400, help='number of batches for interval printing')
parser.add_argument('--stepSize', type=int, default=20, help='number of steps after which learning rate reduces')
parser.add_argument('--use_type_rep', type=bool, default=False , help='using type representation as input')

parser.add_argument('--vocabSize', type=int, default=10000 , help='max size of input and output vocabulary')
parser.add_argument('--hiddenSize', type=int, default=512 , help='hidden size of lstm layer')
parser.add_argument('--tokenEmbeddingSize', type=int, default=256 , help='embedding size of input token')
parser.add_argument('--tagEmbeddingSize', type=int, default=256, help='dimension reduction for tag prediction')
parser.add_argument('--lstmLayers', type=int, default=3 , help='numer of lstm layers')
parser.add_argument('--maxSentLen', type=int, default=60 , help='maximum len of sentence for training')
parser.add_argument('--reverseDirection', action='store_true', help='trains autoregressive model in reverse direction')

parser.add_argument('--checkpointf', default='.', help='folder to output model checkpoints')

opt = parser.parse_args()

##################################################################3
##Assigning device and setup
###############################################################3

batch_size = opt.batchSize
max_sent_len = opt.maxSentLen
lr = opt.lr
stepsize = opt.stepSize
epochs = opt.niter
outfolder = opt.outf
use_type_rep = opt.use_type_rep

checkpoint_model_path = os.path.join(opt.checkpointf, 'net_best_weights.pth')

try:
    os.makedirs(opt.outf)
except OSError:
    pass

f = open(os.path.join(outfolder, 'training_logs.txt'), 'w+')

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    f.write("WARNING: You have a CUDA device, so you should probably run with --cuda \n")
    
device = torch.device("cuda:0" if opt.cuda else "cpu")
f.write("using " + str(device) + "\n")
f.flush()

######################################################################

def latent_loss(outputs, target, device):
    """Numerically stable implementation of the language modeling loss

    """
    #target dim # btchsize x numtags x sentLen
    tag_logits = outputs[0] #btchsize x sentlen x numtags
    word_dist_logits = outputs[1] #list #for jth tag -> batch_size, sent_len, j_vocab_size
    
    numtags = len(word_dist_logits)
    btchSize = tag_logits.shape[0]
    sentLen = tag_logits.shape[1]
    
    #calculate loss for tags
    crossEntropy_tag = nn.CrossEntropyLoss(reduction='none')
    taglogitloss = [-crossEntropy_tag(tag_logits.transpose(1,2), torch.zeros((btchSize, sentLen), dtype=torch.long, device=device) + j) for j in range(numtags)]
    
    #calculate loss for words
    ignore_mask = ((target == Vocabulary.TOKEN_NOT_IN_TAGVOCAB) | (target == Vocabulary.PADTOKEN_FOR_TAGVOCAB))
    target_with_ignore = target.clone()
    target_with_ignore[ignore_mask] = -100
    crossEntropy_word = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    wordlogitloss = [-crossEntropy_word(word_logit.transpose(1,2), target_with_ignore[:, j, :])  for j, word_logit in enumerate(word_dist_logits)]
    
    taglogitloss = torch.stack(taglogitloss)
    wordlogitloss = torch.stack(wordlogitloss)
    totalloss = taglogitloss + wordlogitloss
    
    #0 loss for a tag if output word is not present in tag's vocab
    outofvocab_mask = (torch.transpose(target, 0, 1) == Vocabulary.TOKEN_NOT_IN_TAGVOCAB)
    totalloss[outofvocab_mask] = float('-inf')

    finalLoss = -log_sum_exp(totalloss, dim=0)
    
    #mask the loss from tokens, if the output token is not present in even single tag category
    presentInZeroTagMask = torch.all((torch.transpose(target, 1, 2) == Vocabulary.TOKEN_NOT_IN_TAGVOCAB), dim=-1)
    #mask the loss of padding tokens
    paddingMask = (target[:, 0, :] == Vocabulary.PADTOKEN_FOR_TAGVOCAB)
    tokenContributingToZeroLoss = (presentInZeroTagMask | paddingMask)
    num_useful_tokens = (~tokenContributingToZeroLoss).sum().item()
    
    return torch.sum(finalLoss[~tokenContributingToZeroLoss]), num_useful_tokens


def train_model(model, criterion, optimizer, scheduler, device, checkpoint_path, f, verbIter, hyperparams, num_epochs=25):
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
        
        for batch_num, (inputs, target, labels, roberta_inp, attn) in enumerate(dataloaders["train"]):
            
            data_time = time.time() - end
            inputs = inputs.to(device)
            target = target.to(device)
            roberta_inp = roberta_inp.to(device)
            attn = attn.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            batchSize = inputs.size(0)
            n_samples += batchSize

            # forward
            # track history if only in train
            forward_start_time  = time.time()

            with torch.set_grad_enabled(True):
                outputs = model(inputs, roberta_inp, attn)
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
        epoch_loss, tokenaccuracy, sentaccuracy = evaluate(model, criterion, device, dataloaders["valid"])
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
    
    #Save weights of the best model
    torch.save({
        'epoch': 'best_epoch',
        'model_state_dict': best_model_wts,
        'full_metrics': metrics_dict,
        'hyperparams': hyperparams
        }, '%s/net_best_weights.pth' % (checkpoint_path))

    return None

def getTagPredictions(tag_logits):
    predictions = torch.max(tag_logits, dim=-1).indices
    return predictions


def perplexity(avg_epoch_loss):
    return 2**(avg_epoch_loss/np.log(2))

def getTokenAccuracy(tag_logits, labels):
    #tag_logits ->  btchsize x sentlen x numtags
    #labels -> btchsize x sentlen
    predictions = getTagPredictions(tag_logits) #btchsize x sentlen
    mask = ((labels != UNKNOWN_TAG) & (labels != PAD_TAG_ID))
    num = (predictions[mask] == labels[mask]).sum().item()
    den = labels[mask].shape[0]
    return num, den

def getSentenceAccuracy(tag_logits, labels):
    #tag_logits ->  btchsize x sentlen x numtags
    #labels -> btchsize x sentlen
    predictions = getTagPredictions(tag_logits) #btchsize x sentlen
    mask = ((labels != UNKNOWN_TAG) & (labels != PAD_TAG_ID))
    
    sentCount = 0
    for i in range(tag_logits.shape[0]):
        maski = mask[i]
        labeli = labels[i,:]
        predictioni = predictions[i,:]
        result = torch.equal(labeli[maski], predictioni[maski])
        sentCount += result*1
    
    return sentCount, tag_logits.shape[0]


def evaluate(model, criterion, device, validation_loader):  
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    running_word = 0
    running_sent = 0
    total_words = 0
    total_sents = 0
    n_samples = 0
    non_pad_tokens_cache = 0

    # Iterate over data.
    for batch_num, (inputs, targets, labels, roberta_inp, attn) in enumerate(validation_loader):

        inputs = inputs.to(device)
        targets = targets.to(device)
        labels = labels.to(device)
        roberta_inp = roberta_inp.to(device)
        attn = attn.to(device)

        batchSize = inputs.size(0)
        n_samples += batchSize

        outputs = model(inputs, roberta_inp, attn)
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

##############################################################
#Checkpoint loading
#############################################################

#Load forward model
checkpoint_model = torch.load(checkpoint_model_path, map_location=device)
tag2id = checkpoint_model["hyperparams"]["tagtoid"]
id2tag = defaultdict(str)
for tag in tag2id:
    id2tag[tag2id[tag]] = tag

UNKNOWN_TAG = tag2id['UNKNOWN']
PAD_TAG_ID = -51    
    
vocab = checkpoint_model["hyperparams"]["vocab"]
hidden_size = checkpoint_model["hyperparams"]["hidden_size"]
tok_emb_size = checkpoint_model["hyperparams"]["token_embedding"]
tag_emb_size = checkpoint_model["hyperparams"]["tag_emb_size"]
lstm_layers = checkpoint_model["hyperparams"]["lstmLayers"]
dropout_p = checkpoint_model["hyperparams"]["dropout"]
weight_decay = checkpoint_model['hyperparams']['weight_decay']
tag_wise_vocabsize = dict([(tag2id[tup[0]], tup[1][2]) for tup in vocab.tag_specific_vocab.items()])

#####################################################################
## Data loading 
#####################################################################

datasets = {}
dataloaders = {}

train_pickle_file = os.path.join(opt.dataroot, 'train.p')
val_pickle_file = os.path.join(opt.dataroot, 'val.p')
test_pickle_file = os.path.join(opt.dataroot, 'test.p')

with open(train_pickle_file,"rb") as a:
    traindict = pkl.load(a)
with open(val_pickle_file,"rb") as a:
    valdict = pkl.load(a)
with open(test_pickle_file,"rb") as a:
    testdict = pkl.load(a)

with open('tagset.txt') as a:
    alltags = a.read()

alltags = alltags.split('\n')    
alltags = alltags + ['UNKNOWN']
alltags_set = set(alltags)

datasets["train"] = POS_RobertaDataset(traindict, vocab, tag2id, id2tag, 60)
datasets["valid"] = POS_RobertaDataset(valdict, vocab, tag2id, id2tag, None)
datasets["test"] = POS_RobertaDataset(testdict, vocab, tag2id, id2tag, None)

dataloaders["train"] = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=pad_collate_roberta, pin_memory=True)
dataloaders["valid"] = DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False, collate_fn=pad_collate_roberta, pin_memory=True)
dataloaders["test"]  = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, collate_fn=pad_collate_roberta, pin_memory=True)

model = Autoregressive_Roberta_Autoencoder(vocab.vocab_size, tag_wise_vocabsize, hidden_size, tok_emb_size, tag_emb_size, lstm_layers, dropout_p)
model.load_state_dict(checkpoint_model['model_state_dict'])

for param in model.robertaEncoder.parameters():
    param.requires_grad = True

if opt.ngpu > 1:
    f.write("Let's use " +  str(torch.cuda.device_count()) + " GPUs!")
    f.flush()
    model = nn.DataParallel(model)

model = model.to(device)

options = {"vocab":vocab, "hidden_size": hidden_size, "token_embedding":tok_emb_size, 
           "tag_emb_size":tag_emb_size, "lstmLayers": lstm_layers, "tagtoid":tag2id, "reverse":False,
          "dropout": dropout_p, "weight_decay": weight_decay}

criterion = latent_loss

model_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=0.1)
train_model(model, criterion, optimizer, exp_lr_scheduler, device, outfolder, f, opt.verbIter, options, epochs)
f.close()
