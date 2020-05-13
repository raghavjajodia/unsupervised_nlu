import os
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets import POSDataset
import pickle as pkl


def perplexity(avg_epoch_loss):
    return 2**(avg_epoch_loss/np.log(2))

def getTagPredictions(tag_logits):
    #tag_logits dimension -> btchsize x sentLen x numtags
    predictions = torch.max(tag_logits, dim=-1).indices
    return predictions

def getTokenAccuracy(tag_logits, labels, unk_tag, pad_tag, predictions = None):
    #tag_logits ->  btchsize x sentlen x numtags
    #labels -> btchsize x sentlen
    
    if predictions is None:
        predictions = getTagPredictions(tag_logits) #btchsize x sentlen
        
    mask = ((labels != unk_tag) & (labels != pad_tag))
    num = (predictions[mask] == labels[mask]).sum().item()
    den = labels[mask].shape[0]
    return num, den

def getSentenceAccuracy(tag_logits, labels, unk_tag, pad_tag, predictions = None):
    #tag_logits ->  btchsize x sentlen x numtags
    #labels -> btchsize x sentlen
    
    btchsize = labels.shape[0]
    
    if predictions is None:
        predictions = getTagPredictions(tag_logits) #btchsize x sentlen
        
    mask = ((labels != unk_tag) & (labels != pad_tag))
    sentCount = 0
    for i in range(btchsize):
        maski = mask[i]
        labeli = labels[i,:]
        predictioni = predictions[i,:]
        result = torch.equal(labeli[maski], predictioni[maski])
        sentCount += result*1
    
    return sentCount, btchsize

def evaluate_pos_dataset(model, loss_criterion, getAccuracies, device, data_loader):  
    running_loss = 0.0
    running_word = 0
    running_sent = 0
    total_words = 0
    total_sents = 0
    n_samples = 0
    non_pad_tokens_cache = 0

    # Iterate over data.
    for batch_num, (inputs, targets, labels) in enumerate(data_loader):

        inputs = inputs.to(device)
        targets = targets.to(device)
        labels = labels.to(device)

        batchSize = inputs.size(0)
        n_samples += batchSize
        
        #loss_criterion returns full batch loss and denominator
        batchloss, denominator = loss_criterion(model, inputs, targets, labels, device)
        correctToks, correctSentences, totalToks, totalSents = getAccuracies(model, inputs, targets, labels, device)
            
        # statistics
        running_loss += batchloss.item()
        
        running_word += correctToks
        total_words += totalToks
        running_sent += correctSentences
        total_sents += totalSents

        non_pad_tokens_cache += denominator

    # Metrics
    epoch_loss = running_loss / non_pad_tokens_cache
    tokenaccuracy = running_word/total_words
    sentaccuracy = running_sent/total_sents
    return epoch_loss, tokenaccuracy, sentaccuracy