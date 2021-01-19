import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import math
from vocab import Vocabulary
from transformers import RobertaTokenizer

PAD_TAG_ID = -51

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

def pad_collate_roberta(batch):
    # batch is a list of sample tuples
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    input_list = [s[0] for s in batch]
    target_list = [s[1] for s in batch]
    target_labels = [s[2] for s in batch]
    raw_text = [s[3] for s in batch]
    pad_token_input = 2
    pad_token_output = Vocabulary.PADTOKEN_FOR_TAGVOCAB
    pad_token_tags = PAD_TAG_ID
    input_tensor = pad_list_of_tensors(input_list, pad_token_input)
    target_tensor = pad_list_of_tensors(target_list, pad_token_output)
    target_labels = pad_list_of_tensors(target_labels, pad_token_tags)
    
    roberta_input = roberta_tokenizer.batch_encode_plus(raw_text, add_special_tokens=True, pad_to_max_length=True)
    roberta_input_tens = torch.as_tensor(roberta_input['input_ids'])
    roberta_input_att = torch.as_tensor(roberta_input['attention_mask'])
    
    return input_tensor, target_tensor, target_labels, roberta_input_tens, roberta_input_att

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