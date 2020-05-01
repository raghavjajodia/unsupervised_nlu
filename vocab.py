import numpy as np
from collections import Counter

class Vocabulary():
    
    PADDING = '<pad>'
    UNKNOWN = '<unk>'
    BOS = '<bos>'
    EOS = '<eos>'
    
    def __init__(self, data, vocab_size, alltags):
        self.vocab_size = vocab_size
        self.train = data
        self.alltags = alltags
        
        #create generic vocab
        word_freq = Counter([word[0] for word in self.train['tagged_words']])
        self.token2idx, self.idx2token = self.build_dict(word_freq, self.vocab_size)
        self.vocab_size = len(self.token2idx)
        
        #create vocab for every pos tag
        lis = self.train['tagged_words']
        tag_cntr = {}
        for tup in lis:
            if tup[1] in self.alltags:
                if tup[1] not in tag_cntr:
                    tag_cntr[tup[1]] = Counter()
                tag_cntr[tup[1]][tup[0]] += 1
        self.tag_specific_vocab = {}
        for tag in tag_cntr:
            tok2ind, ind2tok = self.build_dict(tag_cntr[tag], None)
            voc_size = len(tok2ind)
            self.tag_specific_vocab[tag] = (tok2ind, ind2tok, voc_size)
  
    def get_vocab_counter(self):
        """
         Use collections.Counter() to get unique words and their counts
         -Args:
            dataset: a pandas dataset of interst
         -Returns:
            vocab_counter: a counter object, in format {word: count}
        """
        vocab_counter = Counter()
        for i in range(len(self.train)):
            vocab_counter.update(self.train[i])
        return vocab_counter

    def build_dict(self, vocab, vocab_size = None):
        """
        Generate word-index-dict
        -Args:
            vocab_counter: Counter object generated from a dataset of interest
        -Returns:
            target_word_count: number of words to be included in the corpus
        """
        # prune vocab
        if vocab_size is not None:
            vocab = list(map(lambda tup: tup[0], vocab.most_common(vocab_size)))
        else:
            vocab = list(vocab.keys())
        
        vocab = [Vocabulary.BOS, Vocabulary.EOS, Vocabulary.PADDING, Vocabulary.UNKNOWN] + vocab
        token2idx = dict(zip(vocab, range(len(vocab))))
        idx2token = {v:k for k,v in token2idx.items()}
        return token2idx, idx2token
  
    def get_id(self, token):
        if token in self.token2idx:
            return self.token2idx[token]
        else:
            return self.token2idx[Vocabulary.UNKNOWN]
        
    def get_id_tag(self, token, tag):
        if token in self.tag_specific_vocab[tag][0]:
            return self.tag_specific_vocab[tag][0][token]
        else:
            return self.tag_specific_vocab[tag][0][Vocabulary.UNKNOWN]
        
    
    def get_token(self, idx):
        return self.idx2token[idx]
    
    def decode_idx_seq(self, l):
        return list(map(lambda tokid: self.idx2token[tokid], l))
    
    def encode_token_seq(self, l):
        return list(map(lambda tok: self.get_id(tok), l))
    
    def encode_token_seq_tag(self, l, tag):
        return [self.get_id_tag(tok, tag) for tok in l]