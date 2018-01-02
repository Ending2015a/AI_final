import os
import time
import random
import numpy as np
import tensorflow as tf


def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)

    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

print(position_encoding(10, 5))


class MemNet(object):
    def __init__(self, enc_map, dec_map, vocab_size, dim_embed=512, n_hop=6, story_size=20, sentence_encoding=position_encoding):

        self.enc_map = enc_map
        self.dec_map = dec_map
        self.vocab_size = vocab_size
        self.dim_embed = dim_embed
        self.n_hop = n_hop
        self.story_size = story_size

        self.sent_encoding = sentence_encoding
    
    def _input_embedding(self, inputs, reuse=tf.AUTO_REUSE):
        pass

    def _output_embedding(self, ouputs, reuse=tf.AUTO_REUSE):
        pass

    def _query_embedding(self, query, reuse=tf.AUTO_REUSE):
        pass

    def build_memory(self):
        
        for hop in range(self.n_hop):
            pass
        

    def build_model(self):
        pass

    def build_sampler(self):
        pass
            
