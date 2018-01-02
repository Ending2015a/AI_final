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

class MemNet(object):
    def __init__(self, vocab_size, embed_size=512, n_hop=6, memory_size=20, sentence_size=216, 
            sentence_encoding = position_encoding,
            emb_initializer = tf.random_normal_initializer(stddev=0.1)):

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.n_hop = n_hop
        self.memory_size = memory_size
        self.sentence_size = sentence_size

        self.sent_encoding = sentence_encoding
        self.emb_initializer = emb_initializer

        self._encoding = tf.constant(self.sent_encoding(self.sentence_size, self.embed_size), name='encoding') # [sentence_size, embed_size]
        self.build_input()

    def _inputs_embedding(self, inputs, hop=0, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('embedding', reuse=reuse):
            if hop==0:
                A = tf.get_variable('A', [self.vocab_size, self.embed_size], initializer=self.emb_initializer)
            else: # use adjacent weight tying A^{k+1} = C^k
                A = tf.get_variable('C_{}'.format(hop-1), [self.vocab_size, self.embed_size], initializer=self.emb_initializer)
            
            x = tf.nn.embedding_lookup(A, inputs, name='input_vector')
            return x

    def _outputs_embedding(self, outputs, hop=0, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('embedding', reuse=reuse):
            C = tf.get_variable('C_{}'.format(hop), [self.vocab_size, self.embed_size], initializer=self.emb_initializer)

            x = tf.nn.embedding_lookup(C, outputs, name='output_vector')
            return x

    def _query_embedding(self, query, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('embedding', reuse=reuse): # use adjacent weight tying B = A
            B = tf.get_variable('A', [self.vocab_size, self.embed_size], initializer=self.emb_initializer)

            x = tf.nn.embedding_lookup(B, query, name='query_vector')
            return x

    def _unembedding(self, pred, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('embedding', reuse=reuse):
            W = tf.get_variable('C_{}'.format(self.n_hop-1), [self.vocab_size, self.embed_size], initializer=self.emb_initializer)

            WT = tf.transpose(W, [1, 0])
            return tf.matmul(pred, WT)

    def build_input(self):
        self._sentences = tf.placeholder(tf.int32, [None, self.memory_size, self.sentence_size], name='sentences')
        self._query = tf.placeholder(tf.int32, [None, self.sentence_size], name='query')
        self._answer = tf.placeholder(tf.int32, [None, self.vocab_size], name='answer')
        

    def build_model(self):
        with tf.variable_scope('MemN2N'):
            emb_q = self._query_embedding(self._query) # [batch_size, sentence_size, embed_size]
            #print('emb_q shape: ', emb_q.get_shape())
            u = tf.reduce_sum(emb_q*self._encoding, 1) # [batch_size, embed_size]
            #print('u shape: ', u.get_shape())

            onehot = tf.one_hot(self._answer, self.vocab_size)
            #print('onehot shape: ', onehot.get_shape())

            for hop in range(self.n_hop):
                emb_i = self._inputs_embedding(self._sentences, hop) # [batch_size, memory_size, sentence_size, embed_size]
                #print('emb_i shape: ', emb_i.get_shape())
                mem_i = tf.reduce_sum(emb_i*self._encoding, 2) # [batch_size, memory_size, embed_size]
                #print('mem_i shape: ', mem_i.get_shape())

                emb_o = self._outputs_embedding(self._sentences, hop) # same as emb_i
                #print('emb_o shape: ', emb_o.get_shape())
                mem_o = tf.reduce_sum(emb_o*self._encoding, 2) # same as mem_i
                #print('mem_o shape: ', mem_o.get_shape())
                
                uT = tf.transpose(tf.expand_dims(u, -1), [0, 2, 1]) # [batch_size, embed_size, 1] -> [batch_size, 1, embed_size]
                #print('uT shape: ', uT.get_shape())

                p = tf.nn.softmax(tf.reduce_sum(mem_i*uT, 2)) # inner product [batch_size, memory_size]
                #print('probs shape: ', p.get_shape())

                p = tf.expand_dims(p, -1) # [batch_size, memory_size, 1]
                #print('probsT shape: ', p.get_shape())

                u = tf.reduce_sum(mem_o*p, 1) + u # [batch_size, embed_size]
                #print('u shape: ', u.get_shape())

            logits = tf.nn.softmax(self._unembedding(u)) #a_hat [batch_size, vocab_size]
            #print('logits shape: ', logits.get_shape())

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot, logits=logits)
            loss = tf.reduce_mean(cross_entropy)

            return loss


    def build_sampler(self):
        with tf.variable_scope('MemN2N'):
            pass


net = MemNet(1000)
net.build_model()
