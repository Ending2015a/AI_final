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
    def __init__(self, vocab_size, embed_size=512, n_hop=3, memory_size=20, sentence_size=216, option_size=10,
            sentence_encoding = position_encoding,
            emb_initializer = tf.random_normal_initializer(stddev=0.1)):

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.n_hop = n_hop
        self.memory_size = memory_size
        self.sentence_size = sentence_size
        self.option_size = option_size

        self.sent_encoding = sentence_encoding
        self.emb_initializer = emb_initializer

        self._encoding = tf.constant(self.sent_encoding(self.sentence_size, self.embed_size), name='encoding') # [sentence_size, embed_size]

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

    def _fc(self, inputs, num_out, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            input_shape = inputs.get_shape()
            feed_in = input_shape[-1].value
            weights = tf.get_variable('weights', [feed_in, num_out], initializer=tf.truncated_normal_initializer(stddev=5e-2))
            biases = tf.get_variable('biases', [num_out], initializer=tf.constant_initializer(0.0))

            x = tf.nn.xw_plus_b(inputs, weights, biases, name=name)
            return x

    def build_model(self, sentences=None, query=None, answer=None):

        if sentences == None:
            sentences = tf.placeholder(tf.int32, [None, self.memory_size, self.sentence_size], name='sentences')

        if query == None:
            query = tf.placeholder(tf.int32, [None, self.option_size, self.sentence_size], name='query')

        if answer == None:
            answer = tf.placeholder(tf.int32, [None], name='answer')

        with tf.variable_scope('MemN2N'):
            emb_q = self._query_embedding(query) # [batch_size, option_size, sentence_size, embed_size]
            u = tf.reduce_sum(emb_q*self._encoding, 2) # [batch_size, option_size, embed_size]

            onehot = tf.one_hot(answer, self.option_size) # [batch_size, option_size]

            for hop in range(self.n_hop):
                emb_i = self._inputs_embedding(sentences, hop) # [batch_size, memory_size, sentence_size, embed_size]
                mem_i = tf.reduce_sum(emb_i*self._encoding, 2) # [batch_size, memory_size, embed_size]
                mem_i = tf.expand_dims(mem_i, 1) # [batch_size, 1, memory_size, embed_size]

                emb_o = self._outputs_embedding(sentences, hop) # same as emb_i
                mem_o = tf.reduce_sum(emb_o*self._encoding, 2) # same as mem_i
                mem_o = tf.expand_dims(mem_o, 1) # [batch_size, 1, memory_size, embed_size]
                
                uT = tf.transpose(tf.expand_dims(u, -1), [0, 1, 3, 2])
                # [batch_size * option_size, embed_size, 1] -> [batch_size, option_size, 1, embed_size]

                p = tf.nn.softmax(tf.reduce_sum(mem_i*uT, 3)) # inner product [batch_size, option_size, memory_size]
                p = tf.expand_dims(p, -1) # [batch_size, option_size, memory_size, 1]

                o = tf.reduce_sum(mem_o*p, 2) # [batch_size, option_size, embed_size]
                u = o + u # [batch_size, option_size, embed_size]

            #logits = tf.nn.softmax(self._unembedding(u)) #a_hat [batch_size * option_size, vocab_size]
            a_hat = tf.reshape(u, [-1, self.option_size * self.embed_size]) # [batch_size, option_size * embed_size]

            logits = self._fc(a_hat, self.option_size, 'fc2')

            selection = tf.argmax(logits, 1)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot, logits=logits)
            loss = tf.reduce_mean(cross_entropy)
            
            self.onehot = logits


            class Handle(object):
                pass

            handle = Handle()
            handle.sentences = sentences
            handle.query = query
            handle.answer = answer
            handle.selection = selection

            return handle, loss


    def build_sampler(self, sentences=None, query=None):

        if sentences == None:
            sentences = tf.placeholder(tf.int32, [None, self.memory_size, self.sentence_size], name='sentences')

        if query == None:
            query = tf.placeholder(tf.int32, [None, self.option_size, self.sentence_size], name='query')

        with tf.variable_scope('MemN2N'):
            emb_q = self._query_embedding(query) # [batch_size, option_size, sentence_size, embed_size]
            u = tf.reduce_sum(emb_q*self._encoding, 2) # [batch_size, option_size, embed_size]

            for hop in range(self.n_hop):
                emb_i = self._inputs_embedding(sentences, hop) # [batch_size, memory_size, sentence_size, embed_size]
                mem_i = tf.reduce_sum(emb_i*self._encoding, 2) # [batch_size, memory_size, embed_size]
                mem_i = tf.expand_dims(mem_i, 1) # [batch_size, 1, memory_size, embed_size]

                emb_o = self._outputs_embedding(sentences, hop) # same as emb_i
                mem_o = tf.reduce_sum(emb_o*self._encoding, 2) # same as mem_i
                mem_o = tf.expand_dims(mem_o, 1) # [batch_size, 1, memory_size, embed_size]
                
                uT = tf.transpose(tf.expand_dims(u, -1), [0, 1, 3, 2])
                # [batch_size * option_size, embed_size, 1] -> [batch_size, option_size, 1, embed_size]

                p = tf.nn.softmax(tf.reduce_sum(mem_i*uT, 3)) # inner product [batch_size, option_size, memory_size]
                p = tf.expand_dims(p, -1) # [batch_size, option_size, memory_size, 1]

                o = tf.reduce_sum(mem_o*p, 2) # [batch_size, option_size, embed_size]

                u = o + u # [batch_size, option_size, embed_size]

            #logits = tf.nn.softmax(self._unembedding(u)) #a_hat [batch_size * option_size, vocab_size]
            a_hat = tf.reshape(u, [-1, self.option_size * self.embed_size]) # [batch_size, option_size * embed_size]

            logits = self._fc(a_hat, self.option_size, 'fc2')

            selection = tf.argmax(logits, 1)

            class Handle(object):
                pass

            handle = Handle()
            handle.sentences = sentences
            handle.query = query
            handle.selection = selection

            return handle, selection



