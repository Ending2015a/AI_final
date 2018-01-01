#!/usr/bin/python3
import _pickle as cPickle

path = 'encode/cbtest_NE_train.pkl'

# load question list
questions = cPickle.load(open(path, 'rb'))
# load decode map
decode_map = cPickle.load(open('dec_map.pkl', 'rb'))

def decode(ids, dec_map):
    return dec_map[ids]

def decode_str(ids, dec_map):
    return ' '.join([dec_map[x] for x in ids])



q = questions[0]

sents = q['sentences']
ques = q['question']
ans = q['answer']
opts = q['options']

# print sentences
for idx, sent in enumerate(sents):
    print('{}: {}'.format(idx, decode_str(sent, decode_map)))

# print question
print('Q: {}'.format(decode_str(ques, decode_map)))

# print options
print('opt: {}'.format( [decode(x, decode_map) for x in opts] ))

# print answer
print('A: {}'.format(decode(ans, decode_map)))

