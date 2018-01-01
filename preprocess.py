#!/usr/bin/python3

import os
import time
import string
import numpy as np
import _pickle as cPickle
import argparse


from tqdm import tqdm
from tqdm import trange

#====================

blank = 'XXXXX'
data_path = 'AI_Course_Final/CBTest/data'
output_path = 'questions/'
encode_map_path = 'enc_map.pkl'
decode_map_path = 'dec_map.pkl'
vocab_path = 'vocab.pkl'
vocab_threshold = 1

exclusive_vocab = ['', 'XXXXX']

train_list = ['cbtest_CN_train.txt', 
            'cbtest_NE_train.txt', 
            'cbtest_P_train.txt', 
            'cbtest_V_train.txt']

valid_list = ['cbtest_CN_valid_2000ex.txt',
            'cbtest_NE_valid_2000ex.txt',
            'cbtest_P_valid_2000ex.txt',
            'cbtest_V_valid_2000ex.txt']

test_list = ['cbtest_CN_test_2500ex.txt',
            'cbtest_NE_test_2500ex.txt',
            'cbtest_P_test_2500ex.txt',
            'cbtest_V_test_2500ex.txt']

#====================

parser = argparse.ArgumentParser(description='Rreprocessing Input Data')
parser.add_argument('-p', '--progress', type=str, nargs='+', default=['all'])

args = parser.parse_args()

#====================

if not os.path.exists(output_path):
    os.makedirs(output_path)

def read_data(filename):

    def is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    lines = []
    with open(filename, 'r') as file:
        raw_lines = file.readlines()
        for line in tqdm(raw_lines, desc='Read', ncols=80):
            if is_int(line.split(' ', 1)[0]):
                lines.append(line)

    return lines

def parse_questions(lines):
    question_list = []
    assert(len(lines) % 21 == 0)
    exc = set(string.punctuation)
    def parse(lines):
        sent = []
        def filter(line):
            line = ''.join(ch if ch not in exc else ' ' for ch in line)
            return ' '.join(line.split())

        for i in range(20):
            line = lines[i].split(' ', 1)[1]
            sent.append( filter(line) )
        last = [x for x in lines[-1].split('\t') if x!='' ]
        q = filter(last[0].split(' ', 1)[1])
        a = ''.join(ch for ch in last[1] if ch not in exc)
        option = []
        for x in last[2].split('|'):
            o = ''.join(ch for ch in x.replace('\n', '') if ch not in exc)
            if o != '':
                option.append(o)


        question = {'sentences': sent, 
                    'question': q, 
                    'answer': a, 
                    'options': option }
        return question

    for i in trange(len(lines) // 21, desc='Parsing', ncols=80):
        question_list.append(parse(lines[ i*21 : (i+1)*21]))

    return question_list


def create_vocab(qs, exc=[], thres=50):
    vocab = {}

    def add(word):
        word = word.lower()
        if word in exc:
            return
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

    for q in tqdm(qs, desc='vocab', ncols=80):
        for line in q['sentences']:
            for word in line.split():
                add(word)
        for word in q['question'].split():
            add(word)
        for word in q['options']:
            add(word)
        add(q['answer'])

    return vocab


def build_mapping(vocab, thres=50):
    def add(enc_map, dec_map, voc):
        enc_map[voc] = len(dec_map)
        dec_map[len(dec_map)] = voc
        return enc_map, dec_map

    enc_map, dec_map = {}, {}
    for voc in ['<st>', '<ed>', '<rare>']:
        enc_map, dec_map = add(enc_map, dec_map, voc)
    for voc, cnt in tqdm(vocab.items(), desc='map', ncols=80):
        if cnt < thres:
            enc_map[voc] = enc_map['<rare>']
        else:
            enc_map, dec_map = add(enc_map, dec_map, voc)

    return enc_map, dec_map
    

def encode(word, e_map):
    word = word.lower()
    return e_map[word]

def encode_str(sent, e_map):
    ids = [e_map[x.lower()] for x in sent.split() if x.lower() in e_map ]
    return ids

def decode_str(ids, d_map):
    return ' '.join([d_map[x] for x in ids])


def print_question(q):
    for sent in q['sentences']:
        print(sent)
    print(q['question'])
    print(q['options'])
    print(q['answer'])

if __name__ == '__main__':

    if 'parse' in args.progress or 'all' in args.progress:
        print('Parsing input data...')
        for name in train_list + valid_list + test_list:
            n, _ = os.path.splitext(name)
            filename = os.path.join(data_path, name)
            print('Reading data: ', filename)

            # read input data
            lines = read_data(filename)

            print('Read {} lines'.format(len(lines)))

            print('Parsing data: ', filename)

            # parse to questing format
            question_list = parse_questions(lines)

            print('Result: {}/{} (actual/expected)'.format(len(question_list), len(lines)//21))

            outputfile = os.path.join(output_path, n + '.pkl')
            print('Save to {}'.format(outputfile))

            # save to file
            with open(outputfile, 'wb') as f:
                cPickle.dump(question_list, f)

        print('DONE !!')

    gen_pkl_file = [ os.path.join(output_path, x) 
                        for x in os.listdir(output_path) if x.endswith('.pkl') ]

    if 'dict' in args.progress or 'all' in args.progress:
        print('Generating Dictionary...')

        qs = []
        for file in gen_pkl_file:
            qs.extend(cPickle.load( open(file, 'rb') ))

        # create vocabulary
        try:
            vocab = create_vocab(qs, exc=exclusive_vocab, thres=vocab_threshold)
        except Exception as q:
            print_question(q)   

        print('DONE !!')
        print('Total: {} words'.format(len(vocab)))
        x = np.array(list(vocab.values()))
        print('Thres total (>={}): {} words'.format(vocab_threshold, np.sum(x[(-x).argsort()] >= vocab_threshold)))
        
        print('Save dictionary to {}'.format(vocab_path))
        # save to file
        cPickle.dump(vocab, open(vocab_path, 'wb'))


    vocab = cPickle.load( open(vocab_path, 'rb') )

    if 'map' in args.progress or 'all' in args.progress:

        print('Buliding voc mapping...')

        x = np.array(list(vocab.values()))
        vocab_size = len(vocab)
        vocab_thres_size = np.sum(x[(-x).argsort()] >= vocab_threshold)

        enc_map, dec_map = build_mapping(vocab, vocab_threshold)

        print('Encode map size: {}/{} (actual/expected)'.format(len(enc_map), vocab_size+3))
        print('Decode map size: {}/{} (actual/expected)'.format(len(dec_map), vocab_thres_size+3))

        print('Save encode map to {}'.format(encode_map_path))
        cPickle.dump(enc_map, open(encode_map_path, 'wb'))
        print('Save decode map to {}'.format(decode_map_path))
        cPickle.dump(dec_map, open(decode_map_path, 'wb'))

        print('DONE !!')

    print('')
    print('========== FORMAT ==========')
    print('questions')
    print('  |-- sentences (a string list contains 20 lines of sentences)')
    print('  |-- question (a string which is the main question)')
    print('  |-- answer (a string which is the answer of this question)')
    print('  |-- options (a string list contains multiple options of this question)')


#filename = os.path.join(data_path, train_list[0])
#lines = read_data(filename)
#print('filename: ', filename)
#print('total line: ', len(lines))

#for i in range(21, 42):
#    print(lines[i])

#question_list = parse_questions(lines)
#print('total questions: ', len(question_list))

#question = question_list[1]

#for i in range(20):
#    print('{}: {}'.format(i+1, question['sentences'][i]))
    
#print('Q: {}'.format(question['question']))
#print('Opt: {}'.format(question['options']))
#print('A: {}'.format(question['answer']))

#for name in train_list + valid_list + test_list:
#
#    filename = os.path.join(data_path, name)
#    lines = read_data(filename)
#
#    print('filename: ', filename)
#    print('total line: ', len(lines))
#    for i in range(21):
#        print(lines[i])



