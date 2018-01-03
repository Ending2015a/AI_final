import _pickle as cPickle
from model import MemNet
from solver import Solver
from preprocess import get_encoded_train_data
from preprocess import get_encoded_validation_data
from preprocess import get_encoded_test_data

#=============================
encode_map = 'enc_map.pkl'
decode_map = 'dec_map.pkl'
#=============================

def main():

    print("Restoring data...")
    dataset = {'train': get_encoded_train_data(),
                'val': get_encoded_validation_data()}

    print('Restoring map...')
    enc_map = cPickle.load(open(encode_map, 'rb'))
    dec_map = cPickle.load(open(decode_map, 'rb'))
    vocab_size = len(dec_map)

    print('Bulid Model...')
    model = MemNet(vocab_size = vocab_size,
                    embed_size = 512,
                    n_hop = 10,
                    memory_size = 20,
                    sentence_size = 216,
                    option_size = 10)

    print('Bulid Solver...')
    solver = Solver(model, dataset, enc_map, dec_map,
                    n_epochs = 500,
                    batch_size = 32,
                    learning_rate = 0.01,
                    log_path = './log/',
                    model_path = './checkpoint/',
                    restore_path = './checkpoint/',
                    eval_epoch = 1,
                    save_epoch = 1,
                    print_step = 5,
                    summary_step = 10)

    solver.train()

if __name__ == '__main__':
    main()
