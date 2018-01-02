import os
import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from tqdm import trange
from preprocess import decode_str

class Solver(object):
    def __init__(self, model, data, enc_map, dec_map, **kwargs):

        self.model = model
        self.data = data
        self.enc_map = enc_map
        self.dec_map = dec_map

        self.n_epochs = kwargs.pop('n_epochs', 500)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.save_epoch = kwargs.pop('save_epoch', 1)
        self.print_step = kwargs.pop('print_step', 5)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.restore_path = kwargs.pop('restore_path', None)
        self.eval_epoch = kwargs.pop('eval_epoch', 0)
        self.summary_step = kwargs.pop('summary_step', 10)

        self.sentence_size = model.sentence_size
        self.option_size = model.option_size

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def build_dataset(self, data_list):
        def padding_sentence(sentence, size, padding_word=0):
            length = len(sentence)
            crop = max(length-size, 0)
            pad = max(size-length, 0)
            sentence = sentence[0:length-crop] + [padding_word]*pad
            return sentence

        sents_list = []
        qwos_list = []
        idx_list = []

        for q in tqdm(data_list, desc='gen', ncols=80):
            sents = q['sentences']
            qwos = q['q_with_o']
            idx = q['index']

            for i in range(len(sents)):
                sents[i] = padding_sentence(sents[i], self.sentence_size)
                assert(len(sents[i]) == self.sentence_size)

            for i in range(len(qwos)):
                qwos[i] = padding_sentence(qwos[i], self.sentence_size)
                assert(len(qwos[i]) == self.sentence_size)
            
            sents = sents[0:self.sentence_size]
            qwos = qwos[0:self.option_size]

            assert(len(sents) == 20)
            assert(len(qwos) == 10)

            sents_list.append(sents)
            qwos_list.append(qwos)
            idx_list.append([idx])


        sent, qwo, idx = tf.train.slice_input_producer(
                [sents_list, qwos_list, idx_list], capacity=self.batch_size * 8)

        sent_batch, qwo_batch, idx_batch = tf.train.shuffle_batch(
                [sent, qwo, idx],
                batch_size = self.batch_size,
                num_threads = 8,
                capacity = self.batch_size * 5,
                min_after_dequeue = self.batch_size * 2)

        return sent_batch, qwo_batch, idx_batch


    def train(self):

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        
        print('Generating Dataset...')
        print('This may take a while...')
        train_sent, train_query, train_answer = self.build_dataset(self.data['train']);
        test_sent, test_query, test_answer = self.build_dataset(self.data['val']);

        # train dataset
        n_examples = len(self.data['train'])
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))

        test_examples = len(self.data['val'])
        test_iters_per_epoch = int(np.ceil(float(test_examples)/self.batch_size))

        del self.data

        print('DONE !!')

        print('Building model...')

        train_handle, loss = self.model.build_model(train_sent, train_query, train_answer)
        test_handle, generated_answer = self.model.build_sampler(test_sent, test_query)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
            grads = optimizer.compute_gradients(loss)
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradient', grad)
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        summary_op = tf.summary.merge_all()

        print('DONE !!')

        print('======= INFO =======')
        print('The number of epoch: ', self.n_epochs)
        print('Data size: ', n_examples)
        print('Batch_size', self.batch_size)
        print('Iterations per epoch: ', n_iters_per_epoch)
        print('')

        print('Start Session...')
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

            print('Start queue runners...')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            print('Try to restore model...')
            if self.restore_path is not None:
                latest_ckpt = tf.train.latest_checkpoint(self.restore_path)
                if not latest_ckpt:
                    print('Not found any checkpoint in ', self.restore_path)
                else:
                    print('Found pretrained model ', latest_ckpt)
                    saver.restore(sess, latest_ckpt)

            prev_loss = -1
            curr_loss = 0

            print('Start training !!!')

            try:
                save_point = 0
                for e in range(self.n_epochs):
                    start_epoch_time = time.time()
                    start_iter_time = time.time()
                    for i in range(n_iters_per_epoch):

                        op = [global_step, train_handle.query, train_handle.selection, train_handle.answer, loss, train_op]
                        step_, q_, s_, a_, loss_, _ = sess.run(op)

                        curr_loss += loss_

                        if (i+1) % self.summary_step == 0:
                            summary = sess.run(summary_op)
                            summary_writer.add_summary(summary, global_step=step_)

                        if (i+1) % self.print_step == 0:
                            elapsed_iter_time = time.time() - start_iter_time
                            print('[epoch {} | iter {}/{} | step {} | save point {}] loss: {:.5f}, elapsed time: {:.4f}'.format(
                                    e+1, i+1, n_iters_per_epoch, step_, save_point, loss_, elapsed_iter_time))

                            _selection = decode_str(q_[0][int(s_[0])], self.dec_map)
                            _answer = decode_str(q_[0][int(a_[0])], self.dec_map)

                            print('  Answer: {}, {}'.format(int(a_[0]), _answer))
                            print('  Select: {}, {}'.format(int(s_[0]), _selection))

                            start_iter_time = time.time()

                    print('  [epoch {0} | iter {1}/{1} | step {2} | save point {6}] End. prev loss: {3:.5f}, cur loss: {4:.5f}, elapsed time: {5:.4f}'.format(
                                e+1, n_iters_per_epoch, step_, prev_loss, curr_loss, time.time() - start_epoch_time, save_point))

                    prev_loss = curr_loss
                    curr_loss = 0


                    if (e+1) % self.save_epoch == 0:
                        save_point = step_
                        saver.save(sess, os.path.join(self.model_path, 'model'), global_step=global_step)
                        print('model-%s saved. ' % (step_))

            except KeyboardInterrupt:
                print('Interrupt!!')
            finally:
                print('End training step, saving model')
                saver.save(sess, os.path.join(self.model_path, 'model'), global_step=global_step)
                print('model-%s saved.' % (step_))

                coord.request_stop()
                coord.join(threads)



    def test(self):
        pass



