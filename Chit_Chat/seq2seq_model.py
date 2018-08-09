# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
from word_id import Word_Id_Map
import jieba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Seq2SeqModel(object):
    def __init__(self, hidden_size,layers,batch_size,seq_length,embedding_size,learning_rate,num_encoder_symbols,num_decoder_symbols,is_pred):
        self.hidden_size = hidden_size
        self.layers = layers
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.num_encoder_symbols = num_encoder_symbols
        self.num_decoder_symbols = num_decoder_symbols
        self.is_pred = is_pred

        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length])
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length])

        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length])
        self.weights = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_length])

        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.layers)

        if is_pred=='false':
            self.feed_previous = False
        else:
            self.feed_previous = True
        results,states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            tf.unstack(self.encoder_inputs, axis=1),
            tf.unstack(self.decoder_inputs, axis=1),
            self.cell,
            self.num_encoder_symbols,
            self.num_decoder_symbols,
            self.embedding_size,
            feed_previous=self.feed_previous
        )

        logits = tf.stack(results, axis=1)


        if self.is_pred=='true':
            self.pred = tf.argmax(logits, axis=2)
        else:
            self.loss = tf.contrib.seq2seq.sequence_loss(logits, targets=self.targets, weights=self.weights)
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)



    def train(self,sess,epochs, epoch):
        model_dir = './model'
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        if epoch>0:
            saver.restore(sess, './model/model.ckpt-'+str(epoch))


        while epoch < epochs:
            epoch = epoch + 1
            print("epoch:", epoch)
            train_x, train_y, train_target = loadQA()
            print(len(train_x[0]),len(train_y[0]))
            print(len(train_x)/self.batch_size)
            x = []
            loss_set = []
            for step in range(len(train_x) // self.batch_size):
                print("step:", step)
                train_encoder_inputs = train_x[step * self.batch_size:step * self.batch_size + self.batch_size, :]
                train_decoder_inputs = train_y[step * self.batch_size:step * self.batch_size + self.batch_size, :]
                # print(train_encoder_inputs[0])
                # print(train_decoder_inputs[0])
                ########## adjust weights
                train_weights = np.ones(shape=[self.batch_size, self.seq_length], dtype=np.float32)
                train_weights = adjust_weights(train_weights, train_decoder_inputs)
                ##########

                train_targets = train_target[step * self.batch_size:step * self.batch_size + self.batch_size, :]
                # print(train_targets[0])
                op = sess.run(self.train_op, feed_dict={self.encoder_inputs: train_encoder_inputs, self.targets: train_targets,
                                                   self.weights: train_weights, self.decoder_inputs: train_decoder_inputs})
                cost = sess.run(self.loss, feed_dict={self.encoder_inputs: train_encoder_inputs, self.targets: train_targets,
                                                 self.weights: train_weights, self.decoder_inputs: train_decoder_inputs})
                if step % 50 ==0:
                    loss_set.append(cost)
                    x.append(step/10)
                print("loss:", cost)

            # if epoch % 10 == 0:
            saver.save(sess, model_dir + '/model.ckpt', global_step=epoch)
            plt.plot(x, loss_set)
            plt.savefig("data/loss"+str(epoch)+".png")

    def test(self,sess,qsentence, epoch):
        # print("Enter")
        saver = tf.train.Saver()
        module_file = "./model/Chit_chat/model.ckpt-"+str(epoch)
        saver.restore(sess, module_file)
        map = Word_Id_Map()
        encoder_input = map.sentence2ids(cut_word(qsentence))
        # print(encoder_input)

        encoder_input = encoder_input + [3 for i in range(0, self.seq_length - len(encoder_input))]
        encoder_input = np.asarray([np.asarray(encoder_input)])
        decoder_input = np.zeros([1, self.seq_length])
        # print('encoder_input : ', encoder_input)
        # print('decoder_input : ', decoder_input)
        pred_value = sess.run(self.pred, feed_dict={self.encoder_inputs: encoder_input, self.decoder_inputs: decoder_input})
        # print(pred_value[0])
        sentence = map.ids2sentence(pred_value[0])
        # print(sentence)
        sentence = sentence_decode(sentence)
        return sentence

def adjust_weights(weights, inputs):
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            if inputs[i][j] == 3 or inputs[i][j] == 1:
                weights[i][j] = 0.0
    return weights


def sentence_decode(sentence):
    string = ""
    for i in range(len(sentence)):
        if sentence[i] == '<eos>':
            # string+="."
            break
        if sentence[i] == '<go>' or sentence[i] =='<pad>' or sentence[i] =='unk':
            continue
        string+= sentence[i]
    return string

def loadQA():
    train_x = np.load('./data/idx_q.npy', mmap_mode='r')
    train_y = np.load('./data/idx_a.npy', mmap_mode='r')
    train_target = np.load('./data/idx_o.npy', mmap_mode='r')
    # print(train_x[1])
    # print(train_y[1])
    # print(train_target[1])
    return train_x, train_y, train_target


def cut_word(sentence):
    # sentence = sentence.lower()
    # seg_list = jieba.cut(sentence)
    # return tf.compat.as_str("/".join(seg_list)).split('/')
    seg_list = []
    for word in sentence:
        # print(word)
        seg_list.append(word)
    return seg_list

def main():
    loadQA()

if __name__ == "__main__":
    main()