# -*- coding: UTF-8 -*-
import tensorflow as tf
from seq2seq_model import Seq2SeqModel


batch_size =  1280
sequence_length = 50
hidden_size = 256
num_layers = 2
num_encoder_symbols = 5004  # 'UNK' and '<go>' and '<eos>' and '<pad>'
num_decoder_symbols = 5004
embedding_size = 256 # 词向量维度
learning_rate = 0.001

def main():
    epochs = 50
    graph = tf.Graph()
    with graph.as_default():
        model = Seq2SeqModel(hidden_size,num_layers,batch_size,sequence_length,embedding_size,
                             learning_rate,num_encoder_symbols,num_decoder_symbols,'false')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(graph=graph, config=config) as session:
            model.train(session,epochs, epoch= 16)

if __name__== '__main__':
    main()


