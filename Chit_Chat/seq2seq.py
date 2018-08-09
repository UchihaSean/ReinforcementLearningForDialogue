# -*- coding: UTF-8 -*-
import tensorflow as tf
from seq2seq_model import Seq2SeqModel

class seq2seq():
    def __init__(self, sequence_length = 50, batch_size = 128, hidden_size = 256, num_layers=2, num_encoder_symbbols = 5004,
                 num_decoder_symbols=5004, embedding_size = 256, learning_rate = 0.001):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_encoder_symbols = num_encoder_symbbols  # 'UNK' and '<go>' and '<eos>' and '<pad>'
        self.num_decoder_symbols = num_decoder_symbols
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate


    def ask_question(self, question):
        graph = tf.Graph()
        with graph.as_default():
            model = Seq2SeqModel(self.hidden_size, self.num_layers, self.batch_size, self.sequence_length, self.embedding_size,
                                 self.learning_rate, self.num_encoder_symbols, self.num_decoder_symbols, 'true')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(graph=graph, config=config) as session:
                reply = model.test(session, question, epoch=15)

        return reply

def main():
    seq = seq2seq()
    print(seq.ask_question("你好"))

if __name__ == "__main__":
    main()