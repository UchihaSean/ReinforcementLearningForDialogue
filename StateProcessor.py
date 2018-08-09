# This Python file uses the following encoding: utf-8
import tensorflow as tf
class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    def __init__(self, word_dict, word_embedding, sequence_length):
        self.word_dict = word_dict
        self.word_embedding = word_embedding
        self.sequence_length = sequence_length
        self.state_id = tf.placeholder(shape=[self.sequence_length], dtype=tf.int32)
        self.state_embedding = tf.nn.embedding_lookup(self.word_embedding, self.state_id)

    def generate_state_embedding(self, sess, state):
        """
        state to embedding
        """
        state_id = self.generate_state_id(state)

        return sess.run(self.state_embedding, {self.state_id: state_id})

    def generate_state_id(self, state):
        """
        state to id
        """
        state_id = [0 for _ in range(self.sequence_length)]
        if state==None: return state_id
        for i, word in enumerate(state):
            # print(i, word)
            if i== self.sequence_length: break
            if word in self.word_dict:
                state_id[i] = self.word_dict[word]
        # print(state_id)

        return state_id


    def set_embedding(self, word_embedding):
        self.word_embedding = word_embedding