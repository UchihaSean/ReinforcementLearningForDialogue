# This Python file uses the following encoding: utf-8
import tensorflow as tf
import os
class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", VALID_ACTIONS = None, word_embedding = None, sequence_length = None):
        self.scope = scope
        self.VALID_ACTIONS = VALID_ACTIONS
        self.sequence_length = sequence_length
        self.word_embedding = word_embedding

        with tf.variable_scope(scope):
            self.build_model()

    def build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.X_pl = tf.placeholder(shape=[None, self.sequence_length], dtype=tf.int32, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        batch_size = tf.shape(self.X_pl)[0]

        self.W = tf.get_variable(name="embeddings", shape=self.word_embedding.shape, dtype=tf.float32,
                                 initializer=tf.constant_initializer(self.word_embedding), trainable=True)

        embedding = tf.nn.embedding_lookup(self.W, self.X_pl)
        embedding_mean = tf.reduce_mean(embedding, axis=1)

        # Fully connected layers
        fc1 = tf.contrib.layers.fully_connected(embedding_mean, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(self.VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        self.gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), self.gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())


    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        return sess.run(self.predictions, {self.X_pl: s})

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
        global_step, _, loss = sess.run(
            [ tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        # actions_pl,gather_indices, action_predictions=sess.run(
        #     [self.actions_pl,self.gather_indices,self.action_predictions],
        #     feed_dict)
        # print(actions_pl, gather_indices, action_predictions)
        return loss