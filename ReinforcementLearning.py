# This Python file uses the following encoding: utf-8
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque, namedtuple
from StateProcessor import StateProcessor
from Estimator import Estimator
from ModelParametersCopier import ModelParametersCopier
from Environment import Environment
import QA.Data as Data
from QA.TFIDF import TFIDF
from QA.CNN_train import CNN
from Chit_Chat.seq2seq import seq2seq
import csv
import sys


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    num_episodes,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    VALID_ACTIONS=None,
                    save_model_every=1000,
                    save_model_start=0
                    ):
    Transition = namedtuple("Transition", ["state_id", "action", "reward", "next_state_id", "done"])

    # The replay memory
    replay_memory = []

    # Make model copier object
    estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

    # Restore the model
    saver = tf.train.Saver()
    if save_model_start > 0:
        saver.restore(sess, "./model/Rl/model" + str(save_model_start) + ".ckpt")

    # Get the current time step
    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    state = env.reset()
    state_processor.set_embedding(q_estimator.W)

    for i in range(replay_memory_init_size):
        state_id = state_processor.generate_state_id(state)
        action_probs = policy(sess, state_id, epsilons[min(total_t, epsilon_decay_steps - 1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(sess, VALID_ACTIONS[action])
        next_state_id = state_processor.generate_state_id(next_state)

        replay_memory.append(Transition(state_id, action, reward, next_state_id, done))
        if done:
            state = env.reset()

        else:
            state = next_state

    print("Replay memory initial finished")

    for i_episode in range(num_episodes):

        # Reset the environment
        state = env.reset()

        loss_set = []

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                estimator_copy.make(sess)
                print("\nCopied model parameters to target network.")

            # Take a step in the environment
            state_id = state_processor.generate_state_id(state)
            action_probs = policy(sess, state_id, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(sess, VALID_ACTIONS[action])
            next_state_id = state_processor.generate_state_id(next_state)

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state_id, action, reward, next_state_id, done))

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            state_id_batch, action_batch, reward_batch, next_state_id_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets
            q_values_next = target_estimator.predict(sess, next_state_id_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(
                q_values_next, axis=1)

            # Perform gradient descent update
            state_id_batch = np.array(state_id_batch)
            loss = q_estimator.update(sess, state_id_batch, action_batch, targets_batch)
            loss_set.append(loss)
            total_t += 1
            if done:
                break

            state = next_state

        print("Episode " + str(i_episode) + ", total_t " + str(total_t) + ", loss " + str(np.mean(loss_set)))

        if i_episode % save_model_every == 0:
            saver.save(sess, "./model/RL/model" + str(i_episode / save_model_every) + ".ckpt")


def train():
    """
    Train RL model
    """
    VALID_ACTIONS = [0, 1]
    sequence_length = 50
    save_model_start = 0

    tf.reset_default_graph()

    # Get word embeding
    word_dict, word_embedding = Data.read_single_word_embedding("data/single_word_embedding")

    # Create estimators
    q_estimator = Estimator(scope="q", VALID_ACTIONS=[0, 1], word_embedding=word_embedding,
                            sequence_length=sequence_length)
    target_estimator = Estimator(scope="target_q", VALID_ACTIONS=[0, 1], word_embedding=word_embedding,
                                 sequence_length=sequence_length)

    # State processor
    state_processor = StateProcessor(word_dict, word_embedding, 50)

    # Environment
    env = Environment(file_name="data/jd_chat.txt", state_processor=state_processor, sequence_length=sequence_length)

    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Run it!
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        deep_q_learning(sess=sess,
                        env=env,
                        q_estimator=q_estimator,
                        target_estimator=target_estimator,
                        state_processor=state_processor,
                        num_episodes=6000,
                        replay_memory_size=5000,
                        replay_memory_init_size=200,
                        update_target_estimator_every=3000,
                        epsilon_start=1.0,
                        epsilon_end=0.1,
                        epsilon_decay_steps=5000,
                        discount_factor=0.99,
                        batch_size=64,
                        VALID_ACTIONS=VALID_ACTIONS,
                        save_model_every=200,
                        save_model_start=save_model_start)


def predict(state, q_estimator, state_processor, tfidf, cnn, seq, sequence_length, save_model_start, VALID_ACTIONS,
            epsilon):
    """
    Make prediction for the state
    """

    if len(state) > sequence_length:
        state = state[:sequence_length]

    # ---------------

    # Run it!
    action = 0
    with tf.Session() as sess:
        # sess.run(tf.initialize_all_variables())
        policy = make_epsilon_greedy_policy(
            q_estimator,
            len(VALID_ACTIONS))
        # Restore the model
        saver = tf.train.Saver()
        if save_model_start > 0:
            saver.restore(sess, "./model/Rl/model" + str(save_model_start) + ".ckpt")
        # Take a step in the environment
        state_id = state_processor.generate_state_id(state)
        action_probs = policy(sess, state_id, epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        response = "嗯".decode("utf-8")
        if action == 0:
            top_k = 3
            tfidf_response_id, tfidf_response = tfidf.ask_response(state, top_k=top_k)
            if len(tfidf_response) != 0:
                # cnn_response = cnn.ask_response(state, top_k, tfidf_response_id)
                # response = cnn_response[0]
                response = tfidf_response[0]
        else:
            response = seq.ask_question(state)

    return action, response


def rl_output(input_file_name, output_file_name):
    """
    file output prediction for each state
    """
    input_file = open(input_file_name)
    rl_qs = []
    rl_as = []
    for i, line in enumerate(input_file.readlines()):
        if i % 2 == 0:
            rl_qs.append(line.split(" ")[-1].strip().decode("utf-8"))
        else:
            rl_as.append(line.split(" ")[-1].strip().decode("utf-8"))
        if i > 1000: break
    random.seed(12345)
    VALID_ACTIONS = [0, 1]
    sequence_length = 50
    save_model_start = 22
    epsilon = 0.0
    tf.reset_default_graph()

    # Get word embeding
    word_dict, word_embedding = Data.read_single_word_embedding("data/single_word_embedding")

    # Create estimators
    q_estimator = Estimator(scope="q", VALID_ACTIONS=[0, 1], word_embedding=word_embedding,
                            sequence_length=sequence_length)
    # State processor
    state_processor = StateProcessor(word_dict, word_embedding, 50)

    # Read Data
    questions, pred_questions, answers, pred_answers = Data.read_pred_data("data/pred_jd_chat.csv")
    # Build word --> sentence dictionary
    word_sentence_dict = Data.generate_word_sentence_dict(pred_questions)

    tfidf = TFIDF(questions, pred_questions, answers, pred_answers, word_sentence_dict)
    cnn = CNN(questions, pred_questions, answers, pred_answers, word_sentence_dict, isTrain=False)

    # ---------------

    # Chit chat
    seq = seq2seq()

    # Output
    with open(output_file_name, 'w') as csvfile:
        filednames = ['Question', 'Real Reply', 'Reply', 'Action', 'Score']
        writer = csv.DictWriter(csvfile, fieldnames=filednames)
        writer.writeheader()
        for i in range(100):
            print("i: " + str(i))
            # t = int(np.random.random() * len(questions))
            t = i
            print("question: " + str(rl_qs[t].encode("utf-8")))
            dict = {'Score': "",
                    'Question': rl_qs[t].encode("utf-8"),
                    'Real Reply': rl_as[t].encode("utf-8"),
                    }
            action, reply = predict(rl_qs[t], q_estimator, state_processor, tfidf, cnn, seq, sequence_length,
                                    save_model_start, VALID_ACTIONS, epsilon)
            print("action: " + str(action))
            print("reply: " + str(reply.encode("utf-8")))
            dict["Reply"] = reply.encode("utf-8")
            dict['Action'] = action
            writer.writerow(dict)


def text_communication():
    # Prepare

    random.seed(12345)
    VALID_ACTIONS = [0, 1]
    sequence_length = 50
    save_model_start = 4
    epsilon = 0.0
    tf.reset_default_graph()

    # Get word embeding
    word_dict, word_embedding = Data.read_single_word_embedding("data/single_word_embedding")

    # Create estimators
    q_estimator = Estimator(scope="q", VALID_ACTIONS=[0, 1], word_embedding=word_embedding,
                            sequence_length=sequence_length)
    # State processor
    state_processor = StateProcessor(word_dict, word_embedding, 50)

    # Read Data
    questions, pred_questions, answers, pred_answers = Data.read_pred_data("data/pred_jd_chat.csv")
    # Build word --> sentence dictionary
    word_sentence_dict = Data.generate_word_sentence_dict(pred_questions)

    tfidf = TFIDF(questions, pred_questions, answers, pred_answers, word_sentence_dict)
    # cnn = CNN(questions, pred_questions, answers, pred_answers, word_sentence_dict, isTrain=False)
    cnn = None

    # ---------------

    # Chit chat
    seq = seq2seq()

    sys.stdout.write("> ")
    sys.stdout.flush()

    input_seq = sys.stdin.readline()

    while input_seq:
        if input_seq.strip() == "end": break
        # print(input_seq.strip())
        action, reply = predict(input_seq.strip().decode("utf-8"), q_estimator, state_processor, tfidf, cnn, seq,
                                sequence_length, save_model_start, VALID_ACTIONS, epsilon)
        # print(str(action) + " " + reply.encode("utf-8"))
        print(reply.encode("utf-8"))
        sys.stdout.write("> ")
        sys.stdout.flush()
        input_seq = sys.stdin.readline()


def main():
    # train()
    rl_output("data/jd_chat.txt", "data/rl_output.csv")
    # text_communication()


if __name__ == "__main__":
    main()
