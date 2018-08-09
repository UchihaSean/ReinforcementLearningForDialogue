# This Python file uses the following encoding: utf-8
from ActionInference import ActionInference
import random
import numpy as np
from StateProcessor import StateProcessor
import QA.Data as Data
import tensorflow as tf

class Environment():
    def __init__(self, file_name, state_processor, sequence_length):
        self.data = []
        self.session = -1
        self.session_step = -1
        self.ActionInference = ActionInference()
        self.state_processor = state_processor
        self.sequence_length = sequence_length
        session_id = 0
        session_data = []

        # Generate data for all sessions
        file = open(file_name)
        for line in file.readlines():
            lis = line.strip().split(" ")
            if int(lis[2]) != session_id and session_id!=0:
                self.data.append(session_data)
                session_data = []

            session_id = int(lis[2])
            session_data.append(lis[-1].decode("utf-8"))

    def next_session_data(self):
        """
        Generate data of the next session
        """
        self.session = int(np.random.random()*len(self.data))

        return self.data[self.session]

    def reset(self):
        """
        Initial with a new session
        """
        session_data = self.next_session_data()
        self.session_step = 0
        return session_data[0]
    def step(self, sess, action):
        """
        Go ahead one step
        return next state, reward and done
        """
        next_state, reward, done = None, None, False
        positive_reward = 1
        negative_reward = -1

        # Question and answer
        state = self.data[self.session][self.session_step]
        positive_reply = self.data[self.session][self.session_step+1]
        negative_reply = random_sample_reply(self.data)

        # Next step
        self.session_step+=2

        # End of session
        if self.session_step >= len(self.data[self.session])-1:
            done = True

        # Not end, update next state
        if not done:
            next_state = self.data[self.session][self.session_step]


        # Make action
        if len(state)>self.sequence_length:
            state= state[:self.sequence_length]
        action_reply = self.ActionInference.make_action(state, action)

        # Reward calculate
        # reward = positive_reward * self.score(sess,action_reply, positive_reply)+ negative_reward * self.score(sess,action_reply, negative_reply)
        reward = positive_reward * self.score(sess, action_reply, positive_reply)
        # print(state)
        # print(action_reply)
        # print(action,reward)

        return next_state, reward, done, action_reply

    def score(self, sess, reply_x, reply_y):
        """
        Calculate similarity between two replies
        """
        # embedding = self.state_processor.generate_state_embedding(sess,reply_x)
        reply_x_embedding =  average(self.state_processor.generate_state_embedding(sess,reply_x))
        reply_y_embedding = average(self.state_processor.generate_state_embedding(sess,reply_y))
        similarity = cosine_similarity(reply_x_embedding, reply_y_embedding)
        return similarity

def average(x):
    pad_word_embedding = list(np.zeros(300))
    word_embedding = []
    for i in range(len(x)):
        if (x[i] == pad_word_embedding).all():
            break
        word_embedding.append(x[i])
    # print("length = "+str(len(word_embedding)))
    if word_embedding==[]: return pad_word_embedding
    return np.average(word_embedding)




def cosine_similarity(x, y):
    """
    Cosine similarity score
    """
    mul = np.sqrt(np.dot(x,x)*np.dot(y,y))
    if mul == 0: return 0
    return np.dot(x,y)/mul

def random_sample_reply(data):
    session= int(len(data) * random.random())
    session_step = int(len(data[session]) * random.random())
    return data[session][session_step]





def main():
    # Get word embeding
    word_dict, word_embedding = Data.read_single_word_embedding("data/single_word_embedding")

    # State processor
    state_processor = StateProcessor(word_dict, word_embedding, 50)
    # env = Environment("data/jd_chat.txt", state_processor)
    # print(env.step(0))

if __name__ == '__main__':
    main()
